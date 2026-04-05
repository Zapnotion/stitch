"""
backend/lora_manager.py — Smart LoRA registry, auto-selection, and download management.

The registry (app/data/lora_registry.json) lists known ACEStep 1.5 compatible
LoRAs with keyword triggers.  The auto-selector scores each LoRA against the
user's style prompt and returns the best match(es) above a confidence threshold.

Download uses huggingface_hub (already installed as an ACEStep dependency).
Each downloaded LoRA lives in  models_dir/loras/<lora_id>/ as:
    adapter_model.safetensors
    adapter_config.json          (synthesised if not provided by repo)
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

from app.backend.logger import log

# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

_REGISTRY_PATH = Path(__file__).parent.parent / "data" / "lora_registry.json"


def load_registry() -> dict:
    try:
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning(f"[lora_manager] Could not load registry: {exc}")
        return {"loras": [], "checkpoints": []}


# ---------------------------------------------------------------------------
# Auto-selection
# ---------------------------------------------------------------------------

def score_lora(lora: dict, prompt: str) -> float:
    """
    Return a match score 0.0–1.0 for a LoRA against a style prompt.

    Scoring strategy:
    - Multi-word trigger phrases score higher than single words (specificity bonus).
    - Whole-word boundary matching avoids 'pop' matching inside 'topography'.
    - Score is capped at 1.0.
    """
    if not prompt:
        return 0.0
    prompt_lower = prompt.lower()
    triggers = lora.get("triggers", [])
    if not triggers:
        return 0.0

    import re as _re
    score = 0.0
    for trigger in triggers:
        t = trigger.lower()
        words = t.split()
        # Use word-boundary matching for single-word triggers to avoid false positives
        if len(words) == 1:
            if _re.search(r'\b' + _re.escape(t) + r'\b', prompt_lower):
                score += 0.2
        else:
            if t in prompt_lower:
                # Multi-word match: more words = higher specificity bonus
                score += 0.2 + (len(words) - 1) * 0.15

    return min(score, 1.0)


def recommended_weight(lora: dict) -> float:
    """Return the registry recommended_weight for a LoRA, defaulting to 0.65."""
    return float(lora.get("recommended_weight", 0.65))


def auto_select_loras(prompt: str, max_loras: int = 2,
                      threshold: float = 0.2) -> list[dict]:
    """
    Return up to max_loras LoRAs whose triggers match the prompt,
    above the confidence threshold, sorted by score descending.
    Returns empty list if nothing matches.
    """
    registry = load_registry()
    scored = []
    for lora in registry.get("loras", []):
        s = score_lora(lora, prompt)
        if s >= threshold:
            scored.append((s, lora))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [lora for _, lora in scored[:max_loras]]


def lora_local_path(lora_id: str, models_dir: str | Path) -> Optional[Path]:
    """
    Return the local path to a downloaded LoRA folder, or None if not downloaded.
    Expected layout: models_dir/loras/<lora_id>/adapter_model.safetensors
    """
    candidate = Path(models_dir) / "loras" / lora_id
    if (candidate / "adapter_model.safetensors").exists():
        return candidate
    return None


def lora_is_downloaded(lora_id: str, models_dir: str | Path) -> bool:
    return lora_local_path(lora_id, models_dir) is not None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

class LoRADownloadError(Exception):
    pass


def download_lora(lora: dict, models_dir: str | Path,
                  progress_cb=None) -> Path:
    """
    Download a LoRA from HuggingFace into models_dir/loras/<lora_id>/.

    progress_cb(message: str) — optional progress callback.

    Returns the local folder path.
    Raises LoRADownloadError on failure.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise LoRADownloadError("huggingface_hub not installed")

    lora_id  = lora["id"]
    hf_repo  = lora["hf_repo"]
    filename = lora.get("filename", "adapter_model.safetensors")
    dest_dir = Path(models_dir) / "loras" / lora_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _cb(msg: str):
        log.info(f"[lora_dl] {msg}")
        if progress_cb:
            progress_cb(msg)

    _cb(f"Downloading {lora['name']} from {hf_repo}…")

    try:
        # Download the weights file
        local_path = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            local_dir=str(dest_dir),
        )
        # Rename to canonical name expected by ACEStep
        canonical = dest_dir / "adapter_model.safetensors"
        if Path(local_path).resolve() != canonical.resolve():
            shutil.copy2(local_path, canonical)

        # Download adapter_config.json if available, else synthesise a minimal one
        try:
            hf_hub_download(
                repo_id=hf_repo,
                filename="adapter_config.json",
                local_dir=str(dest_dir),
            )
            _cb("Downloaded adapter_config.json")
        except Exception:
            _synthesise_adapter_config(dest_dir)
            _cb("Synthesised adapter_config.json (not in repo)")

        _cb(f"Downloaded {lora['name']} → {dest_dir}")
        return dest_dir

    except Exception as exc:
        raise LoRADownloadError(f"Download failed: {exc}") from exc


def _synthesise_adapter_config(dest_dir: Path) -> None:
    """
    Write a minimal PEFT adapter_config.json compatible with ACEStep's loader.
    ACEStep uses PeftConfig.from_pretrained() which needs at least peft_type + r.
    """
    config = {
        "peft_type": "LORA",
        "r": 64,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none",
        "task_type": "FEATURE_EXTRACTION"
    }
    (dest_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# LoRA merging (offline, for combining two LoRAs)
# ---------------------------------------------------------------------------

def merge_loras(lora_paths: list[Path], weights: list[float],
                output_dir: Path, progress_cb=None) -> Path:
    """
    Merge multiple LoRA adapter_model.safetensors files by weighted averaging
    of their tensors.  Writes merged result to output_dir/adapter_model.safetensors.

    weights: list of floats (must sum > 0, will be normalised)
    Returns the output path.

    Requires safetensors library (installed with ACEStep).
    """
    try:
        from safetensors.torch import load_file, save_file
        import torch
    except ImportError:
        raise LoRADownloadError("safetensors / torch not available for merge")

    if len(lora_paths) != len(weights):
        raise ValueError("lora_paths and weights must have same length")
    if not lora_paths:
        raise ValueError("No LoRAs to merge")

    def _cb(msg: str):
        log.info(f"[lora_merge] {msg}")
        if progress_cb:
            progress_cb(msg)

    _cb(f"Merging {len(lora_paths)} LoRA(s)…")

    # Normalise weights
    total = sum(weights)
    if total <= 0:
        raise ValueError("LoRA weights must sum to > 0")
    norm_weights = [w / total for w in weights]

    # Load all tensors
    loaded = []
    for path in lora_paths:
        adapter = path / "adapter_model.safetensors"
        if not adapter.exists():
            raise LoRADownloadError(f"adapter_model.safetensors not found in {path}")
        _cb(f"Loading {path.name}…")
        loaded.append(load_file(str(adapter)))

    # Merge: weighted average on shared keys, include-only for unique keys
    all_keys = set()
    for tensors in loaded:
        all_keys.update(tensors.keys())

    merged: dict = {}
    for key in all_keys:
        present = [(tensors[key], norm_weights[i])
                   for i, tensors in enumerate(loaded)
                   if key in tensors]
        if len(present) == 1:
            merged[key] = present[0][0]
        else:
            # Weighted sum, renormalised by sum of contributing weights
            w_sum = sum(w for _, w in present)
            merged[key] = sum(
                t.float() * (w / w_sum) for t, w in present
            ).to(present[0][0].dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "adapter_model.safetensors"
    save_file(merged, str(out_path))

    # Copy adapter_config.json from first LoRA that has one
    for path in lora_paths:
        cfg = path / "adapter_config.json"
        if cfg.exists():
            shutil.copy2(cfg, output_dir / "adapter_config.json")
            break
    else:
        _synthesise_adapter_config(output_dir)

    _cb(f"Merged LoRA saved → {out_path}")
    return output_dir


# ---------------------------------------------------------------------------
# Checkpoint download
# ---------------------------------------------------------------------------

def download_checkpoint(checkpoint: dict, models_dir: str | Path,
                        progress_cb=None) -> None:
    """
    Download an ACEStep checkpoint (base/turbo/xl) via huggingface_hub snapshot.
    This just triggers the download — ACEStep will load it from models_dir on next run.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise LoRADownloadError("huggingface_hub not installed")

    def _cb(msg: str):
        log.info(f"[checkpoint_dl] {msg}")
        if progress_cb:
            progress_cb(msg)

    hf_repo   = checkpoint["hf_repo"]
    subfolder = checkpoint.get("subfolder", "")
    dest      = Path(models_dir) / checkpoint["id"]

    _cb(f"Downloading {checkpoint['name']} from {hf_repo}…")
    _cb(f"This may take a while (~{checkpoint.get('size_gb', '?')} GB)…")

    try:
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=[f"{subfolder}/*"] if subfolder else None,
            local_dir=str(dest),
            ignore_patterns=["*.msgpack", "flax_model*"],
        )
        _cb(f"Downloaded {checkpoint['name']} → {dest}")
    except Exception as exc:
        raise LoRADownloadError(f"Checkpoint download failed: {exc}") from exc
