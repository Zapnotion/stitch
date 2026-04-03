"""
backend/ace_step_v15.py — ACE-Step 1.5 integration via subprocess worker.

Uses the same architecture as ace_step.py (v1): spawns _ace_worker_v15.py
in .venv_model_v15 (Python 3.11) as a subprocess, communicates via JSON
lines on stdout. This avoids Python version conflicts and is proven to work.

The previous approach (REST API server via `python -m acestep.api_server`)
was invented and does not exist in the real ACE-Step package.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from app.backend.logger import log
from app.models.generation import (
    CoverRequest,
    GenerationMode,
    GenerationProgress,
    GenerationResult,
    RepairRequest,
    TextGenerationRequest,
    VocalBackingRequest,
)

ProgressCallback = Callable[[GenerationProgress], None]

REPO_ROOT = Path(__file__).parent.parent.parent


def _find_v15_python() -> Optional[str]:
    """Find Python in .venv_model_v15."""
    candidates = [
        REPO_ROOT / ".venv_model_v15" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv_model_v15" / "bin" / "python",
        REPO_ROOT / ".venv_model_v15" / "bin" / "python3",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _worker_script() -> str:
    return str(Path(__file__).parent / "_ace_worker_v15.py")


class ACEStepV15:
    """
    Proxy that runs ACE-Step 1.5 in .venv_model_v15 via subprocess,
    using the same JSON-lines protocol as ACEStepPipeline (v1).
    """

    def __init__(self, models_dir: str, device: str = "auto") -> None:
        from app.config import cfg
        self.models_dir = models_dir
        self.device     = device
        self.lm_model   = cfg.lm_model
        self._python    = _find_v15_python()
        self._loaded    = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, progress_cb: Optional[ProgressCallback] = None) -> bool:
        """Check that ACE-Step 1.5's real inference API is importable in the v1.5 venv."""
        if not self._python:
            log.warning("[v1.5] .venv_model_v15 not found — stub mode")
            return False

        # Verify the actual classes used at generation time are all importable.
        # This catches partial installs where acestep itself is present but a
        # dep like matplotlib or numba is still missing.
        check_code = (
            "from acestep.handler import AceStepHandler; "
            "from acestep.llm_inference import LLMHandler; "
            "from acestep.inference import GenerationParams, GenerationConfig, generate_music; "
            "print('ok')"
        )
        try:
            r = subprocess.run(
                [self._python, "-c", check_code],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "PYTHONUTF8": "1"},
            )
            ok = r.returncode == 0 and "ok" in r.stdout
            if ok:
                log.info("[v1.5] ACE-Step 1.5 API verified — ready")
            else:
                err = r.stderr.strip()
                log.warning(f"[v1.5] ACE-Step 1.5 API check failed")
                for line in err.splitlines()[-10:]:
                    if line.strip():
                        log.warning(f"[v1.5]   {line}")
            self._loaded = ok
            return ok
        except Exception as exc:
            log.warning(f"[v1.5] ACE-Step 1.5 check failed: {exc}")
            return False

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def vram_used_gb(self) -> float:
        return 0.0

    # ------------------------------------------------------------------
    # Core subprocess runner (mirrors ACEStepPipeline._run_worker)
    # ------------------------------------------------------------------

    def _run_worker(
        self,
        payload: dict,
        total_variations: int,
        progress_cb: Optional[ProgressCallback],
    ) -> list[GenerationResult]:

        if not self._loaded:
            log.info("[v1.5] Stub mode — writing silent audio")
            return self._stub_results(payload, total_variations)

        payload_json = json.dumps(payload)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               "PYTHONUNBUFFERED": "1",
               "ACESTEP_LM_MODEL_PATH": self.lm_model}
        results: list[GenerationResult] = []

        try:
            proc = subprocess.Popen(
                [self._python, "-u", _worker_script(), payload_json],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                env=env,
            )

            stderr_lines: list[str] = []

            def _drain_stderr():
                for line in proc.stderr:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    stderr_lines.append(stripped)
                    low = stripped.lower()
                    if any(k in low for k in ("error", "traceback", "exception")):
                        log.error(f"[v1.5-worker] {stripped}")
                    elif any(k in low for k in ("warning", "downloading", "loading")):
                        log.info(f"[v1.5-worker] {stripped}")
                    else:
                        log.debug(f"[v1.5-worker] {stripped}")

            t = threading.Thread(target=_drain_stderr, daemon=True)
            t.start()

            for raw in proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    log.debug(f"[v1.5-worker] {line}")
                    continue

                mtype = msg.get("type")
                if mtype == "progress" and progress_cb:
                    msg_text = msg.get("message", "")
                    if msg_text:
                        log.info(f"[v1.5-gen] {msg_text}")
                    progress_cb(GenerationProgress(
                        variation_index  = msg.get("variation_index", 1),
                        total_variations = msg.get("total_variations", total_variations),
                        step             = msg.get("step", 0),
                        total_steps      = msg.get("total_steps", 30),
                        message          = msg_text,
                    ))
                elif mtype == "result":
                    results.append(GenerationResult(
                        variation_index = msg["variation_index"],
                        audio_path      = msg["audio_path"],
                        seed            = msg.get("seed", 0),
                        duration_secs   = msg.get("duration_secs", 0.0),
                        mode            = GenerationMode(msg.get("mode", "text")),
                        style_prompt    = msg.get("style_prompt", ""),
                    ))
                elif mtype == "error":
                    log.error(f"[v1.5-worker] {msg.get('message')}")

            proc.wait()
            t.join(timeout=2)

            if not results:
                log.warning("[v1.5] Worker returned no results — using stubs")
                if stderr_lines:
                    log.warning("[v1.5] Worker stderr:")
                    for line in stderr_lines:
                        log.warning(f"[v1.5]   {line}")
                return self._stub_results(payload, total_variations)

            log.info(f"[v1.5] Worker returned {len(results)} result(s)")

        except Exception as exc:
            log.error(f"[v1.5] Subprocess failed: {exc}")
            return self._stub_results(payload, total_variations)

        return results

    # ------------------------------------------------------------------
    # Public generation methods
    # ------------------------------------------------------------------

    def generate_text(
        self,
        request: TextGenerationRequest,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> list[GenerationResult]:
        return self._run_worker({
            "mode":         "text",
            "style_prompt": request.style_prompt,
            "lyrics":        request.user_lyrics,
            "lyrics_mode":   request.lyrics_mode.value,
            "lyrics_prompt": request.lyrics_prompt,
            "output_type":   request.output_type.value,
            "duration":     request.duration_secs,
            "variations":   request.variations,
            "seed":         request.seed,
            "lora":         request.lora,
            "output_dir":   request.output_dir,
            "models_dir":   self.models_dir,
            "device":       self.device,
        }, request.variations, progress_cb)

    def generate_cover(
        self,
        request: CoverRequest,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> list[GenerationResult]:
        # Derive duration from source audio length if not set
        try:
            import soundfile as sf
            duration = sf.info(request.source_audio_path).duration
        except Exception:
            duration = 30.0
        return self._run_worker({
            "mode":            "cover",
            "source_audio":    request.source_audio_path,
            "target_style":    request.target_style,
            "new_lyrics":      request.new_lyrics,
            "follow_strength": request.follow_strength,
            "duration":        duration,
            "variations":      request.variations,
            "seed":            request.seed,
            "output_dir":      request.output_dir,
            "models_dir":      self.models_dir,
            "device":          self.device,
        }, request.variations, progress_cb)

    def generate_vocal_backing(
        self,
        request: VocalBackingRequest,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> list[GenerationResult]:
        # Derive duration from vocal audio length
        try:
            import soundfile as sf
            duration = sf.info(request.vocal_audio_path).duration
        except Exception:
            duration = 30.0
        return self._run_worker({
            "mode":          "vocal",
            "vocal_audio":   request.vocal_audio_path,
            "backing_style": request.backing_style,
            "duration":      duration,
            "variations":    request.variations,
            "seed":          request.seed,
            "output_dir":    request.output_dir,
            "models_dir":    self.models_dir,
            "device":        self.device,
        }, request.variations, progress_cb)

    def repair(
        self,
        request: RepairRequest,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> GenerationResult:
        results = self._run_worker({
            "mode":         "repair",
            "source_audio": request.source_audio_path,
            "repair_mode":  request.mode.value,
            "region_start": request.region_start_sec,
            "region_end":   request.region_end_sec,
            "hint_prompt":  request.hint_prompt,
            "output_dir":   request.output_dir,
            "models_dir":   self.models_dir,
            "device":       self.device,
        }, 1, progress_cb)
        return results[0] if results else self._stub_results(
            {"mode": "repair", "output_dir": request.output_dir}, 1)[0]

    # ------------------------------------------------------------------
    # Stub fallback
    # ------------------------------------------------------------------

    def _stub_results(self, payload: dict, n: int) -> list[GenerationResult]:
        import numpy as np
        import soundfile as sf

        out_dir  = Path(payload.get("output_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        mode     = payload.get("mode", "text")
        duration = float(payload.get("duration", 10.0))
        results  = []

        for i in range(1, n + 1):
            seed  = int(time.time()) + i
            fname = f"{mode}_v{i:02d}_{seed}_{uuid.uuid4().hex[:6]}.wav"
            path  = out_dir / fname
            sr    = 44100
            sf.write(str(path), __import__('numpy').zeros(
                (int(sr * min(duration, 10)), 2), dtype="float32"), sr)
            results.append(GenerationResult(
                variation_index = i,
                audio_path      = str(path),
                seed            = seed,
                duration_secs   = min(duration, 10.0),
                mode            = GenerationMode(mode),
                style_prompt    = payload.get("style_prompt", ""),
            ))
        return results
