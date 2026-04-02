"""
backend/ace_step.py — ACE-Step wrapper using subprocess isolation.

ACE-Step runs in its own venv (.venv_model) to avoid dependency conflicts
with the UI venv. All generation is done by spawning _ace_worker.py
in the model venv's Python interpreter.

All file paths come from caller (resolved via config.py). Nothing hardcoded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


def _find_model_python(version: str = "v1") -> str:
    """
    Locate Python in the appropriate model venv.
    version="v1"  -> .venv_model  (Python 3.10)
    version="v1.5" -> .venv_model_v15 (Python 3.11)
    Falls back to current interpreter if not found.
    """
    from app.config import cfg
    ver = version or cfg.ace_step_version

    repo_root = Path(__file__).parent.parent.parent
    if ver == "v1.5":
        candidates = [
            repo_root / ".venv_model_v15" / "Scripts" / "python.exe",
            repo_root / ".venv_model_v15" / "bin" / "python",
        ]
        venv_name = ".venv_model_v15"
    else:
        candidates = [
            repo_root / ".venv_model" / "Scripts" / "python.exe",
            repo_root / ".venv_model" / "bin" / "python",
            repo_root / ".venv_model" / "bin" / "python3",
        ]
        venv_name = ".venv_model"

    for c in candidates:
        if c.exists():
            return str(c)
    log.warning(f"{venv_name} not found — using current interpreter")
    return sys.executable


def _worker_script() -> str:
    return str(Path(__file__).parent / "_ace_worker.py")


class ACEStepPipeline:
    """
    Proxy that runs ACE-Step in an isolated model venv via subprocess.
    The UI venv stays lean; only soundfile/numpy/librosa live there.
    """

    def __init__(self, model_id: str, device: str, models_dir: str) -> None:
        from app.config import cfg
        self.model_id   = model_id
        self.device     = device
        self.models_dir = models_dir
        self.version    = cfg.ace_step_version
        self._python    = _find_model_python(self.version)
        log.info(f"ACE-Step version: {self.version}, python: {self._python}")
        self._loaded    = self._check_available()

    def _check_available(self) -> bool:
        try:
            result = subprocess.run(
                [self._python, "-c", "import acestep; print('ok')"],
                capture_output=True, text=True, timeout=15,
                env={**os.environ, "PYTHONUTF8": "1"},
            )
            ok = result.returncode == 0 and "ok" in result.stdout
            if ok:
                log.info(f"ACE-Step available via {Path(self._python).parent.parent.name}")
            else:
                log.warning("ACE-Step not found in model venv — stub mode")
            return ok
        except Exception as exc:
            log.warning(f"ACE-Step check failed: {exc}")
            return False

    def load(self, progress_cb=None) -> bool:
        self._loaded = self._check_available()
        return self._loaded

    def unload(self) -> None:
        pass  # subprocess manages its own memory

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def vram_used_gb(self) -> float:
        try:
            r = subprocess.run(
                [self._python, "-c",
                 "import torch; print(torch.cuda.memory_allocated()/1e9 "
                 "if torch.cuda.is_available() else 0)"],
                capture_output=True, text=True, timeout=5,
                env={**os.environ, "PYTHONUTF8": "1"},
            )
            return float(r.stdout.strip())
        except Exception:
            return 0.0

    # -----------------------------------------------------------------------
    # Core subprocess runner
    # -----------------------------------------------------------------------

    def _run_worker(
        self,
        payload: dict,
        total_variations: int,
        progress_cb: Optional[ProgressCallback],
    ) -> list[GenerationResult]:

        if not self._loaded:
            log.info("Stub mode — writing silent audio")
            return self._stub_results(payload, total_variations)

        payload_json = json.dumps(payload)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        results: list[GenerationResult] = []

        try:
            proc = subprocess.Popen(
                [self._python, "-u", _worker_script(), payload_json],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                env={**env, "PYTHONUNBUFFERED": "1"},
            )

            # Drain stderr in a background thread to prevent pipe-buffer deadlock
            # on Windows (64KB buffer fills quickly with model loading output).
            stderr_lines: list[str] = []

            def _drain_stderr():
                for line in proc.stderr:
                    stripped = line.strip()
                    if stripped:
                        stderr_lines.append(stripped)
                        # Surface important worker stderr lines as INFO
                        low = stripped.lower()
                        if any(k in low for k in ("error", "warning", "downloading", "loading")):
                            log.info(f"[worker] {stripped}")
                        else:
                            log.debug(f"[worker] {stripped}")

            import threading as _threading
            stderr_thread = _threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            for raw in proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    log.debug(f"[worker] {line}")
                    continue

                t = msg.get("type")
                if t == "progress" and progress_cb:
                    msg_text = msg.get("message", "")
                    if msg_text:
                        log.info(f"[gen] {msg_text}")
                    progress_cb(GenerationProgress(
                        variation_index  = msg.get("variation_index", 1),
                        total_variations = msg.get("total_variations", total_variations),
                        step             = msg.get("step", 0),
                        total_steps      = msg.get("total_steps", 50),
                        message          = msg_text,
                    ))
                elif t == "result":
                    results.append(GenerationResult(
                        variation_index = msg["variation_index"],
                        audio_path      = msg["audio_path"],
                        seed            = msg.get("seed", 0),
                        duration_secs   = msg.get("duration_secs", 0.0),
                        mode            = GenerationMode(msg.get("mode", "text")),
                        style_prompt    = msg.get("style_prompt", ""),
                    ))
                elif t == "error":
                    log.error(f"Worker: {msg.get('message')}")

            proc.wait()
            stderr_thread.join(timeout=2)

            if not results:
                log.warning("Worker returned no results — using stubs")
                log.debug(f"[worker stderr tail] {stderr_lines[-5:] if stderr_lines else '(none)'}")
                return self._stub_results(payload, total_variations)
            
            log.info(f"Worker returned {len(results)} result(s): {[r.audio_path for r in results]}")

        except Exception as exc:
            log.error(f"Subprocess failed: {exc}")
            return self._stub_results(payload, total_variations)

        return results

    # -----------------------------------------------------------------------
    # Public generation methods
    # -----------------------------------------------------------------------

    def generate_text(self, request: TextGenerationRequest,
                      progress_cb: Optional[ProgressCallback] = None) -> list[GenerationResult]:
        return self._run_worker({
            "mode": "text",
            "style_prompt": request.style_prompt,
            "lyrics": request.user_lyrics,
            "lyrics_mode": request.lyrics_mode.value,
            "duration": request.duration_secs,
            "variations": request.variations,
            "seed": request.seed,
            "lora": request.lora,
            "output_dir": request.output_dir,
            "model_id": self.model_id,
            "models_dir": self.models_dir,
            "device": self.device,
        }, request.variations, progress_cb)

    def generate_cover(self, request: CoverRequest,
                       progress_cb: Optional[ProgressCallback] = None) -> list[GenerationResult]:
        return self._run_worker({
            "mode": "cover",
            "source_audio": request.source_audio_path,
            "target_style": request.target_style,
            "new_lyrics": request.new_lyrics,
            "follow_strength": request.follow_strength,
            "variations": request.variations,
            "seed": request.seed,
            "output_dir": request.output_dir,
            "model_id": self.model_id,
            "models_dir": self.models_dir,
            "device": self.device,
        }, request.variations, progress_cb)

    def generate_vocal_backing(self, request: VocalBackingRequest,
                                progress_cb: Optional[ProgressCallback] = None) -> list[GenerationResult]:
        return self._run_worker({
            "mode": "vocal",
            "vocal_audio": request.vocal_audio_path,
            "backing_style": request.backing_style,
            "variations": request.variations,
            "seed": request.seed,
            "output_dir": request.output_dir,
            "model_id": self.model_id,
            "models_dir": self.models_dir,
            "device": self.device,
        }, request.variations, progress_cb)

    def repair(self, request: RepairRequest,
               progress_cb: Optional[ProgressCallback] = None) -> GenerationResult:
        results = self._run_worker({
            "mode": "repair",
            "source_audio": request.source_audio_path,
            "repair_mode": request.mode.value,
            "region_start": request.region_start_sec,
            "region_end": request.region_end_sec,
            "hint_prompt": request.hint_prompt,
            "output_dir": request.output_dir,
            "model_id": self.model_id,
            "models_dir": self.models_dir,
            "device": self.device,
        }, 1, progress_cb)
        return results[0] if results else self._stub_results({"mode": "repair", "output_dir": request.output_dir}, 1)[0]

    # -----------------------------------------------------------------------
    # Stub fallback (model unavailable)
    # -----------------------------------------------------------------------

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
            sf.write(str(path), __import__('numpy').zeros((int(sr * min(duration, 10)), 2), dtype='float32'), sr)
            results.append(GenerationResult(
                variation_index = i,
                audio_path      = str(path),
                seed            = seed,
                duration_secs   = min(duration, 10.0),
                mode            = GenerationMode(mode),
                style_prompt    = payload.get("style_prompt", ""),
            ))
        return results
