"""
backend/ace_step.py — Wrapper around ACE-Step for all generation modes.

ACE-Step GitHub: https://github.com/ace-step/ACE-Step
All file paths come from caller (resolved via config.py). Nothing hardcoded.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from app.models.generation import (
    CoverRequest,
    GenerationMode,
    GenerationProgress,
    GenerationResult,
    RepairRequest,
    TextGenerationRequest,
    VocalBackingRequest,
)


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[GenerationProgress], None]


# ---------------------------------------------------------------------------
# ACE-Step wrapper
# ---------------------------------------------------------------------------

class ACEStepPipeline:
    """
    Wraps ACE-Step inference.

    The actual ACE-Step package is imported lazily on first use so the app
    starts fast even before models are downloaded.

    If ACE-Step is not yet installed the wrapper logs a clear error and
    returns stub results so the UI stays functional during development.
    """

    def __init__(self, model_id: str, device: str, models_dir: str) -> None:
        self.model_id   = model_id
        self.device     = device
        self.models_dir = Path(models_dir)
        self._pipeline  = None
        self._loaded    = False

    # --- Loading ------------------------------------------------------------

    def load(self, progress_cb: Optional[ProgressCallback] = None) -> bool:
        """Load model into VRAM. Returns True on success."""
        try:
            from acestep.pipeline import ACEStepPipeline as _ACEPipeline  # type: ignore

            self._pipeline = _ACEPipeline.from_pretrained(
                self.model_id,
                cache_dir=str(self.models_dir),
                torch_dtype="auto",
            ).to(self.device)
            self._loaded = True
            return True
        except ImportError:
            print("[ace_step] ACE-Step package not found — running in STUB mode.")
            print("           Install via: pip install acestep")
            self._loaded = False
            return False
        except Exception as exc:
            print(f"[ace_step] Failed to load model: {exc}")
            self._loaded = False
            return False

    def unload(self) -> None:
        """Release VRAM."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loaded = False
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def vram_used_gb(self) -> float:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e9
        except ImportError:
            pass
        return 0.0

    # --- Output path helper -------------------------------------------------

    def _out_path(self, output_dir: str, mode: str, variation: int, fmt: str = "wav") -> Path:
        stamp = int(time.time())
        uid   = uuid.uuid4().hex[:6]
        name  = f"{mode}_v{variation:02d}_{stamp}_{uid}.{fmt}"
        return Path(output_dir) / name

    # --- Generation ---------------------------------------------------------

    def generate_text(
        self,
        request: TextGenerationRequest,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> list[GenerationResult]:
        results = []

        for i in range(1, request.variations + 1):
            seed = (request.seed or int(time.time())) + i - 1
            out  = self._out_path(request.output_dir, "gen", i)

            if progress_cb:
                progress_cb(GenerationProgress(
                    variation_index=i,
                    total_variations=request.variations,
                    step=0, total_steps=50,
                    message=f"Generating variation {i}/{request.variations}…",
                ))

            duration = float(request.duration_secs)

            if self._loaded and self._pipeline is not None:
                try:
                    audio = self._pipeline(
                        prompt=request.style_prompt,
                        lyrics=request.user_lyrics if request.lyrics_mode.value == "user" else None,
                        duration=duration,
                        seed=seed,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                    )
                    self._pipeline.save_audio(audio, str(out))
                    duration = self._get_audio_duration(str(out))
                except Exception as exc:
                    print(f"[ace_step] Generation error (var {i}): {exc}")
                    self._write_stub_audio(out, duration)
            else:
                self._write_stub_audio(out, duration)

            results.append(GenerationResult(
                variation_index=i,
                audio_path=str(out),
                seed=seed,
                duration_secs=duration,
                mode=GenerationMode.TEXT,
                style_prompt=request.style_prompt,
            ))

        return results

    def generate_cover(
        self,
        request: CoverRequest,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> list[GenerationResult]:
        results = []

        for i in range(1, request.variations + 1):
            seed = (request.seed or int(time.time())) + i - 1
            out  = self._out_path(request.output_dir, "cover", i)

            if progress_cb:
                progress_cb(GenerationProgress(
                    variation_index=i,
                    total_variations=request.variations,
                    step=0, total_steps=50,
                    message=f"Generating cover variation {i}/{request.variations}…",
                ))

            source_duration = self._get_audio_duration(request.source_audio_path)

            if self._loaded and self._pipeline is not None:
                try:
                    audio = self._pipeline.cover(
                        source_audio=request.source_audio_path,
                        target_prompt=request.target_style,
                        new_lyrics=request.new_lyrics or None,
                        follow_strength=request.follow_strength,
                        seed=seed,
                    )
                    self._pipeline.save_audio(audio, str(out))
                    source_duration = self._get_audio_duration(str(out))
                except Exception as exc:
                    print(f"[ace_step] Cover error (var {i}): {exc}")
                    self._write_stub_audio(out, source_duration)
            else:
                self._write_stub_audio(out, source_duration)

            results.append(GenerationResult(
                variation_index=i,
                audio_path=str(out),
                seed=seed,
                duration_secs=source_duration,
                mode=GenerationMode.COVER,
                style_prompt=request.target_style,
            ))

        return results

    def generate_vocal_backing(
        self,
        request: VocalBackingRequest,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> list[GenerationResult]:
        results = []

        for i in range(1, request.variations + 1):
            seed = (request.seed or int(time.time())) + i - 1
            out  = self._out_path(request.output_dir, "vox", i)

            if progress_cb:
                progress_cb(GenerationProgress(
                    variation_index=i,
                    total_variations=request.variations,
                    step=0, total_steps=50,
                    message=f"Generating backing variation {i}/{request.variations}…",
                ))

            vocal_duration = self._get_audio_duration(request.vocal_audio_path)

            if self._loaded and self._pipeline is not None:
                try:
                    audio = self._pipeline.vocal_backing(
                        vocal_audio=request.vocal_audio_path,
                        backing_prompt=request.backing_style,
                        seed=seed,
                    )
                    self._pipeline.save_audio(audio, str(out))
                    vocal_duration = self._get_audio_duration(str(out))
                except Exception as exc:
                    print(f"[ace_step] Vocal backing error (var {i}): {exc}")
                    self._write_stub_audio(out, vocal_duration)
            else:
                self._write_stub_audio(out, vocal_duration)

            results.append(GenerationResult(
                variation_index=i,
                audio_path=str(out),
                seed=seed,
                duration_secs=vocal_duration,
                mode=GenerationMode.VOCAL,
                style_prompt=request.backing_style,
            ))

        return results

    def repair(
        self,
        request: RepairRequest,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> GenerationResult:
        out = self._out_path(request.output_dir, "repair", 1)

        if progress_cb:
            progress_cb(GenerationProgress(
                variation_index=1, total_variations=1,
                step=0, total_steps=50,
                message="Running repair pass…",
            ))

        source_duration = self._get_audio_duration(request.source_audio_path)

        if self._loaded and self._pipeline is not None:
            try:
                audio = self._pipeline.inpaint(
                    source_audio=request.source_audio_path,
                    region_start=request.region_start_sec,
                    region_end=request.region_end_sec,
                    hint_prompt=request.hint_prompt or None,
                )
                self._pipeline.save_audio(audio, str(out))
                source_duration = self._get_audio_duration(str(out))
            except Exception as exc:
                print(f"[ace_step] Repair error: {exc}")
                self._write_stub_audio(out, source_duration)
        else:
            self._write_stub_audio(out, source_duration)

        return GenerationResult(
            variation_index=1,
            audio_path=str(out),
            seed=0,
            duration_secs=source_duration,
            mode=GenerationMode.REPAIR,
            style_prompt=request.hint_prompt,
        )

    # --- Audio helpers ------------------------------------------------------

    def _get_audio_duration(self, path: str) -> float:
        try:
            import soundfile as sf
            info = sf.info(path)
            return info.duration
        except Exception:
            return 30.0

    def _write_stub_audio(self, out_path: Path, duration: float = 10.0) -> None:
        """Write a short silent WAV so the UI has a real file to reference."""
        try:
            import numpy as np
            import soundfile as sf

            sr      = 44100
            samples = int(sr * min(duration, 10.0))
            silence = np.zeros((samples, 2), dtype=np.float32)
            sf.write(str(out_path), silence, sr)
        except Exception as exc:
            print(f"[ace_step] Could not write stub audio: {exc}")
            out_path.touch()
