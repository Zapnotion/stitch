"""
backend/separator.py — Two-pass stem separation.

Pass 1: audio-separator (UVR MDX-Net / BS-Roformer) for vocals vs instrumental.
         Best-in-class vocal isolation — SDR ~12.9 vs Demucs ~8.5.
Pass 2: Demucs htdemucs_6s on the instrumental for drums/bass/guitar/piano/other.

Falls back gracefully: if audio-separator unavailable, uses Demucs only.
If Demucs also unavailable, returns stub silent files.

Models auto-download on first use (~100-300 MB each).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from app.backend.logger import log


# ---------------------------------------------------------------------------
# Best available vocal separation model (BS-Roformer, SDR 12.9)
# ---------------------------------------------------------------------------
VOCAL_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

# Demucs model for instrument breakdown
DEMUCS_INSTRUMENT_MODEL = "htdemucs_6s"  # gives drums/bass/guitar/piano/other/vocals


class StemSeparator:
    """
    Two-pass separator.  Preferred over raw Demucs for vocal quality.
    """

    def __init__(
        self,
        device: str,
        output_base_dir: str,
        model_cache_dir: str,
    ) -> None:
        self.device          = device
        self.output_base_dir = Path(output_base_dir)
        self.model_cache_dir = Path(model_cache_dir) / "separator_models"
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self._has_separator  = self._check_separator()
        self._has_demucs     = self._check_demucs()
        log.info(
            f"StemSeparator: audio-separator={'yes' if self._has_separator else 'no'}, "
            f"demucs={'yes' if self._has_demucs else 'no'}"
        )

    # -----------------------------------------------------------------------
    # Availability checks
    # -----------------------------------------------------------------------

    def _check_separator(self) -> bool:
        try:
            r = subprocess.run(
                [sys.executable, "-c", "import audio_separator; print('ok')"],
                capture_output=True, text=True, timeout=10,
            )
            return r.returncode == 0 and "ok" in r.stdout
        except Exception:
            return False

    def _check_demucs(self) -> bool:
        try:
            r = subprocess.run(
                [sys.executable, "-m", "demucs", "--help"],
                capture_output=True, timeout=10,
            )
            return r.returncode == 0
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._has_separator or self._has_demucs

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def separate(
        self,
        audio_path: str,
        job_id: Optional[str] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> dict[str, str]:
        source  = Path(audio_path)
        job_id  = job_id or source.stem
        out_dir = self.output_base_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        def _cb(msg: str):
            log.info(f"[stems] {msg}")
            if progress_cb:
                progress_cb(msg)

        if not source.exists():
            raise FileNotFoundError(f"Source audio not found: {audio_path}")

        if not self.is_available:
            _cb("No separator available — returning silent stubs")
            return self._stub_stems(source, out_dir)

        stems: dict[str, str] = {}

        # ------------------------------------------------------------------
        # Pass 1 — Vocals vs Instrumental (audio-separator preferred)
        # ------------------------------------------------------------------
        if self._has_separator:
            _cb(f"Pass 1/2 — Vocal isolation ({VOCAL_MODEL})…")
            vocal_path, instrumental_path = self._run_separator(
                source, out_dir, _cb
            )
            if vocal_path and instrumental_path:
                stems["vocals"]       = vocal_path
                stems["instrumental"] = instrumental_path
                _cb(f"Pass 1 done — vocals isolated")
            else:
                _cb("Pass 1 failed — falling back to Demucs for vocals")
                stems = self._run_demucs_full(source, out_dir, _cb)
                return stems
        else:
            _cb("audio-separator not available — using Demucs for all stems")
            return self._run_demucs_full(source, out_dir, _cb)

        # ------------------------------------------------------------------
        # Pass 2 — Instrument breakdown on the instrumental stem
        # ------------------------------------------------------------------
        if self._has_demucs and instrumental_path:
            _cb("Pass 2/2 — Instrument breakdown (Demucs htdemucs_6s)…")
            instrument_stems = self._run_demucs_instruments(
                Path(instrumental_path), out_dir, _cb
            )
            stems.update(instrument_stems)
            _cb("Pass 2 done — all stems ready")

        return stems

    # -----------------------------------------------------------------------
    # audio-separator pass
    # -----------------------------------------------------------------------

    def _run_separator(
        self,
        source: Path,
        out_dir: Path,
        cb: Callable,
    ) -> tuple[Optional[str], Optional[str]]:
        try:
            from audio_separator.separator import Separator

            sep = Separator(
                output_dir=str(out_dir),
                model_file_dir=str(self.model_cache_dir),
                output_format="WAV",
                normalization_threshold=0.9,
            )
            cb(f"Loading {VOCAL_MODEL} (downloads ~200 MB on first use)…")
            sep.load_model(model_filename=VOCAL_MODEL)
            output_files = sep.separate(str(source))

            # audio-separator returns list of output paths
            # Naming convention: {stem}_(Vocals).wav / {stem}_(Instrumental).wav
            vocal_path        = None
            instrumental_path = None
            for f in output_files:
                f_lower = f.lower()
                if "vocal" in f_lower and "instrumental" not in f_lower:
                    vocal_path = f
                elif "instrumental" in f_lower or "no_vocals" in f_lower:
                    instrumental_path = f

            if not vocal_path or not instrumental_path:
                cb(f"Unexpected output files: {output_files}")
                # Try positional fallback
                if len(output_files) == 2:
                    instrumental_path, vocal_path = sorted(output_files)

            return vocal_path, instrumental_path

        except Exception as exc:
            cb(f"audio-separator error: {exc}")
            return None, None

    # -----------------------------------------------------------------------
    # Demucs passes
    # -----------------------------------------------------------------------

    def _run_demucs_full(
        self,
        source: Path,
        out_dir: Path,
        cb: Callable,
    ) -> dict[str, str]:
        """Full 4-stem Demucs separation when audio-separator unavailable."""
        cb("Running Demucs htdemucs_ft (full stems)…")
        self._run_demucs_cmd(source, out_dir, "htdemucs_ft", cb)
        stem_dir = out_dir / "htdemucs_ft" / source.stem
        return self._collect_demucs_stems(stem_dir, ["vocals", "drums", "bass", "other"])

    def _run_demucs_instruments(
        self,
        source: Path,
        out_dir: Path,
        cb: Callable,
    ) -> dict[str, str]:
        """6-stem Demucs on the instrumental for fine breakdown."""
        self._run_demucs_cmd(source, out_dir, DEMUCS_INSTRUMENT_MODEL, cb)
        stem_dir = out_dir / DEMUCS_INSTRUMENT_MODEL / source.stem
        # Exclude vocals stem (it's the leaked bleed from the instrumental)
        return self._collect_demucs_stems(
            stem_dir, ["drums", "bass", "guitar", "piano", "other"]
        )

    def _run_demucs_cmd(
        self,
        source: Path,
        out_dir: Path,
        model: str,
        cb: Callable,
    ) -> None:
        device = "cuda" if "cuda" in self.device else "cpu"
        cmd = [
            sys.executable, "-m", "demucs",
            "--name", model,
            "--out", str(out_dir),
            "--device", device,
            str(source),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                cb(f"Demucs error: {result.stderr[-300:]}")
        except subprocess.TimeoutExpired:
            cb("Demucs timed out after 10 minutes")
        except Exception as exc:
            cb(f"Demucs failed: {exc}")

    def _collect_demucs_stems(
        self, stem_dir: Path, labels: list[str]
    ) -> dict[str, str]:
        stems = {}
        for label in labels:
            p = stem_dir / f"{label}.wav"
            if p.exists():
                stems[label] = str(p)
            else:
                log.warning(f"[stems] Expected stem not found: {p}")
        return stems

    # -----------------------------------------------------------------------
    # Stub fallback
    # -----------------------------------------------------------------------

    def _stub_stems(self, source: Path, out_dir: Path) -> dict[str, str]:
        import numpy as np
        import soundfile as sf
        sr      = 44100
        silence = np.zeros((sr * 5, 2), dtype=np.float32)
        stems   = {}
        for label in ["vocals", "drums", "bass", "guitar", "piano", "other"]:
            p = out_dir / f"{label}.wav"
            sf.write(str(p), silence, sr)
            stems[label] = str(p)
        return stems
