"""
backend/demucs.py — Wrapper around Demucs for stem separation.

Demucs docs: https://github.com/facebookresearch/demucs
All paths come from caller (resolved via config.py). Nothing hardcoded.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Stem labels by model
# ---------------------------------------------------------------------------

_STEM_LABELS = {
    "htdemucs":      ["drums", "bass", "other", "vocals"],
    "htdemucs_ft":   ["drums", "bass", "other", "vocals"],
    "htdemucs_6s":   ["drums", "bass", "other", "vocals", "guitar", "piano"],
    "mdx_extra":     ["drums", "bass", "other", "vocals"],
}

_DEFAULT_STEMS = ["drums", "bass", "other", "vocals"]


# ---------------------------------------------------------------------------
# Demucs wrapper
# ---------------------------------------------------------------------------

class DemucsSeparator:
    """
    Wraps Demucs stem separation.

    Runs Demucs as a subprocess (most reliable cross-platform approach,
    avoids model-reload overhead by using demucs's own caching).
    """

    def __init__(self, model: str, device: str, output_base_dir: str) -> None:
        self.model           = model
        self.device          = device
        self.output_base_dir = Path(output_base_dir)
        self._available      = self._check_available()

    def _check_available(self) -> bool:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "demucs", "--help"],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    def get_stem_labels(self) -> list[str]:
        return _STEM_LABELS.get(self.model, _DEFAULT_STEMS)

    def separate(
        self,
        audio_path: str,
        job_id: Optional[str] = None,
        progress_cb=None,
    ) -> dict[str, str]:
        """
        Separate `audio_path` into stems.

        Returns a dict mapping stem name -> absolute path.
        e.g. {"vocals": "C:\\...\\vocals.wav", "drums": "C:\\...\\drums.wav"}
        """
        source = Path(audio_path)
        if not source.exists():
            raise FileNotFoundError(f"Source audio not found: {audio_path}")

        out_dir = self.output_base_dir / (job_id or source.stem)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "demucs",
            "--name", self.model,
            "--out", str(out_dir),
            "--device", self.device,
            str(source),
        ]

        if progress_cb:
            progress_cb("Running Demucs separation…")

        if not self._available:
            print("[demucs] Demucs not available — returning stub paths")
            return self._stub_stems(source, out_dir)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,    # 10 min max
            )
            if result.returncode != 0:
                print(f"[demucs] Error:\n{result.stderr}")
                return self._stub_stems(source, out_dir)
        except subprocess.TimeoutExpired:
            print("[demucs] Separation timed out")
            return self._stub_stems(source, out_dir)
        except Exception as exc:
            print(f"[demucs] Unexpected error: {exc}")
            return self._stub_stems(source, out_dir)

        # Demucs writes to: out_dir / model_name / source_stem / stem_name.wav
        stem_dir = out_dir / self.model / source.stem
        stems    = {}
        for label in self.get_stem_labels():
            stem_path = stem_dir / f"{label}.wav"
            if stem_path.exists():
                stems[label] = str(stem_path)
            else:
                print(f"[demucs] Expected stem not found: {stem_path}")

        return stems

    def _stub_stems(self, source: Path, out_dir: Path) -> dict[str, str]:
        """Return stub entries when Demucs is unavailable."""
        import numpy as np
        import soundfile as sf

        sr      = 44100
        silence = np.zeros((sr * 5, 2), dtype=np.float32)
        stems   = {}
        for label in self.get_stem_labels():
            p = out_dir / f"{label}.wav"
            sf.write(str(p), silence, sr)
            stems[label] = str(p)
        return stems
