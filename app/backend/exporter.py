"""
backend/exporter.py — Audio export worker.

Converts WAV → MP3 using pydub (which wraps ffmpeg).
Falls back to a plain file copy with a warning if ffmpeg is absent.

All paths passed in — nothing hardcoded.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal


class ExportWorker(QThread):
    """
    Exports src_path → dest_path.
    Supports WAV passthrough and WAV→MP3 via pydub+ffmpeg.
    """

    finished = Signal(str)   # dest_path on success
    error    = Signal(str)   # error message on failure
    done     = Signal()

    def __init__(
        self,
        src_path:  str,
        dest_path: str,
        fmt:       str = "wav",   # "wav" | "mp3"
        bitrate:   int = 320,
    ) -> None:
        super().__init__()
        self._src     = src_path
        self._dest    = dest_path
        self._fmt     = fmt.lower()
        self._bitrate = bitrate

    def run(self) -> None:
        try:
            if self._fmt == "mp3":
                self._export_mp3()
            else:
                self._export_wav()
            self.finished.emit(self._dest)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.done.emit()

    # -----------------------------------------------------------------------

    def _export_wav(self) -> None:
        import shutil
        shutil.copy2(self._src, self._dest)

    def _export_mp3(self) -> None:
        try:
            from pydub import AudioSegment  # type: ignore
            audio = AudioSegment.from_file(self._src)
            audio.export(
                self._dest,
                format="mp3",
                bitrate=f"{self._bitrate}k",
                tags={"software": "Stitch"},
            )
        except ImportError:
            # pydub not installed — try ffmpeg directly via subprocess
            self._export_mp3_subprocess()
        except Exception as exc:
            raise RuntimeError(f"MP3 export failed: {exc}") from exc

    def _export_mp3_subprocess(self) -> None:
        import subprocess
        import shutil as sh

        ffmpeg = sh.which("ffmpeg")
        if not ffmpeg:
            # Last resort: copy as WAV and rename, warn user
            wav_dest = Path(self._dest).with_suffix(".wav")
            import shutil
            shutil.copy2(self._src, str(wav_dest))
            raise RuntimeError(
                "ffmpeg not found — saved as WAV instead. "
                "Install ffmpeg and add it to PATH for MP3 export."
            )

        result = subprocess.run(
            [
                ffmpeg, "-y",
                "-i", self._src,
                "-codec:a", "libmp3lame",
                "-b:a", f"{self._bitrate}k",
                self._dest,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")
