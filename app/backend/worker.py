"""
backend/worker.py — QThread workers for all generation modes.

All inference runs off the main thread so the UI stays responsive.
Workers emit typed Qt signals back to the UI.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from app.backend.ace_step import ACEStepPipeline
from app.backend.demucs import DemucsSeparator
from app.models.generation import (
    CoverRequest,
    GenerationProgress,
    GenerationResult,
    RepairRequest,
    TextGenerationRequest,
    VocalBackingRequest,
)


# ---------------------------------------------------------------------------
# Base worker
# ---------------------------------------------------------------------------

class BaseWorker(QThread):
    progress = Signal(object)   # GenerationProgress
    result   = Signal(object)   # list[GenerationResult] or GenerationResult
    error    = Signal(str)
    done     = Signal()


# ---------------------------------------------------------------------------
# Text generation worker
# ---------------------------------------------------------------------------

class TextGenerationWorker(BaseWorker):
    def __init__(self, pipeline: ACEStepPipeline, request: TextGenerationRequest) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._request  = request

    def run(self) -> None:
        try:
            results = self._pipeline.generate_text(
                self._request,
                progress_cb=lambda p: self.progress.emit(p),
            )
            self.result.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.done.emit()


# ---------------------------------------------------------------------------
# Cover / restyle worker
# ---------------------------------------------------------------------------

class CoverWorker(BaseWorker):
    def __init__(self, pipeline: ACEStepPipeline, request: CoverRequest) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._request  = request

    def run(self) -> None:
        try:
            results = self._pipeline.generate_cover(
                self._request,
                progress_cb=lambda p: self.progress.emit(p),
            )
            self.result.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.done.emit()


# ---------------------------------------------------------------------------
# Vocal backing worker
# ---------------------------------------------------------------------------

class VocalBackingWorker(BaseWorker):
    def __init__(self, pipeline: ACEStepPipeline, request: VocalBackingRequest) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._request  = request

    def run(self) -> None:
        try:
            results = self._pipeline.generate_vocal_backing(
                self._request,
                progress_cb=lambda p: self.progress.emit(p),
            )
            self.result.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.done.emit()


# ---------------------------------------------------------------------------
# Repair worker
# ---------------------------------------------------------------------------

class RepairWorker(BaseWorker):
    def __init__(self, pipeline: ACEStepPipeline, request: RepairRequest) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._request  = request

    def run(self) -> None:
        try:
            result = self._pipeline.repair(
                self._request,
                progress_cb=lambda p: self.progress.emit(p),
            )
            self.result.emit([result])
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.done.emit()


# ---------------------------------------------------------------------------
# Stem separation worker
# ---------------------------------------------------------------------------

class StemWorker(QThread):
    stems_ready = Signal(str, object)  # (audio_path, dict[stem->path])
    error       = Signal(str)
    done        = Signal()

    def __init__(
        self,
        separator: DemucsSeparator,
        audio_path: str,
        job_id: str,
    ) -> None:
        super().__init__()
        self._separator  = separator
        self._audio_path = audio_path
        self._job_id     = job_id

    def run(self) -> None:
        try:
            stems = self._separator.separate(
                self._audio_path,
                job_id=self._job_id,
            )
            self.stems_ready.emit(self._audio_path, stems)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.done.emit()


# ---------------------------------------------------------------------------
# Model loader worker
# ---------------------------------------------------------------------------

class ModelLoaderWorker(QThread):
    loaded  = Signal(bool)   # True = success
    message = Signal(str)
    done    = Signal()

    def __init__(self, pipeline: ACEStepPipeline) -> None:
        super().__init__()
        self._pipeline = pipeline

    def run(self) -> None:
        try:
            self.message.emit("Loading ACE-Step model…")
            ok = self._pipeline.load()
            self.loaded.emit(ok)
        except Exception as exc:
            self.message.emit(f"Load failed: {exc}")
            self.loaded.emit(False)
        finally:
            self.done.emit()
