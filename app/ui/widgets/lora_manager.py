"""
ui/widgets/lora_manager.py — LoRA Manager panel (Settings → LoRAs tab).

Shows the registry of known ACEStep LoRAs, their download status,
lets the user download them individually, and shows all local LoRAs
(including user-provided ones).

Auto-selection logic lives in backend/lora_manager.py and is used
from generate_page.py — this panel is purely for management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from app.config import cfg
from app.backend.logger import log


# ---------------------------------------------------------------------------
# Download worker
# ---------------------------------------------------------------------------

class _DownloadWorker(QThread):
    progress = Signal(str)
    finished = Signal(bool, str)   # (success, message)

    def __init__(self, task: str, data: dict) -> None:
        super().__init__()
        self._task = task   # "lora" | "checkpoint"
        self._data = data

    def run(self) -> None:
        from app.backend.lora_manager import (
            download_lora, download_checkpoint, LoRADownloadError
        )
        try:
            if self._task == "lora":
                download_lora(
                    self._data,
                    cfg.models_dir,
                    progress_cb=lambda m: self.progress.emit(m),
                )
            else:
                download_checkpoint(
                    self._data,
                    cfg.models_dir,
                    progress_cb=lambda m: self.progress.emit(m),
                )
            self.finished.emit(True, "Download complete.")
        except LoRADownloadError as exc:
            self.finished.emit(False, str(exc))
        except Exception as exc:
            self.finished.emit(False, f"Unexpected error: {exc}")


# ---------------------------------------------------------------------------
# Single LoRA row
# ---------------------------------------------------------------------------

class _LoRARow(QWidget):
    download_requested = Signal(dict)

    def __init__(self, lora: dict, models_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self._lora      = lora
        self._models_dir = models_dir
        self._build_ui()

    def _build_ui(self) -> None:
        from app.backend.lora_manager import lora_is_downloaded
        downloaded = lora_is_downloaded(self._lora["id"], self._models_dir)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(8)

        # Status dot
        dot = QLabel("●")
        dot.setFixedWidth(14)
        dot.setStyleSheet(f"color: {'#1D9E75' if downloaded else '#444'};")
        row.addWidget(dot)

        # Name + description
        info = QVBoxLayout()
        info.setSpacing(1)
        name_lbl = QLabel(self._lora["name"])
        name_lbl.setObjectName("StemLabel")
        desc_lbl = QLabel(self._lora.get("description", "")[:80] + "…"
                          if len(self._lora.get("description", "")) > 80
                          else self._lora.get("description", ""))
        desc_lbl.setObjectName("HintLabel")
        desc_lbl.setWordWrap(True)
        info.addWidget(name_lbl)
        info.addWidget(desc_lbl)

        # Tags
        tags_row = QHBoxLayout()
        tags_row.setSpacing(4)
        for tag in self._lora.get("tags", [])[:4]:
            tag_lbl = QLabel(tag)
            tag_lbl.setObjectName("LoRATag")
            tags_row.addWidget(tag_lbl)
        tags_row.addStretch()
        info.addLayout(tags_row)

        row.addLayout(info, 1)

        # Size
        size_lbl = QLabel(f"{self._lora.get('size_mb', '?')} MB")
        size_lbl.setObjectName("FieldLabelDim")
        size_lbl.setFixedWidth(55)
        row.addWidget(size_lbl)

        # Action button
        if downloaded:
            status_btn = QPushButton("✓ Ready")
            status_btn.setObjectName("ActionBtnActive")
            status_btn.setEnabled(False)
            status_btn.setFixedWidth(80)
        else:
            status_btn = QPushButton("⬇ Download")
            status_btn.setObjectName("MixerRenderBtn")
            status_btn.setFixedWidth(90)
            _lora = self._lora
            status_btn.clicked.connect(lambda: self.download_requested.emit(_lora))
        row.addWidget(status_btn)
        self._action_btn = status_btn

    def mark_downloaded(self) -> None:
        self._action_btn.setText("✓ Ready")
        self._action_btn.setObjectName("ActionBtnActive")
        self._action_btn.setEnabled(False)
        self._action_btn.style().unpolish(self._action_btn)
        self._action_btn.style().polish(self._action_btn)


# ---------------------------------------------------------------------------
# Checkpoint row
# ---------------------------------------------------------------------------

class _CheckpointRow(QWidget):
    download_requested = Signal(dict)

    def __init__(self, checkpoint: dict, models_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self._cp         = checkpoint
        self._models_dir = models_dir
        self._build_ui()

    def _is_downloaded(self) -> bool:
        folder = self._models_dir / self._cp["id"]
        return folder.exists() and any(folder.iterdir())

    def _build_ui(self) -> None:
        downloaded = self._is_downloaded()

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(8)

        dot = QLabel("●")
        dot.setFixedWidth(14)
        dot.setStyleSheet(f"color: {'#1D9E75' if downloaded else '#444'};")
        row.addWidget(dot)

        info = QVBoxLayout()
        info.setSpacing(1)
        name_lbl = QLabel(self._cp["name"])
        name_lbl.setObjectName("StemLabel")
        desc_lbl = QLabel(self._cp.get("description", ""))
        desc_lbl.setObjectName("HintLabel")
        desc_lbl.setWordWrap(True)

        vram_lbl = QLabel(f"VRAM: ~{self._cp.get('vram_gb', '?')} GB  ·  "
                          f"Size: {self._cp.get('size_gb', '?')} GB")
        vram_lbl.setObjectName("FieldLabelDim")

        info.addWidget(name_lbl)
        info.addWidget(desc_lbl)
        info.addWidget(vram_lbl)
        row.addLayout(info, 1)

        if downloaded:
            btn = QPushButton("✓ Ready")
            btn.setObjectName("ActionBtnActive")
            btn.setEnabled(False)
            btn.setFixedWidth(80)
        else:
            btn = QPushButton("⬇ Download")
            btn.setObjectName("MixerRenderBtn")
            btn.setFixedWidth(90)
            _cp = self._cp
            btn.clicked.connect(lambda: self.download_requested.emit(_cp))
        row.addWidget(btn)
        self._action_btn = btn


# ---------------------------------------------------------------------------
# Main LoRA Manager panel
# ---------------------------------------------------------------------------

class LoRAManagerPanel(QWidget):
    """
    Displayed inside Settings → LoRAs tab.
    Shows registry LoRAs + checkpoints with download buttons.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._workers: list[_DownloadWorker] = []
        self._build_ui()

    def _build_ui(self) -> None:
        from app.backend.lora_manager import load_registry
        registry = load_registry()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Status bar ---
        self._status_lbl = QLabel("Select a LoRA or checkpoint to download.")
        self._status_lbl.setObjectName("MixerStatus")
        self._status_lbl.setContentsMargins(12, 6, 12, 6)
        root.addWidget(self._status_lbl)

        self._progress = QProgressBar()
        self._progress.setObjectName("GenProgress")
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setFixedHeight(3)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # --- Checkpoints section ---
        cp_hdr = self._section_header("BASE MODELS / CHECKPOINTS")
        lay.addWidget(cp_hdr)

        cp_note = QLabel(
            "Larger models produce higher quality music. XL models require 9–12 GB VRAM. "
            "Base/Turbo run on 4 GB. Downloaded models are selectable in Generation quality settings."
        )
        cp_note.setObjectName("HintLabel")
        cp_note.setWordWrap(True)
        cp_note.setContentsMargins(12, 4, 12, 8)
        lay.addWidget(cp_note)

        for cp in registry.get("checkpoints", []):
            row = _CheckpointRow(cp, Path(cfg.models_dir))
            row.download_requested.connect(lambda d: self._start_download("checkpoint", d))
            lay.addWidget(row)
            lay.addWidget(self._divider())

        # --- LoRAs section ---
        lora_hdr = self._section_header("STYLE LoRAs")
        lay.addWidget(lora_hdr)

        lora_note = QLabel(
            "LoRAs specialise the model for specific genres or vocal styles. "
            "They must be used with the Base model (Quality mode). "
            "Auto-select will pick matching LoRAs from your style prompt automatically."
        )
        lora_note.setObjectName("HintLabel")
        lora_note.setWordWrap(True)
        lora_note.setContentsMargins(12, 4, 12, 8)
        lay.addWidget(lora_note)

        self._lora_rows: dict[str, _LoRARow] = {}
        for lora in registry.get("loras", []):
            row = _LoRARow(lora, Path(cfg.models_dir))
            row.download_requested.connect(lambda d: self._start_download("lora", d))
            lay.addWidget(row)
            lay.addWidget(self._divider())
            self._lora_rows[lora["id"]] = row

        lay.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    # ---- Helpers -----------------------------------------------------------

    def _section_header(self, text: str) -> QWidget:
        w = QWidget()
        w.setObjectName("MixerHeader")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(12, 8, 12, 8)
        lbl = QLabel(text)
        lbl.setObjectName("StemsHeader")
        lay.addWidget(lbl)
        lay.addStretch()
        return w

    def _divider(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setObjectName("Divider")
        return f

    # ---- Download ----------------------------------------------------------

    def _start_download(self, task: str, data: dict) -> None:
        self._status_lbl.setText(f"Downloading {data['name']}…")
        self._progress.setVisible(True)

        w = _DownloadWorker(task, data)
        w.progress.connect(self._status_lbl.setText)
        w.finished.connect(lambda ok, msg, d=data, t=task: self._on_finished(ok, msg, d, t))
        w.start()
        self._workers.append(w)
        self._workers = [w for w in self._workers if not w.isFinished()]

    def _on_finished(self, ok: bool, msg: str, data: dict, task: str) -> None:
        self._progress.setVisible(False)
        self._status_lbl.setText(msg if ok else f"Error: {msg}")
        if ok and task == "lora":
            row = self._lora_rows.get(data.get("id"))
            if row:
                row.mark_downloaded()
