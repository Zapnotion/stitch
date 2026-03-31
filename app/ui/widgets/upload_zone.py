"""
ui/widgets/upload_zone.py — Drag-and-drop audio upload zone.
Emits file_selected(path: str) when a file is dropped or browsed.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget,
)


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg", ".m4a"}


class UploadZone(QWidget):
    """
    Shows either a dashed drop-zone or a loaded-file pill.
    Emits file_selected(str) when a valid audio file is chosen.
    Emits file_cleared() when user removes the loaded file.
    """

    file_selected = Signal(str)
    file_cleared  = Signal()

    def __init__(self, label: str = "Drop audio file here", parent=None) -> None:
        super().__init__(parent)
        self._label       = label
        self._loaded_path = ""
        self.setAcceptDrops(True)
        self._build_ui()

    # --- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self._drop_widget = self._make_drop_widget()
        self._loaded_widget = self._make_loaded_widget()
        self._loaded_widget.setVisible(False)

        self._layout.addWidget(self._drop_widget)
        self._layout.addWidget(self._loaded_widget)

    def _make_drop_widget(self) -> QWidget:
        w = QWidget()
        w.setObjectName("DropZone")
        w.setCursor(Qt.PointingHandCursor)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(3)
        lay.setAlignment(Qt.AlignCenter)

        icon = QLabel("🎵")
        icon.setAlignment(Qt.AlignCenter)
        lay.addWidget(icon)

        lbl = QLabel(self._label)
        lbl.setObjectName("DropLabel")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)

        sub = QLabel("WAV · MP3 · FLAC · AIFF")
        sub.setObjectName("DropSub")
        sub.setAlignment(Qt.AlignCenter)
        lay.addWidget(sub)

        browse = QLabel("or browse files")
        browse.setObjectName("DropBrowse")
        browse.setAlignment(Qt.AlignCenter)
        lay.addWidget(browse)

        # Make the whole area clickable
        w.mousePressEvent = lambda _: self._browse()
        return w

    def _make_loaded_widget(self) -> QWidget:
        w = QWidget()
        w.setObjectName("LoadedFile")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)

        self._dot = QLabel("●")
        self._dot.setObjectName("LoadedDot")
        self._dot.setFixedWidth(12)
        lay.addWidget(self._dot)

        self._name_lbl = QLabel("")
        self._name_lbl.setObjectName("LoadedName")
        lay.addWidget(self._name_lbl, 1)

        self._dur_lbl = QLabel("")
        self._dur_lbl.setObjectName("LoadedDur")
        lay.addWidget(self._dur_lbl)

        remove = QPushButton("remove")
        remove.setObjectName("RemoveBtn")
        remove.setFlat(True)
        remove.clicked.connect(self._clear)
        lay.addWidget(remove)

        return w

    # --- Public API ---------------------------------------------------------

    @property
    def loaded_path(self) -> str:
        return self._loaded_path

    def set_file(self, path: str) -> None:
        """Programmatically load a file."""
        self._load_file(path)

    def clear(self) -> None:
        self._clear()

    # --- Internal -----------------------------------------------------------

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio file", "",
            "Audio files (*.wav *.mp3 *.flac *.aiff *.aif *.ogg *.m4a)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        p = Path(path)
        if p.suffix.lower() not in AUDIO_EXTS:
            return
        self._loaded_path = path
        self._name_lbl.setText(p.name)
        self._dur_lbl.setText(self._get_duration(path))
        self._drop_widget.setVisible(False)
        self._loaded_widget.setVisible(True)
        self.file_selected.emit(path)

    def _clear(self) -> None:
        self._loaded_path = ""
        self._drop_widget.setVisible(True)
        self._loaded_widget.setVisible(False)
        self.file_cleared.emit()

    def _get_duration(self, path: str) -> str:
        try:
            import soundfile as sf
            info = sf.info(path)
            total = int(info.duration)
            m, s = divmod(total, 60)
            return f"{m}:{s:02d}"
        except Exception:
            return ""

    # --- Drag-and-drop ------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and Path(urls[0].toLocalFile()).suffix.lower() in AUDIO_EXTS:
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._load_file(path)
