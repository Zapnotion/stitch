"""
ui/widgets/result_card.py — A single generation result card widget.
Shows waveform, controls, stem tray. Emits signals for parent to handle actions.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from app.models.generation import GenerationResult
from app.ui.widgets.audio_player import AudioPlayer
from app.ui.widgets.waveform import WaveformWidget


class ResultCard(QWidget):
    """Card displaying a single GenerationResult."""

    star_toggled    = Signal(object)   # GenerationResult
    download_wav    = Signal(object)
    download_mp3    = Signal(object)
    stems_requested = Signal(object)
    repaint_requested = Signal(object)

    def __init__(self, result: GenerationResult, parent=None) -> None:
        super().__init__(parent)
        self.result = result
        self._stems_visible = False
        self._build_ui()
        self._load_waveform()

    # --- Build --------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        frame = QFrame()
        frame.setObjectName("ResultCard")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(10, 10, 10, 10)
        frame_layout.setSpacing(6)

        # --- Top row: star + title + seed + badges ---
        top = QHBoxLayout()
        top.setSpacing(6)

        self.star_btn = QPushButton("☆")
        self.star_btn.setFixedSize(20, 20)
        self.star_btn.setFlat(True)
        self.star_btn.setObjectName("StarBtn")
        self.star_btn.clicked.connect(self._toggle_star)
        top.addWidget(self.star_btn)

        self.title_lbl = QLabel(f"Variation {self.result.variation_index}")
        self.title_lbl.setObjectName("CardTitle")
        top.addWidget(self.title_lbl)

        self.seed_lbl = QLabel(f"seed {self.result.seed}")
        self.seed_lbl.setObjectName("SeedLabel")
        top.addStretch()

        self.dur_badge = QLabel(self.result.duration_label)
        self.dur_badge.setObjectName("Badge")
        top.addWidget(self.seed_lbl)
        top.addWidget(self.dur_badge)

        frame_layout.addLayout(top)

        # --- Waveform ---
        self.waveform = WaveformWidget(selectable=False)
        self.waveform.setFixedHeight(40)
        frame_layout.addWidget(self.waveform)

        # --- Audio player ---
        self.player = AudioPlayer()
        frame_layout.addWidget(self.player)

        # --- Action row ---
        actions = QHBoxLayout()
        actions.setSpacing(4)

        self.stems_btn = QPushButton("Stems ▾")
        self.stems_btn.setObjectName("ActionBtn")
        self.stems_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stems_btn.clicked.connect(self._on_stems)

        self.repaint_btn = QPushButton("Repaint region")
        self.repaint_btn.setObjectName("ActionBtn")
        self.repaint_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.repaint_btn.clicked.connect(lambda: self.repaint_requested.emit(self.result))

        self.wav_btn = QPushButton("⬇ WAV")
        self.wav_btn.setObjectName("ActionBtn")
        self.wav_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.wav_btn.clicked.connect(lambda: self.download_wav.emit(self.result))

        self.mp3_btn = QPushButton("⬇ MP3")
        self.mp3_btn.setObjectName("ActionBtn")
        self.mp3_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mp3_btn.clicked.connect(lambda: self.download_mp3.emit(self.result))

        for btn in [self.stems_btn, self.repaint_btn, self.wav_btn, self.mp3_btn]:
            actions.addWidget(btn)

        frame_layout.addLayout(actions)

        # --- Stems tray (hidden by default) ---
        self.stems_tray = QWidget()
        self.stems_tray.setObjectName("StemsTray")
        self.stems_tray.setVisible(False)
        stems_layout = QVBoxLayout(self.stems_tray)
        stems_layout.setContentsMargins(0, 8, 0, 0)
        stems_layout.setSpacing(4)

        hdr = QLabel("STEMS")
        hdr.setObjectName("StemsHeader")
        stems_layout.addWidget(hdr)

        self._stems_rows_layout = QVBoxLayout()
        self._stems_rows_layout.setSpacing(3)
        stems_layout.addLayout(self._stems_rows_layout)

        frame_layout.addWidget(self.stems_tray)

        root.addWidget(frame)

    def _load_waveform(self) -> None:
        from pathlib import Path
        if Path(self.result.audio_path).exists():
            self.waveform.load_audio(self.result.audio_path)
            self.player.load(self.result.audio_path)
            # Drive waveform playhead from player scrub position (0–1000 range)
            self.player._scrub.valueChanged.connect(
                lambda v: self.waveform.set_playhead(v / 1000.0)
            )
            self.player.playback_stopped.connect(
                lambda _: self.waveform.set_playhead(-1.0)
            )

    # --- Stem display -------------------------------------------------------

    def populate_stems(self, stems: dict[str, str]) -> None:
        """Called after stem separation completes."""
        self.result.stems = stems
        # Clear existing rows
        while self._stems_rows_layout.count():
            item = self._stems_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        stem_colors = {
            "vocals": "#85B7EB",
            "drums":  "#EF9F27",
            "bass":   "#1D9E75",
            "other":  "#B09ECC",
            "guitar": "#F07070",
            "piano":  "#70C8F0",
        }

        for name, path in stems.items():
            row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {stem_colors.get(name, '#888')};")
            dot.setFixedWidth(14)

            lbl = QLabel(name.capitalize())
            lbl.setObjectName("StemLabel")
            lbl.setFixedWidth(60)

            mini_wave = WaveformWidget(selectable=False)
            mini_wave.setFixedHeight(18)
            from pathlib import Path as _P
            if _P(path).exists():
                mini_wave.load_audio(path)

            dl_btn = QPushButton("⬇")
            dl_btn.setObjectName("StemDlBtn")
            dl_btn.setFixedWidth(28)
            _path = path
            dl_btn.clicked.connect(lambda _, p=_path: self._download_stem(p))

            row.addWidget(dot)
            row.addWidget(lbl)
            row.addWidget(mini_wave)
            row.addWidget(dl_btn)

            container = QWidget()
            container.setLayout(row)
            self._stems_rows_layout.addWidget(container)

        self.stems_tray.setVisible(True)
        self._stems_visible = True
        self.stems_btn.setText("Stems ▴")

    def _download_stem(self, path: str) -> None:
        from PySide6.QtWidgets import QFileDialog
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save stem", path, "Audio files (*.wav *.mp3)"
        )
        if dest:
            import shutil
            shutil.copy2(path, dest)

    # --- Actions ------------------------------------------------------------

    def _toggle_star(self) -> None:
        self.result.starred = not self.result.starred
        self.star_btn.setText("★" if self.result.starred else "☆")
        self.star_toggled.emit(self.result)

    def _on_stems(self) -> None:
        if self._stems_visible:
            self.stems_tray.setVisible(False)
            self._stems_visible = False
            self.stems_btn.setText("Stems ▾")
        elif self.result.stems:
            self.populate_stems(self.result.stems)
        else:
            self.stems_requested.emit(self.result)
            self.stems_btn.setText("Separating…")
            self.stems_btn.setEnabled(False)

    def enable_stems_btn(self) -> None:
        self.stems_btn.setEnabled(True)
        self.stems_btn.setText("Stems ▾")
