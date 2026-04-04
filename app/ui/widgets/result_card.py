"""
ui/widgets/result_card.py — A single generation result card widget.
Shows waveform, controls, stem tray, and lyric timeline. Emits signals for parent to handle actions.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from app.models.generation import GenerationResult
from app.ui.widgets.audio_player import AudioPlayer
from app.ui.widgets.lyric_timeline import LyricTimelineWidget
from app.ui.widgets.waveform import WaveformWidget


class ResultCard(QWidget):
    """Card displaying a single GenerationResult."""

    star_toggled    = Signal(object)   # GenerationResult
    download_wav    = Signal(object)
    download_mp3    = Signal(object)
    stems_requested = Signal(object)
    repaint_requested = Signal(object)
    # Phase 1: word-level repair and section regen routed up to GeneratePage
    repair_word_requested        = Signal(object, float, float, str)   # result, start, end, word
    regenerate_section_requested = Signal(object, float, float, str)   # result, start, end, label
    # Phase 3: mix rendered — carries the new audio path
    mix_rendered = Signal(object, str)   # (GenerationResult, new_audio_path)

    def __init__(self, result: GenerationResult, parent=None) -> None:
        super().__init__(parent)
        self.result = result
        self._stems_visible = False
        self._build_ui()
        self._load_waveform()
        self._try_load_alignment()

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

        # --- Waveform (taller for easier interaction) ---
        self.waveform = WaveformWidget(selectable=False)
        self.waveform.setFixedHeight(56)
        frame_layout.addWidget(self.waveform)

        # --- Audio player ---
        self.player = AudioPlayer()
        frame_layout.addWidget(self.player)

        # --- Primary action row: download buttons are prominent ---
        primary_actions = QHBoxLayout()
        primary_actions.setSpacing(6)

        self.wav_btn = QPushButton("⬇ WAV")
        self.wav_btn.setObjectName("DlBtn")
        self.wav_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.wav_btn.clicked.connect(lambda: self.download_wav.emit(self.result))

        self.mp3_btn = QPushButton("⬇ MP3")
        self.mp3_btn.setObjectName("DlBtn")
        self.mp3_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mp3_btn.clicked.connect(lambda: self.download_mp3.emit(self.result))

        primary_actions.addWidget(self.wav_btn)
        primary_actions.addWidget(self.mp3_btn)
        frame_layout.addLayout(primary_actions)

        # --- Secondary action row: utility actions, smaller weight ---
        secondary_actions = QHBoxLayout()
        secondary_actions.setSpacing(4)

        self.stems_btn = QPushButton("Stems")
        self.stems_btn.setObjectName("ActionBtn")
        self.stems_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stems_btn.clicked.connect(self._on_stems)

        self.repaint_btn = QPushButton("Repaint region")
        self.repaint_btn.setObjectName("ActionBtn")
        self.repaint_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.repaint_btn.clicked.connect(lambda: self.repaint_requested.emit(self.result))

        # Phase 3: Mixer button — only visible once stems exist
        self.mixer_btn = QPushButton("🎚 Mixer")
        self.mixer_btn.setObjectName("ActionBtn")
        self.mixer_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mixer_btn.setVisible(False)
        self.mixer_btn.clicked.connect(self._on_mixer)

        secondary_actions.addWidget(self.stems_btn)
        secondary_actions.addWidget(self.repaint_btn)
        secondary_actions.addWidget(self.mixer_btn)
        frame_layout.addLayout(secondary_actions)

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

        # --- Stem Mixer panel (Phase 3) — hidden until mixer button clicked ---
        self._mixer_divider = QFrame()
        self._mixer_divider.setObjectName("Divider")
        self._mixer_divider.setFixedHeight(1)
        self._mixer_divider.setVisible(False)
        frame_layout.addWidget(self._mixer_divider)

        self._mixer_panel = None  # lazy-created on first open
        self._mixer_panel_placeholder = QWidget()
        self._mixer_panel_placeholder.setVisible(False)
        frame_layout.addWidget(self._mixer_panel_placeholder)
        self._frame_layout_ref = frame_layout  # keep ref for lazy insertion

        # --- Lyric timeline (Phase 1) — hidden until alignment sidecar exists ---
        self._timeline_divider = QFrame()
        self._timeline_divider.setObjectName("Divider")
        self._timeline_divider.setFixedHeight(1)
        self._timeline_divider.setVisible(False)
        frame_layout.addWidget(self._timeline_divider)

        self.timeline = LyricTimelineWidget()
        self.timeline.setVisible(False)
        self.timeline.seek_requested.connect(self._on_timeline_seek)
        self.timeline.repair_word_requested.connect(
            lambda s, e, w: self.repair_word_requested.emit(self.result, s, e, w)
        )
        self.timeline.regenerate_section_requested.connect(
            lambda s, e, lbl: self.regenerate_section_requested.emit(self.result, s, e, lbl)
        )
        frame_layout.addWidget(self.timeline)

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
            # Drive lyric timeline position from player
            self.player._scrub.valueChanged.connect(self._on_scrub_for_timeline)
            self.player.playback_stopped.connect(lambda _: self.timeline.stop_tracking())

        # Phase 3: show mixer button if stems are already populated (e.g. from session restore)
        if self.result.stems:
            self.mixer_btn.setVisible(True)

    def _on_scrub_for_timeline(self, slider_val: int) -> None:
        """Convert scrub slider position (0–1000) to seconds and push to timeline."""
        dur_ms = self.player._player.duration()
        if dur_ms > 0:
            sec = slider_val / 1000.0 * dur_ms / 1000.0
            self.timeline.set_position(sec)

    def _on_timeline_seek(self, sec: float) -> None:
        """Seek the audio player to a time in seconds (from timeline click)."""
        dur_ms = self.player._player.duration()
        if dur_ms > 0:
            target_ms = int(sec * 1000)
            self.player._player.setPosition(target_ms)

    def _try_load_alignment(self) -> None:
        """
        Load the alignment sidecar if it already exists alongside the audio file.
        Called at card construction time; also callable after a fresh alignment run.
        """
        from app.backend.aligner import read_sidecar
        sidecar = None
        # Prefer path stored on the result (set by AlignmentWorker)
        if self.result.alignment_path:
            from pathlib import Path
            import json
            try:
                with open(self.result.alignment_path, "r", encoding="utf-8") as f:
                    sidecar = json.load(f)
            except (OSError, ValueError):
                pass
        # Fall back to scanning next to the audio file
        if sidecar is None:
            sidecar = read_sidecar(self.result.audio_path)

        if sidecar:
            self.timeline.load_alignment(sidecar)
            self.timeline.setVisible(True)
            self._timeline_divider.setVisible(True)

    def refresh_alignment(self) -> None:
        """Called by GeneratePage after AlignmentWorker finishes on this result."""
        self._try_load_alignment()

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
        self.stems_btn.setObjectName("ActionBtnActive")

        # Phase 3: reveal mixer button now that stems exist
        self.mixer_btn.setVisible(True)

    def _on_mixer(self) -> None:
        """Toggle the stem mixer panel open/closed (Phase 3)."""
        from app.ui.widgets.stem_mixer import StemMixerPanel

        if self._mixer_panel is None:
            # Lazy-create and insert before the placeholder
            self._mixer_panel = StemMixerPanel(self.result)
            self._mixer_panel.mix_rendered.connect(
                lambda path: self.mix_rendered.emit(self.result, path)
            )
            lay = self._frame_layout_ref
            idx = lay.indexOf(self._mixer_panel_placeholder)
            lay.insertWidget(idx, self._mixer_panel)
            self._mixer_divider.setVisible(True)
            self._mixer_panel.setVisible(True)
            self.mixer_btn.setText("🎚 Mixer ▴")
            self.mixer_btn.setObjectName("ActionBtnActive")
            self.mixer_btn.style().unpolish(self.mixer_btn)
            self.mixer_btn.style().polish(self.mixer_btn)
        else:
            visible = not self._mixer_panel.isVisible()
            self._mixer_panel.setVisible(visible)
            self._mixer_divider.setVisible(visible)
            if visible:
                self.mixer_btn.setText("🎚 Mixer ▴")
                self.mixer_btn.setObjectName("ActionBtnActive")
            else:
                self.mixer_btn.setText("🎚 Mixer")
                self.mixer_btn.setObjectName("ActionBtn")
            self.mixer_btn.style().unpolish(self.mixer_btn)
            self.mixer_btn.style().polish(self.mixer_btn)

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
            self.stems_btn.setObjectName("ActionBtn")
            self.stems_btn.style().unpolish(self.stems_btn)
            self.stems_btn.style().polish(self.stems_btn)
        elif self.result.stems:
            self.populate_stems(self.result.stems)
        else:
            self.stems_requested.emit(self.result)
            self.stems_btn.setText("Separating…")
            self.stems_btn.setEnabled(False)

    def enable_stems_btn(self) -> None:
        self.stems_btn.setEnabled(True)
        self.stems_btn.setText("Stems ▾")
        self.stems_btn.setObjectName("ActionBtn")
        self.stems_btn.style().unpolish(self.stems_btn)
        self.stems_btn.style().polish(self.stems_btn)
