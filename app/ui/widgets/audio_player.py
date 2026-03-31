"""
ui/widgets/audio_player.py — Inline audio player widget.

Uses PySide6's QMediaPlayer for playback — no external deps.
Emits playback_started / playback_stopped signals for parent coordination.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, QUrl, Qt, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton,
    QSlider, QWidget,
)


class AudioPlayer(QWidget):
    """
    Compact transport bar: play/pause, scrub slider, time label.
    Attach to any audio file path via load().
    """

    playback_started = Signal(str)   # path
    playback_stopped = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._path    = ""
        self._seeking = False

        self._player = QMediaPlayer(self)
        self._audio  = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._audio.setVolume(1.0)

        self._player.playbackStateChanged.connect(self._on_state_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.errorOccurred.connect(self._on_error)

        self._build_ui()

        # Poll timer as backup for position updates
        self._poll = QTimer(self)
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._poll_position)

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._play_btn = QPushButton("▶")
        self._play_btn.setObjectName("PlayBtn")
        self._play_btn.setFixedSize(26, 26)
        self._play_btn.clicked.connect(self._toggle_play)
        lay.addWidget(self._play_btn)

        self._scrub = QSlider(Qt.Horizontal)
        self._scrub.setObjectName("ScrubSlider")
        self._scrub.setRange(0, 1000)
        self._scrub.setValue(0)
        self._scrub.sliderPressed.connect(self._on_scrub_press)
        self._scrub.sliderReleased.connect(self._on_scrub_release)
        self._scrub.sliderMoved.connect(self._on_scrub_move)
        lay.addWidget(self._scrub, 1)

        self._time_lbl = QLabel("0:00 / 0:00")
        self._time_lbl.setObjectName("TimeLabel")
        self._time_lbl.setFixedWidth(80)
        lay.addWidget(self._time_lbl)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def load(self, path: str) -> None:
        """Load a new audio file. Does not auto-play."""
        if not Path(path).exists():
            return
        self._path = path
        self._player.setSource(QUrl.fromLocalFile(path))
        self._play_btn.setText("▶")
        self._scrub.setValue(0)
        self._time_lbl.setText("0:00 / 0:00")

    def stop(self) -> None:
        self._player.stop()
        self._play_btn.setText("▶")
        self._poll.stop()

    def set_volume(self, v: float) -> None:
        self._audio.setVolume(max(0.0, min(1.0, v)))

    # -----------------------------------------------------------------------
    # Playback control
    # -----------------------------------------------------------------------

    def _toggle_play(self) -> None:
        if not self._path:
            return
        state = self._player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
            self._play_btn.setText("▶")
            self._poll.stop()
        else:
            self._player.play()
            self._play_btn.setText("⏸")
            self._poll.start()
            self.playback_started.emit(self._path)

    def _on_state_changed(self, state) -> None:
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self._play_btn.setText("▶")
            self._scrub.setValue(0)
            self._poll.stop()
            self.playback_stopped.emit(self._path)

    # -----------------------------------------------------------------------
    # Position tracking
    # -----------------------------------------------------------------------

    def _on_position_changed(self, pos_ms: int) -> None:
        if not self._seeking:
            dur = self._player.duration()
            if dur > 0:
                self._scrub.setValue(int(pos_ms / dur * 1000))
        self._update_time_label()

    def _on_duration_changed(self, dur_ms: int) -> None:
        self._update_time_label()

    def _poll_position(self) -> None:
        self._on_position_changed(self._player.position())

    def _update_time_label(self) -> None:
        pos = self._player.position() // 1000
        dur = self._player.duration() // 1000
        self._time_lbl.setText(f"{self._fmt(pos)} / {self._fmt(dur)}")

    @staticmethod
    def _fmt(sec: int) -> str:
        m, s = divmod(max(sec, 0), 60)
        return f"{m}:{s:02d}"

    # -----------------------------------------------------------------------
    # Scrubbing
    # -----------------------------------------------------------------------

    def _on_scrub_press(self) -> None:
        self._seeking = True

    def _on_scrub_release(self) -> None:
        self._seeking = False
        dur = self._player.duration()
        if dur > 0:
            target = int(self._scrub.value() / 1000 * dur)
            self._player.setPosition(target)

    def _on_scrub_move(self, val: int) -> None:
        dur = self._player.duration()
        if dur > 0:
            pos_sec = int(val / 1000 * dur / 1000)
            self._time_lbl.setText(f"{self._fmt(pos_sec)} / {self._fmt(dur // 1000)}")

    # -----------------------------------------------------------------------
    # Error
    # -----------------------------------------------------------------------

    def _on_error(self, error, msg: str) -> None:
        print(f"[player] Playback error: {msg}")
