"""
ui/widgets/waveform.py — Waveform display widget with drag-to-select region support.
Reads audio file using soundfile / librosa. All paths passed in, nothing hardcoded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


# ---------------------------------------------------------------------------
# Waveform widget
# ---------------------------------------------------------------------------

class WaveformWidget(QWidget):
    """
    Displays a stereo/mono waveform from an audio file.
    Emits region_selected(start_sec, end_sec) when user drags a region.
    """

    region_selected = Signal(float, float)   # start_sec, end_sec
    clicked_at      = Signal(float)          # sec — for playhead / scrub

    BAR_COLOR         = QColor("#85B7EB")
    PLAYHEAD_COLOR    = QColor("#FFFFFF")
    BAR_COLOR_HOVER   = QColor("#5A9FD8")
    REGION_FILL       = QColor(239, 159, 39, 45)
    REGION_BORDER     = QColor("#EF9F27")
    BG_COLOR          = QColor("#1E1E1E")
    TIME_LABEL_COLOR  = QColor("#888888")

    def __init__(self, parent=None, selectable: bool = False) -> None:
        super().__init__(parent)
        self._selectable    = selectable
        self._peaks: Optional[np.ndarray] = None  # shape (N,) normalised 0-1
        self._duration      = 0.0
        self._audio_path    = ""

        # Playhead position (0.0–1.0 fraction of duration)
        self._playhead_pos: float = -1.0   # -1 = hidden

        # Region selection state
        self._drag_start_x: Optional[int] = None
        self._drag_end_x:   Optional[int] = None
        self._region_start  = 0.0
        self._region_end    = 0.0

        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if selectable:
            self.setCursor(Qt.CrossCursor)

    # --- Public API ---------------------------------------------------------

    def load_audio(self, path: str) -> None:
        self._audio_path = path
        self._peaks      = None
        self._duration   = 0.0
        self._region_start = 0.0
        self._region_end   = 0.0
        self._load_peaks()
        self.update()

    def clear(self) -> None:
        self._peaks      = None
        self._duration   = 0.0
        self._audio_path = ""
        self.update()

    @property
    def region_start(self) -> float:
        return self._region_start

    @property
    def region_end(self) -> float:
        return self._region_end

    @property
    def duration(self) -> float:
        return self._duration

    def set_playhead(self, fraction: float) -> None:
        """Set playhead position as a 0.0–1.0 fraction of duration. -1 to hide."""
        self._playhead_pos = fraction
        self.update()

    def set_height(self, h: int) -> None:
        self.setFixedHeight(h)

    # --- Peak extraction ----------------------------------------------------

    def _load_peaks(self) -> None:
        if not self._audio_path or not Path(self._audio_path).exists():
            return
        try:
            import soundfile as sf
            import os, sys
            # Suppress ffmpeg/libsndfile stderr noise on Windows
            devnull = open(os.devnull, 'w')
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                data, sr = sf.read(self._audio_path, dtype="float32")
            finally:
                sys.stderr = old_stderr
                devnull.close()
            if data.ndim > 1:
                data = data.mean(axis=1)
            self._duration = len(data) / sr

            # Downsample to ~300 bars
            n_bars = 300
            chunk  = max(len(data) // n_bars, 1)
            peaks  = []
            for i in range(n_bars):
                seg = data[i * chunk: (i + 1) * chunk]
                peaks.append(float(np.abs(seg).max()) if len(seg) else 0.0)
            peaks = np.array(peaks, dtype=np.float32)
            mx = peaks.max()
            if mx > 0:
                peaks /= mx
            self._peaks = peaks
        except Exception as exc:
            print(f"[waveform] Could not load audio: {exc}")
            # Generate placeholder bars
            rng = np.random.default_rng(42)
            self._peaks = rng.uniform(0.15, 0.95, 300).astype(np.float32)
            self._duration = 30.0

    # --- Paint --------------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)

        w = self.width()
        h = self.height()

        p.fillRect(0, 0, w, h, self.BG_COLOR)

        if self._peaks is not None and len(self._peaks):
            self._draw_bars(p, w, h)

        if self._selectable and self._drag_start_x is not None and self._drag_end_x is not None:
            self._draw_region(p, h)

        if self._playhead_pos >= 0:
            self._draw_playhead(p, w, h)

        if self._duration > 0:
            self._draw_time_labels(p, w, h)

        p.end()

    def _draw_bars(self, p: QPainter, w: int, h: int) -> None:
        n     = len(self._peaks)
        bar_w = max(w / n - 0.5, 1)
        cx    = h / 2
        p.setPen(Qt.NoPen)

        for i, peak in enumerate(self._peaks):
            x      = int(i * w / n)
            bh     = max(int(peak * cx * 0.92), 2)
            color  = self.BAR_COLOR
            p.setBrush(color)
            p.drawRect(x, int(cx - bh), int(bar_w), bh * 2)

    def _draw_playhead(self, p: QPainter, w: int, h: int) -> None:
        x = int(self._playhead_pos * w)
        pen = QPen(self.PLAYHEAD_COLOR, 1)
        pen.setStyle(Qt.SolidLine)
        p.setPen(pen)
        p.drawLine(x, 0, x, h)

    def _draw_region(self, p: QPainter, h: int) -> None:
        x1 = min(self._drag_start_x, self._drag_end_x)
        x2 = max(self._drag_start_x, self._drag_end_x)
        p.fillRect(x1, 0, x2 - x1, h, self.REGION_FILL)
        pen = QPen(self.REGION_BORDER, 2)
        p.setPen(pen)
        p.drawLine(x1, 0, x1, h)
        p.drawLine(x2, 0, x2, h)

    def _draw_time_labels(self, p: QPainter, w: int, h: int) -> None:
        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        p.setPen(self.TIME_LABEL_COLOR)
        p.drawText(4, h - 3, self._fmt(0))
        p.drawText(w // 2 - 15, h - 3, self._fmt(self._duration / 2))
        end_lbl = self._fmt(self._duration)
        p.drawText(w - len(end_lbl) * 6 - 4, h - 3, end_lbl)

        if self._selectable and self._drag_start_x is not None and self._drag_end_x is not None:
            p.setPen(QColor("#EF9F27"))
            lbl = f"{self._fmt(self._region_start)} – {self._fmt(self._region_end)}"
            p.drawText(w // 2 - len(lbl) * 3, 14, lbl)

    def _fmt(self, sec: float) -> str:
        s = int(sec)
        m, s = divmod(s, 60)
        return f"{m}:{s:02d}"

    def _x_to_sec(self, x: int) -> float:
        if self._duration <= 0:
            return 0.0
        return max(0.0, min(self._duration, x / self.width() * self._duration))

    # --- Mouse events (region selection) ------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._selectable:
            return
        if event.button() == Qt.LeftButton:
            self._drag_start_x = event.position().toPoint().x()
            self._drag_end_x   = self._drag_start_x
            self._region_start = self._x_to_sec(self._drag_start_x)
            self._region_end   = self._region_start
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._selectable or self._drag_start_x is None:
            return
        self._drag_end_x  = event.position().toPoint().x()
        self._region_end  = self._x_to_sec(self._drag_end_x)
        self._region_start = self._x_to_sec(self._drag_start_x)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self._selectable or self._drag_start_x is None:
            return
        self._drag_end_x = event.position().toPoint().x()
        start = self._x_to_sec(min(self._drag_start_x, self._drag_end_x))
        end   = self._x_to_sec(max(self._drag_start_x, self._drag_end_x))
        if end - start > 0.1:
            self._region_start = start
            self._region_end   = end
            self.region_selected.emit(start, end)
        self.update()
