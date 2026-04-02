"""
ui/widgets/log_panel.py — Collapsible real-time log panel.

Shows a scrolling view of stitch.log, colour-coded by level.
Collapsed by default; user can expand via the toggle button.
"""
from __future__ import annotations
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QHBoxLayout, QPlainTextEdit, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)
from app.backend.log_relay import LogRelayThread

# Colour map by log level keyword
_COLOURS = {
    "ERROR":   "#E05555",
    "WARNING": "#D4A84B",
    "WARN":    "#D4A84B",
    "INFO":    "#7EC2F8",
    "DEBUG":   "#666666",
    "OK":      "#7EC878",
    "DIAG":    "#B07BF8",
}
_DEFAULT_COLOUR = "#888888"


def _colour_for_line(line: str) -> str:
    upper = line.upper()
    for kw, col in _COLOURS.items():
        if kw in upper:
            return col
    return _DEFAULT_COLOUR


class LogPanel(QWidget):
    """Collapsible log panel — add to main window layout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._expanded = False
        self._build_ui()
        self._relay = LogRelayThread(self)
        self._relay.new_line.connect(self._append)
        self._relay.start()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Toggle bar
        bar = QWidget()
        bar.setObjectName("LogBar")
        bar.setFixedHeight(24)
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(10, 0, 10, 0)
        bar_lay.setSpacing(6)

        self._toggle = QPushButton("▶  Console")
        self._toggle.setObjectName("LogToggle")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.clicked.connect(self._on_toggle)
        bar_lay.addWidget(self._toggle)
        bar_lay.addStretch()

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("LogClear")
        self._clear_btn.setVisible(False)
        self._clear_btn.clicked.connect(self._log.clear if hasattr(self, '_log') else lambda: None)
        bar_lay.addWidget(self._clear_btn)

        root.addWidget(bar)

        # Log text area (hidden by default)
        self._log = QPlainTextEdit()
        self._log.setObjectName("LogView")
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setFixedHeight(180)
        self._log.setVisible(False)
        self._log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._clear_btn.clicked.connect(self._log.clear)
        root.addWidget(self._log)

    def _on_toggle(self, checked: bool):
        self._expanded = checked
        self._toggle.setText(("▼" if checked else "▶") + "  Console")
        self._log.setVisible(checked)
        self._clear_btn.setVisible(checked)
        if checked:
            # Scroll to bottom when opened
            self._log.moveCursor(QTextCursor.End)

    @Slot(str)
    def _append(self, line: str):
        colour = _colour_for_line(line)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(colour))
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(line + "\n", fmt)
        if self._expanded:
            self._log.setTextCursor(cursor)
            self._log.ensureCursorVisible()

    def shutdown(self):
        self._relay.stop()
