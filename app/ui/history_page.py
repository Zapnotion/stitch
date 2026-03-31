"""
ui/history_page.py — Session history viewer.

Shows a table of past generations loaded from session_history.json.
Each row has: timestamp, mode, duration, prompt snippet, seed, and a
"Load into player" button that opens the audio file in a dialog player.
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QScrollArea, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from app.backend.session import session
from app.ui.widgets.audio_player import AudioPlayer


class HistoryPage(QWidget):
    """Read-only history browser."""

    load_into_generate = Signal(dict)   # emitted when "Send to Generate" clicked

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()
        self.refresh()

    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        hdr = QWidget()
        hdr.setObjectName("RightHeader")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(14, 10, 14, 10)
        title = QLabel("Generation History")
        title.setObjectName("RightTitle")
        hdr_lay.addWidget(title)
        hdr_lay.addStretch()

        self._refresh_btn = QPushButton("↻  Refresh")
        self._refresh_btn.setObjectName("ActionBtn")
        self._refresh_btn.clicked.connect(self.refresh)
        hdr_lay.addWidget(self._refresh_btn)

        self._clear_btn = QPushButton("Clear history")
        self._clear_btn.setObjectName("ActionBtn")
        self._clear_btn.clicked.connect(self._on_clear)
        hdr_lay.addWidget(self._clear_btn)

        root.addWidget(hdr)

        # Split: table on left, mini-player on right
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Table
        self._table = QTableWidget()
        self._table.setObjectName("HistoryTable")
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["Time", "Mode", "Duration", "Prompt", "Seed", ""]
        )
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.horizontalHeader().setDefaultSectionSize(90)
        self._table.setColumnWidth(0, 110)
        self._table.setColumnWidth(1, 70)
        self._table.setColumnWidth(2, 68)
        self._table.setColumnWidth(4, 70)
        self._table.setColumnWidth(5, 90)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        body.addWidget(self._table, 1)

        # Right detail panel
        detail = QWidget()
        detail.setObjectName("HistoryDetail")
        detail.setFixedWidth(220)
        detail_lay = QVBoxLayout(detail)
        detail_lay.setContentsMargins(14, 14, 14, 14)
        detail_lay.setSpacing(10)

        self._detail_title = QLabel("Select a row")
        self._detail_title.setObjectName("FieldLabel")
        self._detail_title.setWordWrap(True)
        detail_lay.addWidget(self._detail_title)

        self._detail_player = AudioPlayer()
        detail_lay.addWidget(self._detail_player)

        self._detail_path = QLabel("")
        self._detail_path.setObjectName("WaveFilename")
        self._detail_path.setWordWrap(True)
        detail_lay.addWidget(self._detail_path)

        self._send_btn = QPushButton("Send to Generate")
        self._send_btn.setObjectName("GenBtn")
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._on_send)
        detail_lay.addWidget(self._send_btn)

        detail_lay.addStretch()
        body.addWidget(detail)

        root.addLayout(body, 1)

        self._selected_entry: dict | None = None

    # -----------------------------------------------------------------------

    def refresh(self) -> None:
        entries = session.load_history(limit=200)
        self._table.setRowCount(0)
        self._table.setRowCount(len(entries))
        self._entries = entries

        for row, e in enumerate(entries):
            ts   = time.strftime("%m-%d %H:%M", time.localtime(e.get("timestamp", 0)))
            mode = e.get("mode", "")
            dur  = self._fmt(e.get("duration_secs", 0))
            prompt = (e.get("style_prompt", "") or "")[:60]
            seed   = str(e.get("seed", ""))
            path   = e.get("audio_path", "")
            exists = Path(path).exists()

            self._table.setItem(row, 0, self._cell(ts))
            self._table.setItem(row, 1, self._cell(mode))
            self._table.setItem(row, 2, self._cell(dur))
            self._table.setItem(row, 3, self._cell(prompt))
            self._table.setItem(row, 4, self._cell(seed))

            play_btn = QPushButton("▶  Play" if exists else "✕  Missing")
            play_btn.setObjectName("ActionBtn")
            play_btn.setEnabled(exists)
            if exists:
                play_btn.clicked.connect(lambda _, p=path: self._play_path(p))
            self._table.setCellWidget(row, 5, play_btn)
            self._table.setRowHeight(row, 30)

    def _cell(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setForeground(Qt.GlobalColor.lightGray)
        return item

    # -----------------------------------------------------------------------

    def _on_row_selected(self) -> None:
        rows = self._table.selectedItems()
        if not rows:
            return
        row = self._table.currentRow()
        if row < 0 or row >= len(self._entries):
            return
        entry = self._entries[row]
        self._selected_entry = entry
        path = entry.get("audio_path", "")
        prompt = (entry.get("style_prompt", "") or "")[:80]
        self._detail_title.setText(prompt or "(no prompt)")
        self._detail_path.setText(Path(path).name if path else "")
        if Path(path).exists():
            self._detail_player.load(path)
            self._send_btn.setEnabled(True)
        else:
            self._send_btn.setEnabled(False)

    def _play_path(self, path: str) -> None:
        self._detail_player.load(path)

    def _on_send(self) -> None:
        if self._selected_entry:
            self.load_into_generate.emit(self._selected_entry)

    def _on_clear(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Clear History",
            "Clear all session history?\nThis does not delete audio files.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            session.clear()
            self.refresh()

    @staticmethod
    def _fmt(sec: float) -> str:
        s = int(sec)
        m, s = divmod(s, 60)
        return f"{m}:{s:02d}"
