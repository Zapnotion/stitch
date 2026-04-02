"""
backend/log_relay.py — Tails stitch.log and emits lines via a Qt signal.

Used by the in-app log panel. Runs a background QThread that watches
the log file and emits new lines as they appear.
"""
from __future__ import annotations
import os
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from app.config import LOGS_DIR

LOG_PATH = LOGS_DIR / "stitch.log"


class LogRelayThread(QThread):
    new_line = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def run(self):
        # Seek to end first so we only show lines from this session
        pos = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
        while not self._stop:
            try:
                with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        stripped = line.rstrip()
                        if stripped:
                            self.new_line.emit(stripped)
                    pos = f.tell()
            except Exception:
                pass
            self.msleep(250)  # poll 4x/sec — negligible CPU

    def stop(self):
        self._stop = True
        self.wait(1000)
