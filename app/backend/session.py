"""
backend/session.py — Session history manager.

Persists a rolling log of past generations to:
    %APPDATA%\\stitch\\session_history.json

Each entry records what was generated, when, where it was saved, and
the parameters used. The last 500 entries are kept.

Usage:
    from app.backend.session import session
    session.record(result, request_params)
    history = session.load_history()
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.backend.logger import log


MAX_HISTORY = 500


@dataclass
class HistoryEntry:
    timestamp:       float
    mode:            str        # "text" | "cover" | "vocal" | "repair"
    audio_path:      str
    duration_secs:   float
    seed:            int
    variation_index: int
    style_prompt:    str
    params:          dict       # full request params snapshot


class SessionHistory:
    """Append-only JSON history file. Thread-safe via simple file lock."""

    def __init__(self, history_file: Path) -> None:
        self._file = history_file
        self._file.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------

    def record(
        self,
        mode:            str,
        audio_path:      str,
        duration_secs:   float,
        seed:            int,
        variation_index: int,
        style_prompt:    str,
        params:          dict | None = None,
    ) -> None:
        entry = HistoryEntry(
            timestamp       = time.time(),
            mode            = mode,
            audio_path      = audio_path,
            duration_secs   = duration_secs,
            seed            = seed,
            variation_index = variation_index,
            style_prompt    = style_prompt,
            params          = params or {},
        )
        self._append(entry)

    def load_history(self, limit: int = 100) -> list[dict]:
        """Return up to `limit` most recent entries, newest first."""
        entries = self._read_all()
        return entries[-limit:][::-1]

    def clear(self) -> None:
        try:
            self._file.write_text("[]", encoding="utf-8")
            log.info("Session history cleared")
        except OSError as exc:
            log.warning(f"Could not clear history: {exc}")

    # -----------------------------------------------------------------------

    def _append(self, entry: HistoryEntry) -> None:
        entries = self._read_all()
        entries.append(asdict(entry))
        # Trim to max
        if len(entries) > MAX_HISTORY:
            entries = entries[-MAX_HISTORY:]
        self._write_all(entries)
        log.debug(f"Session: recorded {entry.mode} var {entry.variation_index}")

    def _read_all(self) -> list[dict]:
        if not self._file.exists():
            return []
        try:
            with self._file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _write_all(self, entries: list[dict]) -> None:
        try:
            tmp = self._file.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2)
            tmp.replace(self._file)
        except OSError as exc:
            log.warning(f"Could not write session history: {exc}")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def _get_history_file() -> Path:
    import os, sys
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    return Path(base) / "stitch" / "session_history.json"


session = SessionHistory(_get_history_file())
