"""
backend/presets.py — Style preset manager.

Presets are stored as individual .json files in:
    %APPDATA%\\stitch\\presets\\

Each preset captures a generation mode + all its parameters so a user
can recall a favourite setup in one click.

Usage:
    from app.backend.presets import presets
    presets.save("Chase Scene", "text", {"style_prompt": "...", ...})
    all_presets = presets.list_all()
    params = presets.load("Chase Scene")
    presets.delete("Chase Scene")
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from app.backend.logger import log


class PresetManager:

    def __init__(self, presets_dir: Path) -> None:
        self._dir = presets_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------

    def save(self, name: str, mode: str, params: dict) -> None:
        """Save or overwrite a preset by name."""
        data = {
            "name":       name,
            "mode":       mode,
            "params":     params,
            "created_at": time.time(),
        }
        path = self._name_to_path(name)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            log.info(f"Preset saved: {name}")
        except OSError as exc:
            log.warning(f"Could not save preset '{name}': {exc}")
            raise

    def load(self, name: str) -> dict | None:
        """Return preset dict or None if not found."""
        path = self._name_to_path(name)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(f"Could not load preset '{name}': {exc}")
            return None

    def delete(self, name: str) -> bool:
        path = self._name_to_path(name)
        try:
            path.unlink(missing_ok=True)
            log.info(f"Preset deleted: {name}")
            return True
        except OSError as exc:
            log.warning(f"Could not delete preset '{name}': {exc}")
            return False

    def list_all(self) -> list[dict]:
        """Return all presets sorted by creation time (newest first)."""
        result = []
        for p in self._dir.glob("*.json"):
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                result.append(data)
            except Exception:
                pass
        result.sort(key=lambda d: d.get("created_at", 0), reverse=True)
        return result

    def exists(self, name: str) -> bool:
        return self._name_to_path(name).exists()

    # -----------------------------------------------------------------------

    def _name_to_path(self, name: str) -> Path:
        """Sanitise preset name to a safe filename."""
        safe = re.sub(r'[^\w\s\-]', '', name).strip()
        safe = re.sub(r'\s+', '_', safe)[:80] or "preset"
        return self._dir / f"{safe}.json"


# ---------------------------------------------------------------------------
# Singleton — resolved lazily to avoid circular import with config
# ---------------------------------------------------------------------------

def _get_presets_dir() -> Path:
    import os, sys
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    return Path(base) / "stitch" / "presets"


presets = PresetManager(_get_presets_dir())
