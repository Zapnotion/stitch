"""
config.py — Central configuration for Stitch.

All paths are rooted under %APPDATA%\\stitch (Windows) or
~/.local/share/stitch (fallback). No hardcoded paths elsewhere
in the codebase — always import from here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# App-data root
# ---------------------------------------------------------------------------

def _get_appdata_root() -> Path:
    """Return the OS-appropriate user data directory."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    return Path(base) / "stitch"


APP_DATA_DIR: Path = _get_appdata_root()

# Sub-directories
OUTPUTS_DIR:  Path = APP_DATA_DIR / "outputs"
INPUTS_DIR:   Path = APP_DATA_DIR / "inputs"
MODELS_DIR:   Path = APP_DATA_DIR / "models"
STEMS_DIR:    Path = APP_DATA_DIR / "stems"
PRESETS_DIR:  Path = APP_DATA_DIR / "presets"
LOGS_DIR:     Path = APP_DATA_DIR / "logs"
CONFIG_FILE:  Path = APP_DATA_DIR / "config.json"

_ALL_DIRS = [
    APP_DATA_DIR, OUTPUTS_DIR, INPUTS_DIR,
    MODELS_DIR, STEMS_DIR, PRESETS_DIR, LOGS_DIR,
]


# ---------------------------------------------------------------------------
# Default config values
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "theme": "dark",
    "default_variations": 1,
    "default_duration": 15,
    "default_sample_rate": 44100,
    "demucs_model": "htdemucs",
    "device": "auto",          # "auto" | "cuda" | "cpu"
    "output_format": "wav",    # "wav" | "mp3"
    "mp3_bitrate": 320,
    "max_vram_gb": 10,
    "outputs_dir": str(OUTPUTS_DIR),
    "inputs_dir":  str(INPUTS_DIR),
    "models_dir":  str(MODELS_DIR),
    "stems_dir":   str(STEMS_DIR),
    "lm_model":    "acestep-5Hz-lm-1.7B",
    "lyrics_model": "",   # key from LYRICS_MODELS in lyrics_gen.py; ""=use default (CPU 1.5B)
}


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

class StitchConfig:
    """Loaded from / persisted to CONFIG_FILE. All path access goes here."""

    def __init__(self) -> None:
        self._data: dict = {}
        self._ensure_dirs()
        self._load()

    # --- Init helpers -------------------------------------------------------

    def _ensure_dirs(self) -> None:
        for d in _ALL_DIRS:
            d.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if CONFIG_FILE.exists():
            try:
                with CONFIG_FILE.open("r", encoding="utf-8") as f:
                    stored = json.load(f)
                self._data = {**_DEFAULTS, **stored}
            except (json.JSONDecodeError, OSError):
                self._data = dict(_DEFAULTS)
        else:
            self._data = dict(_DEFAULTS)
            self._save()

    def _save(self) -> None:
        try:
            with CONFIG_FILE.open("w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except OSError as exc:
            print(f"[config] Could not save config: {exc}")

    # --- Public API ---------------------------------------------------------

    def get(self, key: str, fallback=None):
        return self._data.get(key, fallback)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self._save()

    def update(self, updates: dict) -> None:
        self._data.update(updates)
        self._save()

    # --- Resolved paths (always return Path objects) -----------------------

    @property
    def outputs_dir(self) -> Path:
        return Path(self._data.get("outputs_dir", str(OUTPUTS_DIR)))

    @property
    def inputs_dir(self) -> Path:
        return Path(self._data.get("inputs_dir", str(INPUTS_DIR)))

    @property
    def models_dir(self) -> Path:
        return Path(self._data.get("models_dir", str(MODELS_DIR)))

    @property
    def stems_dir(self) -> Path:
        return Path(self._data.get("stems_dir", str(STEMS_DIR)))

    @property
    def device(self) -> str:
        val = self._data.get("device", "auto")
        if val == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return val

    @property
    def lm_model(self) -> str:
        return self._data.get("lm_model", _DEFAULTS["lm_model"])

    @property
    def demucs_model(self) -> str:
        return self._data.get("demucs_model", _DEFAULTS["demucs_model"])

    @property
    def default_variations(self) -> int:
        return int(self._data.get("default_variations", 4))

    @property
    def default_duration(self) -> int:
        return int(self._data.get("default_duration", 30))

    @property
    def output_format(self) -> str:
        return self._data.get("output_format", "wav")

    @property
    def mp3_bitrate(self) -> int:
        return int(self._data.get("mp3_bitrate", 320))

    def __repr__(self) -> str:
        return f"StitchConfig({CONFIG_FILE})"


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------

cfg = StitchConfig()
