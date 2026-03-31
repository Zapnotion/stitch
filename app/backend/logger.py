"""
backend/logger.py — Application logger.

Writes rotating logs to %APPDATA%\\stitch\\logs\\stitch.log
Max 5 MB per file, keeps 3 backups.
Also mirrors to stdout in debug mode.

Usage:
    from app.backend.logger import log
    log.info("Model loaded")
    log.warning("VRAM low")
    log.error("Generation failed", exc_info=True)
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _build_logger(log_dir: Path, debug: bool = False) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "stitch.log"

    logger = logging.getLogger("stitch")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if logger.handlers:
        return logger   # already configured (re-import guard)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    fh = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Console handler (stdout) — always show WARNING+ unless debug
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.DEBUG if debug else logging.WARNING)
    logger.addHandler(ch)

    return logger


def _get_log_dir() -> Path:
    """Import-safe — avoids circular import with config."""
    import os
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    return Path(base) / "stitch" / "logs"


# Debug mode: set STITCH_DEBUG=1 in environment
_debug = "STITCH_DEBUG" in __import__("os").environ

log: logging.Logger = _build_logger(_get_log_dir(), debug=_debug)
