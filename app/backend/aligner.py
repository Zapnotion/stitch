"""
backend/aligner.py — Word-level lyric alignment via whisper-timestamped.

Produces a JSON sidecar alongside every generated WAV:
    outputs/text_v01_1234_abc123.wav
    outputs/text_v01_1234_abc123.alignment.json

Sidecar schema:
    {
      "words": [{"word": "mountain", "start": 1.23, "end": 1.61, "confidence": 0.82}],
      "sections": [{"label": "Verse 1", "start": 0.0, "end": 14.5}]
    }

Falls back gracefully if whisper-timestamped is not installed — every
call path checks ALIGNMENT_AVAILABLE before touching the library.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Availability guard — fallback pattern from the brief
# ---------------------------------------------------------------------------

try:
    import whisper_timestamped as _wt
    ALIGNMENT_AVAILABLE = True
except ImportError:
    ALIGNMENT_AVAILABLE = False

# Whisper model handle — loaded once and reused across calls.
# CPU-only; never competes with ACEStep's GPU budget.
_model: object = None
_model_name: str = ""


def _load_model(name: str = "base") -> object:
    """Load (or return cached) whisper model."""
    global _model, _model_name
    if _model is None or _model_name != name:
        _model = _wt.load_model(name, device="cpu")
        _model_name = name
    return _model


# ---------------------------------------------------------------------------
# Section extraction helpers
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"\[([^\]]+)\]")


def _extract_sections(lyrics: str, words: list[dict]) -> list[dict]:
    """
    Map [Verse 1], [Chorus] etc. from the original lyrics onto time positions
    by finding the nearest word boundary after each section tag.

    Strategy: count non-tag words to find the word index where each section
    starts, then look up that word's start time.
    """
    sections: list[dict] = []
    lines = lyrics.splitlines()

    word_idx = 0
    pending_label: Optional[str] = None
    pending_word_idx: int = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tag_match = _SECTION_RE.fullmatch(stripped)
        if tag_match:
            # Flush previous section
            if pending_label is not None and word_idx > pending_word_idx:
                start_t = words[pending_word_idx]["start"] if pending_word_idx < len(words) else 0.0
                end_t   = words[word_idx - 1]["end"]       if word_idx - 1 < len(words) else start_t
                sections.append({"label": pending_label, "start": start_t, "end": end_t})
            pending_label    = tag_match.group(1)
            pending_word_idx = word_idx
        else:
            # Count the words on this line
            word_idx += len(stripped.split())

    # Flush final section
    if pending_label is not None and words:
        start_t = words[pending_word_idx]["start"] if pending_word_idx < len(words) else 0.0
        end_t   = words[-1]["end"]
        sections.append({"label": pending_label, "start": start_t, "end": end_t})

    return sections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align(
    audio_path: str,
    lyrics: str,
    model_name: str = "base",
) -> Optional[dict]:
    """
    Run forced alignment on *audio_path* against *lyrics*.

    Returns the sidecar dict, or None if whisper-timestamped is not
    installed or if alignment fails for any reason.

    The sidecar is NOT written to disk here — the caller (worker.py) owns
    that responsibility so it can resolve the correct output path via cfg.
    """
    if not ALIGNMENT_AVAILABLE:
        return None

    if not Path(audio_path).exists():
        return None

    try:
        model  = _load_model(model_name)
        result = _wt.transcribe(
            model,
            audio_path,
            language="en",
            task="transcribe",
            # word-level timestamps are the whole point
            compute_word_confidence=True,
            trust_whisper_timestamps=True,
        )

        # Flatten all segments → word list
        words: list[dict] = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append({
                    "word":       w.get("text", "").strip(),
                    "start":      round(float(w.get("start", 0.0)), 3),
                    "end":        round(float(w.get("end",   0.0)), 3),
                    "confidence": round(float(w.get("confidence", 1.0)), 3),
                })

        sections = _extract_sections(lyrics, words) if lyrics.strip() else []

        return {"words": words, "sections": sections}

    except Exception as exc:  # noqa: BLE001
        from app.backend.logger import log
        log.warning(f"[aligner] Alignment failed for {audio_path}: {exc}")
        return None


def write_sidecar(audio_path: str, sidecar: dict) -> str:
    """
    Write *sidecar* as a .alignment.json file next to *audio_path*.
    Returns the sidecar file path string.
    """
    sidecar_path = Path(audio_path).with_suffix("").with_suffix(".alignment.json")
    # Handle double-extension case: text_v01.wav → text_v01.alignment.json
    # Path.with_suffix replaces the last suffix, so we need one call:
    sidecar_path = Path(str(Path(audio_path)) + ".alignment.json")
    # Canonical form: strip the original extension first
    sidecar_path = Path(audio_path).with_name(Path(audio_path).stem + ".alignment.json")

    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)

    return str(sidecar_path)


def read_sidecar(audio_path: str) -> Optional[dict]:
    """
    Load the alignment sidecar for *audio_path* if it exists.
    Returns the dict, or None.
    """
    sidecar_path = Path(audio_path).with_name(Path(audio_path).stem + ".alignment.json")
    if not sidecar_path.exists():
        return None
    try:
        with sidecar_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def sidecar_path_for(audio_path: str) -> Path:
    """Return the expected sidecar Path for a given audio file."""
    return Path(audio_path).with_name(Path(audio_path).stem + ".alignment.json")
