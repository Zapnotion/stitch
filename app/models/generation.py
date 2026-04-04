"""
models/generation.py — Pydantic-style dataclasses for generation requests and results.
No paths hardcoded — all paths passed in as strings resolved by config.py upstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GenerationMode(str, Enum):
    TEXT      = "text"
    COVER     = "cover"
    VOCAL     = "vocal"
    REPAIR    = "repair"


class OutputType(str, Enum):
    WITH_VOCALS   = "with_vocals"
    INSTRUMENTAL  = "instrumental"


class LyricsMode(str, Enum):
    AI_WRITES = "ai_writes"
    USER      = "user"


class RepairMode(str, Enum):
    REGION    = "region"
    FULL_FILE = "full_file"


# ---------------------------------------------------------------------------
# Request dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TextGenerationRequest:
    style_prompt:      str
    output_type:       OutputType        = OutputType.WITH_VOCALS
    lyrics_mode:       LyricsMode        = LyricsMode.AI_WRITES
    user_lyrics:       str               = ""
    lyrics_prompt:     str               = ""    # AI writes: what to sing about
    lyrics_creativity: float             = 0.5   # 0.0=safe/predictable, 1.0=wild
    lyrics_adherence:  float             = 0.7   # 0.0=loose, 1.0=tight on topic
    lyrics_model:      str               = ""    # key from LYRICS_MODELS; ""=default
    duration_secs:     int               = 30
    variations:        int               = 4
    seed:              Optional[int]     = None
    lora:              Optional[str]     = None
    output_dir:        str               = ""    # resolved by caller from cfg
    # Phase 2 musical parameters
    bpm:               Optional[int]     = None  # 40–200; None = free
    key:               str               = ""    # e.g. "C major", "A minor"; "" = free
    time_signature:    str               = ""    # e.g. "4/4", "6/8"; "" = free
    exclusions:        str               = ""    # negative prompt; appended as "Avoid: ..."


@dataclass
class CoverRequest:
    source_audio_path: str           # resolved absolute path
    target_style:      str
    new_lyrics:        str           = ""
    follow_strength:   float         = 0.7  # 0=free restyle, 1=preserve structure
    variations:        int           = 4
    seed:              Optional[int] = None
    output_dir:        str           = ""


@dataclass
class VocalBackingRequest:
    vocal_audio_path: str            # dry vocal, absolute path
    backing_style:    str
    variations:       int            = 4
    seed:             Optional[int]  = None
    output_dir:       str            = ""


@dataclass
class RepairRequest:
    source_audio_path: str           # absolute path
    mode:              RepairMode    = RepairMode.REGION
    region_start_sec:  float         = 0.0
    region_end_sec:    float         = 0.0
    hint_prompt:       str           = ""
    output_dir:        str           = ""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    variation_index:   int
    audio_path:        str           # absolute path to output file
    seed:              int           = 0
    duration_secs:     float         = 0.0
    mode:              GenerationMode = GenerationMode.TEXT
    style_prompt:      str           = ""
    lyrics:            str           = ""   # original lyrics used for generation (needed by aligner)
    score:             float         = 0.0  # future: auto-quality score
    starred:           bool          = False
    stems:             dict[str, str] = field(default_factory=dict)
    # stems = {"vocals": "/path/...", "drums": "...", "bass": "...", "other": "..."}
    alignment_path:    Optional[str] = None  # path to .alignment.json sidecar; None until aligned

    @property
    def filename(self) -> str:
        return Path(self.audio_path).name

    @property
    def duration_label(self) -> str:
        total = int(self.duration_secs)
        m, s = divmod(total, 60)
        return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Progress signal payload
# ---------------------------------------------------------------------------

@dataclass
class GenerationProgress:
    variation_index: int
    total_variations: int
    step:            int
    total_steps:     int
    message:         str = ""

    @property
    def overall_pct(self) -> float:
        var_done = (self.variation_index - 1) / max(self.total_variations, 1)
        step_pct = self.step / max(self.total_steps, 1) / self.total_variations
        return min((var_done + step_pct) * 100, 100.0)
