"""
backend/lyrics_gen.py — Subject-driven lyric generation for Stitch.

The lyrics_prompt ("what should the AI sing about?") is treated as the
actual subject/story of the song. The style_prompt sets the genre tone.

Approach:
  - Extract key nouns, verbs, adjectives from the lyrics_prompt directly
  - Build lines that reference those actual words/concepts
  - Use genre vocabulary for tone/imagery colour
  - Guarantee variety between variations via seed randomisation
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Optional, Callable

ProgressCb = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------------
# Genre tone banks — colour/texture only, not the main subject
# ---------------------------------------------------------------------------

_GENRE: dict[str, dict] = {
    "pop":        {"feel": ["electric", "alive", "free", "lost", "whole"],
                   "colour": ["neon lights", "summer rain", "midnight sky", "open road"]},
    "rock":       {"feel": ["fearless", "raw", "defiant", "wild", "unbroken"],
                   "colour": ["dark highway", "burning stage", "breaking glass", "loud guitar"]},
    "r&b":        {"feel": ["tender", "aching", "devoted", "warm", "smooth"],
                   "colour": ["candlelight", "velvet night", "golden hour", "slow burn"]},
    "hip-hop":    {"feel": ["hungry", "focused", "real", "relentless", "determined"],
                   "colour": ["late nights", "city block", "top floor", "the grind"]},
    "hiphop":     {"feel": ["hungry", "focused", "real", "relentless", "determined"],
                   "colour": ["late nights", "city block", "top floor", "the grind"]},
    "rap":        {"feel": ["hungry", "focused", "real", "relentless", "determined"],
                   "colour": ["late nights", "city block", "top floor", "the grind"]},
    "country":    {"feel": ["homesick", "grateful", "proud", "simple", "warm"],
                   "colour": ["dirt road", "front porch", "fireflies", "pickup truck"]},
    "electronic": {"feel": ["euphoric", "hypnotic", "infinite", "electric", "free"],
                   "colour": ["bass drop", "neon pulse", "synth wave", "pulsing lights"]},
    "edm":        {"feel": ["euphoric", "hypnotic", "infinite", "electric", "free"],
                   "colour": ["bass drop", "neon pulse", "synth wave", "pulsing lights"]},
    "folk":       {"feel": ["honest", "quiet", "true", "simple", "rooted"],
                   "colour": ["open field", "wooden bridge", "old song", "long road"]},
    "jazz":       {"feel": ["smoky", "cool", "melancholy", "warm", "longing"],
                   "colour": ["blue smoke", "late bar", "slow trumpet", "rainy street"]},
    "sad":        {"feel": ["hollow", "broken", "numb", "tired", "resigned"],
                   "colour": ["empty chair", "grey morning", "faded photograph", "silent phone"]},
    "upbeat":     {"feel": ["joyful", "radiant", "carefree", "unstoppable", "bright"],
                   "colour": ["sunshine", "open windows", "dancing feet", "clear skies"]},
}

_DEFAULT_GENRE = {"feel": ["alive", "lost", "free", "tired", "whole"],
                  "colour": ["open road", "city lights", "midnight hour", "falling rain"]}


def _get_genre(style: str) -> dict:
    low = style.lower()
    for key, bank in _GENRE.items():
        if key in low:
            return bank
    return _DEFAULT_GENRE


# ---------------------------------------------------------------------------
# Subject extraction from lyrics_prompt
# ---------------------------------------------------------------------------

# Stop words to ignore when pulling content words
_STOP = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "about","that","this","it","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","not","no","so","if","as","by","from","up",
    "into","than","then","when","where","who","what","how","very","just",
    "some","any","all","my","your","our","their","its","me","him","her","us",
    "them","i","you","he","she","we","they","get","got","let","make","made",
    "sing","sung","song","music","lyrics","write","about","should","ai",
}

def _extract_subject_words(text: str) -> list[str]:
    """Pull meaningful content words from the lyrics prompt."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [w for w in words if w not in _STOP and len(w) > 2]


def _subject_phrases(text: str) -> list[str]:
    """
    Extract 2-3 word phrases from the prompt that can slot into lyrics.
    e.g. "lady telling children off" → ["telling children", "children off", "lady telling"]
    """
    words = re.findall(r"[a-zA-Z']+", text.lower())
    content = [w for w in words if w not in _STOP and len(w) > 2]
    phrases = []
    for i in range(len(content) - 1):
        phrases.append(f"{content[i]} {content[i+1]}")
    return phrases


# ---------------------------------------------------------------------------
# Line construction — subject-first approach
# ---------------------------------------------------------------------------

# Verse line patterns — {subj} = key subject word/phrase, {feel} = genre feel word,
# {colour} = genre colour image. Lines are designed to be grammatically complete.
_VERSE_PATTERNS = [
    "Every day I think about {subj} and I don't know why",
    "There's something about {subj} that I can't leave behind",
    "I keep coming back to {subj} like a broken song",
    "The thought of {subj} follows me all night long",
    "I never understood {subj} until it was too late",
    "They say {subj} changes everything — I think they're right",
    "I've spent so long with {subj} I don't know who I am",
    "The truth about {subj} is harder than it seems",
    "What do you do when {subj} is all you've got?",
    "I close my eyes and all I see is {subj}",
    "Living with {subj} day after day after day",
    "How many times can {subj} bring me to my knees?",
    # Feel-coloured variants
    "I feel {feel} every time {subj} crosses my mind",
    "Something {feel} stirs in me when I think of {subj}",
    "There's a {feel} truth in {subj} I cannot escape",
    "I've been {feel} since {subj} changed the way I see",
]

_CHORUS_PATTERNS = [
    "This is the part where {subj} breaks me open wide",
    "All roads lead back to {subj} — I've tried to run",
    "{subj} is the thing I carry when the day is done",
    "We all know {subj}, we just don't say it out loud",
    "And still {subj} remains — through everything, it stays",
    "Through every storm {subj} is the only thing that's real",
    "I keep returning to {subj} like a tide to shore",
    "{subj} — that's the word that holds it all in place",
    # Generic strong hooks
    "Hold on — we'll get through this somehow",
    "This is what it means to feel {feel} and still survive",
    "We fall and we rise — {subj} is how we learn",
    "The {colour} reminds me why I carry on",
]

_BRIDGE_PATTERNS = [
    "Maybe {subj} was the lesson all along",
    "I never wanted {subj} to mean this much to me",
    "If I could change {subj} I don't know that I would",
    "All the things I said about {subj} — I meant them all",
    "There's no escaping {subj}, so I'll make my peace",
    "In the end {subj} is what I'll remember most",
]


def _pick(lst: list, used: set) -> str:
    available = [x for x in lst if x not in used]
    if not available:
        used.clear()
        available = lst[:]
    choice = random.choice(available)
    used.add(choice)
    return choice


def _make_line(pattern: str, subj: str, genre: dict,
               used_feel: set, used_colour: set) -> str:
    feel   = _pick(genre["feel"],   used_feel)
    colour = _pick(genre["colour"], used_colour)
    return (pattern
            .replace("{subj}",   subj)
            .replace("{feel}",   feel)
            .replace("{colour}", colour))


def _capitalise(line: str) -> str:
    return line[:1].upper() + line[1:] if line else line


def _build_verse(subj: str, genre: dict, used_v: set,
                 used_feel: set, used_colour: set) -> list[str]:
    lines = []
    for _ in range(4):
        pattern = _pick(_VERSE_PATTERNS, used_v)
        line = _make_line(pattern, subj, genre, used_feel, used_colour)
        lines.append(_capitalise(line))
    return lines


def _build_chorus(subj: str, genre: dict) -> list[str]:
    used_c: set = set()
    used_f: set = set()
    used_col: set = set()
    lines = []
    for _ in range(4):
        pattern = _pick(_CHORUS_PATTERNS, used_c)
        line = _make_line(pattern, subj, genre, used_f, used_col)
        lines.append(_capitalise(line))
    return lines


def _build_bridge(subj: str, genre: dict) -> str:
    used_b: set = set()
    pattern = _pick(_BRIDGE_PATTERNS, used_b)
    return _capitalise(_make_line(pattern, subj, genre, set(), set()))


# ---------------------------------------------------------------------------
# Subject string assembly
# ---------------------------------------------------------------------------

def _build_subject(lyrics_prompt: str, style_prompt: str) -> str:
    """
    Turn the lyrics_prompt into a compact subject phrase for lyrics lines.
    Keeps it short enough to fit naturally mid-sentence (≤6 words ideally).
    """
    lp = lyrics_prompt.strip().strip(".!?")
    if not lp:
        words = [w for w in style_prompt.lower().split() if w not in _STOP]
        return words[0] if words else "this feeling"

    words = lp.split()

    # If it's long (>7 words), trim at clause boundaries
    if len(words) > 7:
        core_words = []
        clause_breaks = {"who","that","which","when","where","after","before",
                         "because","since","until","while","although","but","and"}
        for w in words:
            if w.lower() in clause_breaks and len(core_words) >= 3:
                break
            core_words.append(w)
        lp = " ".join(core_words) if core_words else " ".join(words[:6])

    # If it starts with a gerund (verb+ing), return as-is — reads naturally
    first = lp.split()[0].lower() if lp.split() else ""
    if first.endswith("ing"):
        return lp

    # If it already starts with an article/pronoun, return as-is
    starters = {"the","a","an","my","your","our","their","his","her","its",
                "this","that","being","how","why","what"}
    if first in starters:
        return lp

    # Short noun phrase (≤4 words) — prepend "the"
    if len(lp.split()) <= 4:
        return "the " + lp

    return lp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_lyrics(
    style_prompt:  str,
    lyrics_prompt: str = "",
    models_dir:    "Optional[Path]" = None,
    progress_cb:   ProgressCb = None,
) -> str:
    """
    Generate lyrics driven by lyrics_prompt as the actual subject/story.
    Always returns a non-empty string. No network, no VRAM, no downloads.
    """
    def _cb(msg: str):
        if progress_cb:
            progress_cb(msg)

    _cb("Generating lyrics...")

    genre = _get_genre(style_prompt)
    subj  = _build_subject(lyrics_prompt, style_prompt)

    # Use time-based seed so every generation is different even for same prompt
    # (user expects variation, not the same lyrics every time)
    random.seed(None)

    used_v:      set = set()
    used_feel:   set = set()
    used_colour: set = set()

    v1 = _build_verse(subj, genre, used_v, used_feel, used_colour)
    ch = _build_chorus(subj, genre)
    v2 = _build_verse(subj, genre, used_v, used_feel, used_colour)
    br = _build_bridge(subj, genre)

    parts = (
        ["[Verse 1]"] + v1 + [""]
        + ["[Chorus]"]  + ch + [""]
        + ["[Verse 2]"] + v2 + [""]
        + ["[Chorus]"]  + ch + [""]
        + ["[Bridge]", br, ""]
        + ["[Chorus]"]  + ch
    )

    lyrics = "\n".join(parts)
    _cb(f"Lyrics ready ({len(lyrics.split())} words) — about: {subj!r}")
    return lyrics
