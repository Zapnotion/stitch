"""
backend/lyrics_gen.py — Prompt-driven lyric generation for Stitch.

No network, no VRAM, no downloads, no API keys. Pure Python stdlib.

How it works:
  1. Parse the lyrics_prompt (and style_prompt) for subject matter, mood,
     characters, and setting using keyword extraction.
  2. Pick matching vocabulary banks for those themes.
  3. Build verse/chorus/bridge using line templates that slot in the
     extracted concepts, so lyrics reflect what the user actually asked for.

ACEStep DiT uses these as a structural + semantic guide — real words
about the actual subject produce much better output than a bare scaffold.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Optional, Callable

ProgressCb = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------------
# Theme extraction
# ---------------------------------------------------------------------------

# Maps keywords → theme tags. Multiple tags can match.
_KEYWORD_THEMES: list[tuple[list[str], str]] = [
    # Characters / subjects
    (["lady", "woman", "mother", "mum", "mom", "grandmother", "granny", "auntie"], "woman"),
    (["man", "father", "dad", "grandfather", "grandpa", "uncle", "brother"], "man"),
    (["child", "children", "kids", "boy", "girl", "baby", "toddler", "teenager", "teen"], "children"),
    (["lover", "boyfriend", "girlfriend", "partner", "darling", "babe", "sweetheart"], "lover"),
    # Situations
    (["telling off", "scolding", "lecturing", "angry", "yelling", "shouting", "warning",
      "discipline", "punish", "behave", "naughty", "misbehave", "trouble", "rules",
      "telling", "told off", "in trouble", "behave yourself", "listen to me"], "scolding"),
    (["love", "romance", "heart", "kiss", "missing", "longing", "together", "apart"], "romance"),
    (["party", "dance", "celebrate", "night out", "fun", "crowd", "festival"], "party"),
    (["sad", "grief", "loss", "cry", "tears", "heartbreak", "lonely", "alone", "miss"], "sadness"),
    (["work", "hustle", "grind", "money", "success", "ambition", "rise", "climb"], "ambition"),
    (["road", "travel", "journey", "drive", "leave", "freedom", "escape", "away"], "journey"),
    (["home", "family", "roots", "belong", "memory", "past", "childhood", "grew up"], "home"),
    (["fight", "battle", "struggle", "overcome", "strong", "survive", "warrior"], "struggle"),
    (["nature", "rain", "storm", "sun", "sky", "ocean", "river", "mountain", "wind"], "nature"),
    (["night", "dark", "stars", "moon", "midnight", "shadow", "dream", "sleep"], "night"),
    # Genre moods (from style_prompt)
    (["pop", "catchy", "anthem"], "pop"),
    (["rock", "punk", "metal", "heavy", "loud", "guitar"], "rock"),
    (["r&b", "rnb", "soul", "smooth", "groove"], "rnb"),
    (["hip hop", "hip-hop", "rap", "trap", "bars", "flow"], "hiphop"),
    (["country", "bluegrass", "twang", "southern", "rural", "cowboy"], "country"),
    (["electronic", "edm", "synth", "dance", "rave", "techno", "house"], "electronic"),
    (["jazz", "blues", "swing", "lounge"], "jazz"),
    (["folk", "acoustic", "indie", "ballad", "storytelling"], "folk"),
]


def _extract_themes(style: str, lyrics_prompt: str) -> list[str]:
    """Return list of matched theme tags, most specific first."""
    combined = f"{style} {lyrics_prompt}".lower()
    found = []
    for keywords, tag in _KEYWORD_THEMES:
        if any(kw in combined for kw in keywords):
            if tag not in found:
                found.append(tag)
    return found if found else ["pop"]  # default


# ---------------------------------------------------------------------------
# Vocabulary banks
# Each bank: images, verbs, adjectives, anchor (memorable chorus hook line)
# ---------------------------------------------------------------------------

_BANKS: dict[str, dict] = {

    "scolding": {
        "images": [
            "pointed finger", "raised voice", "kitchen table", "school report",
            "crossed arms", "the long sigh", "bedroom doorway", "family dinner",
            "sharp tongue", "that look", "the lecture", "another warning",
        ],
        "verbs": [
            "holler", "scold", "warn", "lecture", "remind", "shake", "point",
            "stand firm", "speak plain", "lay down the law", "set it straight",
        ],
        "adj": ["tired", "firm", "righteous", "exhausted", "loving", "serious", "fed up"],
        "anchor": "I said what I said and I meant every word",
        "extra_lines": [
            "How many times do I have to say",
            "You think this is funny but I'm serious today",
            "Don't look at me like that, you know what you did",
            "I wasn't born yesterday, I was once a kid",
            "Clean that up and do it right this time",
            "We've had this conversation about a thousand times",
            "One day you'll understand why I had to be this way",
            "I do this because I love you, not to ruin your day",
        ],
    },

    "woman": {
        "images": [
            "steady hands", "knowing eyes", "worn path", "quiet strength",
            "iron will", "soft voice", "long shadow", "silver thread",
        ],
        "verbs": ["carry", "stand", "hold", "rise", "endure", "speak", "walk", "lead"],
        "adj": ["unshakeable", "patient", "fierce", "graceful", "determined", "wise"],
        "anchor": "she carries the whole world and never shows the weight",
        "extra_lines": [],
    },

    "children": {
        "images": [
            "muddy shoes", "crayon drawings", "loud laughter", "scraped knees",
            "school bags", "little hands", "front yard", "summer afternoons",
        ],
        "verbs": ["run", "play", "grow", "learn", "laugh", "cry", "listen", "disobey"],
        "adj": ["wild", "restless", "innocent", "stubborn", "bright", "young"],
        "anchor": "kids don't come with instructions but love fills in the gaps",
        "extra_lines": [],
    },

    "romance": {
        "images": [
            "candlelight", "your laugh", "first glance", "slow dance",
            "cold sheets", "familiar scent", "old photograph", "intertwined hands",
        ],
        "verbs": ["need", "want", "hold", "lose", "find", "touch", "whisper", "ache"],
        "adj": ["tender", "electric", "devoted", "broken", "whole", "aching"],
        "anchor": "I'd find you in every life I ever had",
        "extra_lines": [],
    },

    "sadness": {
        "images": [
            "empty chair", "faded photograph", "cold coffee", "silent phone",
            "grey morning", "last voicemail", "worn sweater", "winter light",
        ],
        "verbs": ["miss", "remember", "wait", "fade", "lose", "ache", "hold on", "let go"],
        "adj": ["hollow", "broken", "tired", "quiet", "numb", "resigned", "longing"],
        "anchor": "some things you carry long after they're gone",
        "extra_lines": [],
    },

    "ambition": {
        "images": [
            "long nights", "empty wallet", "top floor", "closed doors",
            "concrete jungle", "late bus home", "the grind", "first win",
        ],
        "verbs": ["push", "build", "climb", "grind", "stack", "move", "own", "prove"],
        "adj": ["hungry", "focused", "relentless", "unstoppable", "determined", "real"],
        "anchor": "came from nothing and I'm not going back",
        "extra_lines": [],
    },

    "journey": {
        "images": [
            "open road", "last mile", "faded map", "rear view mirror",
            "distant lights", "unmarked turn", "gravel road", "horizon line",
        ],
        "verbs": ["drive", "leave", "chase", "follow", "wander", "find", "go", "run"],
        "adj": ["restless", "free", "lost", "alive", "untethered", "searching"],
        "anchor": "the road knows where I need to be",
        "extra_lines": [],
    },

    "home": {
        "images": [
            "front porch", "kitchen light", "familiar voice", "old tree",
            "back road", "screen door", "handmade quilt", "Sunday morning",
        ],
        "verbs": ["remember", "return", "belong", "miss", "sit", "stay", "hold", "come home"],
        "adj": ["warm", "steady", "safe", "simple", "true", "rooted", "grateful"],
        "anchor": "some places live inside you all your life",
        "extra_lines": [],
    },

    "struggle": {
        "images": [
            "heavy load", "rising tide", "long night", "broken door",
            "burning bridge", "last thread", "storm wall", "iron ground",
        ],
        "verbs": ["fight", "stand", "hold", "endure", "rise", "break through", "refuse", "survive"],
        "adj": ["fierce", "worn", "unbroken", "defiant", "scarred", "resolute"],
        "anchor": "I've been down before but I always get back up",
        "extra_lines": [],
    },

    "party": {
        "images": [
            "pulsing lights", "spilled drinks", "bass drop", "sweaty crowd",
            "last song", "neon sign", "dance floor", "loud chorus",
        ],
        "verbs": ["dance", "sing", "shout", "move", "feel", "let go", "celebrate", "spin"],
        "adj": ["electric", "euphoric", "reckless", "loud", "carefree", "alive"],
        "anchor": "tonight we live like there's no tomorrow",
        "extra_lines": [],
    },

    "nature": {
        "images": [
            "storm clouds", "river bend", "mountain top", "falling rain",
            "morning frost", "turning leaves", "ocean wave", "still water",
        ],
        "verbs": ["breathe", "stand", "watch", "feel", "move", "grow", "weather", "flow"],
        "adj": ["still", "vast", "ancient", "wild", "clean", "free", "quiet"],
        "anchor": "the earth keeps turning whether we're ready or not",
        "extra_lines": [],
    },

    # Genre tone banks — used as supplement when no subject bank is found
    "rock": {
        "images": ["burning stage", "loud guitar", "breaking glass", "dark highway"],
        "verbs": ["burn", "scream", "break", "roar", "drive", "fight"],
        "adj": ["raw", "wild", "fearless", "defiant"],
        "anchor": "we're not going down without a fight",
        "extra_lines": [],
    },
    "country": {
        "images": ["dirt road", "front porch", "pickup truck", "fireflies"],
        "verbs": ["drive", "sit", "hold", "remember", "come home"],
        "adj": ["simple", "proud", "homesick", "grateful"],
        "anchor": "there's no place like home",
        "extra_lines": [],
    },
    "hiphop": {
        "images": ["block life", "late nights", "top floor", "the hustle"],
        "verbs": ["grind", "stack", "move", "own", "prove"],
        "adj": ["real", "hungry", "focused", "unstoppable"],
        "anchor": "came from nothing now we here",
        "extra_lines": [],
    },
    "rnb": {
        "images": ["candlelight", "slow burn", "velvet night", "golden hour"],
        "verbs": ["want", "need", "touch", "stay", "whisper"],
        "adj": ["smooth", "tender", "devoted", "warm"],
        "anchor": "baby just stay with me tonight",
        "extra_lines": [],
    },
    "electronic": {
        "images": ["bass drop", "neon pulse", "synth wave", "pulsing lights"],
        "verbs": ["pulse", "drop", "rise", "sync", "ignite"],
        "adj": ["euphoric", "electric", "hypnotic", "infinite"],
        "anchor": "let the rhythm take control",
        "extra_lines": [],
    },
    "jazz": {
        "images": ["blue smoke", "late bar", "slow trumpet", "rainy street"],
        "verbs": ["play", "sway", "linger", "drift", "feel"],
        "adj": ["smoky", "cool", "melancholy", "warm"],
        "anchor": "the music says what words never could",
        "extra_lines": [],
    },
    "folk": {
        "images": ["open field", "wooden bridge", "old song", "long road"],
        "verbs": ["walk", "sing", "tell", "carry", "leave", "return"],
        "adj": ["honest", "quiet", "simple", "true"],
        "anchor": "every story's worth the telling once",
        "extra_lines": [],
    },
    "pop": {
        "images": ["neon lights", "summer rain", "midnight sky", "city crowd"],
        "verbs": ["run", "feel", "hold", "breathe", "fall", "rise"],
        "adj": ["electric", "alive", "free", "lost"],
        "anchor": "we'll be okay in the end",
        "extra_lines": [],
    },
    "night": {
        "images": ["midnight sky", "city glow", "empty street", "burning stars"],
        "verbs": ["wander", "drift", "find", "chase", "watch", "stay"],
        "adj": ["still", "restless", "alive", "alone"],
        "anchor": "the night holds every secret I keep",
        "extra_lines": [],
    },
}

_FALLBACK_BANK = _BANKS["pop"]


def _merge_banks(themes: list[str]) -> dict:
    """Merge vocabulary from matched theme banks into one combined bank."""
    merged: dict = {"images": [], "verbs": [], "adj": [], "anchor": "", "extra_lines": []}
    for theme in themes:
        bank = _BANKS.get(theme)
        if bank:
            merged["images"]      += bank["images"]
            merged["verbs"]       += bank["verbs"]
            merged["adj"]         += bank["adj"]
            merged["extra_lines"] += bank.get("extra_lines", [])
            if not merged["anchor"]:
                merged["anchor"] = bank["anchor"]
    if not merged["images"]:
        merged = dict(_FALLBACK_BANK)
    return merged


# ---------------------------------------------------------------------------
# Line construction
# ---------------------------------------------------------------------------

_VERSE_TEMPLATES = [
    "I see the {img} and I feel {adj}",
    "Every time I think of {img} I {verb}",
    "There's something {adj} about the way things change",
    "The {img} reminds me of what I knew",
    "I {verb} through the {img} every single day",
    "Standing here with {adj} hands and nothing left to say",
    "The {img} keeps calling out my name",
    "I've been {adj} for longer than I care to say",
    "We {verb} and we try but it's never the same",
    "Something {adj} lives deep inside this {img}",
    "I {verb} till the {img} fades away",
    "There's a {adj} truth that I cannot escape",
]

_CHORUS_TEMPLATES = [
    "{anchor}",
    "So we {verb} through the {img} side by side",
    "Hold on, we'll {verb} through the night",
    "Let the {img} carry what we can't",
    "We fall and we {verb} and we fall again",
    "Gonna {verb} till the {img} leads us home",
]


def _pick(lst: list, used: set, fallback: list) -> str:
    available = [x for x in lst if x not in used]
    if not available:
        available = fallback or lst
    choice = random.choice(available)
    used.add(choice)
    return choice


def _fill(template: str, bank: dict, used_i: set, used_v: set) -> str:
    img  = _pick(bank["images"], used_i, bank["images"])
    verb = _pick(bank["verbs"],  used_v, bank["verbs"])
    adj  = random.choice(bank["adj"])
    return (template
            .replace("{img}",    img)
            .replace("{verb}",   verb)
            .replace("{adj}",    adj)
            .replace("{anchor}", bank["anchor"]))


def _build_verse(bank: dict, used_i: set, used_v: set,
                 extra: list, extra_used: set) -> list[str]:
    lines = []
    # When a theme has specific extra_lines (e.g. scolding), prefer them
    # strongly so the verse sounds like the actual subject matter.
    use_extra_prob = 0.75 if extra else 0.0
    for _ in range(4):
        use_extra = extra and random.random() < use_extra_prob and len(extra_used) < len(extra)
        if use_extra:
            available = [l for l in extra if l not in extra_used]
            if available:
                line = random.choice(available)
                extra_used.add(line)
                lines.append(line)
                continue
        tpl = random.choice(_VERSE_TEMPLATES)
        lines.append(_fill(tpl, bank, used_i, used_v))
    return lines


def _build_chorus(bank: dict) -> list[str]:
    tpls = random.sample(_CHORUS_TEMPLATES, min(4, len(_CHORUS_TEMPLATES)))
    ui, uv = set(), set()
    return [_fill(t, bank, ui, uv) for t in tpls]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_lyrics(
    style_prompt:  str,
    lyrics_prompt: str = "",
    models_dir:    "Optional[Path]" = None,  # kept for API compat, unused
    progress_cb:   ProgressCb = None,
) -> str:
    """
    Generate lyrics from the style and lyrics prompts.
    Always returns a non-empty string. No network, no VRAM, no downloads.
    """
    def _cb(msg: str):
        if progress_cb:
            progress_cb(msg)

    _cb("Generating lyrics...")

    combined = f"{style_prompt} {lyrics_prompt}".strip()
    themes   = _extract_themes(style_prompt, lyrics_prompt)
    bank     = _merge_banks(themes)

    # Seed from combined prompt so same input → same lyrics (repeatable)
    random.seed(hash(combined) & 0xFFFFFFFF)

    used_i:     set = set()
    used_v:     set = set()
    extra_used: set = set()
    extra = bank.get("extra_lines", [])

    v1 = _build_verse(bank, used_i, used_v, extra, extra_used)
    ch = _build_chorus(bank)
    v2 = _build_verse(bank, used_i, used_v, extra, extra_used)
    br_tpl = random.choice(_VERSE_TEMPLATES)
    br = _fill(br_tpl, bank, set(), set())

    parts = (
        ["[Verse 1]"] + v1 + [""]
        + ["[Chorus]"]  + ch + [""]
        + ["[Verse 2]"] + v2 + [""]
        + ["[Chorus]"]  + ch + [""]
        + ["[Bridge]", br, ""]
        + ["[Chorus]"]  + ch
    )

    lyrics = "\n".join(parts)
    _cb(f"Lyrics ready ({len(lyrics.split())} words, themes: {', '.join(themes)})")
    return lyrics
