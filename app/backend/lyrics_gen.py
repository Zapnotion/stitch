"""
backend/lyrics_gen.py — Offline lyric generation for Stitch.

Primary path:  llama-cpp-python + a small GGUF model (Qwen2.5-Instruct).
Fallback path: template-fill engine (zero dependencies, always works).

The GGUF model is downloaded once from HuggingFace into the user's models_dir
under  lyrics_models/<model_filename>.gguf  and reused on every subsequent run.

Parameters
----------
creativity  : float  0.0–1.0
    Maps to LLM temperature (0.5 → 1.6). Low = safe/predictable,
    high = experimental/surprising.

adherence   : float  0.0–1.0
    How tightly the output sticks to the lyrics_prompt subject.
    1.0 = every line must directly reference the prompt.
    0.0 = treat the prompt as loose inspiration only.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Optional, Callable

ProgressCb = Optional[Callable[[str], None]]


# ---------------------------------------------------------------------------
# Model registry — three tiers
# ---------------------------------------------------------------------------

LYRICS_MODELS: dict[str, dict] = {
    "Qwen2.5-1.5B (CPU · ~1 GB)": {
        "repo":         "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename":     "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "n_gpu_layers": 0,       # pure CPU — touches zero VRAM
        "ctx":          2048,
        "label":        "CPU-friendly",
    },
    "Qwen2.5-3B (Balanced · ~2 GB)": {
        "repo":         "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename":     "qwen2.5-3b-instruct-q4_k_m.gguf",
        "n_gpu_layers": 16,      # partial GPU offload — ~1 GB VRAM optional
        "ctx":          2048,
        "label":        "Balanced",
    },
    "Qwen2.5-7B (GPU · ~4.5 GB)": {
        "repo":         "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename":     "qwen2.5-7b-instruct-q4_k_m.gguf",
        "n_gpu_layers": 28,      # most layers on GPU — ~3-4 GB VRAM
        "ctx":          2048,
        "label":        "GPU-assisted",
    },
}

LYRICS_MODEL_NAMES  = list(LYRICS_MODELS.keys())
DEFAULT_LYRICS_MODEL = LYRICS_MODEL_NAMES[0]   # CPU-friendly safe default


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _model_path(model_name: str, models_dir: Optional[Path]) -> Optional[Path]:
    info = LYRICS_MODELS.get(model_name)
    if not info or models_dir is None:
        return None
    p = Path(models_dir) / "lyrics_models" / info["filename"]
    return p if p.exists() else None


def download_lyrics_model(
    model_name: str,
    models_dir: Path,
    progress_cb: ProgressCb = None,
) -> Optional[Path]:
    """Download the GGUF for model_name. Returns local path or None."""
    def _cb(msg: str):
        if progress_cb:
            progress_cb(msg)

    info = LYRICS_MODELS.get(model_name)
    if not info:
        _cb(f"[lyrics_gen] Unknown model: {model_name!r}")
        return None

    dest_dir = Path(models_dir) / "lyrics_models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / info["filename"]

    if dest.exists():
        _cb(f"[lyrics_gen] Already downloaded: {info['filename']}")
        return dest

    try:
        from huggingface_hub import hf_hub_download
        _cb(f"[lyrics_gen] Downloading {info['filename']} (~{info['repo'].split('/')[-1]})…")
        downloaded = hf_hub_download(
            repo_id  = info["repo"],
            filename = info["filename"],
            local_dir= str(dest_dir),
        )
        src = Path(downloaded)
        if src.resolve() != dest.resolve():
            import shutil
            shutil.move(str(src), str(dest))
        _cb(f"[lyrics_gen] Download complete → {dest.name}")
        return dest
    except Exception as exc:
        _cb(f"[lyrics_gen] Download failed: {exc}")
        return None


def _llm_generate(
    model_name: str,
    models_dir: Optional[Path],
    prompt_user: str,
    system: str,
    temperature: float,
    max_tokens: int = 512,
    progress_cb: ProgressCb = None,
) -> Optional[str]:
    """Run one llama-cpp-python completion. Returns text or None."""
    def _cb(msg: str):
        if progress_cb:
            progress_cb(msg)

    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError:
        _cb("[lyrics_gen] llama-cpp-python not installed — using template fallback")
        return None

    local = _model_path(model_name, models_dir)
    if local is None:
        if models_dir is not None:
            local = download_lyrics_model(model_name, models_dir, progress_cb)
        if local is None:
            _cb("[lyrics_gen] Model not available — using template fallback")
            return None

    info = LYRICS_MODELS.get(model_name, LYRICS_MODELS[DEFAULT_LYRICS_MODEL])

    try:
        _cb(f"[lyrics_gen] Loading {info['filename']} ({info['label']})…")
        llm = Llama(
            model_path   = str(local),
            n_ctx        = info["ctx"],
            n_gpu_layers = info["n_gpu_layers"],
            verbose      = False,
        )
        _cb("[lyrics_gen] Generating lyrics…")
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt_user},
            ],
            temperature = temperature,
            max_tokens  = max_tokens,
            stop        = ["<|im_end|>", "<|endoftext|>"],
        )
        text = out["choices"][0]["message"]["content"].strip()
        return text if text else None
    except Exception as exc:
        _cb(f"[lyrics_gen] LLM error: {exc} — using template fallback")
        return None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_system(adherence: float, style: str) -> str:
    if adherence >= 0.75:
        focus = (
            "Stay tightly focused on the subject the user gives you. "
            "Every single line should directly reference or address that topic. "
            "Do not drift into generic imagery or unrelated themes."
        )
    elif adherence >= 0.4:
        focus = (
            "Use the given subject as the central theme but feel free to use "
            "metaphor and imagery to explore it. Most lines should relate to the topic."
        )
    else:
        focus = (
            "Treat the given subject as loose inspiration only. "
            "Explore tangents, abstract imagery, and unexpected directions. "
            "The topic is a starting point, not a constraint."
        )

    style_note = f" Musical style: {style.strip()}." if style.strip() else ""

    return (
        f"You are a professional songwriter.{style_note} "
        "Write original song lyrics using section tags exactly like this: "
        "[Verse 1], [Chorus], [Verse 2], [Bridge], [Chorus]. "
        "Each section should have 4 lines. "
        "Output ONLY the lyrics — no commentary, no explanations, no extra text. "
        f"{focus}"
    )


def _build_user_prompt(subject: str, style: str) -> str:
    if subject.strip():
        return f"Write song lyrics about: {subject.strip()}"
    if style.strip():
        return f"Write an original song in the style of: {style.strip()}"
    return "Write an original song."


def _creativity_to_temp(creativity: float) -> float:
    """0.0 → 0.5 (tight), 0.5 → 0.9 (natural), 1.0 → 1.6 (wild)"""
    return 0.5 + creativity * 1.1


# ---------------------------------------------------------------------------
# Post-process LLM output
# ---------------------------------------------------------------------------

def _ensure_section_tags(text: str) -> str:
    """Add section tags if the LLM forgot them."""
    if re.search(r"\[(Verse|Chorus|Bridge|Pre-Chorus|Outro)", text, re.I):
        return text
    sections = [s.strip() for s in re.split(r"\n{2,}", text) if s.strip()]
    labels   = ["[Verse 1]", "[Chorus]", "[Verse 2]", "[Chorus]", "[Bridge]", "[Chorus]"]
    parts    = []
    for i, sec in enumerate(sections):
        label = labels[i] if i < len(labels) else f"[Section {i + 1}]"
        parts.extend([label, sec, ""])
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Template fallback engine (zero dependencies)
# ---------------------------------------------------------------------------

_GENRE: dict[str, dict] = {
    "pop":        {"feel": ["electric","alive","free","lost","whole"],
                   "colour": ["neon lights","summer rain","midnight sky","open road"]},
    "rock":       {"feel": ["fearless","raw","defiant","wild","unbroken"],
                   "colour": ["dark highway","burning stage","breaking glass","loud guitar"]},
    "r&b":        {"feel": ["tender","aching","devoted","warm","smooth"],
                   "colour": ["candlelight","velvet night","golden hour","slow burn"]},
    "hip-hop":    {"feel": ["hungry","focused","real","relentless","determined"],
                   "colour": ["late nights","city block","top floor","the grind"]},
    "hiphop":     {"feel": ["hungry","focused","real","relentless","determined"],
                   "colour": ["late nights","city block","top floor","the grind"]},
    "rap":        {"feel": ["hungry","focused","real","relentless","determined"],
                   "colour": ["late nights","city block","top floor","the grind"]},
    "country":    {"feel": ["homesick","grateful","proud","simple","warm"],
                   "colour": ["dirt road","front porch","fireflies","pickup truck"]},
    "electronic": {"feel": ["euphoric","hypnotic","infinite","electric","free"],
                   "colour": ["bass drop","neon pulse","synth wave","pulsing lights"]},
    "edm":        {"feel": ["euphoric","hypnotic","infinite","electric","free"],
                   "colour": ["bass drop","neon pulse","synth wave","pulsing lights"]},
    "folk":       {"feel": ["honest","quiet","true","simple","rooted"],
                   "colour": ["open field","wooden bridge","old song","long road"]},
    "jazz":       {"feel": ["smoky","cool","melancholy","warm","longing"],
                   "colour": ["blue smoke","late bar","slow trumpet","rainy street"]},
    "sad":        {"feel": ["hollow","broken","numb","tired","resigned"],
                   "colour": ["empty chair","grey morning","faded photograph","silent phone"]},
    "upbeat":     {"feel": ["joyful","radiant","carefree","unstoppable","bright"],
                   "colour": ["sunshine","open windows","dancing feet","clear skies"]},
}
_DEFAULT_GENRE = {"feel": ["alive","lost","free","tired","whole"],
                  "colour": ["open road","city lights","midnight hour","falling rain"]}

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

# Subject-referencing patterns
_VERSE_SUBJ = [
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
    "I feel {feel} every time {subj} crosses my mind",
    "Something {feel} stirs in me when I think of {subj}",
    "There's a {feel} truth in {subj} I cannot escape",
    "I've been {feel} since {subj} changed the way I see",
]
# Generic / atmospheric patterns — used when adherence is low
_VERSE_GENERIC = [
    "The {colour} reminds me of everything I left behind",
    "I walk alone through {colour} wondering where it all went",
    "Something {feel} wakes me up at 3 AM",
    "I never asked for {feel} but here we are",
    "Nothing stays the same no matter how hard you try",
    "We were young and reckless underneath the {colour}",
    "Time moves slow when all you do is think",
    "Some nights the {colour} is the only thing that's real",
]

_CHORUS_SUBJ = [
    "This is the part where {subj} breaks me open wide",
    "All roads lead back to {subj} — I've tried to run",
    "{subj} is the thing I carry when the day is done",
    "We all know {subj}, we just don't say it out loud",
    "And still {subj} remains — through everything, it stays",
    "Through every storm {subj} is the only thing that's real",
    "I keep returning to {subj} like a tide to shore",
]
_CHORUS_GENERIC = [
    "Hold on — we'll get through this somehow",
    "This is what it means to feel {feel} and still survive",
    "We fall and we rise — that's how we learn",
    "The {colour} reminds me why I carry on",
    "Everything changes but the feeling stays the same",
]

_BRIDGE_SUBJ = [
    "Maybe {subj} was the lesson all along",
    "I never wanted {subj} to mean this much to me",
    "If I could change {subj} I don't know that I would",
    "All the things I said about {subj} — I meant them all",
    "There's no escaping {subj}, so I'll make my peace",
    "In the end {subj} is what I'll remember most",
]
_BRIDGE_GENERIC = [
    "Maybe this was meant to happen all along",
    "I never thought I'd end up here but here I am",
    "Some things you carry, some things you let go",
    "The hardest part is knowing when to stop holding on",
]


def _get_genre(style: str) -> dict:
    low = style.lower()
    for key, bank in _GENRE.items():
        if key in low:
            return bank
    return _DEFAULT_GENRE


def _pick(lst: list, used: set) -> str:
    avail = [x for x in lst if x not in used]
    if not avail:
        used.clear()
        avail = lst[:]
    c = random.choice(avail)
    used.add(c)
    return c


def _line(pat: str, subj: str, genre: dict, uf: set, uc: set) -> str:
    feel   = _pick(genre["feel"],   uf)
    colour = _pick(genre["colour"], uc)
    return (pat.replace("{subj}", subj)
               .replace("{feel}", feel)
               .replace("{colour}", colour))


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _build_subject(lyrics_prompt: str, style_prompt: str) -> str:
    lp = lyrics_prompt.strip().strip(".!?")
    if not lp:
        words = [w for w in style_prompt.lower().split() if w not in _STOP]
        return words[0] if words else "this feeling"
    words = lp.split()
    if len(words) > 7:
        breaks = {"who","that","which","when","where","after","before",
                  "because","since","until","while","although","but","and"}
        core = []
        for w in words:
            if w.lower() in breaks and len(core) >= 3:
                break
            core.append(w)
        lp = " ".join(core) if core else " ".join(words[:6])
    first = lp.split()[0].lower() if lp.split() else ""
    if first.endswith("ing"):
        return lp
    starters = {"the","a","an","my","your","our","their","his","her","its",
                "this","that","being","how","why","what"}
    if first in starters:
        return lp
    if len(lp.split()) <= 4:
        return "the " + lp
    return lp


def _template_generate(
    style_prompt: str,
    lyrics_prompt: str,
    adherence: float,
) -> str:
    """Template engine. adherence controls subject-slot density."""
    genre = _get_genre(style_prompt)
    subj  = _build_subject(lyrics_prompt, style_prompt)
    random.seed(None)

    # Blend subject vs generic patterns based on adherence
    # adherence 1.0 → 100% subject patterns
    # adherence 0.0 → 100% generic patterns
    def _pool(subj_pool: list, generic_pool: list) -> list:
        if adherence >= 0.75:
            return subj_pool
        if adherence <= 0.25:
            return generic_pool
        # mix proportionally
        n_subj = int(adherence * len(subj_pool))
        mixed  = subj_pool[:n_subj] + generic_pool
        return mixed if mixed else subj_pool

    verse_pool  = _pool(_VERSE_SUBJ, _VERSE_GENERIC)
    chorus_pool = _pool(_CHORUS_SUBJ, _CHORUS_GENERIC)
    bridge_pool = _pool(_BRIDGE_SUBJ, _BRIDGE_GENERIC)

    used_v = set(); used_c = set(); used_b = set()
    used_vf = set(); used_vc = set()

    def verse() -> list[str]:
        lines = []
        for _ in range(4):
            p = _pick(verse_pool, used_v)
            lines.append(_cap(_line(p, subj, genre, used_vf, used_vc)))
        return lines

    ch_pats = []; ch_used = set(); ch_uf = set(); ch_uc = set()
    def chorus() -> list[str]:
        nonlocal ch_pats
        if not ch_pats:
            ch_pats = [_pick(chorus_pool, ch_used) for _ in range(4)]
        return [_cap(_line(p, subj, genre, ch_uf, ch_uc)) for p in ch_pats]

    br_pat  = _pick(bridge_pool, used_b)
    bridge  = _cap(_line(br_pat, subj, genre, set(), set()))

    ch = chorus()
    v1 = verse()
    v2 = verse()

    parts = (
          ["[Verse 1]"]  + v1 + [""]
        + ["[Chorus]"]   + ch + [""]
        + ["[Verse 2]"]  + v2 + [""]
        + ["[Chorus]"]   + ch + [""]
        + ["[Bridge]", bridge, ""]
        + ["[Chorus]"]   + ch
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_lyrics(
    style_prompt:  str,
    lyrics_prompt: str            = "",
    models_dir:    "Optional[Path]" = None,
    progress_cb:   ProgressCb     = None,
    creativity:    float          = 0.5,
    adherence:     float          = 0.7,
    lyrics_model:  str            = "",
) -> str:
    """
    Generate lyrics offline. LLM path first, template fallback if unavailable.
    Always returns a non-empty string.
    """
    def _cb(msg: str):
        if progress_cb:
            progress_cb(msg)

    model_name  = lyrics_model if lyrics_model in LYRICS_MODELS else DEFAULT_LYRICS_MODEL
    temperature = _creativity_to_temp(creativity)

    _cb("Generating lyrics…")
    system   = _build_system(adherence, style_prompt)
    user_msg = _build_user_prompt(lyrics_prompt, style_prompt)

    result = _llm_generate(
        model_name  = model_name,
        models_dir  = models_dir,
        prompt_user = user_msg,
        system      = system,
        temperature = temperature,
        progress_cb = _cb,
    )

    if result and result.strip():
        lyrics = _ensure_section_tags(result)
        _cb(f"Lyrics ready ({len(lyrics.split())} words)")
        return lyrics

    # Fallback
    _cb("Using template fallback for lyrics…")
    lyrics = _template_generate(style_prompt, lyrics_prompt, adherence)
    _cb(f"Lyrics ready ({len(lyrics.split())} words) [template]")
    return lyrics
