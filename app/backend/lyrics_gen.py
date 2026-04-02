"""
backend/lyrics_gen.py — Local LLM lyric generation via llama-cpp-python.

Uses a small GGUF model (Qwen2.5-3B-Instruct, ~2GB) to write structured
verse/chorus/bridge lyrics from a style prompt. Fully offline, no API keys.

Model is auto-downloaded from HuggingFace on first use.
Falls back to a template if llama-cpp-python is not installed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from app.backend.logger import log

# HuggingFace repo + filename for the lyrics LLM
LLM_REPO     = "Qwen/Qwen2.5-3B-Instruct-GGUF"
LLM_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"   # ~2 GB, good quality/speed balance


def _model_path(models_dir: Path) -> Path:
    return models_dir / "lyrics_llm" / LLM_FILENAME


def ensure_model(models_dir: Path) -> Optional[Path]:
    """Download the GGUF if not present. Returns path or None on failure."""
    dest = _model_path(models_dir)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"[lyrics] Downloading {LLM_FILENAME} (~2 GB) — first use only…")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=LLM_REPO,
            filename=LLM_FILENAME,
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
        )
        log.info(f"[lyrics] Model downloaded to {path}")
        return dest
    except Exception as exc:
        log.error(f"[lyrics] Download failed: {exc}")
        return None


_LYRIC_PROMPT = """\
You are a professional songwriter. Write song lyrics for the following style/genre description.

Style: {style}

Requirements:
- Structure: [Verse 1], [Chorus], [Verse 2], [Chorus], [Bridge], [Chorus]
- Each verse: 4 lines
- Chorus: 4 lines (same each time)
- Bridge: 2-4 lines
- Tone and subject matter should match the style description
- Lines should rhyme naturally (AABB or ABAB)
- Do NOT include any commentary — output ONLY the lyrics with section labels

Output format:
[Verse 1]
...
[Chorus]
...
[Verse 2]
...
[Chorus]
...
[Bridge]
...
[Chorus]
...
"""


def generate_lyrics(
    style_prompt: str,
    models_dir: Path,
    progress_cb=None,
) -> str:
    """
    Generate lyrics for `style_prompt` using local LLM.
    Returns lyrics string, or empty string on failure (ACEStep will improvise).
    """
    def _cb(msg: str):
        log.info(f"[lyrics] {msg}")
        if progress_cb:
            progress_cb(msg)

    # Check llama-cpp is available
    try:
        from llama_cpp import Llama
    except ImportError:
        _cb("llama-cpp-python not installed — skipping lyric generation")
        return ""

    model_path = ensure_model(models_dir)
    if not model_path or not model_path.exists():
        _cb("Lyrics model not available — skipping")
        return ""

    _cb(f"Generating lyrics with local LLM…")

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_gpu_layers=-1,    # use all GPU layers available
            verbose=False,
        )

        prompt = _LYRIC_PROMPT.format(style=style_prompt)
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a professional songwriter. Output only song lyrics, no commentary."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=600,
            temperature=0.8,
            top_p=0.9,
            stop=["</s>", "<|im_end|>"],
        )

        lyrics = response["choices"][0]["message"]["content"].strip()
        _cb(f"Lyrics generated ({len(lyrics.split())} words)")
        log.debug(f"[lyrics]\n{lyrics}")
        return lyrics

    except Exception as exc:
        _cb(f"LLM inference failed: {exc}")
        return ""
