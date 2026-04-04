"""
_ace_worker_v15.py — ACE-Step 1.5 generation worker.

Runs inside .venv_model_v15 (Python 3.11). Receives a JSON payload as argv[1],
emits JSON lines to stdout — same protocol as _ace_worker.py.

Real ACE-Step 1.5 API (from acestep/inference.py and handler.py):
  from acestep.handler        import AceStepHandler
  from acestep.llm_inference  import LLMHandler
  from acestep.inference      import GenerationParams, GenerationConfig, generate_music

  dit = AceStepHandler()
  dit.initialize_service(project_root=..., config_path="acestep-v15-turbo", device="cuda")

  llm = LLMHandler()
  llm.initialize(checkpoint_dir=..., lm_model_path="acestep-5Hz-lm-1.7B",
                 backend="pt", device="cuda")

  params = GenerationParams(caption=..., lyrics=..., audio_duration=..., seed=...)
  config = GenerationConfig(batch_size=1, audio_format="wav")
  result = generate_music(dit, llm, params, config, save_dir=...)
  # result.audios -> list of {"path": "...", "key": "...", ...}
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

_stdout_lock = threading.Lock()


def emit(obj: dict) -> None:
    with _stdout_lock:
        print(json.dumps(obj), flush=True)


def progress(variation_index: int, step: int, total_steps: int,
             message: str, total: int = 1) -> None:
    emit({
        "type":             "progress",
        "variation_index":  variation_index,
        "total_variations": total,
        "step":             step,
        "total_steps":      total_steps,
        "message":          message,
    })


def result(variation_index: int, audio_path: str, seed: int,
           duration: float, mode: str, style_prompt: str,
           lyrics: str = "") -> None:
    emit({
        "type":           "result",
        "variation_index": variation_index,
        "audio_path":     audio_path,
        "seed":           seed,
        "duration_secs":  duration,
        "mode":           mode,
        "style_prompt":   style_prompt,
        "lyrics":         lyrics,
    })


def get_audio_duration(path: str) -> float:
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        return 0.0


def out_path(output_dir: str, mode: str, variation: int) -> str:
    stamp = int(time.time())
    uid   = uuid.uuid4().hex[:6]
    return str(Path(output_dir) / f"{mode}_v{variation:02d}_{stamp}_{uid}.wav")


class HeartbeatThread:
    def __init__(self, var_idx, total, step_start, step_end, total_steps,
                 message_fn, interval=5.0):
        self.var_idx     = var_idx
        self.total       = total
        self.step_start  = step_start
        self.step_end    = step_end
        self.total_steps = total_steps
        self.message_fn  = message_fn
        self.interval    = interval
        self._stop       = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _gpu_stats():
        try:
            import torch
            if torch.cuda.is_available():
                used  = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return f" | GPU {used:.1f}/{total:.1f} GB VRAM"
        except Exception:
            pass
        return ""

    def _run(self):
        t0 = time.time()
        progress(self.var_idx, self.step_start, self.total_steps,
                 self.message_fn(0) + self._gpu_stats(), self.total)
        while not self._stop.wait(self.interval):
            elapsed  = time.time() - t0
            span     = max(self.step_end - self.step_start, 1)
            frac     = 1.0 - 0.5 ** (elapsed / (span * 2))
            cur_step = self.step_start + int(frac * span)
            cur_step = min(cur_step, self.step_end - 1)
            progress(self.var_idx, cur_step, self.total_steps,
                     self.message_fn(elapsed) + self._gpu_stats(), self.total)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=3)


# ---------------------------------------------------------------------------
# Resolve checkpoint / project root
# ---------------------------------------------------------------------------

def _resolve_project_root(p: dict) -> str:
    """
    ACE-Step's initialize_service() needs the project root — the directory
    that contains the 'checkpoints' folder (or where they'll be downloaded).
    We map from Stitch's models_dir to the right parent.
    """
    models_dir = p.get("models_dir") or str(
        Path.home() / ".cache" / "ace-step"
    )
    models_path = Path(models_dir)

    # If the user pointed at the checkpoints dir directly, go up one level
    if models_path.name == "checkpoints":
        return str(models_path.parent)

    # If there's an ace_step_v15 subdir, that's the checkpoints dir
    v15_dir = models_path / "ace_step_v15"
    if v15_dir.exists():
        return str(models_path)

    # Default: treat models_dir as the project root
    return str(models_path)


def _select_config(p: dict) -> str:
    """
    Select the right ACE-Step config for the generation task.
    Turbo is fast (8 steps) and works for all tasks.
    Base is required for repaint / extract / lego tasks.
    """
    mode = p.get("mode", "text")
    if mode in ("repair",):
        # repaint/editing tasks need base model capabilities
        return p.get("ace_config", "acestep-v15-base")
    return p.get("ace_config", "acestep-v15-turbo")


def _select_device(p: dict) -> str:
    try:
        import torch
        if torch.cuda.is_available():
            device_str = str(p.get("device", ""))
            if ":" in device_str:
                return device_str  # e.g. "cuda:0"
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Handler initialisation
# ---------------------------------------------------------------------------

_dit_handler = None
_llm_handler = None
_init_config = None   # (project_root, config_path, device) tuple


def _get_handlers(p: dict):
    """
    Initialise (or reuse) AceStepHandler + LLMHandler.
    Returns (dit_handler, llm_handler).
    LLMHandler uses 'pt' backend on Windows (no vllm), 'vllm' on Linux.
    """
    global _dit_handler, _llm_handler, _init_config

    project_root = _resolve_project_root(p)
    config_path  = _select_config(p)
    device       = _select_device(p)
    this_config  = (project_root, config_path, device)

    if _dit_handler is not None and _init_config == this_config:
        return _dit_handler, _llm_handler

    from acestep.handler       import AceStepHandler
    from acestep.llm_inference import LLMHandler

    # ---- DiT handler ----
    dit = AceStepHandler()
    progress(1, 2, 10, f"Initialising ACE-Step 1.5 DiT ({config_path}) on {device}...")
    dit.initialize_service(
        project_root=project_root,
        config_path=config_path,
        device=device,
    )

    # ---- LLM handler ----
    # On Windows with 'pt' backend:
    #   - 1.7B hangs during weight loading (no Triton/CUDA graphs, loads
    #     layer-by-layer into VRAM and can take 20+ minutes or stall forever)
    #   - 0.6B loads reliably in ~30s and produces good lyrics
    # Use 0.6B as the safe default on Windows; allow override via env var.
    lm_backend = os.environ.get("ACESTEP_LM_BACKEND", "")
    if not lm_backend:
        lm_backend = "pt" if sys.platform == "win32" else "vllm"

    lm_model_env = os.environ.get("ACESTEP_LM_MODEL_PATH", "")
    if lm_model_env:
        lm_model = lm_model_env
    elif sys.platform == "win32" and lm_backend == "pt":
        # 0.6B is the safe choice for Windows pt backend — reliably loads in ~30s
        lm_model = "acestep-5Hz-lm-0.6B"
    else:
        lm_model = "acestep-5Hz-lm-1.7B"

    # LLM init timeout — if initialize() doesn't return within this many seconds
    # we abandon it and run in DiT-only mode (no AI lyrics, but generation works).
    # 180s default: 0.6B loads in ~30s normally, but needs ~90s on first run
    # (tokenizer download). 1.7B on pt backend may never finish — it will always
    # hit this timeout, which is intentional. Generation still succeeds without it.
    LLM_INIT_TIMEOUT = int(os.environ.get("STITCH_LLM_TIMEOUT", "180"))

    progress(1, 5, 10,
             f"Initialising LLM ({lm_model}, backend={lm_backend}, timeout={LLM_INIT_TIMEOUT}s)...")

    os.environ.setdefault("ACESTEP_INIT_LLM", "auto")

    llm = LLMHandler()
    llm_result = [None]   # [0] = result or exception, captured from thread
    llm_exc    = [None]

    def _init_llm():
        try:
            r = llm.initialize(
                checkpoint_dir=project_root,
                lm_model_path=lm_model,
                backend=lm_backend,
                device=device,
            )
            llm_result[0] = r
        except Exception as e:
            llm_exc[0] = e

    t = threading.Thread(target=_init_llm, daemon=True)
    t.start()
    t.join(timeout=LLM_INIT_TIMEOUT)

    if t.is_alive():
        # Still running after timeout — the pt backend hung loading weights.
        # We cannot kill the thread cleanly, but we abandon it and continue.
        # IMPORTANT: Generation will still complete — just without AI lyrics.
        # The CoT/thinking flags are NOT passed when llm=None, so there is no
        # secondary hang during generate_music().
        emit({"type": "error",
              "message": f"[LLM init timed out after {LLM_INIT_TIMEOUT}s]\n"
                         f"The {lm_model} model did not finish loading.\n"
                         f"Continuing in DiT-only mode — music will still generate,\n"
                         f"but lyrics will use a structural scaffold instead of AI-written text.\n"
                         f"Tip: switch to the 0.6B model in Settings → Models for faster LLM loading."})
        llm = None
    elif llm_exc[0] is not None:
        import traceback as tb
        emit({"type": "error",
              "message": f"[LLM init failed] {type(llm_exc[0]).__name__}: {llm_exc[0]}\n"
                         f"AI Writes lyrics disabled this session.\n"
                         f"{tb.format_exc()}"})
        llm = None
    else:
        init_result = llm_result[0]
        if isinstance(init_result, tuple):
            status_msg, init_ok = init_result
        else:
            init_ok = True
            status_msg = "ok"
        if init_ok:
            progress(1, 8, 10, f"LLM ready ({lm_model})")
        else:
            emit({"type": "error",
                  "message": f"[LLM init failed] success=False: {status_msg}\n"
                              f"AI Writes lyrics disabled this session."})
            llm = None

    _dit_handler = dit
    _llm_handler = llm
    _init_config = this_config

    progress(1, 9, 10, f"ACE-Step 1.5 ready on {device}")
    return dit, llm


# ---------------------------------------------------------------------------
# Build GenerationParams for each task type
# ---------------------------------------------------------------------------

def _make_params(p: dict, seed: int, variation_idx: int, total: int,
                 llm_handler=None):
    """
    Construct a GenerationParams object from our payload.

    For text mode we handle three distinct paths:
      - lyrics_mode=ai_writes + output_type=with_vocals:
            Use create_sample() — the LM writes caption, lyrics, bpm, key, etc.
            from the style_prompt as a natural-language query.
      - lyrics_mode=ai_writes + output_type=instrumental:
            Use create_sample() with instrumental=True.
      - lyrics_mode=user + output_type=with_vocals:
            Pass user-supplied lyrics directly. Caption = style_prompt.
      - lyrics_mode=user + output_type=instrumental:
            Pass lyrics="[Instrumental]" — ACE-Step convention for no vocals.

    We use inspect to guard against version differences in the dataclass.
    """
    from acestep.inference import GenerationParams

    sig    = inspect.signature(GenerationParams)
    fields = set(sig.parameters.keys())

    def f(**kwargs):
        return {k: v for k, v in kwargs.items() if k in fields}

    mode         = p.get("mode", "text")
    lyrics_mode  = p.get("lyrics_mode", "ai_writes")   # "ai_writes" | "user"
    output_type  = p.get("output_type", "with_vocals")  # "with_vocals" | "instrumental"
    instrumental = (output_type == "instrumental")

    # GenerationParams uses "audio_duration" in some versions and "duration"
    # in others. Introspect the actual installed signature to handle both.
    duration_val = float(p.get("duration", 30.0))
    if "audio_duration" in fields:
        base = f(seed=seed, audio_duration=duration_val)
    elif "duration" in fields:
        base = f(seed=seed, duration=duration_val)
    else:
        base = f(seed=seed)  # no duration field — model will use its default

    # Vocal scaffold used when no lyrics are available but vocals are wanted.
    # An empty string tells ACE-Step "no vocals" — we need at least a structural
    # hint so the DiT knows to generate a sung vocal track.
    VOCAL_SCAFFOLD = "[Verse]\n[Chorus]"

    if mode == "text":
        if lyrics_mode == "ai_writes" and llm_handler is not None:
            # Full LM path — use create_sample to generate caption + lyrics
            # from the user's query. If create_sample fails (common on Windows
            # pt backend due to constrained decoding issues), fall back to
            # format_sample which takes an explicit caption and enriches it —
            # a safer path that still uses the LLM for lyrics generation.
            lp = p.get("lyrics_prompt", "").strip()
            query = lp if lp else p.get("style_prompt", "")

            sample = None

            # --- Attempt 1: create_sample (full auto mode) ---
            try:
                from acestep.inference import create_sample
                raw = create_sample(
                    llm_handler             = llm_handler,
                    query                   = query,
                    instrumental            = instrumental,
                    use_constrained_decoding = False,  # skip FSM — more stable on pt backend
                )
                if raw and raw.success:
                    sample = raw
                else:
                    reason = getattr(raw, "error", None) or "success=False, no error detail"
                    emit({"type": "error",
                          "message": f"[create_sample attempt 1] {reason} — trying format_sample"})
            except Exception as exc:
                import traceback
                emit({"type": "error",
                      "message": f"[create_sample attempt 1 exception] {type(exc).__name__}: {exc}\n"
                                 f"{traceback.format_exc()}\nTrying format_sample..."})

            # --- Attempt 2: format_sample (caption+lyrics enrichment mode) ---
            if sample is None:
                try:
                    from acestep.inference import format_sample
                    scaffold = "[Instrumental]" if instrumental else VOCAL_SCAFFOLD
                    raw2 = format_sample(
                        llm_handler             = llm_handler,
                        caption                 = p.get("style_prompt", ""),
                        lyrics                  = scaffold,
                        use_constrained_decoding = False,
                    )
                    if raw2 and raw2.success:
                        sample = raw2
                    else:
                        reason2 = getattr(raw2, "error", None) or "success=False"
                        emit({"type": "error",
                              "message": f"[format_sample attempt 2] {reason2} — using vocal scaffold directly"})
                except Exception as exc2:
                    import traceback
                    emit({"type": "error",
                          "message": f"[format_sample attempt 2 exception] {type(exc2).__name__}: {exc2}\n"
                                     f"{traceback.format_exc()}\nUsing vocal scaffold directly."})

            # --- Apply result or final fallback ---
            if sample is not None:
                lyrics_out = sample.lyrics or ("[Instrumental]" if instrumental else VOCAL_SCAFFOLD)
                # Phase 2: user-set bpm/key override LM-generated values if provided.
                # Time signature and exclusions are always appended to caption.
                caption_out = sample.caption or p.get("style_prompt", "")
                time_sig = p.get("time_signature", "").strip()
                excl     = p.get("exclusions", "").strip()
                if time_sig:
                    caption_out = f"{caption_out}, {time_sig} time signature"
                if excl:
                    caption_out = f"{caption_out}. Avoid: {excl}"

                user_bpm = p.get("bpm")
                user_key = p.get("key", "").strip()
                base.update(f(
                    caption        = caption_out,
                    lyrics         = lyrics_out,
                    bpm            = int(user_bpm) if user_bpm else sample.bpm,
                    keyscale       = user_key     if user_key  else sample.keyscale,
                    vocal_language = sample.language,
                ))
            else:
                caption_fb = p.get("style_prompt", "")
                time_sig   = p.get("time_signature", "").strip()
                excl       = p.get("exclusions", "").strip()
                if time_sig:
                    caption_fb = f"{caption_fb}, {time_sig} time signature".lstrip(", ")
                if excl:
                    caption_fb = f"{caption_fb}. Avoid: {excl}".lstrip(". ")
                base.update(f(
                    caption = caption_fb,
                    lyrics  = "[Instrumental]" if instrumental else VOCAL_SCAFFOLD,
                ))
                user_bpm = p.get("bpm")
                user_key = p.get("key", "").strip()
                if user_bpm:
                    base.update(f(bpm=int(user_bpm)))
                if user_key:
                    base.update(f(keyscale=user_key))
        else:
            # User-provided lyrics (or LLM unavailable).
            # If user typed lyrics, use them. If user typed nothing but still
            # wants vocals (with_vocals selected), use the scaffold so the DiT
            # generates a vocal track rather than defaulting to instrumental.
            # "[Instrumental]" is ACE-Step's convention for no-vocals generation.
            user_lyrics = p.get("lyrics") or ""
            if instrumental:
                lyrics_out = "[Instrumental]"
            elif user_lyrics:
                lyrics_out = user_lyrics
            else:
                lyrics_out = VOCAL_SCAFFOLD

            # Build caption: start with style_prompt, then append Phase 2 hints.
            # Time signature has no native GenerationParams field — append as text.
            # Exclusions become "Avoid: ..." which ACEStep's caption encoder responds to.
            caption = p.get("style_prompt", "")
            time_sig = p.get("time_signature", "").strip()
            excl     = p.get("exclusions", "").strip()
            if time_sig:
                caption = f"{caption}, {time_sig} time signature".lstrip(", ")
            if excl:
                caption = f"{caption}. Avoid: {excl}".lstrip(". ")

            base.update(f(
                caption = caption,
                lyrics  = lyrics_out,
            ))

            # BPM and key — pass through inspect-safe f() so missing fields
            # on older ACEStep installs are silently dropped.
            bpm = p.get("bpm")
            key = p.get("key", "").strip()
            if bpm:
                base.update(f(bpm=int(bpm)))
            if key:
                # ACEStep uses "keyscale" field; value format: "C major" / "A minor"
                base.update(f(keyscale=key))

    elif mode == "cover":
        base.update(f(
            caption          = p.get("target_style", ""),
            lyrics           = p.get("new_lyrics") or "",
            reference_audio  = p.get("source_audio"),
            task             = "cover",
        ))

    elif mode == "vocal":
        base.update(f(
            caption          = p.get("backing_style", ""),
            lyrics           = "",
            reference_audio  = p.get("vocal_audio"),
            task             = "singing2accompaniment",
        ))

    elif mode == "repair":
        base.update(f(
            caption          = p.get("hint_prompt") or "",
            lyrics           = "",
            reference_audio  = p.get("source_audio"),
            task             = "repaint",
            repaint_start    = float(p.get("region_start", 0.0)),
            repaint_end      = float(p.get("region_end", 30.0)),
        ))

    # Infer steps — turbo default is 8, base default is 32.
    # Turbo was designed for low step counts; 8 steps is optimal.
    # More steps waste VRAM and time without quality gain on turbo.
    config_path = _select_config(p)
    is_turbo = "turbo" in config_path
    default_steps = 8 if is_turbo else 32
    infer_steps = int(p.get("infer_steps", default_steps))
    base.update(f(infer_step=infer_steps))

    # For turbo: guidance_scale must be 1.0 (no CFG).
    # ACEStep overrides this internally anyway, but setting it explicitly
    # avoids entering the CFG path before the override kicks in.
    if is_turbo:
        base.update(f(guidance_scale=1.0))

    # thinking=True enables ACE-Step's internal LLM chain-of-thought pipeline.
    # CRITICAL: Only set this when the LLM handler is actually initialised.
    # If thinking=True but llm_initialized=False, ACEStep enters a CoT stall
    # that runs for the full 600s generation timeout before failing. This was
    # the root cause of the "GPU inflates then hangs forever" symptom.
    llm_available = llm_handler is not None
    if mode == "text" and lyrics_mode != "instrumental" and llm_available:
        base.update(f(
            thinking         = True,
            use_cot_caption  = True,
            use_cot_metas    = True,
            use_cot_language = (not instrumental),
        ))

    return GenerationParams(**base)


def _make_config(p: dict):
    from acestep.inference import GenerationConfig
    sig    = inspect.signature(GenerationConfig)
    fields = set(sig.parameters.keys())

    def f(**kwargs):
        return {k: v for k, v in kwargs.items() if k in fields}

    return GenerationConfig(**f(
        batch_size   = 1,       # we loop variations ourselves
        audio_format = "wav",   # Stitch expects WAV
    ))


# ---------------------------------------------------------------------------
# TOTAL_STEPS constant for progress reporting
# ---------------------------------------------------------------------------
TOTAL_STEPS = 30


# ---------------------------------------------------------------------------
# Core generation runner
# ---------------------------------------------------------------------------

def _run_generation(p: dict, var_idx: int, total: int, seed: int,
                    save_dir: str, mode_label: str) -> str | None:
    """
    Run one variation. Returns the path of the generated file, or None on failure.
    """
    from acestep.inference import generate_music

    dit, llm = _get_handlers(p)

    params = _make_params(p, seed, var_idx, total, llm_handler=llm)
    config = _make_config(p)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    with HeartbeatThread(var_idx, total, 10, TOTAL_STEPS, TOTAL_STEPS,
                         lambda t, i=var_idx, n=total:
                         f"Generating {i}/{n} — {int(t)}s elapsed"):
        gen_result = generate_music(dit, llm, params, config, save_dir=save_dir)

    if not gen_result or not gen_result.success:
        emit({"type": "error",
              "message": f"generate_music returned failure for variation {var_idx}"})
        return None

    # result.audios is a list of dicts with 'path' key
    audios = getattr(gen_result, "audios", None) or []
    if not audios:
        emit({"type": "error", "message": f"No audio in result for variation {var_idx}"})
        return None

    return str(audios[0]["path"])


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def _generate_lyrics_pre(p: dict) -> None:
    """
    Generate lyrics BEFORE the heavy DiT model loads — zero VRAM, fast.
    Writes result back into p["lyrics"] so _make_params picks it up as
    user-supplied lyrics (bypassing the broken ACEStep LLM path entirely).

    Only runs when lyrics_mode=ai_writes and no user lyrics are provided.
    Skipped for instrumental output.
    """
    lyrics_mode = p.get("lyrics_mode", "ai_writes")
    output_type = p.get("output_type", "with_vocals")
    user_lyrics = (p.get("lyrics") or "").strip()

    if lyrics_mode != "ai_writes":
        return  # user supplied their own lyrics
    if output_type == "instrumental":
        return  # no lyrics needed
    if user_lyrics:
        return  # already have lyrics

    try:
        # lyrics_gen lives in the UI venv — import from source tree
        _root = Path(__file__).parent.parent.parent
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from app.backend.lyrics_gen import generate_lyrics

        style      = p.get("style_prompt", "")
        lp         = p.get("lyrics_prompt", "")
        creativity = float(p.get("lyrics_creativity", 0.5))
        adherence  = float(p.get("lyrics_adherence",  0.7))
        lyr_model  = p.get("lyrics_model", "")
        models_dir = Path(p["models_dir"]) if p.get("models_dir") else None

        def _cb(msg: str):
            progress(1, 1, TOTAL_STEPS, msg)

        lyrics = generate_lyrics(
            style_prompt  = style,
            lyrics_prompt = lp,
            models_dir    = models_dir,
            progress_cb   = _cb,
            creativity    = creativity,
            adherence     = adherence,
            lyrics_model  = lyr_model,
        )

        if lyrics and lyrics.strip():
            p["lyrics"]      = lyrics
            p["lyrics_mode"] = "user"   # tell _make_params to use them directly
    except Exception as exc:
        import traceback
        emit({"type": "error",
              "message": f"[lyrics_gen] {type(exc).__name__}: {exc}\n"
                         f"{traceback.format_exc()}\nUsing scaffold fallback."})


def run_text(p: dict) -> None:
    total = p.get("variations", 1)
    out_dir = p["output_dir"]

    # Step 1: Generate lyrics before loading the heavy model (fast, zero VRAM)
    _generate_lyrics_pre(p)

    # Step 2: Load DiT (the only heavy step)
    with HeartbeatThread(1, total, 0, 10, TOTAL_STEPS,
                         lambda t: f"Loading model... ({int(t)}s)"):
        _get_handlers(p)

    for i in range(1, total + 1):
        seed = (p.get("seed") or int(time.time())) + i - 1

        path = _run_generation(p, i, total, seed, out_dir, "gen")
        if path is None:
            continue

        dur   = get_audio_duration(path)
        fsize = Path(path).stat().st_size // 1024 if Path(path).exists() else 0
        progress(i, TOTAL_STEPS, TOTAL_STEPS,
                 f"Variation {i}/{total} done — {dur:.1f}s | {fsize} KB | seed {seed}", total)
        result(i, path, seed, dur, "text", p.get("style_prompt", ""),
               lyrics=p.get("lyrics", ""))


def run_cover(p: dict) -> None:
    total   = p.get("variations", 1)
    out_dir = p["output_dir"]

    with HeartbeatThread(1, total, 0, 10, TOTAL_STEPS,
                         lambda t: f"Loading model... ({int(t)}s)"):
        _get_handlers(p)

    for i in range(1, total + 1):
        seed = (p.get("seed") or int(time.time())) + i - 1

        path = _run_generation(p, i, total, seed, out_dir, "cover")
        if path is None:
            continue

        dur = get_audio_duration(path)
        progress(i, TOTAL_STEPS, TOTAL_STEPS, f"Cover {i}/{total} done — {dur:.1f}s", total)
        result(i, path, seed, dur, "cover", p.get("target_style", ""))


def run_vocal(p: dict) -> None:
    total   = p.get("variations", 1)
    out_dir = p["output_dir"]

    with HeartbeatThread(1, total, 0, 10, TOTAL_STEPS,
                         lambda t: f"Loading model... ({int(t)}s)"):
        _get_handlers(p)

    for i in range(1, total + 1):
        seed = (p.get("seed") or int(time.time())) + i - 1

        path = _run_generation(p, i, total, seed, out_dir, "vox")
        if path is None:
            continue

        dur = get_audio_duration(path)
        progress(i, TOTAL_STEPS, TOTAL_STEPS, f"Backing {i}/{total} done — {dur:.1f}s", total)
        result(i, path, seed, dur, "vocal", p.get("backing_style", ""))


def run_repair(p: dict) -> None:
    out_dir = p["output_dir"]
    seed    = int(time.time())

    with HeartbeatThread(1, 1, 0, 10, TOTAL_STEPS,
                         lambda t: f"Loading model... ({int(t)}s)"):
        _get_handlers(p)

    path = _run_generation(p, 1, 1, seed, out_dir, "repair")
    if path is None:
        return

    dur = get_audio_duration(path)
    progress(1, TOTAL_STEPS, TOTAL_STEPS, f"Repair done — {dur:.1f}s", 1)
    result(1, path, seed, dur, "repair", p.get("hint_prompt", ""))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        emit({"type": "error", "message": "No payload provided"})
        sys.exit(1)

    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        emit({"type": "error", "message": f"Bad payload JSON: {exc}"})
        sys.exit(1)

    mode = payload.get("mode", "text")
    Path(payload.get("output_dir", ".")).mkdir(parents=True, exist_ok=True)

    try:
        if mode == "text":
            run_text(payload)
        elif mode == "cover":
            run_cover(payload)
        elif mode == "vocal":
            run_vocal(payload)
        elif mode == "repair":
            run_repair(payload)
        else:
            emit({"type": "error", "message": f"Unknown mode: {mode}"})
            sys.exit(1)
    except ImportError as exc:
        emit({"type": "error", "message": f"Import error: {exc}"})
        sys.exit(1)
    except Exception as exc:
        import traceback
        emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        emit({"type": "error", "message": traceback.format_exc()})
        sys.exit(1)


if __name__ == "__main__":
    main()
