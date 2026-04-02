"""
_ace_worker.py — ACE-Step generation worker.

Runs inside .venv_model. Receives a JSON payload as argv[1], runs generation,
emits JSON lines to stdout.

ACEStepPipeline.__call__ signature (acestep 0.2.0):
  format, audio_duration, prompt, lyrics, infer_step, guidance_scale,
  scheduler_type, cfg_type, omega_scale, manual_seeds, guidance_interval,
  guidance_interval_decay, min_guidance_scale, use_erg_tag, use_erg_lyric,
  use_erg_diffusion, oss_steps, guidance_scale_text, guidance_scale_lyric,
  audio2audio_enable, ref_audio_strength, ref_audio_input,
  lora_name_or_path, lora_weight, retake_seeds, retake_variance,
  task, repaint_start, repaint_end, src_audio_path,
  edit_target_prompt, edit_target_lyrics, edit_n_min, edit_n_max, edit_n_avg,
  save_path, batch_size, debug
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

# Stdout lock — shared between main thread and HeartbeatThread to prevent
# interleaved JSON lines on Windows (which doesn't have atomic write like Linux).
_stdout_lock = threading.Lock()


def _install_hf_progress_hook() -> None:
    """
    Monkey-patch huggingface_hub's file download to emit progress messages
    so the user can see what's being downloaded during first-run model fetch.
    Only patches once; safe to call multiple times.
    """
    try:
        import huggingface_hub.file_download as fd
        if getattr(fd, "_stitch_patched", False):
            return

        _orig_http_get = fd.http_get

        def _patched_http_get(url, temp_file, *args, **kwargs):
            filename = url.split("/")[-1].split("?")[0]
            progress(1, 0, 10, f"Downloading: {filename}")
            result = _orig_http_get(url, temp_file, *args, **kwargs)
            size_mb = temp_file.tell() / 1e6 if hasattr(temp_file, "tell") else 0
            if size_mb > 1:
                progress(1, 0, 10, f"Downloaded: {filename} ({size_mb:.0f} MB)")
            return result

        fd.http_get = _patched_http_get
        fd._stitch_patched = True
    except Exception:
        pass  # If patching fails, silently continue — it's just cosmetic


def emit(obj: dict) -> None:
    with _stdout_lock:
        print(json.dumps(obj), flush=True)


def progress(variation_index: int, step: int, total_steps: int,
             message: str, total: int = 1) -> None:
    emit({
        "type": "progress",
        "variation_index": variation_index,
        "total_variations": total,
        "step": step,
        "total_steps": total_steps,
        "message": message,
    })


def result(variation_index: int, audio_path: str, seed: int,
           duration: float, mode: str, style_prompt: str) -> None:
    emit({
        "type": "result",
        "variation_index": variation_index,
        "audio_path": audio_path,
        "seed": seed,
        "duration_secs": duration,
        "mode": mode,
        "style_prompt": style_prompt,
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
    """
    Emits a progress tick every N seconds while a blocking pipe() call runs.
    Uses threading.Event.wait() — sleeps efficiently, no spinning.
    All stdout writes go through _stdout_lock to prevent interleaving.
    """
    def __init__(self, var_idx: int, total: int,
                 step_start: int, step_end: int, total_steps: int,
                 message_fn, interval: float = 5.0):
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
    def _gpu_stats() -> str:
        """Read GPU memory usage — pure CUDA query, no compute load."""
        try:
            import torch
            if torch.cuda.is_available():
                used  = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                util  = f"{used:.1f}/{total:.1f} GB VRAM"
                return f" | GPU {util}"
        except Exception:
            pass
        return ""

    def _run(self):
        t0 = time.time()
        # Emit immediately so the UI shows something right away,
        # then tick every interval seconds after that.
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


def _report_hf_download(models_dir: str) -> None:
    _install_hf_progress_hook()
    """
    Check which model files are already cached and warn if a large download
    is about to happen. ACEStepPipeline gives no download progress itself.
    """
    from pathlib import Path
    required = [
        "ace_step_transformer",
        "music_dcae_f8c8",
        "music_vocoder",
        "umt5-base",
    ]
    missing = [r for r in required if not (Path(models_dir) / r).exists()]
    total_cached = sum(
        f.stat().st_size
        for f in Path(models_dir).rglob("*")
        if f.is_file()
    ) if Path(models_dir).exists() else 0

    if missing:
        progress(1, 0, 10,
            f"Downloading model files (first run — ~7GB). "
            f"Missing: {', '.join(missing)}. "
            f"This may take 10-30 minutes depending on connection speed.")
    else:
        cached_gb = total_cached / 1e9
        progress(1, 1, 10, f"Model cached ({cached_gb:.1f} GB). Loading into GPU…")


def make_pipe(p: dict):
    import torch
    from acestep.pipeline_ace_step import ACEStepPipeline

    # Always resolve device here — the UI venv has no torch so config.device
    # falls back to "cpu" even on a CUDA machine. Check directly.
    if torch.cuda.is_available():
        device_str = str(p.get("device", ""))
        device_id  = int(device_str.split(":")[-1]) if ":" in device_str else 0
        dtype      = "bfloat16"  # native on Ampere+ (RTX 3000/4000 series)
        label      = f"GPU: {torch.cuda.get_device_name(device_id)}"
    else:
        device_id  = 0
        dtype      = "float32"
        label      = "CPU (no CUDA detected)"

    models_dir = p.get("models_dir") or str(
        Path.home() / ".cache" / "ace-step" / "checkpoints"
    )

    # Report download status BEFORE the blocking constructor call
    _report_hf_download(models_dir)

    progress(1, 2, 10, f"Initialising pipeline on {label}…")
    pipe = ACEStepPipeline(
        checkpoint_dir=models_dir,
        device_id=device_id,
        dtype=dtype,
    )
    progress(1, 9, 10, f"Pipeline ready on {label}")
    return pipe


INFER_STEPS = 60

INFER_DEFAULTS = dict(
    infer_step=INFER_STEPS,
    guidance_scale=15.0,
    scheduler_type="euler",
    cfg_type="apg",
    omega_scale=10.0,
    guidance_interval=0.5,
    guidance_interval_decay=0.0,
    min_guidance_scale=3.0,
    use_erg_tag=True,
    use_erg_lyric=True,
    use_erg_diffusion=True,
    oss_steps=None,
    guidance_scale_text=0.0,
    guidance_scale_lyric=0.0,
    lora_name_or_path="none",
    lora_weight=1.0,
    debug=False,
)

TOTAL_STEPS = 10 + INFER_STEPS  # 10 pseudo-steps for load, 60 for inference


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def _get_lyrics(p: dict) -> str:
    """
    Return lyrics to use for generation.
    Priority: user-provided > LLM-generated > empty (model improvises).
    """
    user_lyrics = (p.get("lyrics") or "").strip()
    if user_lyrics:
        emit({"type": "progress", "variation_index": 1, "total_variations": 1,
              "step": 0, "total_steps": TOTAL_STEPS,
              "message": f"Using provided lyrics ({len(user_lyrics.split())} words)"})
        return user_lyrics

    lyrics_mode = p.get("lyrics_mode", "ai_writes")
    if lyrics_mode == "instrumental":
        return ""  # intentionally no lyrics

    # Try local LLM generation
    try:
        from pathlib import Path as _Path
        import sys as _sys
        # lyrics_gen lives in the UI venv, not model venv — import from path
        _root = _Path(__file__).parent.parent.parent
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from app.backend.lyrics_gen import generate_lyrics

        models_dir = _Path(p.get("models_dir") or
                           _Path.home() / ".cache" / "ace-step" / "checkpoints")
        style = p.get("style_prompt", "")

        def _lcb(msg):
            emit({"type": "progress", "variation_index": 1,
                  "total_variations": 1, "step": 1,
                  "total_steps": TOTAL_STEPS, "message": msg})

        lyrics = generate_lyrics(style, models_dir.parent, progress_cb=_lcb)
        if lyrics:
            return lyrics
    except Exception as exc:
        emit({"type": "progress", "variation_index": 1, "total_variations": 1,
              "step": 1, "total_steps": TOTAL_STEPS,
              "message": f"Lyric generation skipped: {exc}"})

    return ""


def run_text(p: dict) -> None:
    total = p.get("variations", 1)

    # Step 0: generate lyrics before loading the heavy model
    lyrics = _get_lyrics(p)

    with HeartbeatThread(1, total, 0, 10, TOTAL_STEPS,
                         lambda t: f"Loading model… ({int(t)}s)"):
        pipe = make_pipe(p)

    for i in range(1, total + 1):
        seed = (p.get("seed") or int(time.time())) + i - 1
        path = out_path(p["output_dir"], "gen", i)

        with HeartbeatThread(i, total, 10, TOTAL_STEPS, TOTAL_STEPS,
                             lambda t, i=i, total=total:
                             f"Generating {i}/{total} — {int(t)}s elapsed"):
            pipe(
                prompt=p.get("style_prompt", ""),
                lyrics=lyrics,
                audio_duration=float(p.get("duration", 15)),
                manual_seeds=[seed],
                save_path=path,
                format="wav",
                task="text2music",
                batch_size=1,
                **INFER_DEFAULTS,
            )

        dur = get_audio_duration(path)
        fsize = Path(path).stat().st_size // 1024 if Path(path).exists() else 0
        progress(i, TOTAL_STEPS, TOTAL_STEPS,
                 f"✓ Variation {i}/{total} — {dur:.1f}s | {fsize} KB | seed {seed} | {Path(path).name}",
                 total)
        result(i, path, seed, dur, "text", p.get("style_prompt", ""))


def run_cover(p: dict) -> None:
    total = p.get("variations", 1)

    with HeartbeatThread(1, total, 0, 10, TOTAL_STEPS,
                         lambda t: f"Loading model… ({int(t)}s)"):
        pipe = make_pipe(p)

    for i in range(1, total + 1):
        seed = (p.get("seed") or int(time.time())) + i - 1
        path = out_path(p["output_dir"], "cover", i)

        with HeartbeatThread(i, total, 10, TOTAL_STEPS, TOTAL_STEPS,
                             lambda t, i=i, total=total:
                             f"Generating cover {i}/{total} — {int(t)}s elapsed"):
            pipe(
                prompt=p.get("target_style", ""),
                lyrics=p.get("new_lyrics") or "",
                audio_duration=float(p.get("duration", 15)),
                manual_seeds=[seed],
                save_path=path,
                format="wav",
                task="custom",
                audio2audio_enable=True,
                ref_audio_input=p["source_audio"],
                ref_audio_strength=float(p.get("follow_strength", 0.7)),
                batch_size=1,
                **INFER_DEFAULTS,
            )

        dur = get_audio_duration(path)
        progress(i, TOTAL_STEPS, TOTAL_STEPS, f"Cover {i} done — {dur:.1f}s audio", total)
        result(i, path, seed, dur, "cover", p.get("target_style", ""))


def run_vocal(p: dict) -> None:
    total = p.get("variations", 1)

    with HeartbeatThread(1, total, 0, 10, TOTAL_STEPS,
                         lambda t: f"Loading model… ({int(t)}s)"):
        pipe = make_pipe(p)

    for i in range(1, total + 1):
        seed = (p.get("seed") or int(time.time())) + i - 1
        path = out_path(p["output_dir"], "vox", i)

        with HeartbeatThread(i, total, 10, TOTAL_STEPS, TOTAL_STEPS,
                             lambda t, i=i, total=total:
                             f"Generating backing {i}/{total} — {int(t)}s elapsed"):
            pipe(
                prompt=p.get("backing_style", ""),
                lyrics="",
                audio_duration=float(p.get("duration", 15)),
                manual_seeds=[seed],
                save_path=path,
                format="wav",
                task="singing2accompaniment",
                audio2audio_enable=True,
                ref_audio_input=p["vocal_audio"],
                ref_audio_strength=0.7,
                batch_size=1,
                **INFER_DEFAULTS,
            )

        dur = get_audio_duration(path)
        progress(i, TOTAL_STEPS, TOTAL_STEPS, f"Backing {i} done — {dur:.1f}s audio", total)
        result(i, path, seed, dur, "vocal", p.get("backing_style", ""))


def run_repair(p: dict) -> None:
    with HeartbeatThread(1, 1, 0, 10, TOTAL_STEPS,
                         lambda t: f"Loading model… ({int(t)}s)"):
        pipe = make_pipe(p)

    path         = out_path(p["output_dir"], "repair", 1)
    region_start = float(p.get("region_start", 0.0))
    region_end   = float(p.get("region_end", 15.0))

    with HeartbeatThread(1, 1, 10, TOTAL_STEPS, TOTAL_STEPS,
                         lambda t: f"Repairing — {int(t)}s elapsed"):
        pipe(
            prompt=p.get("hint_prompt") or "",
            lyrics="",
            audio_duration=region_end - region_start,
            manual_seeds=[int(time.time())],
            save_path=path,
            format="wav",
            task="repaint",
            src_audio_path=p["source_audio"],
            repaint_start=int(region_start),
            repaint_end=int(region_end),
            retake_variance=0.5,
            batch_size=1,
            **INFER_DEFAULTS,
        )

    dur = get_audio_duration(path)
    progress(1, TOTAL_STEPS, TOTAL_STEPS, f"Repair done — {dur:.1f}s audio", 1)
    result(1, path, 0, dur, "repair", p.get("hint_prompt", ""))


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
        emit({"type": "error", "message": f"ACE-Step not installed in model venv: {exc}"})
        sys.exit(1)
    except Exception as exc:
        import traceback
        emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        emit({"type": "error", "message": traceback.format_exc()})
        sys.exit(1)


if __name__ == "__main__":
    main()
