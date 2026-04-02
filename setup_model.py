"""
setup_model.py — Sets up the ACE-Step 1.5 model environment for Stitch.

Uses an embedded Python 3.11 distribution (no system install required).
ACE-Step 1.5 requires Python 3.11+ for its LLM backend.

Exit codes:
  0 — ready
  1 — fatal error
  2 — not available (stub mode)
"""

import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent
CUDA_INDEX    = "https://download.pytorch.org/whl/cu124"

# Embedded Python 3.11 — self-contained, no Windows install, no PATH changes
PY311_VERSION = "3.11.9"
PY311_URL     = f"https://www.python.org/ftp/python/{PY311_VERSION}/python-{PY311_VERSION}-embed-amd64.zip"
PY311_DIR     = SCRIPT_DIR / ".python311"
PY311_EXE     = PY311_DIR / "python.exe"

# v1.5 venv
VENV          = SCRIPT_DIR / ".venv_model_v15"
VENV_PY       = VENV / "Scripts" / "python.exe"

env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def run(args, capture=False):
    return subprocess.run(args, env=env, capture_output=capture, text=True)

def pip(py, *args):
    return run([py, "-m", "pip"] + list(args))

def query(py, code):
    r = run([py, "-c", code], capture=True)
    return r.stdout.strip(), r.stderr.strip(), r.returncode == 0


# ---------------------------------------------------------------------------
# Step 1: Embedded Python 3.11
# ---------------------------------------------------------------------------
def ensure_python311() -> bool:
    if PY311_EXE.exists():
        print(f"[OK] Embedded Python 3.11 already present")
        return True

    print(f"[..] Downloading embedded Python {PY311_VERSION} (~25 MB)...")
    PY311_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = PY311_DIR / "py311.zip"

    try:
        urllib.request.urlretrieve(PY311_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(PY311_DIR)
        zip_path.unlink(missing_ok=True)

        # Enable site-packages (required to install packages)
        for pth in PY311_DIR.glob("python3*._pth"):
            content = pth.read_text()
            pth.write_text(content.replace("#import site", "import site"))

        # Bootstrap pip
        get_pip = PY311_DIR / "get-pip.py"
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip)
        r = run([str(PY311_EXE), str(get_pip), "--quiet"])
        get_pip.unlink(missing_ok=True)
        if r.returncode != 0:
            print("[WARN] pip bootstrap failed")
            return False

        print(f"[OK] Python {PY311_VERSION} embedded dist ready")
        return True

    except Exception as exc:
        print(f"[WARN] Could not set up embedded Python 3.11: {exc}")
        shutil.rmtree(PY311_DIR, ignore_errors=True)
        return False


# ---------------------------------------------------------------------------
# Step 2: Create .venv_model_v15
# ---------------------------------------------------------------------------
def ensure_venv() -> bool:
    if VENV_PY.exists():
        print("[OK] .venv_model_v15 already exists")
        return True

    print("[..] Creating .venv_model_v15...")
    # Use virtualenv (more reliable than venv with embedded Python)
    pip(str(PY311_EXE), "install", "virtualenv", "--quiet")
    r = run([str(PY311_EXE), "-m", "virtualenv", str(VENV), "--quiet"])
    if r.returncode != 0:
        print("[ERROR] Could not create .venv_model_v15")
        return False
    print("[OK] .venv_model_v15 created")
    return True


# ---------------------------------------------------------------------------
# Step 3: CUDA torch in v1.5 venv
# ---------------------------------------------------------------------------
def ensure_cuda_torch() -> None:
    vpy = str(VENV_PY)
    _, _, cuda_ok = query(vpy, "import torch; assert torch.cuda.is_available()")
    if cuda_ok:
        out, _, _ = query(vpy, "import torch; print(torch.__version__)")
        print(f"[OK] PyTorch CUDA already available ({out})")
        return

    if not shutil.which("nvidia-smi"):
        print("[INFO] No NVIDIA GPU — will use CPU (slow)")
        return

    print("[..] Installing PyTorch CUDA 12.4 into v1.5 venv...")
    print("     (may take several minutes on first run)")
    r = pip(vpy, "install", "torch", "torchvision", "torchaudio",
            "--index-url", CUDA_INDEX, "--force-reinstall", "--quiet")
    if r.returncode != 0:
        print("[WARN] CUDA torch install failed")
        return

    out, _, ok = query(vpy,
        "import torch\n"
        "print(torch.__version__)\n"
        "assert torch.cuda.is_available()\n"
        "print('[OK] GPU:', torch.cuda.get_device_name(0))\n"
    )
    print(out if ok else "[WARN] CUDA torch installed but GPU not detected")


# ---------------------------------------------------------------------------
# Step 4: ACE-Step 1.5 + ALL its runtime dependencies
# ---------------------------------------------------------------------------

# Complete set of ACE-Step 1.5 deps from pyproject.toml required for inference.
# Excluded intentionally:
#   - torch/torchvision/torchaudio  — handled in Step 3
#   - nano-vllm                     — not on PyPI, Linux/source only
#   - mlx / mlx-lm                  — Apple Silicon macOS only
#   - flash-attn                    — handled separately below
#   - peft / lycoris-lora / lightning / tensorboard — training only
#   - modelscope                    — optional China mirror, pulls many heavy deps
#   - torchcodec                    — not available on Windows (platform-guarded below)
#
# Key: Python import name   Value: pip install spec
ACE_STEP_DEPS = {
    # NOTE: transformers must stay <4.58.0 — ACE-Step breaks on 4.58+
    "transformers":            "transformers>=4.51.0,<4.58.0",
    "diffusers":               "diffusers",
    "gradio":                  "gradio==6.2.0",
    "matplotlib":              "matplotlib>=3.7.5",
    "scipy":                   "scipy>=1.10.1",
    "soundfile":               "soundfile>=0.13.1",
    "loguru":                  "loguru>=0.7.3",
    "einops":                  "einops>=0.8.1",
    "accelerate":              "accelerate>=1.12.0",
    "fastapi":                 "fastapi>=0.110.0",
    "uvicorn":                 "uvicorn[standard]>=0.27.0",
    "numba":                   "numba>=0.63.1",
    "vector_quantize_pytorch": "vector-quantize-pytorch>=1.27.15",
    # torchao must be pinned to exactly 0.13.0 for PyTorch 2.6.x compatibility.
    # - torchao <0.10 lacks Float8WeightOnlyConfig, which transformers imports at
    #   module load time — causes an AttributeError before any model loads.
    # - torchao >=0.14 requires PyTorch 2.7+ (register_constant / pytree API).
    # - torchao 0.13.0 is the highest version officially supporting PyTorch 2.6.0
    #   per the torchao compatibility table: https://github.com/pytorch/ao/issues/2919
    # ACE-Step does not use torchao for inference — it is only pulled in
    # transitively by transformers for optional quantization support.
    "torchao":                 "torchao==0.13.0",
    "diskcache":               "diskcache",
    "toml":                    "toml",
    "omegaconf":               "omegaconf",
    "safetensors":             "safetensors",
    "pydantic":                "pydantic",
    "huggingface_hub":         "huggingface_hub>=0.20.0",
    "librosa":                 "librosa",
}

# torchcodec is not available on Windows — only install on Linux/Mac
if sys.platform != "win32":
    ACE_STEP_DEPS["torchcodec"] = "torchcodec>=0.9.1"


def _install_flash_attn(vpy: str) -> None:
    """
    flash-attn speeds up LLM token generation substantially.
    On Windows + Python 3.11 + CUDA, use the pre-built wheel from ACE-Step's
    own requirements.txt. On Linux x86_64, install from PyPI. Otherwise skip.
    """
    _, _, already = query(vpy, "import flash_attn")
    if already:
        print("[OK] flash-attn already installed")
        return

    if sys.platform == "win32":
        # Pre-built wheel listed in ACE-Step 1.5 requirements.txt
        # Matches: Python 3.11, CUDA 12.8, torch 2.7.1, Windows AMD64
        win_wheel = (
            "https://github.com/sdbds/flash-attention-for-windows/releases/download/"
            "2.8.2/flash_attn-2.8.2+cu128torch2.7.1cxx11abiFALSEfullbackward"
            "-cp311-cp311-win_amd64.whl"
        )
        print("[..] Installing flash-attn for Windows (pre-built wheel)...")
        r = pip(vpy, "install", win_wheel, "--quiet")
        if r.returncode == 0:
            print("[OK] flash-attn installed")
        else:
            print("[INFO] flash-attn install failed — generation works but LM step may be slower")

    elif sys.platform == "linux" and platform.machine() == "x86_64":
        print("[..] Installing flash-attn for Linux...")
        r = pip(vpy, "install", "flash-attn", "--quiet")
        if r.returncode == 0:
            print("[OK] flash-attn installed")
        else:
            print("[INFO] flash-attn install failed — generation works but LM step may be slower")
    else:
        print("[INFO] flash-attn not available on this platform — skipping")


def _check_and_install_deps(vpy: str) -> None:
    """
    Verify every dep in ACE_STEP_DEPS is importable and install any missing ones.
    Idempotent — safe to call on every launch.
    """
    missing = []
    for import_name, install_spec in ACE_STEP_DEPS.items():
        _, _, ok = query(vpy, f"import {import_name}")
        if not ok:
            missing.append(install_spec)

    if not missing:
        print("[OK] v1.5 runtime deps satisfied")
        return

    print(f"[..] Installing {len(missing)} missing v1.5 dep(s)...")
    for dep in missing:
        r = pip(vpy, "install", dep, "--quiet")
        status = "[OK]" if r.returncode == 0 else "[WARN]"
        print(f"     {status} {dep}")


def ensure_ace_step_v15() -> bool:
    vpy = str(VENV_PY)

    # Install ALL deps first so that when we install acestep with --no-deps
    # everything it needs is already present, and pip can't silently pull
    # nano-vllm or wrong-version packages.
    print("[..] Checking v1.5 runtime deps...")
    _check_and_install_deps(vpy)

    # flash-attn (optional but recommended for speed)
    _install_flash_attn(vpy)

    _, _, ok = query(vpy, "import acestep; print(acestep.__version__)")
    if ok:
        print("[OK] ACE-Step 1.5 already installed")
        # Re-run dep check in case a prior install left gaps
        _check_and_install_deps(vpy)
        return True

    print("[..] Installing ACE-Step 1.5 from GitHub...")
    print("     (This may take a few minutes)")

    # Always use --no-deps: we have already installed all required packages
    # above. This prevents pip from trying to resolve nano-vllm (not on PyPI),
    # overwriting our pinned transformers, or re-downloading torch.
    r = pip(vpy, "install",
            "git+https://github.com/ace-step/ACE-Step-1.5.git",
            "--no-deps", "--quiet")

    if r.returncode == 0:
        print("[OK] ACE-Step 1.5 installed")
        # Re-run dep check — installing acestep may reveal new transitive deps
        _check_and_install_deps(vpy)
        return True

    print("[WARN] ACE-Step 1.5 install failed. Stitch will run in STUB mode.")
    return False


# ---------------------------------------------------------------------------
# Step 5: audio-separator in UI venv for stems
# ---------------------------------------------------------------------------
def ensure_audio_separator() -> None:
    ui_py = str(SCRIPT_DIR / ".venv" / "Scripts" / "python.exe")
    if not Path(ui_py).exists():
        return

    _, _, ok = query(ui_py, "import audio_separator")
    if ok:
        print("[OK] audio-separator already installed")
        return

    print("[..] Installing audio-separator for stem separation...")
    r = pip(ui_py, "install", "audio-separator[gpu]", "--quiet")
    print("[OK] audio-separator installed" if r.returncode == 0
          else "[WARN] audio-separator install failed — stems will use Demucs only")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print()

    if not ensure_python311():
        print("[ERROR] Cannot set up Python 3.11 — ACE-Step 1.5 unavailable")
        sys.exit(2)

    if not ensure_venv():
        sys.exit(2)

    ensure_cuda_torch()

    ok = ensure_ace_step_v15()

    ensure_audio_separator()

    # Final GPU report
    vpy = str(VENV_PY)
    out, _, _ = query(vpy,
        "import torch\n"
        "if torch.cuda.is_available():\n"
        "    name = torch.cuda.get_device_name(0)\n"
        "    vram = torch.cuda.get_device_properties(0).total_memory / 1e9\n"
        "    print(f'[OK] GPU: {name} ({vram:.0f} GB VRAM)')\n"
        "else:\n"
        "    print('[INFO] Generation will use CPU (slow)')\n"
    )
    print(out)

    sys.exit(0 if ok else 2)
