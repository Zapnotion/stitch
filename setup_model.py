"""
setup_model.py — Model environment setup for Stitch.

Handles:
  1. CUDA torch install in .venv_model (force-reinstall to replace +cpu build)
  2. ACE-Step v1 install (PyPI → GitHub fallback)
  3. ACE-Step v1.5 setup (optional — embedded Python 3.11 + separate venv)

Run by run.bat. Uses the BASE system Python (not a venv).

Exit codes:
  0 — at least v1 is working
  1 — fatal error
  2 — v1 unavailable (stub mode)
"""

import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
MODEL_PY    = str(SCRIPT_DIR / ".venv_model" / "Scripts" / "python.exe")
CUDA_INDEX  = "https://download.pytorch.org/whl/cu124"
env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

# Python 3.11 embedded distribution for v1.5 venv
PY311_VERSION = "3.11.9"
PY311_URL     = f"https://www.python.org/ftp/python/{PY311_VERSION}/python-{PY311_VERSION}-embed-amd64.zip"
PY311_DIR     = SCRIPT_DIR / ".python311"
PY311_EXE     = PY311_DIR / "python.exe"
VENV_V15      = SCRIPT_DIR / ".venv_model_v15"
VENV_V15_PY   = VENV_V15 / "Scripts" / "python.exe"


def run(args, capture=False):
    return subprocess.run(args, env=env, capture_output=capture, text=True)

def pip(model_py, *args):
    return run([model_py, "-m", "pip"] + list(args))

def query(py, code):
    r = run([py, "-c", code], capture=True)
    return r.stdout.strip(), r.stderr.strip(), r.returncode == 0


# ---------------------------------------------------------------------------
# Step 1: CUDA torch for .venv_model
# ---------------------------------------------------------------------------
def ensure_cuda_torch():
    _, _, cuda_ok = query(MODEL_PY, "import torch; assert torch.cuda.is_available()")
    if cuda_ok:
        out, _, _ = query(MODEL_PY, "import torch; print(torch.__version__)")
        print(f"[OK] PyTorch CUDA already available ({out})")
        return

    if not shutil.which("nvidia-smi"):
        print("[INFO] No NVIDIA GPU detected - will use CPU")
        return

    print("[..] NVIDIA GPU found. Installing PyTorch with CUDA 12.4 support...")
    print("     (force-reinstalling to replace +cpu build — may take a few minutes)")
    r = pip(MODEL_PY, "install", "torch", "torchvision", "torchaudio",
            "--index-url", CUDA_INDEX, "--force-reinstall", "--quiet")
    if r.returncode != 0:
        print("[WARN] CUDA torch install failed")
        return

    out, _, ok = query(MODEL_PY,
        "import torch\n"
        "print(torch.__version__)\n"
        "assert torch.cuda.is_available()\n"
        "print('[OK] GPU:', torch.cuda.get_device_name(0))\n"
    )
    print(out if ok else "[WARN] CUDA torch installed but GPU still not detected")


# ---------------------------------------------------------------------------
# Step 2: ACE-Step v1
# ---------------------------------------------------------------------------
def ensure_ace_step_v1():
    _, _, ok = query(MODEL_PY, "import acestep")
    if ok:
        print("[OK] ace-step v1 already installed")
        return True

    print("[..] Installing ACE-Step v1 into model environment...")
    print("     (This may take a few minutes)")

    r = pip(MODEL_PY, "install", "ace-step", "--quiet")
    if r.returncode == 0:
        print("[OK] ace-step v1 installed from PyPI")
        return True

    print("[WARN] PyPI install failed (upstream setup.py bug). Trying GitHub...")
    r = pip(MODEL_PY, "install",
            "git+https://github.com/ace-step/ACE-Step.git", "--quiet")
    if r.returncode == 0:
        print("[OK] ace-step v1 installed from GitHub")
        return True

    print("[WARN] ace-step v1 could not be installed. Stitch will run in STUB mode.")
    return False


# ---------------------------------------------------------------------------
# Step 3: Embedded Python 3.11 + ACE-Step v1.5 (optional)
# ---------------------------------------------------------------------------
def ensure_python311():
    """Download embedded Python 3.11 if not already present."""
    if PY311_EXE.exists():
        return True

    print(f"[..] Downloading embedded Python {PY311_VERSION} for ACE-Step v1.5…")
    print(f"     (~25 MB — only needed once)")
    PY311_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = PY311_DIR / "python311.zip"
    try:
        urllib.request.urlretrieve(PY311_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(PY311_DIR)
        zip_path.unlink(missing_ok=True)

        # The embedded dist needs pip — patch ._pth to allow site-packages
        pth_files = list(PY311_DIR.glob("python3*._pth"))
        for pth in pth_files:
            content = pth.read_text()
            if "#import site" in content:
                pth.write_text(content.replace("#import site", "import site"))

        # Install pip into embedded Python
        get_pip = PY311_DIR / "get-pip.py"
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip)
        subprocess.run([str(PY311_EXE), str(get_pip), "--quiet"], env=env)
        get_pip.unlink(missing_ok=True)

        print(f"[OK] Python {PY311_VERSION} embedded dist ready")
        return True
    except Exception as exc:
        print(f"[WARN] Could not set up embedded Python 3.11: {exc}")
        shutil.rmtree(PY311_DIR, ignore_errors=True)
        return False


def ensure_ace_step_v15():
    """Set up .venv_model_v15 with ACE-Step 1.5 using embedded Python 3.11."""
    if not ensure_python311():
        return False

    # Create venv using embedded Python via pip's venv
    if not VENV_V15_PY.exists():
        print("[..] Creating .venv_model_v15 for ACE-Step v1.5…")
        # Embedded Python can't run -m venv — use virtualenv instead
        r = run([str(PY311_EXE), "-m", "pip", "install", "virtualenv", "--quiet"])
        if r.returncode != 0:
            print("[WARN] Could not install virtualenv for v1.5")
            return False
        r = run([str(PY311_EXE), "-m", "virtualenv", str(VENV_V15), "--quiet"])
        if r.returncode != 0:
            print("[WARN] Could not create v1.5 venv")
            return False
        print("[OK] .venv_model_v15 created")

    vpy = str(VENV_V15_PY)

    # Install CUDA torch first
    _, _, cuda_ok = query(vpy, "import torch; assert torch.cuda.is_available()")
    if not cuda_ok and shutil.which("nvidia-smi"):
        print("[..] Installing CUDA torch into v1.5 venv…")
        pip(vpy, "install", "torch", "torchvision", "torchaudio",
            "--index-url", CUDA_INDEX, "--force-reinstall", "--quiet")

    # Install ACE-Step 1.5
    _, _, v15_ok = query(vpy, "import acestep")
    if not v15_ok:
        print("[..] Installing ACE-Step v1.5 from GitHub…")
        r = pip(vpy, "install",
                "git+https://github.com/ace-step/ACE-Step-1.5.git", "--quiet")
        if r.returncode == 0:
            print("[OK] ACE-Step v1.5 installed")
        else:
            print("[WARN] ACE-Step v1.5 install failed")
            return False
    else:
        print("[OK] ACE-Step v1.5 already installed")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not Path(MODEL_PY).exists():
        print(f"[ERROR] Model venv not found at {MODEL_PY}")
        sys.exit(1)

    ensure_cuda_torch()
    v1_ok = ensure_ace_step_v1()

    # v1.5 is optional — failure doesn't block v1
    print()
    print("[..] Setting up ACE-Step v1.5 environment (optional — can take a while)…")
    v15_ok = ensure_ace_step_v15()
    if v15_ok:
        print("[OK] ACE-Step v1.5 environment ready")
    else:
        print("[INFO] ACE-Step v1.5 not available — v1 only mode")

    # Final GPU report
    out, _, _ = query(MODEL_PY,
        "import torch\n"
        "if torch.cuda.is_available():\n"
        "    print('[OK] GPU:', torch.cuda.get_device_name(0))\n"
        "else:\n"
        "    print('[INFO] Generation will use CPU (slow)')\n"
    )
    print(out)

    sys.exit(0 if v1_ok else 2)
