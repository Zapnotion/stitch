@echo off
setlocal enabledelayedexpansion

echo.
echo  ================================================
echo   Stitch  -  AI Music Workstation
echo  ================================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: ---------------------------------------------------------------------------
:: 1. Check Python 3.10+
:: ---------------------------------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Install Python 3.10+ from https://python.org
    echo         Check "Add Python to PATH" during install.
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER%
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    if %%a LSS 3 ( echo [ERROR] Need Python 3.10+. & pause & exit /b 1 )
    if %%a EQU 3 if %%b LSS 10 ( echo [ERROR] Need Python 3.10+. Found %PY_VER%. & pause & exit /b 1 )
)

:: ---------------------------------------------------------------------------
:: 2. Main UI venv (.venv)
:: ---------------------------------------------------------------------------
echo.
echo  [Step 1/4] Setting up UI environment...
call :setup_venv ".venv" "requirements.txt"
if errorlevel 1 ( pause & exit /b 1 )

:: ---------------------------------------------------------------------------
:: 2b. llama-cpp-python (offline lyrics LLM — CPU or CUDA wheel)
:: ---------------------------------------------------------------------------
echo.
echo  [Step 1b/4] Checking llama-cpp-python (offline lyrics engine)...
set "VENV_PY=.venv\Scripts\python.exe"

:: Check if already installed
"%VENV_PY%" -c "import llama_cpp" >nul 2>&1
if not errorlevel 1 (
    echo [OK] llama-cpp-python already installed
    goto :llama_done
)

:: Detect CUDA availability via torch (already installed in the venv)
set CUDA_AVAILABLE=0
"%VENV_PY%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 set CUDA_AVAILABLE=1

if "%CUDA_AVAILABLE%"=="1" (
    echo [..] GPU detected - installing llama-cpp-python with CUDA support...
    echo      This may take a minute on first run.
    set "CMAKE_ARGS=-DGGML_CUDA=on"
    set "FORCE_CMAKE=1"
    "%VENV_PY%" -m pip install llama-cpp-python --upgrade --quiet ^
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    if errorlevel 1 (
        echo [WARN] CUDA llama-cpp-python failed - falling back to CPU wheel...
        "%VENV_PY%" -m pip install llama-cpp-python --upgrade --quiet
    )
) else (
    echo [..] No GPU - installing CPU-only llama-cpp-python...
    "%VENV_PY%" -m pip install llama-cpp-python --upgrade --quiet
)

"%VENV_PY%" -c "import llama_cpp" >nul 2>&1
if errorlevel 1 (
    echo [WARN] llama-cpp-python install failed.
    echo        AI lyrics will use the built-in template fallback until fixed.
) else (
    echo [OK] llama-cpp-python ready
)

:llama_done

:: ---------------------------------------------------------------------------
:: 2c. whisper-timestamped (lyric alignment — CPU-only, Phase 1)
:: ---------------------------------------------------------------------------
echo.
echo  [Step 1c/4] Checking whisper-timestamped (lyric alignment)...

:: Check if already installed
"%VENV_PY%" -c "import whisper_timestamped" >nul 2>&1
if not errorlevel 1 (
    echo [OK] whisper-timestamped already installed
    goto :whisper_done
)

:: whisper-timestamped is CPU-only — no CUDA variant needed.
:: It depends on openai-whisper which in turn needs ffmpeg on PATH.
echo [..] Installing whisper-timestamped (CPU-only, ~150 MB model on first use)...
"%VENV_PY%" -m pip install whisper-timestamped --upgrade --quiet

"%VENV_PY%" -c "import whisper_timestamped" >nul 2>&1
if errorlevel 1 (
    echo [WARN] whisper-timestamped install failed.
    echo        Lyric alignment and the timeline view will be unavailable.
    echo        Generation still works normally without it.
) else (
    echo [OK] whisper-timestamped ready
)

:whisper_done

:: ---------------------------------------------------------------------------
:: 2d. pyloudnorm (broadcast loudness normalisation — Phase 3)
:: ---------------------------------------------------------------------------
echo  [Step 1d/4] Checking pyloudnorm...
"%VENV_PY%" -c "import pyloudnorm" >nul 2>&1
if not errorlevel 1 (
    echo [OK] pyloudnorm already installed
    goto :pyloudnorm_done
)
echo [..] Installing pyloudnorm...
"%VENV_PY%" -m pip install pyloudnorm --upgrade --quiet
"%VENV_PY%" -c "import pyloudnorm" >nul 2>&1
if errorlevel 1 (
    echo [WARN] pyloudnorm install failed. Loudness normalisation will use RMS fallback.
) else (
    echo [OK] pyloudnorm ready
)
:pyloudnorm_done

:: ---------------------------------------------------------------------------
:: 2e. pyrubberband (vocal pitch correction — Phase 3)
::     Needs the rubberband CLI on PATH or rubberband.dll bundled in app/libs/.
::     Falls back to librosa.effects.pitch_shift if not importable.
:: ---------------------------------------------------------------------------
echo  [Step 1e/4] Checking pyrubberband...
"%VENV_PY%" -c "import pyrubberband" >nul 2>&1
if not errorlevel 1 (
    echo [OK] pyrubberband already installed
    goto :pyrubberband_done
)
echo [..] Installing pyrubberband...
"%VENV_PY%" -m pip install pyrubberband --upgrade --quiet
"%VENV_PY%" -c "import pyrubberband" >nul 2>&1
if errorlevel 1 (
    echo [WARN] pyrubberband install failed. Pitch correction will fall back to librosa.
) else (
    echo [OK] pyrubberband ready
)
:pyrubberband_done


:: ---------------------------------------------------------------------------
echo.
echo  [Step 2/4] Setting up model environment...
call :setup_venv ".venv_model" "requirements_model.txt"
if errorlevel 1 ( pause & exit /b 1 )

:: ---------------------------------------------------------------------------
:: 4+5. GPU torch + ACE-Step (handled by setup_model.py to avoid batch bugs)
:: ---------------------------------------------------------------------------
echo.
echo  [Step 3/4] Setting up ACE-Step 1.5 environment...
python setup_model.py
set SETUP_CODE=%ERRORLEVEL%
if %SETUP_CODE% EQU 1 (
    echo [ERROR] setup_model.py hit a fatal error - cannot continue.
    pause & exit /b 1
)
if %SETUP_CODE% GEQ 2 (
    echo [WARN] ACE-Step unavailable - Stitch will run in STUB mode.
)
::after_model_setup

:: ---------------------------------------------------------------------------
:: 6. AppData folders
:: ---------------------------------------------------------------------------
set "APP_DATA=%APPDATA%\stitch"
for %%D in (
    "%APP_DATA%" "%APP_DATA%\outputs" "%APP_DATA%\inputs"
    "%APP_DATA%\models" "%APP_DATA%\models\loras" "%APP_DATA%\models\lyrics_models"
    "%APP_DATA%\stems" "%APP_DATA%\presets" "%APP_DATA%\logs"
) do ( if not exist %%D mkdir %%D >nul 2>&1 )
echo [OK] App data: %APP_DATA%

:: ---------------------------------------------------------------------------
:: 7. Launch
:: ---------------------------------------------------------------------------
echo.
echo  Starting Stitch...
echo.

:: STITCH_DEBUG=1 ensures all log levels (INFO+DEBUG) are visible in the
:: Console panel inside the app and in this window. Remove to reduce noise.
set STITCH_DEBUG=1

:: Activate the UI venv for the main process
call ".venv\Scripts\activate.bat"
python main.py
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% NEQ 0 (
    echo.
    echo  ================================================
    echo   Stitch exited with code %EXIT_CODE%
    echo   Log: %APP_DATA%\logs\stitch.log
    echo  ================================================
    echo.
    pause
)
endlocal
exit /b %EXIT_CODE%


:: ---------------------------------------------------------------------------
:: Subroutine: setup_venv <venv_dir> <requirements_file>
:: ---------------------------------------------------------------------------
:setup_venv
set "VENV_DIR=%~1"
set "REQS=%~2"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if exist "%VENV_DIR%\" (
    "%VENV_PY%" --version >nul 2>&1
    if errorlevel 1 (
        echo [WARN] %VENV_DIR% is broken - rebuilding...
        rmdir /s /q "%VENV_DIR%" >nul 2>&1
    ) else (
        echo [OK] %VENV_DIR% is healthy
        goto :check_reqs
    )
)

echo [..] Creating %VENV_DIR%...
python -m venv "%VENV_DIR%"
if errorlevel 1 ( echo [ERROR] Failed to create %VENV_DIR%. & exit /b 1 )
echo [OK] %VENV_DIR% created

:check_reqs
"%VENV_PY%" -m pip install --upgrade pip --quiet 2>nul

echo [..] Checking %REQS%...
set _ERRS=0
for /f "usebackq eol=# tokens=* delims=" %%L in ("%REQS%") do (
    set "LINE=%%L"
    if not "!LINE!"=="" (
        for /f "tokens=1 delims=><=~! " %%N in ("!LINE!") do set "PKG=%%N"
        "%VENV_PY%" -m pip show "!PKG!" >nul 2>&1
        if errorlevel 1 (
            echo      Installing !LINE!
            "%VENV_PY%" -m pip install "!LINE!" --quiet
            if errorlevel 1 ( echo [WARN] Could not install !LINE! & set _ERRS=1 )
        )
    )
)
if "!_ERRS!"=="1" (
    echo [WARN] Some packages in %REQS% failed to install.
) else (
    echo [OK] %REQS% satisfied
)
exit /b 0
