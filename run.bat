@echo off
setlocal enabledelayedexpansion

echo.
echo  ================================================
echo   Stitch  -  AI Music Workstation
echo  ================================================
echo.

:: --- Locate script directory (repo root) ---
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: ---------------------------------------------------------------------------
:: 1. Check Python
:: ---------------------------------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Install Python 3.10+ from https://python.org
    echo         Check "Add Python to PATH" during install.
    echo.
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER%

:: Require 3.10+
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    if %%a LSS 3 ( echo [ERROR] Python 3.10+ required. & pause & exit /b 1 )
    if %%a EQU 3 if %%b LSS 10 ( echo [ERROR] Python 3.10+ required. Found %PY_VER%. & pause & exit /b 1 )
)

:: ---------------------------------------------------------------------------
:: 2. Create or repair virtual environment
:: ---------------------------------------------------------------------------
set "VENV_PY=.venv\Scripts\python.exe"

if exist ".venv\" (
    "%VENV_PY%" --version >nul 2>&1
    if errorlevel 1 (
        echo [WARN] Existing venv is broken - rebuilding...
        rmdir /s /q ".venv" >nul 2>&1
        goto :create_venv
    )
    echo [OK] Virtual environment is healthy
    goto :activate
)

:create_venv
echo [..] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause & exit /b 1
)
echo [OK] Virtual environment created

:activate
call ".venv\Scripts\activate.bat"
if errorlevel 1 ( echo [ERROR] Failed to activate venv. & pause & exit /b 1 )
echo [OK] Virtual environment activated

:: ---------------------------------------------------------------------------
:: 3. Upgrade pip
:: ---------------------------------------------------------------------------
python -m pip install --upgrade pip --quiet 2>nul

:: ---------------------------------------------------------------------------
:: 4. Install requirements - check each package, install only if missing
:: ---------------------------------------------------------------------------
echo [..] Checking requirements...
set INSTALL_ERRORS=0

for /f "usebackq eol=# tokens=* delims=" %%L in ("requirements.txt") do (
    set "LINE=%%L"
    :: Skip blank lines
    if not "!LINE!"=="" (
        :: Extract package name (everything before >= <= == ~= [space])
        for /f "tokens=1 delims=><=~ " %%N in ("!LINE!") do set "PKG_NAME=%%N"
        python -m pip show "!PKG_NAME!" >nul 2>&1
        if errorlevel 1 (
            echo      Installing !LINE! ...
            python -m pip install "!LINE!" --quiet
            if errorlevel 1 (
                echo [WARN] Failed to install: !LINE!
                set INSTALL_ERRORS=1
            )
        )
    )
)

if "!INSTALL_ERRORS!"=="1" (
    echo.
    echo [WARN] Some packages failed. Stitch may run in limited mode.
    echo        Common fix: install PyTorch manually first from https://pytorch.org
    echo.
) else (
    echo [OK] All requirements satisfied
)

:: ---------------------------------------------------------------------------
:: 5. GPU check (informational)
:: ---------------------------------------------------------------------------
python -c "import torch; assert torch.cuda.is_available(); print('[OK] GPU:', torch.cuda.get_device_name(0))" 2>nul
if errorlevel 1 (
    echo [INFO] CUDA GPU not detected - will run on CPU ^(slow for generation^)
    echo        Install CUDA PyTorch from: https://pytorch.org/get-started/locally/
)

:: ---------------------------------------------------------------------------
:: 6. Ensure AppData folders exist
:: ---------------------------------------------------------------------------
set "APP_DATA=%APPDATA%\stitch"
for %%D in (
    "%APP_DATA%"
    "%APP_DATA%\outputs"
    "%APP_DATA%\inputs"
    "%APP_DATA%\models"
    "%APP_DATA%\models\loras"
    "%APP_DATA%\stems"
    "%APP_DATA%\presets"
    "%APP_DATA%\logs"
) do (
    if not exist %%D mkdir %%D >nul 2>&1
)
echo [OK] App data: %APP_DATA%

:: ---------------------------------------------------------------------------
:: 7. Launch Stitch
:: ---------------------------------------------------------------------------
echo.
echo  Starting Stitch...
echo.

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
