@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: VFI-gui Auto-Detect Launcher
:: This script auto-detects GPU type and launches VFI-gui with the appropriate runtime

set "SCRIPT_DIR=%~dp0"
set "VFI_ROOT=%SCRIPT_DIR%.."
set "CUDA_RUNTIME=%VFI_ROOT%\runtime\cuda"
set "XPU_RUNTIME=%VFI_ROOT%\runtime\xpu"
set "CUDA_PYTHON=%CUDA_RUNTIME%\Scripts\python.exe"
set "XPU_PYTHON=%XPU_RUNTIME%\Scripts\python.exe"

echo [VFI-gui] Auto-detecting GPU type...

:: Check CUDA runtime availability first
set "USE_RUNTIME=cpu"
set "USE_PYTHON="

if exist "%CUDA_RUNTIME%\pyvenv.cfg" (
    if exist "%CUDA_PYTHON%" (
        :: Test CUDA availability
        "%CUDA_PYTHON%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
        if !ERRORLEVEL! EQU 0 (
            set "USE_RUNTIME=cuda"
            set "USE_PYTHON=%CUDA_PYTHON%"
            echo [VFI-gui] Detected: NVIDIA CUDA GPU
        )
    )
)

:: Check XPU runtime if CUDA not found
if "%USE_RUNTIME%"=="cpu" (
    if exist "%XPU_RUNTIME%\pyvenv.cfg" (
        if exist "%XPU_PYTHON%" (
            :: Test XPU availability
            "%XPU_PYTHON%" -c "import torch; exit(0 if hasattr(torch, 'xpu') and torch.xpu.is_available() else 1)" 2>nul
            if !ERRORLEVEL! EQU 0 (
                set "USE_RUNTIME=xpu"
                set "USE_PYTHON=%XPU_PYTHON%"
                echo [VFI-gui] Detected: Intel XPU GPU
            )
        )
    )
)

:: Launch with detected runtime
set "MAIN_PY=%SCRIPT_DIR%main.py"

if "%USE_RUNTIME%"=="cuda" (
    echo [VFI-gui] Starting with CUDA runtime...
    set "VIRTUAL_ENV=%CUDA_RUNTIME%"
    set "PATH=%CUDA_RUNTIME%\Scripts;%PATH%"
) else if "%USE_RUNTIME%"=="xpu" (
    echo [VFI-gui] Starting with XPU runtime...
    set "VIRTUAL_ENV=%XPU_RUNTIME%"
    set "PATH=%XPU_RUNTIME%\Scripts;%PATH%"
    set "ONEAPI_DEVICE_SELECTOR=*:gpu"
) else (
    echo [VFI-gui] No GPU detected, starting with CPU mode...
    echo [VFI-gui] Warning: CPU mode may be slow for video processing.
    :: Use system Python or first available runtime
    if exist "%CUDA_PYTHON%" (
        set "USE_PYTHON=%CUDA_PYTHON%"
        set "VIRTUAL_ENV=%CUDA_RUNTIME%"
    ) else if exist "%XPU_PYTHON%" (
        set "USE_PYTHON=%XPU_PYTHON%"
        set "VIRTUAL_ENV=%XPU_RUNTIME%"
    ) else (
        echo [ERROR] No Python runtime found.
        echo Please install at least one runtime environment.
        pause
        exit /b 1
    )
)

:: Check main.py exists
if not exist "%MAIN_PY%" (
    echo [ERROR] main.py not found: %MAIN_PY%
    pause
    exit /b 1
)

"%USE_PYTHON%" "%MAIN_PY%" %*

exit /b %ERRORLEVEL%
