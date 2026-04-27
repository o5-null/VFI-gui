@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: VFI-gui CUDA Runtime Launcher
:: This script launches VFI-gui using the NVIDIA CUDA runtime environment

set "SCRIPT_DIR=%~dp0"
set "VFI_ROOT=%SCRIPT_DIR%.."
set "CUDA_RUNTIME=%VFI_ROOT%\runtime\cuda"
set "PYTHON_EXE=%CUDA_RUNTIME%\Scripts\python.exe"
set "MAIN_PY=%SCRIPT_DIR%main.py"

:: Check if CUDA runtime exists
if not exist "%CUDA_RUNTIME%\pyvenv.cfg" (
    echo [ERROR] CUDA runtime not found: %CUDA_RUNTIME%
    echo Please ensure the CUDA runtime environment is installed.
    pause
    exit /b 1
)

:: Check if Python executable exists
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo Please ensure the CUDA runtime environment is properly set up.
    pause
    exit /b 1
)

:: Check if main.py exists
if not exist "%MAIN_PY%" (
    echo [ERROR] main.py not found: %MAIN_PY%
    pause
    exit /b 1
)

:: Set environment variables
set "VIRTUAL_ENV=%CUDA_RUNTIME%"
set "PATH=%CUDA_RUNTIME%\Scripts;%PATH%"

:: Launch VFI-gui with CUDA runtime
echo [VFI-gui] Starting with CUDA runtime...
echo [VFI-gui] Python: %PYTHON_EXE%
echo [VFI-gui] Runtime: %CUDA_RUNTIME%

"%PYTHON_EXE%" "%MAIN_PY%" %*

exit /b %ERRORLEVEL%
