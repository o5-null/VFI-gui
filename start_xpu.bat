@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: VFI-gui XPU Runtime Launcher
:: This script launches VFI-gui using the Intel XPU runtime environment

set "SCRIPT_DIR=%~dp0"
set "VFI_ROOT=%SCRIPT_DIR%.."
set "XPU_RUNTIME=%VFI_ROOT%\runtime\xpu"
set "PYTHON_EXE=%XPU_RUNTIME%\Scripts\python.exe"
set "MAIN_PY=%SCRIPT_DIR%main.py"

:: Check if XPU runtime exists
if not exist "%XPU_RUNTIME%\pyvenv.cfg" (
    echo [ERROR] XPU runtime not found: %XPU_RUNTIME%
    echo Please ensure the XPU runtime environment is installed.
    pause
    exit /b 1
)

:: Check if Python executable exists
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    echo Please ensure the XPU runtime environment is properly set up.
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
set "VIRTUAL_ENV=%XPU_RUNTIME%"
set "PATH=%XPU_RUNTIME%\Scripts;%PATH%"

:: Set Intel GPU specific environment variables
set "ONEAPI_DEVICE_SELECTOR=*:gpu"
set "ZE_ENABLE_VALIDATION_LAYER=1"

:: Launch VFI-gui with XPU runtime
echo [VFI-gui] Starting with Intel XPU runtime...
echo [VFI-gui] Python: %PYTHON_EXE%
echo [VFI-gui] Runtime: %XPU_RUNTIME%

"%PYTHON_EXE%" "%MAIN_PY%" %*

exit /b %ERRORLEVEL%
