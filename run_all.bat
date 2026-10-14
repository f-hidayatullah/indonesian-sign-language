@echo off
setlocal
set "SCRIPT_DIR=%~dp0"

if not exist "%SCRIPT_DIR%run_backend.bat" (
    echo [ERROR] run_backend.bat tidak ditemukan di "%SCRIPT_DIR%"
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%run_ngrok.bat" (
    echo [ERROR] run_ngrok.bat tidak ditemukan di "%SCRIPT_DIR%"
    pause
    exit /b 1
)

start "SIBI Backend" /D "%SCRIPT_DIR%" cmd /k call "%SCRIPT_DIR%run_backend.bat"
start "SIBI ngrok" /D "%SCRIPT_DIR%" cmd /k call "%SCRIPT_DIR%run_ngrok.bat"
