@echo off
setlocal
set "NGROK_EXE=C:\Users\FIRMAN HIDAYATULLAH\Downloads\ngrok-v3-stable-windows-amd64\ngrok.exe"

if not exist "%NGROK_EXE%" (
    echo [ERROR] ngrok.exe tidak ditemukan:
    echo "%NGROK_EXE%"
    echo.
    echo Perbaiki path NGROK_EXE di run_ngrok.bat sesuai lokasi ngrok.exe di komputer Anda.
    pause
    exit /b 1
)

echo [INFO] Starting ngrok tunnel to backend port 8000...
"%NGROK_EXE%" http 8000
