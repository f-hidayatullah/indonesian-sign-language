@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python virtual environment tidak ditemukan:
    echo "%PYTHON_EXE%"
    echo.
    echo Pastikan folder .venv ada di root project atau install dependency terlebih dahulu.
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%app.py" (
    echo [ERROR] app.py tidak ditemukan di:
    echo "%SCRIPT_DIR%"
    pause
    exit /b 1
)

rem ── Cek Keberadaan Model YOLO (train_v4_negative3 atau train default) ────────
set "YOLO_EXIST=0"
if exist "%SCRIPT_DIR%runs\detect\train_v4_negative3\weights\best.pt" set "YOLO_EXIST=1"
if exist "%SCRIPT_DIR%runs\detect\train\weights\best.pt" set "YOLO_EXIST=1"

if "%YOLO_EXIST%"=="0" (
    echo [ERROR] Model YOLO tidak ditemukan di runs\detect\train_v4_negative3\ atau runs\detect\train\
    pause
    exit /b 1
)

if not exist "%SCRIPT_DIR%artifacts\rf_model.pkl" (
    echo [ERROR] Model Random Forest tidak ditemukan:
    echo "%SCRIPT_DIR%artifacts\rf_model.pkl"
    pause
    exit /b 1
)

rem ── Cek apakah port 8000 sudah dipakai ─────────────────────────────────────
set "PORT_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do set "PORT_PID=%%P"

if defined PORT_PID (
    echo [INFO] Port 8000 sedang dipakai oleh process PID %PORT_PID%.
    echo [INFO] Menghentikan proses lama secara otomatis...
    taskkill /PID %PORT_PID% /F >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Gagal menghentikan proses PID %PORT_PID%.
        echo         Coba tutup manual dari Task Manager lalu jalankan kembali.
        pause
        exit /b 1
    )
    echo [INFO] Proses lama berhasil dihentikan. Menunggu port bebas...
    timeout /t 2 /nobreak >nul
)

rem ── Persiapan direktori ─────────────────────────────────────────────────────
if not exist "%SCRIPT_DIR%artifacts\ultralytics" mkdir "%SCRIPT_DIR%artifacts\ultralytics"
set "ULTRALYTICS_CONFIG_DIR=%SCRIPT_DIR%artifacts\ultralytics"

rem ── Jalankan server ─────────────────────────────────────────────────────────
pushd "%SCRIPT_DIR%"
echo.
echo ============================================================
echo   SIBI Alphabet Recognition - Backend Server
echo ============================================================
echo   Akses sistem di browser: http://localhost:8000
echo ============================================================
echo.
"%PYTHON_EXE%" -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info
set "EXIT_CODE=%ERRORLEVEL%"
popd

echo.
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Backend berhenti dengan exit code %EXIT_CODE%.
) else (
    echo [INFO] Backend berhenti.
)
pause
exit /b %EXIT_CODE%
