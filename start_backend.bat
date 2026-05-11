@echo off
title CS462 - Launcher
cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2

echo Starting Flask server...

:: รัน Flask ใน window ใหม่
start "CS462 Flask" cmd /k "cd /d "%~dp0" && set PYTHONIOENCODING=utf-8 && set TF_ENABLE_ONEDNN_OPTS=0 && set TF_CPP_MIN_LOG_LEVEL=2 && python app.py"

:: รอให้ Flask โหลด model (TF ใช้เวลา ~5-10 วิ)
echo Waiting for server to start...
ping -n 8 127.0.0.1 >nul

:: เปิด browser
start "" "http://localhost:5000"

echo Done! Browser opened. Close the Flask window to stop the server.
timeout /t 3 >nul
