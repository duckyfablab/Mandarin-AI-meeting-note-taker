@echo off
title TaiwanSpeak Local Notes
cd /d "C:\Users\Admin\Desktop\meeting_note_taker"

:: Open the web UI in your browser 2 seconds after the server starts
start "" /b cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000"

:: Use venv if present, otherwise fall back to global Python
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" app_online.py
) else (
    python app_online.py
)

pause