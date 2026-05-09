@echo off
REM Launch the Engine Removal Forecast inference stack.
REM Double-click this file from Windows Explorer; or run it from any shell.

cd /d "%~dp0"
title Engine Removal Forecast - servers

echo.
echo === Engine Removal Forecast =================================
echo Working dir: %CD%
echo Starting FastAPI backend on port 8001 + static frontend on 8000
echo.
echo Open http://127.0.0.1:8000/index.html in your browser once you
echo see "Serving static files on port 8000..." below.
echo.
echo Press Ctrl+C in this window to stop both servers.
echo =============================================================
echo.

python start_servers.py

echo.
echo Servers stopped. Press any key to close this window.
pause >nul
