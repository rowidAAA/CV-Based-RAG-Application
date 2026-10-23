@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python virtual environment...
    py -m venv .venv
    if errorlevel 1 goto :failed
)

echo Installing or updating application dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo Starting the CV RAG Assistant...
start "CV RAG Assistant" "http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
goto :eof

:failed
echo.
echo Startup failed. Check the message above, then press any key to close.
pause >nul
