@echo off
rem Garden Trials launcher for Windows. Double-click to start.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is not installed. Get it from https://www.python.org/downloads/ & pause & exit /b 1)
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Could not install the packages. Check your internet connection and try again.
  pause
  exit /b 1
)
if not defined OLLAMA_MODEL set OLLAMA_MODEL=gemma3:1b
where ollama >nul 2>nul
if errorlevel 1 (
  echo Ollama is not installed. Get it from https://ollama.com/download
  echo Starting anyway. Tick "Run without Gemma" in the sidebar, or set OLLAMA_HOST if Ollama runs on another machine.
) else (
  ollama pull %OLLAMA_MODEL% || echo Could not download the model. Is Ollama running? You can still tick "Run without Gemma" in the app.
)
python -m streamlit run app.py
pause
