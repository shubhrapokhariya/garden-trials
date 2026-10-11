#!/usr/bin/env bash
# Garden Trials launcher for macOS and Linux:  bash run.sh
set -e
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Python 3 is not installed. Get it from https://www.python.org/downloads/"; exit 1; }
python3 -m pip install -r requirements.txt
if command -v ollama >/dev/null; then
  ollama pull "${OLLAMA_MODEL:-gemma3:1b}" || echo "Could not download the model. Is Ollama running? You can still tick 'Run without Gemma' in the app."
else
  echo "Ollama is not installed (https://ollama.com/download). Starting anyway."
  echo "Tick 'Run without Gemma' in the sidebar, or set OLLAMA_HOST if Ollama runs on another machine."
fi
python3 -m streamlit run app.py
