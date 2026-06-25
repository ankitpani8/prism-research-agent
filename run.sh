#!/usr/bin/env bash
# One-command local run: sets up venv (if needed) and launches the Streamlit demo.
set -euo pipefail

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
  echo "No .env found — copy .env.example to .env and add your GEMINI_API_KEY."
  exit 1
fi

streamlit run streamlit_app.py
