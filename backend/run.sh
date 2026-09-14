#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements-min.txt
else
  source .venv/bin/activate
fi
exec uvicorn main:app --host 127.0.0.1 --port 8000 --reload
