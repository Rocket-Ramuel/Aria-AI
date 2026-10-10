#!/usr/bin/env bash
# Start Aria from this folder:  ./start-aria-linux.sh
# (or double-click it and choose "Run").
#
# The first time, it sets Aria up: it needs the internet and takes a few
# minutes. After that she starts in seconds.

cd "$(dirname "$0")" || exit 1

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && \
     "$c" -c 'import sys; sys.exit(sys.version_info[:2] < (3, 10))' 2>/dev/null; then
    PY="$c"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Aria needs Python 3.10 or newer. On Debian or Ubuntu:"
  echo "  sudo apt install python3 python3-venv python3-tk"
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Setting Aria up for the first time; this takes a few minutes."
  "$PY" -m venv .venv || {
    echo "Couldn't make a Python environment. On Debian or Ubuntu:"
    echo "  sudo apt install python3-venv python3-tk"
    exit 1
  }
fi
if ! cmp -s requirements.txt .venv/aria-requirements.txt; then
  .venv/bin/python -m pip install --upgrade pip || exit 1
  # PyTorch without the NVIDIA libraries (gigabytes) unless there's an NVIDIA GPU.
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu || exit 1
  fi
  .venv/bin/python -m pip install -r requirements.txt pypdf || exit 1
  cp requirements.txt .venv/aria-requirements.txt
fi

exec .venv/bin/python -m aria.app "$@"
