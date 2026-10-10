#!/bin/bash
# Double-click this file to start Aria from this folder.
#
# The first time, it sets Aria up: it needs the internet and takes a few
# minutes. After that she starts in seconds. Closing this window stops Aria;
# she saves what she has learned first.

cd "$(dirname "$0")" || exit 1
printf '\033]0;Aria\007'

stop() {
  echo
  read -r -p "Press Return to close this window. " _
  exit 1
}

# Python from python.org (or Homebrew), newest first. Not /usr/bin/python3:
# on a Mac without developer tools that only offers to install them.
PY=""
for c in /Library/Frameworks/Python.framework/Versions/3.1{3,2,1,0}/bin/python3 \
         /opt/homebrew/bin/python3 /usr/local/bin/python3; do
  if [ -x "$c" ] && "$c" -c 'import sys; sys.exit(sys.version_info[:2] < (3, 10))' 2>/dev/null; then
    PY="$c"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Aria needs Python 3.10 or newer, and this Mac doesn't have it yet."
  echo "Your browser is opening python.org: download the macOS installer, run it,"
  echo "then double-click this file again."
  open "https://www.python.org/downloads/macos/"
  stop
fi

if [ ! -x .venv/bin/python ]; then
  echo "Setting Aria up for the first time. This downloads about 100 MB and takes"
  echo "a few minutes; next time she starts straight away."
  echo
  "$PY" -m venv .venv || stop
fi
if ! cmp -s requirements.txt .venv/aria-requirements.txt; then
  .venv/bin/python -m pip install --upgrade pip || stop
  EXTRA=""
  # The last PyTorch for Intel Macs (2.2) needs NumPy 1.
  [ "$(uname -m)" = x86_64 ] && EXTRA="numpy<2"
  .venv/bin/python -m pip install -r requirements.txt pypdf $EXTRA || stop
  cp requirements.txt .venv/aria-requirements.txt
fi

exec .venv/bin/python -m aria.app "$@"
