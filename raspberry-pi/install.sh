#!/usr/bin/env bash
# Install Aria on a Raspberry Pi so that she runs all the time, for every
# device on your network. Run it from the Aria-AI folder, as your normal user:
#
#     ./raspberry-pi/install.sh
#
# It asks for your password when it needs to install things. Run it again
# after `git pull` to update her; what she has learned is kept.
#
# Options:
#   --port N            serve on port N instead of 8000
#   --allow-host NAME   another name she may be reached by, such as a
#                       Tailscale name (repeatable)
#   --no-service        set her up and check her, but don't start her with
#                       the Pi (run `.venv/bin/python -m aria.pi` yourself)

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
PORT=8000
EXTRA=()
SERVICE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --allow-host) EXTRA+=(--allow-host "$2"); shift 2 ;;
    --no-service) SERVICE=0; shift ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)"; exit 2 ;;
  esac
done

step() { printf '\n== %s\n' "$*"; }

if [ "$(id -u)" -eq 0 ]; then
  echo "Run this as your normal user, not with sudo: it asks for your password"
  echo "when it needs it, and Aria keeps her memory in your home folder."
  exit 1
fi

case "$(uname -m)" in
  aarch64|arm64|x86_64) ;;
  armv7l|armv6l)
    echo "Aria needs the 64-bit version of Raspberry Pi OS (PyTorch, the AI"
    echo "library she's built on, has no 32-bit version). Reinstall the Pi with"
    echo "\"Raspberry Pi OS (64-bit)\" in Raspberry Pi Imager, then run this again."
    exit 1 ;;
  *) echo "Aria doesn't know how to run on a $(uname -m) machine."; exit 1 ;;
esac

MEM_MB=$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)
if [ "$MEM_MB" -lt 1800 ]; then
  echo "Warning: this machine has ${MEM_MB} MB of memory. Aria uses about 1 GB on"
  echo "top of the system, so she may be very slow here, or not start."
fi

step "Installing what Aria needs from the system (asks for your password)"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip curl >/dev/null
python3 -c 'import sys; sys.exit(sys.version_info[:2] < (3, 10))' || {
  echo "Aria needs Python 3.10 or newer; this system has $(python3 --version)."
  exit 1
}

step "Installing PyTorch and Aria's other parts (the first time takes a while)"
[ -x .venv/bin/python ] || python3 -m venv .venv
if ! cmp -s requirements.txt .venv/aria-requirements.txt; then
  .venv/bin/python -m pip install -q --upgrade pip
  # The CPU-only build: a Pi has no GPU PyTorch can use.
  .venv/bin/python -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu \
    || .venv/bin/python -m pip install -q torch
  .venv/bin/python -m pip install -q -r requirements.txt pypdf
  cp requirements.txt .venv/aria-requirements.txt
fi

step "Checking that Aria works here: loading, chatting, reading, saving"
# In a temporary folder, so it doesn't touch her real memory.
.venv/bin/python -m aria app --self-test

ADDRESSES=("http://$(hostname -s).local:$PORT")
for ip in $(hostname -I 2>/dev/null); do
  case "$ip" in *:*) ;; *) ADDRESSES+=("http://$ip:$PORT") ;; esac
done

if [ "$SERVICE" -eq 0 ]; then
  step "Done"
  echo "Start her with:  .venv/bin/python -m aria.pi --port $PORT ${EXTRA[*]:-}"
  exit 0
fi

step "Setting Aria up to start whenever this machine turns on"
sudo tee /etc/systemd/system/aria.service >/dev/null <<EOF
[Unit]
Description=Aria, a small AI that learns as you talk to her
After=network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$REPO
ExecStart="$REPO/.venv/bin/python" -m aria.pi --port $PORT ${EXTRA[*]:-}
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=5
# Time to finish a learning step and save before being stopped.
TimeoutStopSec=150

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --quiet aria.service
sudo systemctl restart aria.service

for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/api/hello" >/dev/null; then
    step "Aria is running, and will start by herself whenever this machine does"
    echo "On your computer or phone (on the same network), open:"
    for a in "${ADDRESSES[@]}"; do echo "    $a"; done
    echo
    echo "See what she's doing:   journalctl -u aria -f"
    echo "Update her later:       git pull && ./raspberry-pi/install.sh"
    exit 0
  fi
  sleep 1
done
echo "Aria didn't answer within two minutes. What she said:"
sudo journalctl -u aria --no-pager -n 40
exit 1
