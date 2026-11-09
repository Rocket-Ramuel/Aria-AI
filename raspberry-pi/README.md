# Aria on a Raspberry Pi

This is a separate edition of Aria for a Raspberry Pi left switched on: she
runs all the time, starts by herself whenever the Pi does, and every computer
and phone on your home network talks to the same Aria, in a web browser. The
Mac, Windows and Linux apps are unaffected by any of this.

She saves what she has learned every five minutes and whenever she's stopped,
including when the Pi shuts down or reboots, so a power cut costs at most the
last few minutes.

## What you need

- A **Raspberry Pi 5 with 4 or 8 GB** of memory. A Pi 4 with 4 GB or more also
  works, more slowly. Models with 1 or 2 GB don't have enough memory: Aria
  uses about 1 GB on top of the system.
- Its power supply, and a case that keeps it cool.
- A **microSD card of 32 GB or more**, or better, a USB SSD: she rewrites her
  memory file (13 MB) every few minutes while she's learning, and an SSD
  wears far better than an SD card over years of that.
- Your home Wi-Fi or a network cable.

Left idle, a Pi 5 draws about 3 watts: a few dollars of electricity a year.

## Setting it up

1. **Put Raspberry Pi OS (64-bit) on the card** with
   [Raspberry Pi Imager](https://www.raspberrypi.com/software/). In the
   Imager's settings, give the Pi a name (say `aria`), choose a username and
   password, enter your Wi-Fi details, and turn on SSH. It must be the
   **64-bit** version.
2. **Start the Pi** and wait a couple of minutes for it to join the network.
3. **Open a terminal on the Pi.** If it has a screen and keyboard, open
   Terminal there. From your Mac instead: open Terminal and type
   `ssh yourname@aria.local` (your username and the Pi's name), then your
   password.
4. **Copy and paste these lines** into that terminal:

   ```bash
   sudo apt install -y git
   git clone -b claude/keen-einstein-9cy5rx https://github.com/Rocket-Ramuel/Aria-AI.git
   cd Aria-AI
   ./raspberry-pi/install.sh
   ```

   The first time takes a while, since it downloads PyTorch, the AI library
   Aria is built on. It then checks that she works on your Pi (loading,
   chatting, reading a document, saving) before setting her up to start with
   the Pi.
5. **Open her chat** on your computer or phone, on the same network. The
   installer prints the address, normally `http://aria.local:8000` (with your
   Pi's name in place of `aria`).

## Living with her

| to... | type, in the Aria-AI folder on the Pi |
| --- | --- |
| update her to the newest version | `git pull && ./raspberry-pi/install.sh` |
| see what she's doing | `journalctl -u aria -f` (Ctrl-C to stop watching) |
| check she's running | `systemctl status aria` |
| stop her for now | `sudo systemctl stop aria` (she saves first) |
| start her again | `sudo systemctl start aria` |

**Her memory** is in `~/.local/share/aria` on the Pi. Updating never touches
it. To move her to another computer, copy that folder across.

**From outside your home**, install [Tailscale](https://tailscale.com/) (free
for personal use) on the Pi and on your other devices. It links them
privately, so you can reach her from anywhere without putting her on the open
internet. If you reach her by her Tailscale name, tell her that name:
`./raspberry-pi/install.sh --allow-host aria.your-tailnet.ts.net`.

**Privacy:** her chat has no password. Anyone on your network can talk to her,
and what they say can end up in her memory. That's fine at home; don't open
her up to the internet (with port forwarding on your router, for example).

**Speed:** on a Pi 5 expect replies to take around a second, and documents to
take several times longer to read than on a laptop. These are estimates: the
edition is tested on GitHub's 64-bit ARM machines (the same kind of chip as a
Pi, but faster), not on a Pi itself.

## Removing her

```bash
sudo systemctl disable --now aria
sudo rm /etc/systemd/system/aria.service
sudo systemctl daemon-reload
```

Her memory stays in `~/.local/share/aria` until you delete it.

## Without the installer

`python -m aria.pi` (from the Aria-AI folder, with PyTorch installed) runs the
same thing in the foreground; Ctrl-C saves and stops her. `--port` and
`--allow-host` work as for the installer. It works on any always-on Linux
machine, not only a Pi.
