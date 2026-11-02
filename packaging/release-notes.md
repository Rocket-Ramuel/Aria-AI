Aria is a small AI that keeps learning as you talk to her. Open the app and a
small window says she's running; a moment later her chat opens in your web
browser. Close the window (or press Quit) and she saves what she has learned.

## Download

| your computer | download | then |
| --- | --- | --- |
| Mac with Apple Silicon (M1 or later) | **Aria-Mac.dmg** | open it and drag Aria into Applications |
| Windows 10 or 11 | **Aria-Windows-Setup.exe** | run it; Aria appears in the Start menu |
| Linux (64-bit PC) | **Aria-Linux.tar.gz** | extract it and open `Aria/Aria` |

Which Mac do you have? Apple menu → **About This Mac**. "Chip: Apple M…" means
the download works. "Processor: Intel" means the app can't run there; use the
*Start Aria (Mac).command* file in the source code instead (see the
[README](https://github.com/Rocket-Ramuel/Aria-AI/tree/claude/keen-einstein-9cy5rx#the-app)).

The `.zip` files hold the same Mac and Windows apps, for anyone who prefers a
zip to a disk image or an installer.

## The first time you open her

She isn't signed with a paid Apple or Microsoft certificate, so your computer
will be wary once:

- **Mac:** a message says Apple could not check Aria for malicious software.
  Click **Done**, open **System Settings → Privacy & Security**, scroll down to
  the line about Aria, click **Open Anyway**, and confirm.
- **Windows:** "Windows protected your PC". Click **More info**, then
  **Run anyway**.

## Where her memory is

What she learns is kept in your user folder, not in the app, so updating,
reinstalling or deleting the app never touches it:

| system | folder |
| --- | --- |
| macOS | `~/Library/Application Support/Aria` |
| Windows | `%APPDATA%\Aria` |
| Linux | `~/.local/share/aria` |

To move her to another computer, copy that folder across.

Each download was built on a real computer of its kind and checked there
before it was published: it opened, chatted, read a document, saved, quit and
reloaded what it learned (the Windows one after installing).
