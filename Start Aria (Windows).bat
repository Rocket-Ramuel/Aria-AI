@echo off
rem Double-click this file to start Aria from this folder.
rem
rem The first time, it sets Aria up: it needs the internet and takes a few
rem minutes. After that she starts in seconds, in a window of her own.

title Aria
cd /d "%~dp0"

set "PY="
for %%v in (3.12 3.13 3.11 3.10) do (
  if not defined PY py -%%v -c "" >nul 2>nul && set "PY=py -%%v"
)
if not defined PY (
  python -c "import sys; sys.exit(sys.version_info[:2] < (3, 10))" >nul 2>nul && set "PY=python"
)
if not defined PY goto :nopython

if not exist ".venv\Scripts\python.exe" (
  echo Setting Aria up for the first time. This downloads about 200 MB and takes
  echo a few minutes; next time she starts straight away.
  echo.
  %PY% -m venv .venv || goto :failed
)
fc /b requirements.txt .venv\aria-requirements.txt >nul 2>nul || (
  ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :failed
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt pypdf || goto :failed
  copy /y requirements.txt .venv\aria-requirements.txt >nul
)

start "" ".venv\Scripts\pythonw.exe" -m aria.app %*
exit /b 0

:nopython
echo Aria needs Python 3.10 or newer, and this computer doesn't have it yet.
echo Your browser is opening python.org: download the Windows installer, run it
echo (tick "Add python.exe to PATH"), then double-click this file again.
start "" "https://www.python.org/downloads/windows/"
pause
exit /b 1

:failed
echo.
echo Setting Aria up didn't work; the messages above say why.
pause
exit /b 1
