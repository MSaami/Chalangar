#!/bin/zsh
# One-time setup for Game Translator (macOS).
#   ./setup.sh        ask before installing the audio tools
#   ./setup.sh -y     install everything without asking
set -e
cd "$(dirname "$0")"

YES=0
[[ "$1" == "-y" || "$1" == "--yes" ]] && YES=1

bold() { print -P "%B$1%b"; }
ok()   { print -P "  %F{green}✓%f $1"; }
warn() { print -P "  %F{yellow}!%f $1"; }
fail() { print -P "  %F{red}✗%f $1"; exit 1; }
ask()  { (( YES )) && return 0; read -q "?  $1 [y/N] " && { echo; return 0; } || { echo; return 1; } }

[[ "$(uname)" == "Darwin" ]] || fail "Game Translator runs on macOS only."

# ---------- 1. Python ----------
bold "1/4 Python"
PYTHON=""
for cand in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v $cand >/dev/null 2>&1 &&
     $cand -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PYTHON=$(command -v $cand); break
  fi
done
if [[ -z "$PYTHON" ]]; then
  if command -v brew >/dev/null 2>&1 && ask "Python 3.10+ not found. Install python@3.12 with Homebrew?"; then
    brew install python@3.12
    PYTHON="$(brew --prefix)/bin/python3.12"
  else
    fail "Python 3.10 or newer is needed: https://www.python.org/downloads/macos/"
  fi
fi
ok "$PYTHON ($($PYTHON --version 2>&1))"

# ---------- 2. Audio tools ----------
bold "2/4 Audio tools"
if command -v brew >/dev/null 2>&1; then
  if [[ -d /Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver ]]; then
    ok "BlackHole 2ch"
  elif ask "Install BlackHole 2ch (virtual audio cable, needed to hear the game/meeting)?"; then
    brew install --cask blackhole-2ch && ok "BlackHole 2ch (if it does not show up in Sound settings, restart the Mac)"
  else
    warn "BlackHole skipped (game mode can still use a USB sound card / mic)"
  fi
  if command -v SwitchAudioSource >/dev/null 2>&1; then
    ok "SwitchAudioSource"
  elif ask "Install switchaudio-osx (lets the app change the Mac's sound output for you)?"; then
    brew install switchaudio-osx && ok "SwitchAudioSource"
  else
    warn "switchaudio-osx skipped (change Sound output in System Settings yourself)"
  fi
else
  warn "Homebrew not found (https://brew.sh). Install the audio tools by hand:"
  warn "  BlackHole 2ch: https://existential.audio/blackhole/"
  warn "  then later:    brew install switchaudio-osx  (optional)"
fi

# ---------- 3. Python packages ----------
bold "3/4 Python packages (first time: a few minutes)"
if [[ -x .venv/bin/python ]] && ! .venv/bin/python -c 'import sys' 2>/dev/null; then
  warn ".venv is broken (Python moved?) → making a new one"
  rm -rf .venv
fi
[[ -x .venv/bin/python ]] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
ok "packages installed in .venv"

# ---------- 4. App ----------
bold "4/4 App"
chmod +x run.sh "Game Translator.app/Contents/MacOS/GameTranslator"
# a downloaded zip marks the app as "from the internet"; this folder is yours, so clear it
xattr -dr com.apple.quarantine "Game Translator.app" 2>/dev/null || true
ok "Game Translator.app is ready"

echo
bold "Done! Start it with:"
echo "  open \"Game Translator.app\"     (or double-click it, or drag it to the Dock)"
echo "  ./run.sh                       (from the terminal, shows the log)"
echo
echo "Then click 🔑 Keys in the window and paste your OpenRouter or Anthropic API key."
echo "The first start downloads the speech model (~1.2 GB)."
