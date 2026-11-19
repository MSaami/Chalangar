#!/bin/zsh
# Start Game Translator from the terminal (the log stays in this window).
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Not set up yet. Run: ./setup.sh"
  exit 1
fi
exec .venv/bin/python game_translator.py "$@"
