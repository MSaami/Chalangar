# Game Translator

Live **English → Persian** subtitles for games, videos and meetings on macOS, built to help you
improve your English while you play or work.

The app listens to your Mac's audio, turns the speech into English text **on your own Mac**
(Parakeet / Whisper), and sends only that text to an AI model (through OpenRouter or the
Anthropic API) for a natural Persian translation, plus short notes on idioms and slang.

**Two modes**

| Mode | What it does |
|---|---|
| 🎮 **Game** | Translates what you hear in a game or video. Each line also shows useful phrases (📝) and is saved to `game_log_<date>.md` so you can review it later. |
| 💼 **Meeting** | Translates what **others** say, and gives feedback on **your own** English: ✅ when it was good, or ✏️ a more natural version with a 💡 short explanation in Persian. Nothing from meetings is saved to disk. |

```
Mac audio ──► BlackHole ──► speech detection ──► Parakeet / Whisper ──► AI model ──► app window
                              (local)              (local, GPU)        (OpenRouter    (English + فارسی)
                                                                       or Anthropic)
```

---

## Requirements

- **macOS 11+**. An Apple Silicon Mac (M1 or newer) is recommended: speech runs on the GPU.
  Intel Macs work too, using Whisper on the CPU (slower).
- **Python 3.10+** (the setup script can install it with Homebrew).
- **[Homebrew](https://brew.sh)** (recommended, used to install the audio tools).
- An API key from **[OpenRouter](https://openrouter.ai/keys)** *or*
  **[Anthropic](https://console.anthropic.com/settings/keys)**.

## Quick start

```bash
git clone https://github.com/<your-username>/game_translator.git
cd game_translator
./setup.sh
open "Game Translator.app"
```

`setup.sh` does everything once:

1. finds Python 3.10+ (or offers to install it),
2. offers to install **BlackHole 2ch** (virtual audio cable) and **switchaudio-osx**
   (lets the app change the Mac's sound output for you),
3. creates `.venv` and installs the Python packages,
4. gets `Game Translator.app` ready to open.

Use `./setup.sh -y` to say yes to everything. Then:

1. Open the app (double-click **Game Translator.app**, or drag it to the Dock).
2. Click **🔑 Keys**, paste your API key, and click **Save**.
3. Choose the **Provider** you have a key for.
4. Set up the audio (below) and play something.

> The first start downloads the speech model (~1.2 GB). After that it starts in a few seconds.
> macOS asks for microphone access the first time. Allow it, because BlackHole counts as a microphone.

## AI provider and API keys

Choose the provider in the window's **Provider** menu. You can change it at any time; no restart is needed.

| Provider | Models in the Model menu | Key |
|---|---|---|
| **OpenRouter** | GPT-6.1 Terra, GPT-6 Luna, Gemini 3.8 Flash, Claude Haiku, Claude Sonnet | [openrouter.ai/keys](https://openrouter.ai/keys) |
| **Anthropic** | Claude Haiku, Claude Sonnet | [console.anthropic.com](https://console.anthropic.com/settings/keys) |

Click **🔑 Keys** to add or replace a key. Next to each key, the window shows where it comes from:
**✓ saved …abcd** (pasted in the window), **✓ from OPENROUTER_API_KEY** (environment), or
**missing**.

Where keys are looked up, in this order:

1. the key pasted in the window (saved in `settings.json`),
2. the environment variable `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY`,
3. an `export OPENROUTER_API_KEY=...` line in `~/.zshrc`, `~/.zprofile` or `~/.bash_profile`
   (apps opened from Finder don't read these files, so the app reads them itself).

`settings.json` is in `.gitignore`, so your keys are never committed.

**Adding a model:** add one line to `MODEL_CHOICES` at the top of `game_translator.py`:
`key: (label, OpenRouter model id, Anthropic model id or None)`.

## Audio setup

The app needs to **hear** the sound. On a Mac, an app can't record another app's sound directly,
so **BlackHole** acts as a cable: the Mac plays sound *into* BlackHole and the app listens *from* it.

### 🎮 Game / video mode

1. In the window: **Mode → Game**, **Input → BlackHole 2ch**, **Sound out → BlackHole 2ch**.
   (Without switchaudio-osx: set *System Settings → Sound → Output* to BlackHole 2ch yourself.)
2. Now the sound goes into BlackHole, so you hear nothing yet. Click **🎧** to play it back
   to your headphones or speakers. Choose the device in **🎧 to** and set the volume with the slider.

*Alternative:* if your game sound comes from a console through a USB sound card / capture card,
just choose that device as **Input**; you don't need BlackHole.

### 💼 Meeting mode (Zoom, Teams, Meet, …)

Here you need to hear the meeting yourself **and** send it to BlackHole, so you need a
*Multi-Output Device* (one-time setup):

1. Open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup").
2. Click **+** (bottom left) → **Create Multi-Output Device**.
3. Tick **your headphones/speakers** and **BlackHole 2ch**. Put your headphones first and
   tick *Drift Correction* for BlackHole.
4. In the window: **Mode → Meeting**, **Your mic → your real microphone** (not BlackHole),
   **Sound out → Multi-Output Device** (the app picks it automatically the first time).
5. In the meeting app, keep the speaker set to the **system default** (or the Multi-Output Device).

The window labels each line **You** (from your mic) or **Others** (from BlackHole).

> Bluetooth headphones (e.g. AirPods, Sony WH-1000XM) switch to low "phone quality" when their
> mic is in use. For better sound, use the Mac's built-in mic as **Your mic**.

## The window

| Control | What it does |
|---|---|
| ● status | green = listening, orange = loading, red = a problem (the text says what) |
| 🎧 | play the input sound in your headphones (game mode) |
| 📌 | keep the window on top of the game |
| Clear | clear the list |
| Mode | Game or Meeting |
| Input / Your mic | where to listen |
| Sound out | the Mac's sound output for this mode (needs switchaudio-osx) |
| Provider, 🔑 Keys | AI provider and API keys |
| Model | AI model (cheaper ↔ better) |
| Pause | how long a silence ends a sentence (shorter = faster, longer = fuller sentences) |

All settings are saved in `settings.json` and remembered for next time.

## Command line

```bash
./run.sh                       # start with the log in the terminal
./run.sh --browser             # show the subtitles in your browser instead of a window
./run.sh --list-devices        # list audio devices
./run.sh --provider anthropic  # override the provider chosen in the window
./run.sh --engine faster       # speech engine: auto | parakeet | mlx | faster
./run.sh --help                # all options
```

When you open the `.app`, the log goes to `app.log` instead.

## Privacy

- Speech recognition runs **locally**. Audio never leaves your Mac.
- Only the recognized **text** (plus the last few lines as context) is sent to the AI provider you chose.
- Game mode saves lines to `game_log_<date>.md` for your own review. Meeting mode saves nothing,
  and meeting lines are not written to `app.log` either.
- `settings.json` (your API keys), `app.log` and `game_log_*.md` are in `.gitignore`.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Not set up yet" when opening the app | Run `./setup.sh` in the project folder. |
| "API key missing" | Click **🔑 Keys** and paste the key for the provider you selected. |
| "API error: 401" | The key is wrong or expired. Paste a new one. |
| No translations appear | Check that **Input** is BlackHole 2ch and **Sound out** is BlackHole (game) or the Multi-Output Device (meeting). |
| I hear nothing in game mode | Turn on **🎧** and pick your headphones in **🎧 to**. |
| I hear the sound twice | Sound out is the Multi-Output Device while 🎧 is on. In game mode use BlackHole 2ch. |
| BlackHole isn't in the device list | Restart the Mac after installing BlackHole. |
| "Input too loud (distorted)" | Lower the game / Mac volume a bit. |
| Slow or wrong words | Try a longer **Pause**, or `--engine mlx` (Whisper) instead of Parakeet. |
| More detail | Run `./run.sh` and read the log, or open `app.log`. |

## License

[MIT](LICENSE): free to use, change and share.
