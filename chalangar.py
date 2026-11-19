#!/usr/bin/env python3
"""
Chalangar — live English → Persian subtitles for games and videos (macOS).

Flow:  audio input (USB sound card or BlackHole) → speech detection (Silero VAD)
       → Parakeet or Whisper (local) → Claude API → small app window

Setup: ./setup.sh   (see README.md)

Usage:
    ./run.sh                                     # same as below, using the .venv
    python chalangar.py                    # open the app window and start
    python chalangar.py --list-devices     # show audio devices
    python chalangar.py --browser          # show in the browser instead

The input device is the Mac's current input (System Settings → Sound → Input).
AI provider: choose OpenRouter or Anthropic in the window (Provider menu) and paste the keys
under 🔑 Keys; they are saved in settings.json. A key can also come from OPENROUTER_API_KEY /
ANTHROPIC_API_KEY (environment or an export line in ~/.zshrc). --provider overrides the window.
"""
import argparse
import datetime
import json
import multiprocessing as mp
import math
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import anthropic
import numpy as np
import sounddevice as sd

SR = 16000                   # Whisper and VAD want 16 kHz mono
BLOCK_SEC = 0.03             # audio block size (small = less delay when listening)
VAD_EVERY_SEC = 0.2          # how often we look for the end of a sentence

# Common Whisper "hallucinations" on music / silence
# Choices in the window's Model menu, in this order: key -> (label, OpenRouter id, Anthropic id).
# Anthropic id None = only through OpenRouter. To add a model, add a line here.
MODEL_CHOICES = {
    "terra":  ("GPT-6.1 Terra · near Sonnet", "openai/gpt-6.1-terra", None),
    "luna":   ("GPT-6 Luna · cheapest", "openai/gpt-6-luna", None),
    "gemini": ("Gemini 3.8 Flash", "google/gemini-3.8-flash", None),
    "haiku":  ("Claude Haiku · cheap", "anthropic/claude-haiku-5.5", "claude-haiku-5-5"),
    "sonnet": ("Claude Sonnet · best", "anthropic/claude-sonnet-5.5", "claude-sonnet-5-5"),
}
DEFAULT_MODEL = "terra"
MODELS = {k: v[2] for k, v in MODEL_CHOICES.items() if v[2]}           # Anthropic direct
OPENROUTER_MODELS = {k: v[1] for k, v in MODEL_CHOICES.items()}
OPENROUTER_BASE_URL = "https://openrouter.ai/api"   # Anthropic-compatible Messages API
PROVIDERS = {"openrouter": {"label": "OpenRouter", "env": "OPENROUTER_API_KEY"},
             "anthropic": {"label": "Anthropic", "env": "ANTHROPIC_API_KEY"}}
DEFAULT_PAUSE = 1.5

# Words Whisper often "hears" in music or noise when nobody is talking
JUNK = {"", "you", "thank you", "thank you very much", "thanks", "thanks for watching",
        "thank you for watching", "bye", "bye bye", "goodbye", "good bye", "see you",
        "see you next time", "music", "applause", "laughter", "so", "oh", "ah", "um", "hmm",
        "please subscribe", "subtitles by the amara org community"}


def remove_loops(text):
    """Whisper sometimes gets stuck and repeats a phrase many times. Keep it once."""
    text = re.sub(r"\b(.{2,60}?)(?:[\s,.!?]+\1\b){3,}", r"\1", text, flags=re.I)
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        key = sent.strip().lower()
        if len(out) >= 2 and key == out[-1].strip().lower() == out[-2].strip().lower():
            continue          # a sentence 3+ times in a row is a loop: keep 2
        out.append(sent)
    return " ".join(out).strip()


def normalize(text):
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()

SYSTEM_PROMPT = """You translate English video game dialogue into natural, simple, spoken Persian (Farsi) for an English learner (B2 level).

Output format (nothing else):
- Line 1: the Persian translation.
- Only if the English was clearly misheard by speech recognition, add a line:
  ✏️ <the corrected English line>
- Then, only if the line has an idiom, slang, phrasal verb or useful expression, up to 3 lines like:
  📝 <English phrase> = <short Persian meaning>

Rules:
- Output only the final answer. Never explain, comment, or correct yourself.
- Keep names, places and code names as they are (e.g. Bond, MI6, 009).
- The English comes from automatic speech recognition. It may have misheard words or start/end mid-sentence. Translate what was most likely said.
- Use the previous lines only as context. Translate only the new line.
- If the new line is clearly noise or not real speech, reply exactly: SKIP"""

MEETING_PROMPT = SYSTEM_PROMPT.replace(
    "English video game dialogue", "English spoken in a work meeting (a software team)")

SELF_PROMPT = """You are an English coach. The new line is what the learner (a B2-level Persian speaker, backend developer) just said in a work meeting. The text comes from speech recognition.

Output format (nothing else):
- If the line is correct and natural, reply exactly: OK
- Otherwise:
  ✏️ <a corrected, natural version, as a native speaker would say it in a meeting>
  💡 <one short line in simple Persian: what was wrong or unnatural>
  (at most 2 💡 lines)

Rules:
- Focus on grammar, word choice and natural phrasing. Keep the learner's meaning and tone.
- Ignore small things that look like speech-recognition errors, missing punctuation or filler words (um, like).
- Keep technical terms and names as they are.
- Use the previous lines only as context.
- If the line is clearly noise or not real speech, reply exactly: SKIP"""

VIEWER_HTML = """<!doctype html>
<html lang="fa"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chalangar</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600&display=swap">
<style>
  :root { --bg:#111418; --card:#1b2027; --en:#7dd3fc; --fa:#f3f4f6; --note:#fcd34d;
          --muted:#8b95a1; --ok:#22c55e; --wait:#f59e0b; --err:#ef4444; }
  * { box-sizing: border-box; }
  html, body { margin:0; background:var(--bg); color:var(--fa); }
  body { font-family: Vazirmatn, Tahoma, sans-serif; }
  .top { position:sticky; top:0; z-index:1; }
  header { background:var(--bg);
           display:flex; gap:10px; align-items:center; padding:10px 14px;
           border-bottom:1px solid #222831; font-size:13px; color:var(--muted);
           font-family:-apple-system, Helvetica, sans-serif; }
  #dot { width:9px; height:9px; border-radius:50%; background:var(--wait); flex:none; }
  #status { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #status.err { color:var(--err); }
  #settings { background:var(--bg); display:flex; gap:14px; align-items:center; padding:8px 14px;
              border-bottom:1px solid #222831; font-size:13px; color:var(--muted);
              font-family:-apple-system, Helvetica, sans-serif; }
  #settings { flex-wrap:wrap; }
  #settings label { display:inline-flex; gap:6px; align-items:center; white-space:nowrap; }
  .row2 { display:flex; flex-wrap:wrap; gap:14px; align-items:center; padding:6px 14px 8px;
          border-bottom:1px solid #222831; font-size:13px; color:var(--muted); background:var(--bg);
          font-family:-apple-system, Helvetica, sans-serif; }
  .row2 label { display:inline-flex; gap:6px; align-items:center; white-space:nowrap; }
  .row2.off { opacity:0.45; }
  .row2.hidden { display:none; }
  #keysRow input[type=password] { background:#222831; color:var(--fa); border:0; border-radius:8px;
           padding:4px 8px; font-size:13px; width:190px; }
  .keyst { font-size:12px; }
  .keyst.saved, .keyst.env { color:var(--ok); }
  .keyst.missing { color:var(--err); }
  #keysMsg { font-size:12px; }
  input[type=range] { width:150px; accent-color:#6366f1; }
  select { background:#222831; color:var(--fa); border:0; border-radius:8px;
           padding:4px 8px; font-size:13px; }
  button { background:#222831; color:var(--fa); border:0; border-radius:8px;
           padding:5px 9px; font-size:13px; cursor:pointer; }
  button.on { background:#334155; }
  main { padding:12px 14px; }
  .item { background:var(--card); border-radius:12px; padding:12px 16px; margin-bottom:10px; }
  .item:first-child { outline:1px solid #334155; }
  .en { direction:ltr; text-align:left; color:var(--en); font-size:15px;
        font-family:-apple-system, Helvetica, sans-serif; }
  .who { display:inline-block; font-size:11px; padding:1px 7px; border-radius:6px;
         margin-bottom:4px; font-family:-apple-system, Helvetica, sans-serif; }
  .who.You { background:#14532d; color:#bbf7d0; }
  .who.Others { background:#1e3a8a; color:#bfdbfe; }
  .item.mine { border-left:3px solid #22c55e; }
  .ok { direction:ltr; color:#86efac; font-size:14px; margin-top:4px;
        font-family:-apple-system, Helvetica, sans-serif; }
  .fix { direction:ltr; text-align:left; color:#a7f3d0; font-size:14px; margin-top:2px;
         font-family:-apple-system, Helvetica, sans-serif; }
  .fa { direction:rtl; text-align:right; font-size:21px; line-height:1.8; margin-top:4px; }
  .fa.pending { color:var(--muted); font-size:15px; }
  .note { direction:rtl; text-align:right; color:var(--note); font-size:15px; margin-top:2px; }
  .t { direction:ltr; color:#4b5563; font-size:11px; margin-top:4px;
       font-family:-apple-system, Helvetica, sans-serif; }
  #empty { color:var(--muted); text-align:center; margin-top:50px; direction:rtl; }
</style></head>
<body>
<div class="top">
<header>
  <span id="dot"></span><span id="status">Starting…</span>
  <button id="mon" title="Play the sound in your headphones">🎧</button>
  <button id="pin" title="Keep window on top">📌</button>
  <button id="clear" title="Clear the list">Clear</button>
</header>
<div id="settings">
  <label>Mode <select id="mode">
    <option value="game">Game</option>
    <option value="meeting">Meeting</option>
  </select></label>
  <label><span id="inLabel">Input</span> <select id="device"></select></label>
  <label>Sound out <select id="output"></select></label>
  <label>Provider <select id="provider"></select></label>
  <button id="keysBtn" title="API keys">🔑 Keys</button>
  <label>Model <select id="model">
  </select></label>
  <label>Pause <select id="pause"></select></label>
</div>
<div id="keysRow" class="row2 hidden">
  <label>OpenRouter <input id="key_openrouter" type="password" autocomplete="off" spellcheck="false">
    <span id="st_openrouter" class="keyst"></span></label>
  <label>Anthropic <input id="key_anthropic" type="password" autocomplete="off" spellcheck="false">
    <span id="st_anthropic" class="keyst"></span></label>
  <button id="keysSave">Save</button>
  <span id="keysMsg"></span>
</div>
<div id="listenRow" class="row2">
  <label>🎧 to <select id="listen"></select></label>
  <label class="vol">🔈 <input id="volume" type="range" min="0" max="100" step="1"> <span id="volVal"></span></label>
</div>
</div>
<main>
  <div id="empty">صحبتی پخش کنید تا ترجمه اینجا ظاهر شود</div>
  <div id="list"></div>
</main>
<script>
let rev = -1, hideBefore = 0;
const list = document.getElementById('list');
function el(cls, text) { const d = document.createElement('div'); d.className = cls; d.textContent = text; return d; }
function render(entries) {
  const shown = entries.filter(e => e.id >= hideBefore).slice(-80).reverse();
  document.getElementById('empty').style.display = shown.length ? 'none' : '';
  list.replaceChildren(...shown.map(e => {
    const mine = e.who === 'You';
    const d = el('item' + (mine ? ' mine' : ''), '');
    if (e.who) d.append(el('who ' + e.who, e.who));
    d.append(el('en', e.en));
    if (e.fix) d.append(el('fix', '✏️ ' + e.fix));
    if (mine) { if (e.fa === '✅') d.append(el('ok', '✅ Good English')); else if (!e.fix && !e.fa) d.append(el('fa pending', '…')); }
    else if (e.fa) d.append(el('fa', e.fa)); else d.append(el('fa pending', '…'));
    for (const n of e.notes) d.append(el('note', n));
    if (e.t) d.append(el('t', e.t));
    return d;
  }));
}
async function poll() {
  try {
    const r = await fetch('/state?rev=' + rev);
    const data = await r.json();
    if (data.entries) { rev = data.rev; render(data.entries); window._last = data.entries; }
    if (data.volume !== undefined && document.activeElement !== volSl
        && Number(volSl.value) !== data.volume) {      // the Mac's volume keys moved it
      volSl.value = data.volume; document.getElementById('volVal').textContent = data.volume;
    }
    const st = document.getElementById('status');
    st.textContent = data.status; st.title = data.status;
    st.className = data.state === 'error' ? 'err' : '';
    document.getElementById('dot').style.background =
      data.state === 'ok' ? 'var(--ok)' : data.state === 'error' ? 'var(--err)' : 'var(--wait)';
  } catch (e) {
    document.getElementById('dot').style.background = 'var(--err)';
  }
  setTimeout(poll, 250);
}
document.getElementById('clear').onclick = () => {
  const last = window._last || []; hideBefore = last.length ? last[last.length - 1].id + 1 : 0;
  render(last);
};
const pin = document.getElementById('pin');
pin.onclick = async () => {
  if (!window.pywebview) return;
  pin.className = (await window.pywebview.api.toggle_on_top()) ? 'on' : '';
};
const mon = document.getElementById('mon');
mon.onclick = async () => {
  if (!window.pywebview) return;
  mon.className = (await window.pywebview.api.toggle_monitor()) ? 'on' : '';
  loadSettings();
};
window.addEventListener('pywebviewready', async () => {
  pin.className = (await window.pywebview.api.is_on_top()) ? 'on' : '';
  mon.className = (await window.pywebview.api.is_monitor()) ? 'on' : '';
});
const modeSel = document.getElementById('mode');
const modelSel = document.getElementById('model');
const pauseSel = document.getElementById('pause');
for (let i = 1; i <= 20; i++) {
  const v = (i / 10).toFixed(1);
  const o = document.createElement('option'); o.value = v; o.textContent = v + ' s';
  pauseSel.append(o);
}
const devSel = document.getElementById('device');
const outSel = document.getElementById('output');
const listenSel = document.getElementById('listen');
const volSl = document.getElementById('volume');
listenSel.onchange = async () => {
  showSettings(await (await fetch('/listen?listen_on=' + encodeURIComponent(listenSel.value))).json());
};
volSl.oninput = () => {
  document.getElementById('volVal').textContent = volSl.value;
  fetch('/listen?volume=' + volSl.value);
};
const provSel = document.getElementById('provider');
const keysRow = document.getElementById('keysRow');
const PROVS = ['openrouter', 'anthropic'];
let keysOpened = false;
function showKeys(s) {
  if (!s.keys) return;
  for (const p of PROVS) {
    const k = s.keys[p], st = document.getElementById('st_' + p);
    const inp = document.getElementById('key_' + p);
    st.className = 'keyst ' + k.source;
    st.textContent = k.source === 'saved' ? '✓ saved ' + k.hint
                   : k.source === 'env' ? '✓ from ' + k.hint : 'missing';
    inp.placeholder = k.source === 'saved' ? 'paste to replace' : 'paste key';
  }
  // open the keys row by itself the first time the chosen provider has no key
  if (!keysOpened && s.keys[s.provider].source === 'missing') {
    keysOpened = true; keysRow.classList.remove('hidden');
  }
}
async function postProvider(body) {
  const r = await fetch('/provider', {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  if (!r.ok) throw new Error('HTTP ' + r.status);
  modelSel.innerHTML = '';            // the model list depends on the provider
  showSettings(await r.json());
}
provSel.onchange = () => postProvider({provider: provSel.value});
document.getElementById('keysBtn').onclick = () => {
  keysOpened = true; keysRow.classList.toggle('hidden');
};
document.getElementById('keysSave').onclick = async () => {
  const keys = {};
  for (const p of PROVS) {
    const v = document.getElementById('key_' + p).value.trim();
    if (v) keys[p] = v;
  }
  const msg = document.getElementById('keysMsg');
  if (!Object.keys(keys).length) { msg.textContent = 'Paste a key first'; return; }
  try {
    await postProvider({keys});
    for (const p of PROVS) document.getElementById('key_' + p).value = '';
    msg.textContent = 'Saved';
  } catch (e) { msg.textContent = 'Could not save: ' + e.message; }
  setTimeout(() => { msg.textContent = ''; }, 2500);
};
function showSettings(s) {
  if (s.providers && provSel.options.length !== s.providers.length)
    provSel.innerHTML = s.providers.map(p => `<option value="${p.key}">${p.label}</option>`).join('');
  if (s.provider) provSel.value = s.provider;
  showKeys(s);
  if (s.models && modelSel.options.length !== s.models.length)
    modelSel.innerHTML = s.models.map(m => `<option value="${m.key}">${m.label}</option>`).join('');
  modeSel.value = s.mode; modelSel.value = s.model; pauseSel.value = Number(s.pause).toFixed(1);
  document.getElementById('inLabel').textContent = s.mode === 'meeting' ? 'Your mic' : 'Input';
  const opts = [['', 'Mac default']].concat(s.devices
    .filter(n => !(s.mode === 'meeting' && n.toLowerCase().includes('blackhole')))
    .map(n => [n, n]));
  if (s.device && !s.devices.includes(s.device)) opts.push([s.device, s.device + ' (not found)']);
  devSel.replaceChildren(...opts.map(([v, t]) => {
    const o = document.createElement('option'); o.value = v; o.textContent = t; return o; }));
  devSel.value = s.device;
  const outOpts = [['', "Don't change"]].concat(s.outputs.map(n => [n, n]));
  if (s.output && !s.outputs.includes(s.output)) outOpts.push([s.output, s.output + ' (not found)']);
  outSel.replaceChildren(...outOpts.map(([v, t]) => {
    const o = document.createElement('option'); o.value = v; o.textContent = t; return o; }));
  outSel.value = s.output;
  const row = document.getElementById('listenRow');
  row.className = 'row2' + (s.mode === 'meeting' ? ' hidden' : (s.monitor ? '' : ' off'));
  const lOpts = [['', 'Auto (headphones/speakers)']].concat(s.outputs
    .filter(n => !/blackhole|multi-output/i.test(n)).map(n => [n, n]));
  listenSel.replaceChildren(...lOpts.map(([v, t]) => {
    const o = document.createElement('option'); o.value = v; o.textContent = t; return o; }));
  listenSel.value = s.listen_on;
  if (document.activeElement !== volSl) volSl.value = s.volume;
  document.getElementById('volVal').textContent = volSl.value;
  outSel.title = s.can_switch ? 'Where the Mac plays sound in this mode'
                              : 'Needs: brew install switchaudio-osx';
}
async function loadSettings() {
  try { showSettings(await (await fetch('/settings')).json()); }
  catch (e) { setTimeout(loadSettings, 500); }
}
async function saveSettings() {
  const r = await fetch('/set?mode=' + modeSel.value + '&model=' + modelSel.value +
                        '&pause=' + pauseSel.value);
  showSettings(await r.json());
}
modeSel.onchange = saveSettings;
outSel.onchange = async () => {
  showSettings(await (await fetch('/set?output=' + encodeURIComponent(outSel.value))).json());
};
devSel.onchange = async () => {
  showSettings(await (await fetch('/set?device=' + encodeURIComponent(devSel.value))).json());
};
modelSel.onchange = saveSettings;
pauseSel.onchange = saveSettings;
loadSettings();
poll();
</script></body></html>"""


def load_api_key(provider="openrouter", saved_key=""):
    """Key for the provider: the one saved from the window (settings.json), the environment,
    or ~/.zshrc (apps opened from Finder do not read ~/.zshrc)."""
    env = PROVIDERS[provider]["env"]
    key = saved_key or os.environ.get(env)
    if key:
        return key
    for name in (".zshrc", ".zprofile", ".bash_profile"):
        path = Path.home() / name
        if path.exists():
            m = re.search(env + r"=[\"']?([^\"'\s]+)", path.read_text(errors="ignore"))
            if m:
                return m.group(1)
    return None


def find_input_device(name):
    """By name (part of it), or the Mac's current input device when name is empty."""
    if not name:
        idx = sd.default.device[0]
        if idx is None or idx < 0:
            return None, None
        d = sd.query_devices(idx)
        return (idx, d) if d["max_input_channels"] > 0 else (None, None)
    devices = list(enumerate(sd.query_devices()))
    for i, d in devices:                       # exact name first
        if d["name"] == name and d["max_input_channels"] > 0:
            return i, d
    for i, d in devices:
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return i, d
    return None, None


def input_device_names():
    return [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]


def output_device_names():
    return [d["name"] for d in sd.query_devices() if d["max_output_channels"] > 0]


def switch_audio_tool():
    return (shutil.which("SwitchAudioSource")
            or next((p for p in ("/opt/homebrew/bin/SwitchAudioSource",
                                 "/usr/local/bin/SwitchAudioSource") if Path(p).exists()), None))


def set_mac_output(name):
    """Set the Mac's sound output (like System Settings → Sound → Output).
    Returns an error text, or None when it worked."""
    tool = switch_audio_tool()
    if tool is None:
        return "To change Sound out from the app: brew install switchaudio-osx"
    r = subprocess.run([tool, "-t", "output", "-s", name], capture_output=True, text=True)
    if r.returncode != 0:
        return f"Could not switch output to '{name}': {(r.stderr or r.stdout).strip()}"
    # system sounds (alerts) follow too, so nothing plays only into BlackHole by mistake
    subprocess.run([tool, "-t", "system", "-s", name], capture_output=True, text=True)
    return None


class MacVolume:
    """Read and set a device's output volume, like the macOS volume slider (CoreAudio)."""
    BH_RANGE_DB = 64.0     # BlackHole's volume goes linearly in dB from -64 dB (0) to 0 dB (1)

    def __init__(self):
        import ctypes, ctypes.util, struct
        self._c = c = ctypes
        self._ca = c.cdll.LoadLibrary(ctypes.util.find_library("CoreAudio"))
        self._cf = c.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        self._cf.CFStringGetCString.argtypes = [c.c_void_p, c.c_char_p, c.c_long, c.c_uint32]
        self._cf.CFRelease.argtypes = [c.c_void_p]
        four = lambda t: struct.unpack(">I", t.encode())[0]
        self.GLOB, self.OUT = four("glob"), four("outp")
        self.VOL, self.MUTE, self.DEVS, self.NAME, self.STREAMS = (
            four(t) for t in ("volm", "mute", "dev#", "lnam", "stm#"))

        class Addr(c.Structure):
            _fields_ = [("sel", c.c_uint32), ("scope", c.c_uint32), ("elem", c.c_uint32)]
        self._Addr = Addr

    def _get(self, obj, sel, scope, elem, ctype):
        c = self._c
        a, size = self._Addr(sel, scope, elem), c.c_uint32(0)
        if self._ca.AudioObjectGetPropertyDataSize(obj, c.byref(a), 0, None, c.byref(size)):
            return None
        if size.value == 0:
            return []
        buf = (ctype * (size.value // c.sizeof(ctype)))()
        if self._ca.AudioObjectGetPropertyData(obj, c.byref(a), 0, None, c.byref(size), buf):
            return None
        return list(buf)

    def _set(self, obj, sel, scope, elem, value):
        c = self._c
        a = self._Addr(sel, scope, elem)
        return self._ca.AudioObjectSetPropertyData(obj, c.byref(a), 0, None,
                                                   c.sizeof(value), c.byref(value)) == 0

    def device(self, name):
        """CoreAudio id of the OUTPUT device with this name (as sounddevice shows it), or None.
        Bluetooth headsets show up twice with the same name: a mic and an output."""
        c = self._c
        for d in self._get(1, self.DEVS, self.GLOB, 0, c.c_uint32) or []:
            if not self._get(d, self.STREAMS, self.OUT, 0, c.c_uint32):
                continue                      # input only
            ref = self._get(d, self.NAME, self.GLOB, 0, c.c_void_p)
            if not ref or not ref[0]:
                continue
            buf = c.create_string_buffer(256)
            self._cf.CFStringGetCString(ref[0], buf, 256, 0x08000100)   # UTF-8
            self._cf.CFRelease(ref[0])
            if buf.value.decode("utf-8", "ignore") == name:
                return d
        return None

    def _elements(self, dev):
        # one master volume, or one per channel (e.g. Bluetooth headphones)
        if self._get(dev, self.VOL, self.OUT, 0, self._c.c_float) is not None:
            return [0]
        return [e for e in (1, 2) if self._get(dev, self.VOL, self.OUT, e, self._c.c_float)]

    def get(self, dev):
        """Volume 0-1, or None when the device has no volume control."""
        vals = [self._get(dev, self.VOL, self.OUT, e, self._c.c_float)[0]
                for e in self._elements(dev)]
        return sum(vals) / len(vals) if vals else None

    def set(self, dev, v):
        v = self._c.c_float(max(0.0, min(1.0, v)))
        return any([self._set(dev, self.VOL, self.OUT, e, v) for e in self._elements(dev)])

    def muted(self, dev):
        m = self._get(dev, self.MUTE, self.OUT, 0, self._c.c_uint32)
        return bool(m and m[0])

    def set_mute(self, dev, on):
        return self._set(dev, self.MUTE, self.OUT, 0, self._c.c_uint32(1 if on else 0))


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.astype(np.float32)
    n_out = int(round(len(audio) * dst_rate / src_rate))
    x_old = np.arange(len(audio)) / src_rate
    x_new = np.arange(n_out) / dst_rate
    return np.interp(x_new, x_old, audio).astype(np.float32)


class Recognizer:
    """Speech → English text. Uses the Mac's GPU (mlx-whisper) when it can,
    otherwise faster-whisper on the CPU. Must be loaded and used in ONE thread."""

    def __init__(self, args):
        self.args = args
        self.engine = None
        self.label = ""

    def load(self, set_status):
        want = self.args.engine
        apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        if want in ("auto", "parakeet") and apple_silicon:
            try:
                import mlx.core as mx
                from parakeet_mlx import Beam, DecodingConfig, from_pretrained
                from parakeet_mlx.audio import get_logmel
                repo = self.args.parakeet_model
                name = repo.split("/")[-1]
                set_status(f"Loading {name} on GPU (first time: ~1.2 GB download)…")
                self._mx, self._get_logmel = mx, get_logmel
                self._pk = from_pretrained(repo)
                self._pk_cfg = DecodingConfig(decoding=Beam(
                    beam_size=5, length_penalty=0.013, patience=3.5, duration_reward=0.67))
                self.engine, self.label = "parakeet", f"{name} · GPU"
                self.transcribe(np.zeros(SR, dtype=np.float32), None)   # warm up
                return
            except Exception as e:
                print(f"[parakeet not used] {e}", flush=True)
                if want == "parakeet":
                    raise
        if want in ("auto", "mlx") and apple_silicon:
            try:
                import mlx_whisper
                self._mlx = mlx_whisper
                repo = self.args.mlx_model
                set_status(f"Loading {repo.split('/')[-1]} on GPU (first time: ~1.6 GB download)…")
                self._mlx.transcribe(np.zeros(SR, dtype=np.float32),
                                     path_or_hf_repo=repo, language="en")
                self.engine, self.label = "mlx", f"{repo.split('/')[-1]} · GPU"
                return
            except Exception as e:
                print(f"[mlx not used] {e}", flush=True)
                if want == "mlx":
                    raise
        from faster_whisper import WhisperModel
        set_status(f"Loading Whisper '{self.args.whisper}' (first time: downloading)…")
        self._fw = WhisperModel(self.args.whisper, device="cpu", compute_type="int8")
        self._fw.transcribe(np.zeros(SR, dtype=np.float32), language="en")
        self.engine, self.label = "faster", f"{self.args.whisper} · CPU"

    @staticmethod
    def _keep(text, no_speech_prob, avg_logprob):
        words = normalize(text)
        if words in JUNK:
            return False
        if no_speech_prob > 0.6 and avg_logprob < -1.0:     # Whisper's own silence rule
            return False
        # very short results are often invented: keep them only when Whisper is sure
        if len(words.split()) <= 2 and (no_speech_prob > 0.3 or avg_logprob < -0.7):
            return False
        return True

    def transcribe(self, audio, prompt):
        if self.engine == "parakeet":
            # parakeet's mel code only works with float32 input; the model itself is bfloat16
            mel = self._get_logmel(self._mx.array(audio, dtype=self._mx.float32),
                                   self._pk.preprocessor_config).astype(self._mx.bfloat16)
            result = self._pk.generate(mel, decoding_config=self._pk_cfg)[0]
            parts = []
            for sent in result.sentences:
                text = sent.text.strip()
                words = normalize(text)
                if words in JUNK:
                    continue
                if len(words.split()) <= 2 and sent.confidence < 0.5:
                    continue       # very short and unsure: probably noise
                parts.append(text)
            print(f"[parakeet] conf={[round(x.confidence, 2) for x in result.sentences]}",
                  flush=True)
            return remove_loops(" ".join(parts).strip())
        if self.engine == "mlx":
            r = self._mlx.transcribe(
                audio, path_or_hf_repo=self.args.mlx_model, language="en",
                initial_prompt=prompt, condition_on_previous_text=False,
                temperature=(0.0, 0.2, 0.4, 0.6), verbose=None)
            parts = [s["text"].strip() for s in r.get("segments", [])
                     if s.get("compression_ratio", 0) <= 2.4
                     and self._keep(s["text"], s.get("no_speech_prob", 0), s.get("avg_logprob", 0))]
        else:
            segments, _ = self._fw.transcribe(
                audio, language="en", beam_size=5, vad_filter=False,
                condition_on_previous_text=False, initial_prompt=prompt,
                without_timestamps=True)
            parts = [s.text.strip() for s in segments if s.compression_ratio <= 2.4
                     if self._keep(s.text, s.no_speech_prob, s.avg_logprob)]
        return remove_loops(" ".join(p for p in parts if p).strip())


class MonitorBuffer:
    """Plays input audio on another device without clicks.

    The input and output devices run on different clocks, so one is always a tiny
    bit faster. Instead of dropping audio or inserting silence (which clicks), we
    resample continuously and nudge the speed by a fraction of a percent to keep
    a steady amount of audio buffered."""

    TARGET_SEC = 0.05        # audio kept in the buffer (adds this much delay)
    # more than this buffered (after a hiccup or at the start): jump back to TARGET at once
    # instead of slowly catching up, which used to leave the sound up to 0.6 s late
    MAX_SEC = 0.15

    def __init__(self, src_rate, out_rate):
        self.src_rate, self.out_rate = src_rate, out_rate
        self.fifo = np.zeros((0, 2), dtype=np.float32)
        self.pos = 0.0           # fractional read position inside fifo
        self.adj = 0.0           # speed correction (e.g. 0.001 = 0.1 % faster)
        self.primed = False
        self.target = int(src_rate * self.TARGET_SEC)
        self.lock = threading.Lock()
        self.last = np.zeros(2, dtype=np.float32)
        self.underruns = 0

    def clear(self):
        with self.lock:
            self.fifo = np.zeros((0, 2), dtype=np.float32)
            self.pos, self.adj, self.primed = 0.0, 0.0, False

    def push(self, stereo):
        with self.lock:
            self.fifo = np.concatenate((self.fifo, stereo))
            if len(self.fifo) > int(self.src_rate * self.MAX_SEC):
                self.fifo = self.fifo[-self.target:]
                self.pos = 0.0

    def pull(self, frames):
        """Return `frames` stereo samples at the output rate."""
        out = np.zeros((frames, 2), dtype=np.float32)
        with self.lock:
            fifo = self.fifo
            if not self.primed:
                if len(fifo) < self.target:
                    return out
                self.primed = True
                fifo = fifo[-self.target:]            # start with exactly TARGET of delay
                self.pos = 0.0
            # gentle speed correction towards the target fill level
            err = (len(fifo) - self.target) / self.target
            want = max(-0.004, min(0.004, err * 0.004))
            self.adj += (want - self.adj) * 0.05
            step = self.src_rate / self.out_rate * (1.0 + self.adj)
            idx = self.pos + step * np.arange(frames)
            i0 = idx.astype(np.int64)
            ok = i0 + 1 < len(fifo)
            n = int(ok.sum()) if not ok.all() else frames
            if n:
                frac = (idx[:n] - i0[:n])[:, None]
                out[:n] = fifo[i0[:n]] * (1.0 - frac) + fifo[i0[:n] + 1] * frac
                self.last = out[n - 1].copy()
            if n < frames:
                # ran dry: fade out instead of a hard cut, then wait to refill
                fade = np.linspace(1.0, 0.0, min(64, frames - n), dtype=np.float32)[:, None]
                out[n:n + len(fade)] = self.last * fade
                self.primed = False
                self.underruns += 1
                self.fifo = fifo[-1:] if len(fifo) else fifo
                self.pos = 0.0
                return out
            new_pos = self.pos + step * frames
            drop = int(new_pos)
            self.fifo = fifo[drop:]
            self.pos = new_pos - drop
        return out



def monitor_process(in_name, out_name, enabled, gain, stop, stats, parent_pid):
    """Runs in its OWN process: plays the input device on the headphones.
    In a separate process the speech model can never make the sound stutter
    (Python runs only one thread at a time inside one process).
    It keeps running in game mode; 🎧 only flips `enabled`, so on/off is instant."""
    def find(name, kind):
        for i, d in enumerate(sd.query_devices()):
            if d["name"] == name and d[kind] > 0:
                return i, d
        return None, None
    in_idx, din = find(in_name, "max_input_channels")
    out_idx, dout = find(out_name, "max_output_channels")
    if in_idx is None or out_idx is None:
        print(f"[monitor] device not found: {in_name} / {out_name}", flush=True)
        return
    buf = MonitorBuffer(int(din["default_samplerate"]), int(dout["default_samplerate"]))

    was_on = [False]

    def on_input(indata, frames, t, status):
        if not enabled.value:
            return
        left = indata[:, 0]
        right = indata[:, 1] if indata.shape[1] > 1 else left
        buf.push(np.stack([left, right], axis=1).astype(np.float32))

    def on_output(outdata, frames, t, status):
        if not enabled.value:
            if was_on[0]:
                buf.clear()             # next time 🎧 starts with fresh sound, not old audio
                was_on[0] = False
            outdata.fill(0)
            return
        was_on[0] = True
        if status.output_underflow:
            stats[1] += 1
        chunk = buf.pull(frames) * gain.value
        if outdata.shape[1] == 1:
            outdata[:, 0] = chunk.mean(axis=1)
        else:
            outdata[:, :2] = chunk

    with sd.InputStream(device=in_idx, channels=min(2, din["max_input_channels"]),
                        samplerate=int(din["default_samplerate"]), dtype="float32",
                        latency="low", callback=on_input), \
         sd.OutputStream(device=out_idx, channels=min(2, dout["max_output_channels"]),
                         samplerate=int(dout["default_samplerate"]), dtype="float32",
                         latency="low", callback=on_output):
        while not stop.wait(0.1):
            stats[0] = buf.underruns
            if os.getppid() != parent_pid:      # the app was closed: stop playing
                break


class Source:
    """One audio input we listen to (game sound, or in a meeting: you / others)."""

    def __init__(self, speaker, dev_idx, dev):
        self.speaker = speaker            # "game", "you" or "others"
        self.dev_idx = dev_idx
        self.name = dev["name"]
        self.src_rate = int(dev["default_samplerate"])
        self.channels = min(2, dev["max_input_channels"])
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stream = None
        self.alive = True


class Translator:
    def __init__(self, args):
        self.args = args
        self.chunk_q: "queue.Queue[tuple]" = queue.Queue()
        self.context = deque(maxlen=20)   # previous lines sent to Claude as context
        self.entries = []                 # {"id","who","en","fix","fa","notes","t"}
        self.rev = 0
        self.lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.status, self.state = "Starting…", "wait"
        self.listening_status = ""
        self.sources = []
        self.recognizer = None
        self.client = None
        self.thinking_required = set()   # models that refuse thinking={"type": "disabled"}
        self.pool = ThreadPoolExecutor(max_workers=3)
        here = Path(__file__).resolve().parent
        self.log_path = here / "logs" / f"game_log_{datetime.date.today()}.md"
        self.settings_path = here / "settings.json"

        saved = self.load_settings()
        self.monitor = saved.get("monitor", False)
        self.mode = saved.get("mode", "game") if saved.get("mode") in ("game", "meeting") else "game"
        # the input chosen in the window, per mode ("" = the Mac's input in System Settings)
        self.inputs = {"game": "", "meeting": ""}
        self.inputs.update({k: v for k, v in saved.get("inputs", {}).items() if k in self.inputs})
        # the Mac's sound output per mode ("" = leave it as it is)
        self.outputs = {"game": "", "meeting": ""}
        self.outputs.update({k: v for k, v in saved.get("outputs", {}).items() if k in self.outputs})
        self.listen_on = saved.get("listen_on", "")       # 🎧 device ("" = first non-BlackHole)
        self.volume = int(saved.get("volume", 70))         # 🎧 volume 0-100
        self.output_error = None
        self.provider = saved.get("provider") if saved.get("provider") in PROVIDERS else "openrouter"
        if args.provider:                   # command line wins
            self.provider = args.provider
        # keys pasted in the window; empty = use the environment / ~/.zshrc
        self.keys = {p: saved.get(f"{p}_key", "") for p in PROVIDERS}
        self.client_error = None
        self.model_key = saved.get("model", DEFAULT_MODEL)
        if self.model_key not in self.models:
            self.model_key = DEFAULT_MODEL if DEFAULT_MODEL in self.models else "haiku"
        self.pause = float(saved.get("pause", DEFAULT_PAUSE))
        if args.pause is not None:          # command line wins
            self.pause = args.pause
        if args.model:
            self.model_key = next((k for k, v in {**MODELS, **OPENROUTER_MODELS}.items()
                                   if v == args.model), "custom")
            self.custom_model = args.model

        # Headphone playback ("monitor"), game mode only
        self.out_rate = None
        self.out_name = ""
        self.mon_proc = None               # the 🎧 playback process (runs in game mode)
        self.mon_enabled = self.mon_gain = None
        # 🎧 volume = the headphones' real macOS volume, and the volume keys work.
        # In game mode the Mac plays into BlackHole, so the keys change BlackHole's volume;
        # we copy that to the headphones and undo BlackHole's quieter sound for the
        # speech recognizer and 🎧 (bh_gain), so translation never gets quieter.
        try:
            self.macvol = MacVolume() if platform.system() == "Darwin" else None
        except Exception as e:
            print(f"[volume] CoreAudio not available: {e}", flush=True)
            self.macvol = None
        self.bh_dev = None                 # BlackHole's CoreAudio id
        self.hp_dev = None                 # the 🎧 device's id, if it has a volume control
        self.bh_gain = 1.0                 # undoes BlackHole's volume (1 = BlackHole at 100 %)
        self._bh_seen = None               # BlackHole volume at the last check
        self._bh_touched = False           # we changed BlackHole's volume (reset it on exit)

    # ---------- settings ----------
    def load_settings(self):
        try:
            return json.loads(self.settings_path.read_text())
        except Exception:
            return {}

    def save_settings(self):
        try:
            self.settings_path.write_text(json.dumps(
                {"monitor": self.monitor, "model": self.model_key, "pause": self.pause,
                 "mode": self.mode, "inputs": self.inputs, "outputs": self.outputs,
                 "listen_on": self.listen_on, "volume": self.volume, "provider": self.provider,
                 **{f"{p}_key": k for p, k in self.keys.items() if k}}))
        except Exception:
            pass

    @property
    def model_id(self):
        return self.models.get(self.model_key) or getattr(self, "custom_model", self.models["haiku"])

    @property
    def models(self):
        """key -> model id for the current provider (Anthropic direct has only the Claude ones)."""
        return OPENROUTER_MODELS if self.provider == "openrouter" else MODELS

    def set_listen(self, listen_on=None, volume=None):
        """Change where 🎧 plays and how loud, without restarting the audio input."""
        if volume is not None:
            self.volume = max(0, min(100, int(float(volume))))
            self.apply_volume(self.volume / 100)
        restart = listen_on is not None and listen_on != self.listen_on
        if restart:
            self.listen_on = listen_on
        self.save_settings()
        if restart and self.mode == "game" and self.sources:
            self.stop_monitor()
            self.start_monitor()
            self.refresh_listening_status()

    # ---------- 🎧 volume ----------
    def _game_from_blackhole(self):
        return (self.mode == "game" and bool(self.sources)
                and "blackhole" in self.sources[0].name.lower())

    def apply_volume(self, v):
        """The 🎧 slider: set the level like the Mac's volume keys do."""
        mv = self.macvol
        if mv and self._game_from_blackhole() and self.bh_dev:
            self._bh_touched = True
            mv.set(self.bh_dev, v)        # volume_sync copies it to the headphones
        elif mv and self.hp_dev:
            mv.set(self.hp_dev, v)
        self.update_monitor_gain()

    def update_monitor_gain(self):
        if self.mon_gain is None:
            return
        if self.macvol is None:
            g = (self.volume / 100) ** 2                  # no CoreAudio: plain software volume
        elif self._game_from_blackhole():
            # with a volume control on the headphones they set the level; without one
            # (e.g. HDMI), BlackHole's own volume is the level
            g = self.bh_gain if self.hp_dev else 1.0
        else:
            g = 1.0 if self.hp_dev else (self.volume / 100) ** 2
        self.mon_gain.value = g

    def sync_volume_on_enable(self):
        """When 🎧 turns on, start the volume keys from the level you hear now."""
        mv = self.macvol
        if not (mv and self._game_from_blackhole() and self.bh_dev):
            return
        v = mv.get(self.hp_dev) if self.hp_dev else self.volume / 100
        if v is None:
            return
        self._bh_seen = v                # not a key press: don't copy it back
        self._bh_touched = True
        mv.set(self.bh_dev, v)
        mv.set_mute(self.bh_dev, False)
        self.volume = round(v * 100)

    def volume_sync(self):
        """Runs all the time: follows BlackHole's volume (the Mac's volume keys in game mode)."""
        mv, n = self.macvol, 0
        while True:
            time.sleep(0.1)
            try:
                if n % 20 == 0:                       # devices can come and go
                    self.bh_dev = mv.device("BlackHole 2ch")
                n += 1
                v = mv.get(self.bh_dev) if self.bh_dev else None
                if v is None:
                    self.bh_gain = 1.0
                    continue
                self.bh_gain = 10 ** (MacVolume.BH_RANGE_DB * (1.0 - v) / 20)
                playing = (self.monitor and self.mon_proc is not None
                           and self._game_from_blackhole())
                if mv.muted(self.bh_dev):
                    # the mute key: a muted BlackHole would silence the translator too,
                    # so un-mute it and mute / un-mute the headphones instead
                    mv.set_mute(self.bh_dev, False)
                    if playing and self.hp_dev:
                        mv.set_mute(self.hp_dev, not mv.muted(self.hp_dev))
                if self._bh_seen is None or abs(v - self._bh_seen) > 0.001:
                    self._bh_seen = v
                    if playing:
                        self.volume = round(v * 100)
                        if self.hp_dev:
                            mv.set(self.hp_dev, v)
                            mv.set_mute(self.hp_dev, False)
                self.update_monitor_gain()
            except Exception as e:
                print(f"[volume] {e}", flush=True)
                time.sleep(2)

    def set_provider(self, provider=None, keys=None):
        """Switch provider and/or save keys from the window, then rebuild the API client."""
        for p, k in (keys or {}).items():
            if p in PROVIDERS and isinstance(k, str):
                self.keys[p] = k.strip()
        if provider in PROVIDERS and provider != self.provider:
            self.provider = provider
            if self.model_key not in self.models:      # e.g. GPT chosen, now Anthropic direct
                self.model_key = DEFAULT_MODEL if DEFAULT_MODEL in self.models else "haiku"
        self.save_settings()
        self.setup_client()
        if self.client_error:
            self.set_status(self.client_error, "error")
        elif self.listening_status:
            self.refresh_listening_status()
        else:
            self.set_status(f"{PROVIDERS[self.provider]['label']} ready", "wait")

    def key_info(self):
        """What the window shows about each key (never the key itself)."""
        info = {}
        for p in PROVIDERS:
            if self.keys[p]:
                info[p] = {"source": "saved", "hint": "…" + self.keys[p][-4:]}
            elif load_api_key(p):
                info[p] = {"source": "env", "hint": PROVIDERS[p]["env"]}
            else:
                info[p] = {"source": "missing", "hint": ""}
        return info

    def setup_client(self):
        info = PROVIDERS[self.provider]
        key = load_api_key(self.provider, self.keys[self.provider])
        if not key:
            self.client = None
            self.client_error = f"{info['label']} API key missing. Click 🔑 Keys to add it"
            return
        if self.provider == "openrouter":
            # OpenRouter wants "Authorization: Bearer"; an explicit credential also stops the SDK
            # from picking up ANTHROPIC_* env keys, so the Anthropic key is never sent there
            self.client = anthropic.Anthropic(base_url=OPENROUTER_BASE_URL, auth_token=key,
                                              default_headers={"X-Title": "Chalangar"})
        else:
            self.client = anthropic.Anthropic(api_key=key)
        self.client_error = None
        print(f"[ai] provider={self.provider} model={self.model_id}", flush=True)

    def apply_settings(self, model=None, pause=None, mode=None, device=None, output=None):
        if model in self.models:
            self.model_key = model
        if pause is not None:
            self.pause = min(2.0, max(0.1, round(float(pause), 1)))
        mode_changed = mode in ("game", "meeting") and mode != self.mode
        if mode_changed:
            self.mode = mode
        elif device is not None and device != self.inputs[self.mode]:
            self.inputs[self.mode] = device      # a new input for the current mode
            mode_changed = True                  # restart the audio
        elif output is not None and output != self.outputs[self.mode]:
            self.outputs[self.mode] = output     # a new sound output for the current mode
            mode_changed = True
        self.save_settings()
        print(f"[settings] model={self.model_id} pause={self.pause} mode={self.mode} "
              f"inputs={self.inputs} outputs={self.outputs}", flush=True)
        if mode_changed and self.recognizer is not None:
            threading.Thread(target=self.start_audio, daemon=True).start()

    def set_status(self, text, state="wait"):
        self.status, self.state = text, state
        print(f"[status] {text}", flush=True)

    def refresh_listening_status(self):
        if not self.listening_status:
            return
        if self.client_error:
            self.set_status(self.client_error, "error")
            return
        if self.output_error:
            self.set_status(f"⚠️ {self.output_error}", "error")
            return
        text = self.listening_status
        if self.monitor and self.mon_proc is not None:
            text += f" · 🎧 {self.out_name}"
            try:
                mac_out = sd.query_devices(sd.default.device[1])["name"].lower()
            except Exception:
                mac_out = ""
            if "multi-output" in mac_out:
                text += " · ⚠️ Sound out is Multi-Output: you will hear it twice. Use BlackHole 2ch"
        self.set_status(text, "ok")

    # ---------- start everything (runs in a background thread) ----------
    def start_backend(self):
        try:
            self.setup_client()      # a missing key is reported, the speech model still loads
            if self.macvol:
                threading.Thread(target=self.volume_sync, daemon=True).start()
            if self.client_error:
                self.set_status(self.client_error, "error")

            # the speech model is loaded inside the transcriber thread (GPU needs one thread)
            recognizer = Recognizer(self.args)
            self.model_ready = threading.Event()
            self.model_error = None
            threading.Thread(target=self.transcriber, args=(recognizer,), daemon=True).start()
            self.model_ready.wait()
            if self.model_error:
                self.set_status(f"Speech model error: {self.model_error}", "error")
                return
            self.recognizer = recognizer
            self.start_audio()
        except Exception as e:
            self.set_status(f"Error: {e}", "error")

    def stop_audio(self):
        self.stop_monitor()
        for src in self.sources:
            src.alive = False
            src.audio_q.put(None)
            try:
                src.stream.stop()
                src.stream.close()
            except Exception:
                pass
        self.sources = []

    def find_sources(self):
        """Which inputs to listen to in the current mode. Returns (sources, error)."""
        if self.mode == "game":
            idx, dev = find_input_device(self.args.device or self.inputs["game"])
            if idx is None:
                return [], f"Input '{self.inputs['game'] or 'default'}' not found. Choose another Input"
            return [Source("game", idx, dev)], None

        # meeting: others = BlackHole (the meeting app's sound), you = the Mac's input (mic)
        bh_idx, bh = find_input_device("BlackHole")
        if bh_idx is None:
            return [], "Meeting mode needs BlackHole 2ch (brew install blackhole-2ch)"
        mic_idx, mic = find_input_device(self.inputs["meeting"])
        if mic_idx is None or "blackhole" in mic["name"].lower():
            return [], "Meeting mode: choose your microphone in Input (not BlackHole)"
        return [Source("you", mic_idx, mic), Source("others", bh_idx, bh)], None

    def start_audio(self):
        with self.audio_lock:
            try:
                self.stop_audio()
                if self.mode == "meeting" and not self.outputs["meeting"]:
                    # first time in meeting mode: pick the Multi-Output Device automatically
                    multi = next((n for n in output_device_names()
                                  if "multi-output" in n.lower()), "")
                    if multi:
                        self.outputs["meeting"] = multi
                        self.save_settings()
                self.output_error = None
                if self.outputs[self.mode]:
                    self.output_error = set_mac_output(self.outputs[self.mode])
                    if self.output_error:
                        print(f"[output] {self.output_error}", flush=True)
                try:
                    sd._terminate()
                    sd._initialize()        # see devices added since the app started
                except Exception:
                    pass
                sources, error = self.find_sources()
                if error:
                    self.listening_status = ""
                    self.set_status(error, "error")
                    return
                for src in sources:
                    threading.Thread(target=self.segmenter, args=(src,), daemon=True).start()
                    src.stream = sd.InputStream(
                        device=src.dev_idx, channels=src.channels, samplerate=src.src_rate,
                        blocksize=int(src.src_rate * BLOCK_SEC), dtype="float32",
                        callback=lambda indata, f, t, s, src=src: self.audio_callback(src, indata))
                    src.stream.start()
                self.sources = sources
                label = self.recognizer.label
                if self.mode == "game":
                    self.listening_status = f"Listening · {sources[0].name} · {label}"
                    try:
                        out_name = sd.query_devices(sd.default.device[1])["name"]
                    except Exception:
                        out_name = ""
                    if (out_name and sources[0].name == out_name
                            and "blackhole" not in out_name.lower()):
                        self.listening_status += (" · ⚠️ headphone mic: sound will be phone "
                                                  "quality. Choose another Input")
                    self.start_monitor()          # ready for 🎧, plays only when it is on
                else:
                    self.listening_status = (f"Meeting · You: {sources[0].name} · "
                                             f"Others: {sources[1].name} · {label}")
                self.refresh_listening_status()
            except Exception as e:
                self.set_status(f"Audio error: {e}", "error")

    # ---------- play the input sound in the headphones (game mode) ----------
    def set_monitor(self, on):
        self.monitor = on
        self.save_settings()
        if self.mode != "game" or not self.sources:     # applied when game audio starts
            return on
        if self.mon_proc is None or not self.mon_proc.is_alive():
            self.start_monitor()
        if on:
            self.sync_volume_on_enable()
            self.update_monitor_gain()
        if self.mon_enabled is not None:
            self.mon_enabled.value = on                  # instant: the process is already running
        self.refresh_listening_status()
        return self.monitor

    def pick_listen_device(self):
        """The device 🎧 plays to: the chosen one, else the Mac output if it is a real
        speaker/headphone, else the first output that is not BlackHole / Multi-Output."""
        def ok(d):
            n = d["name"].lower()
            return d["max_output_channels"] > 0 and "blackhole" not in n and "multi-output" not in n
        devices = list(enumerate(sd.query_devices()))
        if self.listen_on:
            for i, d in devices:
                if d["name"] == self.listen_on and ok(d):
                    return i, d
        try:
            i = sd.default.device[1]
            d = sd.query_devices(i)
            if ok(d):
                return i, d
        except Exception:
            pass
        return next(((i, d) for i, d in devices if ok(d)), (None, None))

    def start_monitor(self):
        try:
            src = self.sources[0] if self.sources else None
            if src is None:
                return
            out_idx, d = self.pick_listen_device()
            if out_idx is None:
                self.set_status("🎧: no headphones or speakers found", "error")
                return
            ctx = mp.get_context("spawn")
            self.out_name = d["name"]
            self.hp_dev = None
            if self.macvol:
                dev = self.macvol.device(d["name"])
                if dev is not None and self.macvol.get(dev) is not None:
                    self.hp_dev = dev                # has a real volume control
            self.mon_enabled = ctx.Value("b", False)
            self.mon_gain = ctx.Value("d", 1.0)
            self.mon_stop = ctx.Event()
            self.mon_stats = ctx.Array("i", [0, 0])
            if self.monitor:
                self.sync_volume_on_enable()
            self.update_monitor_gain()
            self.mon_enabled.value = self.monitor
            self.mon_proc = ctx.Process(
                target=monitor_process, daemon=True,
                args=(src.name, d["name"], self.mon_enabled, self.mon_gain, self.mon_stop,
                      self.mon_stats, os.getpid()))
            self.mon_proc.start()
            print(f"[monitor] {src.name} → {d['name']} (own process {self.mon_proc.pid})",
                  flush=True)
        except Exception as e:
            self.monitor = False
            self.mon_proc = None
            self.set_status(f"Headphones error: {e}", "error")

    def stop_monitor(self):
        proc = getattr(self, "mon_proc", None)
        if proc is not None:
            try:
                print(f"[monitor] stopped · gaps={self.mon_stats[0]} "
                      f"underflows={self.mon_stats[1]}", flush=True)
                self.mon_stop.set()
                proc.join(2)
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
        self.mon_proc = None
        self.mon_enabled = self.mon_gain = None
        if self._bh_touched and self.macvol and self.bh_dev:
            # leave BlackHole at 100 % for other apps (the level is on the headphones)
            self.macvol.set(self.bh_dev, 1.0)
            self._bh_touched = False
            self._bh_seen = None

    # ---------- audio capture ----------
    def audio_callback(self, src, indata):
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:, 0]
        mono = mono.copy()
        if self.bh_gain != 1.0 and "blackhole" in src.name.lower():
            mono *= self.bh_gain            # undo BlackHole's volume (the Mac's volume keys)
        src.audio_q.put(mono)

    # ---------- find sentences with a speech detector (works with music behind) ----------
    def segmenter(self, src):
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        vad_opts = VadOptions(threshold=self.args.vad, min_silence_duration_ms=200,
                              min_speech_duration_ms=250, speech_pad_ms=300)
        pre_pad = int(0.25 * SR)   # extra audio before speech so first words are not cut
        carry = np.zeros(0, dtype=np.float32)
        buf = np.zeros(0, dtype=np.float32)
        since_check = 0
        max_len = int(self.args.max_len * SR)
        step = int(VAD_EVERY_SEC * SR)
        unit = src.src_rate // math.gcd(src.src_rate, SR)   # e.g. 3 for 48 kHz

        while src.alive:
            block = src.audio_q.get()
            if block is None:
                break
            carry = np.concatenate((carry, block))
            n_in = len(carry) - len(carry) % unit
            part = resample(carry[:n_in], src.src_rate, SR)
            carry = carry[n_in:]
            buf = np.concatenate((buf, part))
            since_check += len(part)
            if since_check < step:
                continue
            since_check = 0

            try:
                ts = get_speech_timestamps(buf, vad_options=vad_opts)
            except Exception as e:
                print(f"[vad error] {e}", flush=True)
                ts = []

            if not ts:
                buf = buf[-int(1.0 * SR):]          # nothing said: keep only the last 1 s
                continue

            first, last = ts[0]["start"], ts[-1]["end"]
            if len(buf) - last >= int(self.pause * SR):
                cut = last                          # the speaker stopped
            elif len(buf) - first >= max_len:
                # very long talk: cut at the last small gap, or the quietest moment
                gaps = [(ts[i]["end"] + ts[i + 1]["start"]) // 2 for i in range(len(ts) - 1)]
                gaps = [g for g in gaps if g - first >= 1.5 * SR]
                cut = gaps[-1] if gaps else self.quietest_point(buf, first)
            else:
                continue

            audio = buf[max(0, first - pre_pad):cut]
            buf = buf[cut:]
            if len(audio) >= 0.4 * SR and src.alive:
                self.chunk_q.put((audio, time.time(), src.speaker))

    @staticmethod
    def quietest_point(buf, first):
        """A cut point near the end of the buffer where the sound is quietest."""
        win = int(0.05 * SR)
        lo = max(first + int(2 * SR), len(buf) - int(1.5 * SR))
        best, best_rms = len(buf), None
        for s in range(lo, len(buf) - win, win // 2):
            rms = float(np.sqrt(np.mean(buf[s:s + win] ** 2)))
            if best_rms is None or rms < best_rms:
                best, best_rms = s + win // 2, rms
        return best

    # ---------- speech → English text ----------
    def transcriber(self, recognizer):
        try:
            recognizer.load(self.set_status)
        except Exception as e:
            self.model_error = e
        finally:
            self.model_ready.set()
        if self.model_error:
            return

        next_id = 0
        while True:
            audio, t_cut, speaker = self.chunk_q.get()
            t0 = time.time()
            try:
                text = recognizer.transcribe(audio, None)
            except Exception as e:
                print(f"[speech error] {e}", flush=True)
                continue
            if normalize(text) in JUNK or len(text) < 2:
                continue
            # the same short line again and again = noise, not dialogue
            recent = [normalize(e["en"]) for e in self.entries[-6:]]
            if len(text.split()) <= 3 and recent.count(normalize(text)) >= 2:
                continue
            t_speech = time.time() - t0

            who = {"you": "You", "others": "Others"}.get(speaker, "")
            ctx = list(self.context)
            self.context.append(f"{who}: {text}" if who else text)
            with self.lock:
                entry = {"id": next_id, "who": who, "en": text, "fix": "", "fa": "",
                         "notes": [], "t": ""}
                next_id += 1
                self.entries.append(entry)
                self.rev += 1
            # meeting lines are private: show them in a terminal, but keep them out of logs/app.log
            if self.mode != "meeting" or sys.stdout.isatty():
                print(f"{who or 'EN'}: {text}", flush=True)
            if speaker != "you":
                clip = float(np.mean(np.abs(audio) > 0.98))
                if clip > 0.002:
                    self.set_status("⚠️ Input too loud (distorted). Lower the volume a bit.",
                                    "error")
            self.pool.submit(self.translate, entry, ctx, len(audio) / SR, t_speech)

    # ---------- English → Persian (others) or English feedback (you) with Claude ----------
    def ask_claude(self, ctx, text, system):
        kwargs = dict(
            # room for models that must think first (thinking counts toward max_tokens)
            model=self.model_id, max_tokens=1500, system=system,
            messages=[{"role": "user", "content":
                       f"Previous lines:\n{chr(10).join(ctx) or '(none)'}\n\nNew line:\n{text}"}])
        if self.model_id not in self.thinking_required:
            try:
                return self.client.messages.create(thinking={"type": "disabled"}, **kwargs)
            except anthropic.BadRequestError as e:
                # Anthropic says "thinking ...", OpenRouter says "Reasoning is mandatory ..."
                if not any(w in str(e).lower() for w in ("thinking", "reasoning")):
                    raise
                self.thinking_required.add(self.model_id)   # stop sending it for this model
        return self.client.messages.create(**kwargs)

    def translate(self, entry, ctx, audio_sec, t_speech):
        mine = entry["who"] == "You"
        system = SELF_PROMPT if mine else (MEETING_PROMPT if entry["who"] else SYSTEM_PROMPT)
        t1 = time.time()
        if self.client is None:
            self.set_status(self.client_error or "No API client", "error")
            with self.lock:
                entry["fa"] = "⚠️"
                self.rev += 1
            return
        try:
            resp = self.ask_claude(ctx, entry["en"], system)
            out = "".join(b.text for b in resp.content if b.type == "text").strip()
            if self.state == "error":
                self.refresh_listening_status()
        except Exception as e:
            self.set_status(f"API error: {e}", "error")
            with self.lock:
                entry["fa"] = "⚠️"
                self.rev += 1
            return
        t_claude = time.time() - t1
        timing = f"audio {audio_sec:.1f}s · speech {t_speech:.1f}s · claude {t_claude:.1f}s"
        print(f"   {timing}", flush=True)

        if out == "SKIP":
            with self.lock:
                if entry in self.entries:
                    self.entries.remove(entry)
                self.rev += 1
            return

        lines = [l.strip() for l in out.splitlines() if l.strip()]
        lines = [l for l in lines if not (l.endswith(":") and len(l) < 40)]
        fix = next((l[1:].strip() for l in lines if l.startswith("✏")), "")
        fix = fix.lstrip("️").strip()

        if mine:
            # feedback on your own English
            notes = [l for l in lines if l.startswith("💡")][:2]
            fa = "✅" if (out.upper().startswith("OK") or not fix) else ""
        else:
            notes = [l for l in lines if l.startswith("📝")][:3]
            fa = next((l for l in lines if not l.startswith(("📝", "✏", "💡"))), "") or "—"

        with self.lock:
            entry.update(fa=fa, fix=fix, notes=notes, t=timing if self.args.show_times else "")
            self.rev += 1

        if self.mode == "meeting":
            return          # meetings are not saved to a file (privacy)
        self.log_path.parent.mkdir(exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"- **{fix or entry['en']}**\n  - {fa}\n")
            for n in notes:
                f.write(f"  - {n}\n")


def make_handler(translator):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/listen":
                q = dict(p.split("=", 1) for p in url.query.split("&") if "=" in p)
                try:
                    lo = q.get("listen_on")
                    translator.set_listen(unquote(lo) if lo is not None else None, q.get("volume"))
                except ValueError:
                    pass
                url = url._replace(path="/settings", query="")
            if url.path in ("/settings", "/set"):
                q = dict(p.split("=", 1) for p in url.query.split("&") if "=" in p)
                if url.path == "/set":
                    try:
                        dev, out = q.get("device"), q.get("output")
                        translator.apply_settings(q.get("model"), q.get("pause"), q.get("mode"),
                                                  unquote(dev) if dev is not None else None,
                                                  unquote(out) if out is not None else None)
                    except ValueError:
                        pass
                try:
                    devices = input_device_names()
                    outputs = output_device_names()
                except Exception:
                    devices, outputs = [], []
                body = json.dumps({"model": translator.model_key, "pause": translator.pause,
                                   "models": [{"key": k, "label": MODEL_CHOICES[k][0]}
                                              for k in MODEL_CHOICES if k in translator.models],
                                   "mode": translator.mode,
                                   "device": translator.inputs[translator.mode],
                                   "devices": devices,
                                   "output": translator.outputs[translator.mode],
                                   "outputs": outputs,
                                   "can_switch": switch_audio_tool() is not None,
                                   "listen_on": translator.listen_on,
                                   "volume": translator.volume,
                                   "monitor": translator.monitor,
                                   "provider": translator.provider,
                                   "providers": [{"key": k, "label": v["label"]}
                                                 for k, v in PROVIDERS.items()],
                                   "keys": translator.key_info()},
                                  ensure_ascii=False)
                self._send(body.encode("utf-8"), "application/json; charset=utf-8")
                return
            if url.path == "/state":
                q = dict(p.split("=", 1) for p in url.query.split("&") if "=" in p)
                with translator.lock:
                    data = {"status": translator.status, "state": translator.state,
                            "rev": translator.rev, "volume": translator.volume}
                    if q.get("rev") != str(translator.rev):
                        data["entries"] = [dict(e) for e in translator.entries]
                self._send(json.dumps(data, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")
            else:
                self._send(VIEWER_HTML.encode("utf-8"), "text/html; charset=utf-8")

        def do_POST(self):
            # JSON body: {"provider": "...", "keys": {"openrouter": "...", "anthropic": "..."}}
            # Requiring application/json means another website can't post here without a
            # CORS preflight, which this server never answers.
            if (urlparse(self.path).path != "/provider"
                    or not self.headers.get("Content-Type", "").startswith("application/json")):
                self.send_error(400)
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(min(n, 10000)) or b"{}")
                translator.set_provider(data.get("provider"), data.get("keys"))
            except (ValueError, AttributeError):
                self.send_error(400)
                return
            self.path = "/settings"
            self.do_GET()

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    return Handler


class WindowApi:
    """Functions the window's buttons can call."""
    def __init__(self, translator):
        self._t = translator
        self._window = None
        self._on_top = False

    def toggle_monitor(self):
        return self._t.set_monitor(not self._t.monitor)

    def is_monitor(self):
        return self._t.monitor

    def toggle_on_top(self):
        self._on_top = not self._on_top
        self._window.on_top = self._on_top
        return self._on_top

    def is_on_top(self):
        return self._on_top


def main():
    p = argparse.ArgumentParser(description="Chalangar: live English → Persian translator")
    p.add_argument("--device", default="",
                   help="input device name (part of it); default: the Mac's current input")
    p.add_argument("--engine", default="auto", choices=["auto", "parakeet", "mlx", "faster"],
                   help="speech engine: parakeet / mlx (Whisper) = Mac GPU, faster = Whisper on CPU")
    p.add_argument("--parakeet-model", default="mlx-community/parakeet-tdt-0.6b-v2",
                   help="Parakeet model (v2 = English only, most accurate for English)")
    p.add_argument("--mlx-model", default="mlx-community/whisper-large-v3-turbo",
                   help="model for the GPU engine")
    p.add_argument("--whisper", default="large-v3-turbo",
                   help="model for the CPU engine: small.en, medium.en, large-v3-turbo")
    p.add_argument("--model", default=None,
                   help="model id, e.g. openai/gpt-6.1-terra (default: the one chosen in the window)")
    p.add_argument("--provider", default=None, choices=sorted(PROVIDERS),
                   help="AI provider (default: settings.json, else openrouter)")
    p.add_argument("--pause", type=float, default=None,
                   help="seconds of no speech that end a sentence")
    p.add_argument("--max-len", type=float, default=9.0, help="max seconds per chunk")
    p.add_argument("--vad", type=float, default=0.5,
                   help="speech detector sensitivity 0-1 (lower = catches more speech)")
    p.add_argument("--show-times", action="store_true", default=True,
                   help="show timing under each line")
    p.add_argument("--port", type=int, default=8765, help="local port for the window")
    p.add_argument("--browser", action="store_true", help="show in the browser, not a window")
    p.add_argument("--on-top", action="store_true", help="start with the window on top")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    t = Translator(args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(t))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=t.start_backend, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"

    if args.browser:
        webbrowser.open(url)
        print(f"Translations: {url}  (Ctrl+C to stop)")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    else:
        import webview  # pip install pywebview
        api = WindowApi(t)
        api._on_top = args.on_top
        api._window = webview.create_window(
            "Chalangar", url, js_api=api,
            width=460, height=680, min_size=(320, 300),
            on_top=args.on_top, background_color="#111418")
        try:   # show our icon in the Dock instead of the Python icon
            from AppKit import NSApplication, NSImage
            icon = (Path(__file__).resolve().parent /
                    "Chalangar.app/Contents/Resources/AppIcon.icns")
            if icon.exists():
                NSApplication.sharedApplication().setApplicationIconImage_(
                    NSImage.alloc().initWithContentsOfFile_(str(icon)))
        except Exception as e:
            print(f"[icon] {e}", flush=True)
        webview.start()   # returns when the window is closed

    t.stop_monitor()
    print(f"Stopped. Phrases saved in {t.log_path}", flush=True)
    os._exit(0)          # stop audio and background threads right away


if __name__ == "__main__":
    main()
