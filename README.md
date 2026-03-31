# Stitch — Local AI Music Workstation

A local-first desktop GUI for AI music generation, editing, and stem
separation. All data, config, and outputs live in `%APPDATA%\stitch\` —
nothing is hardcoded, nothing leaves your machine.

---

## Quick Start (Windows)

1. Install **Python 3.10+** from https://python.org  
   ✔ Check **"Add Python to PATH"** during install

2. Install **CUDA-enabled PyTorch** from https://pytorch.org/get-started/locally/  
   Select your CUDA version (e.g. CUDA 12.1). Example:
   ```
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. Install **ACE-Step** from source:
   ```
   pip install git+https://github.com/ace-step/ACE-Step.git
   ```

4. Double-click **`run.bat`**

`run.bat` will create a `.venv`, install all remaining deps from
`requirements.txt`, and launch Stitch. If the venv ever gets broken, just
delete `.venv\` and run again.

---

## Features

### Generate tab

| Mode | What it does |
|------|-------------|
| **Text prompt** | Describe a style → generate N audio variations |
| **Cover / restyle** | Upload audio + new style prompt → transform it while keeping structure |
| **Vocal + backing** | Upload a dry vocal → AI generates matching instrumentation |

**Results panel** (right side):
- Waveform preview + inline audio player for every variation
- ★ Star to favourite; sort by Creation order / Starred first / Duration
- **Stems ▾** — runs Demucs to separate into vocals, drums, bass, other
- **Repaint region** — sends the file straight to the Repair tab
- **⬇ WAV / ⬇ MP3** download buttons (MP3 requires ffmpeg)
- **Presets ▾** collapsible strip — save/load/delete named style presets

### Repair tab

- Load any audio file (drag-and-drop or browse)
- Inline audio player for the source file
- **Drag on the waveform** to select a region → AI inpaints only that section
- **Full file** mode for a cleanup pass across the entire track
- After repair: **Before/After** comparison with individual players
- **Keep repaired / Keep original / ⬇ Save as…** actions

### History tab

- Full log of every generation and repair
- Click any row to preview it in the mini-player
- **Send to Generate** button re-populates the form with the original prompt
- Clear history (does not delete audio files)

### Settings (⚙ button, top-right)

Four tabs — all changes written to `config.json` immediately:

| Tab | Settings |
|-----|---------|
| Generation | Default variations, duration, compute device, sample rate |
| Paths | Override output/input/model/stems directories |
| Models | ACE-Step model ID, Demucs model variant |
| Export | Default format (WAV/MP3), MP3 bitrate |

---

## App Data Layout

Everything stored under `%APPDATA%\stitch\` — no files written next to the
executable.

```
%APPDATA%\stitch\
├── config.json            ← all settings (edit here or via ⚙ Settings UI)
├── outputs\               ← generated and repaired audio files
├── inputs\                ← imported/staged audio files
├── models\                ← ACE-Step weights cache (HuggingFace)
│   └── loras\             ← drop .safetensors / .pt LoRA files here
├── stems\                 ← Demucs stem separation outputs
├── presets\               ← named style presets (one JSON per preset)
├── logs\
│   └── stitch.log         ← rotating log (5 MB × 3 backups)
└── session_history.json   ← last 500 generation records
```

To move outputs to a different drive, edit `outputs_dir` in `config.json`
(or use Settings → Paths → Outputs directory).

---

## LoRA Support

Drop `.safetensors` or `.pt` LoRA files into:
```
%APPDATA%\stitch\models\loras\
```
They will appear in the **LoRA** dropdown on the Text prompt panel.
Click **↻** to rescan without restarting.

---

## MP3 Export

MP3 export uses **pydub** (installed automatically) + **ffmpeg**.

Install ffmpeg:
- Windows: https://ffmpeg.org/download.html — add to PATH
- Or via `winget install ffmpeg`

If ffmpeg is absent, the WAV file is saved instead with a warning in the
status bar.

---

## Config Reference (`config.json`)

| Key | Default | Description |
|-----|---------|-------------|
| `device` | `"auto"` | `"auto"` detects CUDA; override with `"cuda"` or `"cpu"` |
| `ace_step_model` | `"ACE-Step/ACE-Step-v1-3.5B"` | HuggingFace repo ID |
| `demucs_model` | `"htdemucs"` | `htdemucs`, `htdemucs_ft`, `htdemucs_6s`, `mdx_extra` |
| `default_variations` | `4` | How many variations to generate per run |
| `default_duration` | `30` | Default clip duration in seconds |
| `output_format` | `"wav"` | `"wav"` or `"mp3"` |
| `mp3_bitrate` | `320` | MP3 bitrate in kbps |
| `outputs_dir` | `%APPDATA%\stitch\outputs` | Override output path |
| `inputs_dir` | `%APPDATA%\stitch\inputs` | Override input path |
| `models_dir` | `%APPDATA%\stitch\models` | Override model cache path |
| `stems_dir` | `%APPDATA%\stitch\stems` | Override stems path |

---

## Stub Mode

If ACE-Step is not installed, Stitch runs in **stub mode** — all UI and
navigation works normally, but generation produces short silent WAV files.
This lets you develop and test the UI without needing a GPU.

Demucs is also optional; if absent, the Stems button produces silent
placeholder files.

---

## Debug Logging

Set the environment variable `STITCH_DEBUG=1` before launching to enable
verbose DEBUG-level logging to both console and log file:

```bat
set STITCH_DEBUG=1
python main.py
```

---

## Project Structure

```
stitch/
├── main.py                        # Entry point + exception hook
├── run.bat                        # Windows launcher + venv setup
├── requirements.txt
├── README.md
└── app/
    ├── config.py                  # All paths + settings singleton (cfg)
    ├── backend/
    │   ├── ace_step.py            # ACE-Step model wrapper (stub-safe)
    │   ├── demucs.py              # Demucs stem separator (subprocess)
    │   ├── exporter.py            # WAV/MP3 export worker (pydub + ffmpeg)
    │   ├── logger.py              # Rotating file logger
    │   ├── presets.py             # Named preset save/load/delete
    │   ├── session.py             # Generation history (session_history.json)
    │   └── worker.py              # QThread workers for all operations
    ├── models/
    │   └── generation.py          # Request/result dataclasses + enums
    └── ui/
        ├── main_window.py         # App shell, topbar, VRAM meter, stylesheet
        ├── generate_page.py       # Generate tab (Text / Cover / Vocal modes)
        ├── repair_page.py         # Repair tab + waveform region selector
        ├── history_page.py        # Session history browser
        ├── settings_dialog.py     # 4-tab settings dialog
        └── widgets/
            ├── audio_player.py    # Inline transport bar (QMediaPlayer)
            ├── preset_panel.py    # Collapsible preset strip
            ├── result_card.py     # Single result card with player + stems
            ├── upload_zone.py     # Drag-and-drop audio file zone
            └── waveform.py        # Waveform painter + drag-to-select
```

---

## Roadmap / Future

- [ ] Batch export (export all starred results at once)
- [ ] Project/session save + load (group of results with shared prompt)
- [ ] Real-time VRAM progress bar during generation steps
- [ ] BPM / key detection display on result cards
- [ ] React frontend port for polished release
