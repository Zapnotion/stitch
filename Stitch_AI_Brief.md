# Stitch — AI Coding Brief
**For:** AI coding assistant (Claude, Cursor, etc.)  
**Project:** Stitch — Local AI Music Workstation  
**Status:** Working prototype. Codebase is clean and well-structured. Do not refactor what isn't broken.

---

## What Stitch is

A Windows desktop application built with **PySide6 (Qt)** and **Python 3.10+**. It provides a local, fully offline GUI for AI music generation via **ACEStep 1.5**, stem separation via **Demucs**, and AI lyric writing via a local GGUF LLM (Qwen2.5). Nothing calls home. No subscriptions. No cloud.

The guiding principles — enforce them in every feature you write:

| Principle | Meaning |
|---|---|
| **Offline first** | Every feature works without internet after initial model download |
| **Fallback always** | If a model isn't downloaded or a library is missing, prior behaviour continues — no crash |
| **Transparency** | The user can always see what the AI produced and why |
| **Composability** | Features build on each other; design with downstream unlocks in mind |
| **VRAM respect** | New models must be CPU-side or use a small, bounded GPU budget — ACEStep already owns the GPU |

---

## Project structure

```
stitch/
├── main.py                             # Entry point + global exception hook
├── run.bat                             # Windows launcher: creates .venv, installs deps, launches
├── requirements.txt
└── app/
    ├── config.py                       # cfg singleton — all paths + settings
    ├── backend/
    │   ├── _ace_worker_v15.py          # ← CORE: subprocess worker; receives JSON argv, emits JSON lines
    │   ├── ace_step_v15.py             # ACEStepV15 wrapper — spawns _ace_worker_v15.py
    │   ├── demucs.py                   # Demucs stem separator (subprocess)
    │   ├── exporter.py                 # WAV/MP3 export (pydub + ffmpeg)
    │   ├── lyrics_gen.py               # GGUF LLM lyric writer
    │   ├── presets.py                  # Named preset save/load/delete
    │   ├── session.py                  # Generation history (session_history.json)
    │   └── worker.py                   # QThread workers for all async operations
    ├── models/
    │   └── generation.py               # Request/result dataclasses + enums
    └── ui/
        ├── main_window.py              # App shell, topbar, nav, VRAM meter, stylesheet
        ├── generate_page.py            # Generate tab (Text / Cover / Vocal modes)
        ├── repair_page.py              # Repair tab + waveform region selector
        ├── history_page.py             # Session history browser
        ├── settings_dialog.py          # 4-tab settings dialog
        └── widgets/
            ├── audio_player.py         # Inline transport bar (QMediaPlayer)
            ├── preset_panel.py         # Collapsible preset strip
            ├── result_card.py          # ← PRIMARY ATTACHMENT POINT for new features
            ├── upload_zone.py          # Drag-and-drop audio input
            └── waveform.py             # Waveform painter + drag-to-select
```

**Key data paths** (all resolved via `cfg` — never hardcode):
- `cfg.outputs_dir` — generated audio
- `cfg.stems_dir` — Demucs stem outputs
- `cfg.models_dir` — ACEStep weights + GGUF models
- `cfg.presets_dir` — per-preset JSON files
- `cfg.appdata_dir / "session_history.json"` — last 500 records

**App data root:** `%APPDATA%\stitch\`

---

## Architecture patterns — follow these exactly

### Generation worker pattern
All heavy operations run in a `QThread` worker in `worker.py`. Workers emit Qt signals for progress and results; the UI thread never blocks. Look at any existing worker (e.g. `TextGenerationWorker`) before writing a new one.

### ACEStep subprocess protocol
`_ace_worker_v15.py` runs as a subprocess inside a separate venv (`.venv_model_v15`). The parent sends a JSON payload as `argv[1]`. The worker emits JSON lines to stdout — `{"type": "progress", ...}` and `{"type": "result", ...}`. `ace_step_v15.py` owns the subprocess lifecycle.

**GenerationParams field detection:** `_ace_worker_v15.py` uses Python `inspect` to check which fields `GenerationParams` actually accepts before passing them. Always apply the same pattern when adding new params (bpm, key, etc.) — do not assume a field exists.

### Fallback pattern
```python
try:
    import whisper_timestamped
    ALIGNMENT_AVAILABLE = True
except ImportError:
    ALIGNMENT_AVAILABLE = False
```
Then gate every alignment-dependent code path behind `ALIGNMENT_AVAILABLE`. The UI shows a "Download alignment model" prompt rather than crashing.

### Stylesheet
All visual styling lives in `main_window.py → _apply_stylesheet()`. Use `setObjectName()` for targeting. Do not use inline `setStyleSheet()` on individual widgets — keep the stylesheet centralised.

### Sidecar pattern (new — establish this in Phase 1)
Every generated audio file gets a JSON sidecar stored alongside it:
```
outputs/text_v01_1234_abc123.wav
outputs/text_v01_1234_abc123.alignment.json
```
Sidecar schema:
```json
{
  "words": [{"word": "mountain", "start": 1.23, "end": 1.61, "confidence": 0.82}],
  "sections": [{"label": "Verse 1", "start": 0.0, "end": 14.5}]
}
```
`GenerationResult` (in `generation.py`) should grow an `alignment_path: Optional[str] = None` field.

---

## Current capabilities (as of April 2026)

| Feature | Status |
|---|---|
| Text-to-music generation | ✅ |
| Cover / restyle | ✅ |
| Vocal + backing | ✅ |
| Region repair (waveform drag-to-select) | ✅ |
| Stem separation (Demucs) | ✅ |
| AI lyrics (GGUF, offline) | ✅ |
| Presets (save/recall full param sets) | ✅ |
| Session history (persistent, 500 records) | ✅ |
| LoRA support | ✅ |

**The gap:** Stitch generates music but cannot show the user what was generated, where it sits in time, or give control over specific moments. The features below close that gap.

---

## Feature roadmap — implement in this order

### Phase 1 — Lyric Alignment & Precision
*Keystone phase. Everything downstream depends on the alignment sidecar.*

**New dependency:** `whisper-timestamped` (offline, CPU, ~150 MB base Whisper model). Add to `requirements.txt` and to `run.bat` install step with the same CUDA-detection guard already used for `llama-cpp-python`.

#### 1.1 Alignment runner
- New file: `app/backend/aligner.py`
- Function: `align(audio_path: str, lyrics: str) -> dict` — returns the sidecar dict
- Uses `whisper_timestamped.transcribe()` with `language="en"`, `task="transcribe"`, word-level timestamps
- Falls back gracefully if `whisper_timestamped` is not installed (returns `None`)
- Called after every successful generation in `worker.py`; sidecar written to disk alongside the WAV

#### 1.2 Lyric timeline widget
- New file: `app/ui/widgets/lyric_timeline.py` — `LyricTimelineWidget(QWidget)`
- Displays lyrics as inline word `<span>` elements inside a `QTextEdit` with `QTextCharFormat` colour markup
- Colour coding by confidence: `≥ threshold` → white; `0.4–threshold` → amber `#E8A84B`; `< 0.4` → red `#E85D5D`; current word → `#FFFFFF` bold
- `set_position(sec: float)` slot: called every ~50ms via `QTimer` from `AudioPlayer` playback position; updates the highlighted word
- Clicking a word seeks the player to that word's `start` time (emit a `seek_requested = Signal(float)` signal)
- Section labels (`[Verse 1]`, `[Chorus]`) rendered as distinct divider rows between words
- Attached below the waveform in `ResultCard` when an alignment sidecar exists

#### 1.3 Confidence flagging banner
- A `QWidget` bar above the timeline: "3 words may be unclear — click to review"
- Clicking jumps playback to the first flagged word
- Collapsible (×)
- Threshold slider lives in Settings → Generation tab (0.0–1.0, default 0.6, key: `alignment_confidence_threshold`)

#### 1.4 Single-word repair
- Right-click a word span in the timeline → context menu: "Repair this word"
- Extracts `start - 0.05s` to `end + 0.05s` from the sidecar (enforce minimum 0.3s window)
- Routes to existing `RepairRequest(mode=REGION, region_start_sec=..., region_end_sec=..., hint_prompt=<intended word>)`
- On success: splice repaired region back into original WAV using `soundfile` + numpy array slicing; save as new variation; re-run aligner; refresh timeline
- Preserve the original — new variation only

#### 1.5 Section-level regeneration
- Right-click a section divider in the timeline → "Regenerate this section"
- Same mechanism as word repair but at section granularity
- Apply 100ms crossfade at both splice boundaries using `scipy.signal`
- Lock icon on each section divider — locked sections are skipped and shown with a subtle border highlight

#### 1.6 Lyric lock/unlock per line
- Lock icon on each section divider (and optionally per word on right-click)
- Locked sections/words are passed as a constraint in the next generation — prepend `[LOCKED]` tag around those lines in the lyrics field; document the convention in a comment

#### 1.7 Phoneme-level edit hints
- Right-click a flagged word → "Edit pronunciation…" → small inline input
- User enters a respelling (e.g. `MAO-ten` for "mountain")
- The respelling is prepended to that word in the lyrics for the next repair or regeneration: `[MAO-ten]mountain`

---

### Phase 2 — Musical Range & Structure
*No new models required. Exposes parameters ACEStep already supports.*

#### 2.1 Tempo, key, time signature
- Add to the Text prompt panel under a collapsible "Musical parameters" section (collapsed by default)
- **BPM slider:** 40–200, default blank (free). Snap markers at 60, 80, 90, 100, 120, 140, 160. "Tap" button sets BPM by tapping
- **Key dropdown:** 24 options (C major … B major, C minor … B minor) + "Free"
- **Time signature selector:** 4/4 (default), 3/4, 6/8, 5/4, 7/8
- Pass through to `GenerationParams` using the existing `inspect`-based field detection. If ACEStep has no native time signature field, append it as a tag in the style prompt (e.g. `"6/8 time signature"`)
- Persist in presets alongside style prompt and lyrics

#### 2.2 Song structure template builder
- A horizontal row of draggable pill widgets above the lyrics field
- Each pill: section type selector (Intro / Verse / Pre-Chorus / Chorus / Bridge / Outro / Instrumental / Break), optional duration target, optional energy badge (Low / Mid / High)
- Add pill via `+` button; remove via `×`; reorder via drag
- Each Verse/Bridge pill has an expandable text area for its specific lyrics; Chorus pills share one lyric block by default
- On generation, serialise to structured lyrics tags: `[Intro]\n[Verse 1]\n<verse lyrics>\n[Chorus]\n<chorus lyrics>` etc., with energy badge appended as a per-section caption hint
- One-click presets: Standard Pop, Ballad, EDM Build, Anthem, Film Theme

#### 2.3 Negative prompting
- A `+Exclusions` expandable field below the style prompt
- Placeholder: `no piano, no autotune, no trap beat`
- Appended to the caption as `Avoid: [content]` — ACEStep's caption conditioning responds to this phrasing
- Persists in presets

#### 2.4 Micro-genre blend wheel
- Two genre combo boxes with a proportion slider between them (0–100%); optional third genre slot
- Genre vocabulary: ~80 curated terms that ACEStep responds well to (build this list from testing; store as a JSON file in `app/data/genres.json`)
- Preview shows the auto-generated style prompt string in real time
- "Manual" toggle switches back to the existing free-text field
- Persists in presets

#### 2.5 Reference track style extraction
- Drop zone (or "Browse…" button) in the Musical Parameters section
- On drop: run `librosa.beat.beat_track()` for BPM, chroma analysis for key (accuracy ~80% on tonal music)
- Map spectral features (centroid, rolloff, zero-crossing rate) to descriptive adjectives (bright/dark, dense/sparse, energetic/calm) and auto-populate the style prompt
- No copying or storing the reference audio — metadata extraction only
- All offline, no new dependencies (librosa is already present)

#### 2.6 Mood arc per section
- Each section pill in the structure builder has an optional mood field (free text or a small set of presets: melancholy, tense, euphoric, contemplative, etc.)
- On generation, each section's mood is appended to its caption hint alongside the energy badge

---

### Phase 3 — Stem Mixer Console
*Demucs stems already exist as files. This gives them a mixing surface.*

#### 3.1 Stem mixer panel
- New file: `app/ui/widgets/stem_mixer.py` — `StemMixerPanel(QWidget)`
- "Open mixer" button appears on `ResultCard` when `result.stems` is populated
- One channel strip per stem (Vocals, Drums, Bass, Other):
  - Vertical `QSlider` fader: −48 dB to +6 dB, default 0 dB. Display dB label above fader
  - `QDial` pan knob: L100–R100, default centre
  - Mute button: zeroes gain for that stem
  - Solo button: mutes all others
- "Preview mix" button: renders a 10-second preview in a background `QThread` using numpy/soundfile; plays inline without saving
- Mix state persisted per result in the session JSON

#### 3.2 Per-stem 3-band EQ
- Expandable EQ section on each channel strip
- Three band gain sliders: Low shelf (200 Hz), Mid peak (1 kHz, frequency adjustable), High shelf (8 kHz)
- Range: −10 dB to +10 dB (cap at ±10 to avoid scipy artefacts; show a "Reset EQ" button prominently)
- Implementation: `scipy.signal.iirfilter` — shelf filters for Low/High, peak filter for Mid. Pre-computed at render time, not real-time
- A small static EQ curve graphic drawn with `QPainter` updates to reflect current settings

#### 3.3 Mix render and export
- "Render mix" button: loads all stem WAVs as numpy arrays, applies gain/pan/EQ in sequence, sums to stereo, writes via `soundfile`
- Loudness normalisation: `pyloudnorm` to −14 LUFS (streaming standard), configurable target in Settings
- True-peak limiting: look-ahead limiter at −0.3 dBTP implemented in numpy
- Format selector: WAV (lossless), MP3 via ffmpeg (if available), FLAC
- Saved as a new variation entry in the result list, labelled "Custom mix", linked to its source generation

#### 3.4 Vocal pitch correction
- Optional toggle on the Vocals channel strip: Off / Subtle / Medium / Tight
- Library: `pyrubberband` (CPU, offline). Requires `rubberband.dll` on Windows — bundle in `app/libs/` or fall back to `librosa.effects.pitch_shift` if not found
- If a key was set in Phase 2, correction snaps to scale tones; otherwise chromatic
- Non-destructive: original vocal stem always preserved; corrected version is a temp file referenced only from the mix session

---

### Phase 4 — Collaboration & Versioning

#### 4.1 Version tree
- Replace the flat result list with a branching version tree
- Fork from any result: "Branch from here" creates a new generation pre-populated with that result's prompt and parameters
- Visual tree rendered in the History tab showing lineage (parent → children)
- Store parent-child relationship in `session_history.json` as `parent_id: Optional[str]`

#### 4.2 Session snapshots
- "Save session" saves full state: prompts, slider values, starred results, stems, mix settings, alignment sidecars
- Saved as a `.stitch` file (ZIP containing a manifest JSON + referenced audio paths)
- "Open session" restores exactly

#### 4.3 A/B comparison mode
- Select two result cards → "Compare" button
- Splits the right panel into two players; single keypress (Space/A/B) switches between them
- Waveforms rendered side by side with a shared time axis

#### 4.4 Annotation layer
- Timestamped notes on any generation: right-click the waveform at a position → "Add note…"
- Notes stored in the alignment sidecar's `annotations` array: `[{"time": 1.23, "text": "guitar too bright here"}]`
- Shown as markers on the waveform; listed below the lyric timeline
- "Use as negative prompt" button on each note — appends the text to the next generation's exclusion field

#### 4.5 Collaborative co-writing mode
- Turn-based lyric building panel in the Generate tab
- AI writes a verse → user edits it → AI writes chorus constrained by that verse
- The lyric generator already has the structure for this; it just needs a turn-based UI with "Accept / Regenerate / Edit" controls per section

---

### Phase 5 — Post-Processing Pipeline

#### 5.1 Mastering chain (standalone, no stems required)
- One-click "Master" button on every result card
- Pipeline: loudness normalisation (pyloudnorm, −14 LUFS) → high-shelf air boost (+1.5 dB at 10 kHz) → gentle limiter (−0.3 dBTP)
- All offline, scipy + pyloudnorm. Output saved as new variation "Mastered"

#### 5.2 Vocal doubling
- In the stem mixer: "Double vocals" toggle on the Vocals channel strip
- Generates a tight double using `pyrubberband` pitch shift (±10–15 cents, slight timing offset)
- Blended at −6 dB under the lead vocal

#### 5.3 Room / reverb matching
- "Match reverb from reference" in the stem mixer
- User drops a reference audio file; system estimates its RT60 (reverb time) using `pyroomacoustics` or a simple autocorrelation method
- Applies a matched reverb impulse response to the generated stems using `scipy.signal.fftconvolve`

#### 5.4 Format-aware export
- Export dialog gains format options: WAV, MP3, FLAC, OGG (for game engines)
- "Stems zip" option: exports all stems in a DAW-ready folder structure with tempo/key metadata embedded as BWF markers (use `soundfile` + manual RIFF chunk writing)
- "Broadcast deliverables" option (requires Phase 2 structure builder): auto-generates 30s, 60s, loopable, fade-out, and sting versions from one generation

---

### Phase 6 — Reference Track Conditioning

#### 6.1 Style cloning from audio
- Drop in any audio file → Stitch extracts: BPM (librosa), key (chroma), instrumentation profile (spectral analysis), energy profile (RMS over time)
- Auto-populates style prompt, BPM, and key fields
- Shows extracted descriptor so user can edit before generating

#### 6.2 Melody extraction and transfer
- Extract the predominant melody from a reference using `librosa.yin()` or `crepe` (offline)
- Convert to a note sequence and pass as `melody_hint` to ACEStep's `GenerationParams` (the parameter exists in the API but is not currently exposed)
- Ethically clear: extracting structure only, not copying audio

#### 6.3 Timbral matching
- Spectral centroid + tilt matching between reference and generated output
- Applied as a post-processing EQ curve on the final mix

---

### Phase 7 — Intelligent Prompt Assistance

#### 7.1 Prompt expansion
- User types a brief idea → system suggests specific instrumentation, vocal character, tempo, key, reference-style descriptors
- Implemented as a call to the local GGUF LLM (already in use for lyrics) with a system prompt instructing it to expand a music production brief
- Presented as editable suggestions, not auto-applied

#### 7.2 Prompt memory
- After each session, record which style prompts produced starred results
- Store in `cfg.appdata_dir / "prompt_memory.json"` (simple key: prompt fragment → star count)
- Suggest similar language in future sessions via a "Previously worked well:" chip strip above the style prompt field

#### 7.3 Style vocabulary guide
- In-app reference panel (accessible via "?" icon near the style prompt field)
- Curated list of genre names, production terms, instrument names, vocal descriptors that ACEStep responds well to
- Searchable; terms are clickable to append to the current style prompt

#### 7.4 Emotional coherence scoring
- After generation, run a small offline classifier that estimates whether the music's sonic mood matches the intended mood from the prompt
- Flag mismatches: "You asked for melancholy but this reads as energetic — regenerate?"
- Model: a fine-tuned audio feature classifier (MFCC + spectral features → mood label); should run in <2s on CPU
- Shown as a badge on the result card; dismissible

---

### Phase 8 — Film, TV & Animation Scoring

#### 8.1 Video import and duration lock
- Drop a video file into a new "Score" tab
- `duration_secs` is locked to the video's exact length
- Video plays back in sync with the generated audio (use `QMediaPlayer` for both, synchronised via a shared position signal)

#### 8.2 Scene marker system
- User places markers on the video timeline: "action peak at 0:23", "resolution at 1:45"
- Markers serialised into the generation prompt as timestamped event hints
- Musical events (swells, drops, key changes) conditioned to land at those timestamps

#### 8.3 Emotional curve mapping
- A simple line graph editor overlaid on the video timeline
- User draws an intensity envelope (0–100%) over time
- Envelope serialised as per-section energy values passed to the structure template builder (Phase 2)

#### 8.4 Cue library mode
- "Generate cue library" button: produces N variations (default 10) at different energy levels from the same theme prompt
- Variations are auto-labelled by detected energy level (quiet / moderate / intense)
- Batch exported as individually named files ready for use in a video editor

#### 8.5 Stinger generation
- Dedicated "Stingers" mode on the Generate tab
- Duration capped at 5–10 seconds
- Batch generate: 5 stingers in one click, auto-labelled

#### 8.6 Loop-to-cue export
- "Broadcast deliverables" button on any result card
- Auto-generates: 30s trim, 60s trim, loopable version (crossfaded tail-to-head), fade-out version, 5s sting
- Uses numpy audio trimming + crossfade; no additional models

---

### Phase 9 — Advanced / Novel Features

#### 9.1 Piano roll for melody hints
- A simple piano roll widget (new tab or expandable panel in Generate)
- User sketches a rough vocal contour — pitch and rough timing, not note-perfect
- Serialised as a MIDI-like note sequence and passed to ACEStep's `melody_hint` parameter
- If `melody_hint` is not in the current `GenerationParams`, document it as pending and expose the field for future ACEStep versions

#### 9.2 Chord progression editor
- Visual chord timeline: bar-by-bar chord slots (C, Am, F, G)
- Chord vocabulary: triads + common 7ths, Roman numeral display optional
- Serialised as a chord string appended to the lyrics/caption for conditioning
- Displayed on the result card after generation as detected chords (via `librosa.feature.chroma_stft`)

#### 9.3 Stem-aware cross-variation compositing
- "Build a mix" mode: user picks stems from different result cards
- "Drums from variation 2, vocals from variation 4" — mix engine loads selected stems, sums them, renders
- Requires all selected variations to have been stem-separated first
- A small drag-and-drop stem compositing surface in the History tab

#### 9.4 Live regeneration during playback
- While a song is playing, the user adjusts style/prompt sliders
- A new generation is queued silently in the background
- When ready, crossfades in at the next section boundary (detected from the alignment sidecar)
- Requires a background generation queue with a cancellable worker

#### 9.5 Adaptive scoring for interactive media
- A small runtime engine (separate from the main app, callable via a local socket or subprocess)
- Accepts state change events: `{"event": "combat_start", "intensity": 0.9}`
- Transitions between pre-generated cues based on event type and intensity
- Output: real-time audio stream or timed file handoff
- Designed for Godot/Unity integration via a simple TCP or file-watch protocol

#### 9.6 Micro-expression editing
- After generation, automatically detect emotional "peak moments": high-energy regions, silence boundaries, spectral flux peaks
- Display as markers on the waveform
- Right-click a marker → "Make this moment bigger" / "Soften this"
- Routes to targeted regional regeneration at that timestamp with a modified energy hint

#### 9.7 Narrative arc mode
- "Album / Soundtrack" mode: generate N related pieces sharing melodic DNA
- User defines the arc: how the emotional intensity and instrumentation should evolve across the set
- Each piece is generated with a shared seed offset and a per-piece mood/energy constraint
- Output: a numbered set of audio files with a shared thematic identity

---

## Technical stack reference

| Library | Purpose | Phase | Install status |
|---|---|---|---|
| `whisper-timestamped` | Word-level forced alignment | 1 | Add to requirements.txt |
| `scipy.signal` | Biquad EQ filters, crossfade | 1, 3 | Already installed (via librosa) |
| `numpy` | Audio array operations | all | Already installed |
| `soundfile` | Multi-channel WAV I/O | all | Already installed |
| `librosa` | BPM, key, spectral analysis | 2, 6 | Already installed |
| `pyloudnorm` | Broadcast loudness normalisation | 3, 5 | Add to requirements.txt |
| `pyrubberband` | Pitch correction, vocal doubling | 3, 5 | Add to requirements.txt; bundle rubberband.dll |

**VRAM budget across all phases:** Phases 1–5 add zero VRAM. Whisper-timestamped is CPU-only. The stem mixer, EQ, mastering, and pitch correction are all numpy/scipy. The only GPU consumer remains ACEStep.

---

## What to preserve at all costs

- The existing `_ace_worker_v15.py` subprocess protocol — do not change the JSON line format
- The `inspect`-based `GenerationParams` field detection — extend it, never bypass it
- The `cfg` singleton for all path resolution — never hardcode paths
- The centralised stylesheet in `main_window.py` — no inline widget styles
- Stub mode: every new feature must degrade gracefully if its model or library is absent
- The tidy aesthetic: the UI is clean and uncluttered. New widgets should feel integrated, not bolted on. When in doubt, hide things behind a collapsible section or a contextual right-click menu rather than adding more permanent chrome to the interface.
