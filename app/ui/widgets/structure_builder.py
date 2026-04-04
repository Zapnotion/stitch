"""
ui/widgets/structure_builder.py — Song structure template builder (Phase 2.2 + 2.6).

A horizontal row of draggable section pills that lets the user define
a song's structure before generating. Each pill carries:
  - Section type  (Intro / Verse / Pre-Chorus / Chorus / Bridge / Outro /
                   Instrumental / Break)
  - Energy badge  (Low / Mid / High — optional)
  - Mood tag      (free text or preset — optional, Phase 2.6)
  - Lyrics block  (expandable text area; Chorus pills share one block)

On serialise(), returns a (lyrics_text, caption_hint) tuple ready to be
merged into the generation request.

One-click structure presets: Standard Pop, Ballad, EDM Build, Anthem,
Film Theme.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMenu, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTION_TYPES = [
    "Intro", "Verse", "Pre-Chorus", "Chorus",
    "Bridge", "Outro", "Instrumental", "Break",
]

ENERGY_OPTIONS   = ["", "Low", "Mid", "High"]
MOOD_PRESETS     = ["", "melancholy", "tense", "euphoric", "contemplative",
                    "nostalgic", "triumphant", "dreamy", "angry", "peaceful"]

# Sections that share a single lyric block across all instances
SHARED_LYRIC_SECTIONS = {"Chorus", "Pre-Chorus"}

# One-click structure presets  (list of (type, energy, mood) tuples)
STRUCTURE_PRESETS: dict[str, list[tuple[str, str, str]]] = {
    "Standard Pop": [
        ("Intro",      "",    ""),
        ("Verse",      "Low", ""),
        ("Pre-Chorus", "Mid", ""),
        ("Chorus",     "High",""),
        ("Verse",      "Low", ""),
        ("Pre-Chorus", "Mid", ""),
        ("Chorus",     "High",""),
        ("Bridge",     "Mid", ""),
        ("Chorus",     "High",""),
        ("Outro",      "Low", ""),
    ],
    "Ballad": [
        ("Intro",  "Low",  "melancholy"),
        ("Verse",  "Low",  "melancholy"),
        ("Chorus", "Mid",  ""),
        ("Verse",  "Low",  ""),
        ("Chorus", "High", ""),
        ("Bridge", "Mid",  "contemplative"),
        ("Chorus", "High", ""),
        ("Outro",  "Low",  ""),
    ],
    "EDM Build": [
        ("Intro",         "Low",  ""),
        ("Verse",         "Mid",  ""),
        ("Break",         "Low",  "tense"),
        ("Instrumental",  "High", "euphoric"),
        ("Break",         "Low",  ""),
        ("Instrumental",  "High", ""),
        ("Outro",         "Mid",  ""),
    ],
    "Anthem": [
        ("Intro",  "Mid",  ""),
        ("Verse",  "Mid",  ""),
        ("Chorus", "High", "triumphant"),
        ("Verse",  "Mid",  ""),
        ("Chorus", "High", "triumphant"),
        ("Bridge", "High", ""),
        ("Chorus", "High", "triumphant"),
    ],
    "Film Theme": [
        ("Intro",        "Low",  "contemplative"),
        ("Verse",        "Low",  ""),
        ("Chorus",       "Mid",  ""),
        ("Verse",        "Mid",  ""),
        ("Chorus",       "High", "triumphant"),
        ("Bridge",       "High", ""),
        ("Instrumental", "High", ""),
        ("Outro",        "Low",  "melancholy"),
    ],
}


# ---------------------------------------------------------------------------
# Single section pill
# ---------------------------------------------------------------------------

class SectionPill(QFrame):
    """One pill in the structure row."""

    remove_requested = Signal(object)   # self
    changed          = Signal()

    def __init__(
        self,
        section_type: str = "Verse",
        energy:       str = "",
        mood:         str = "",
        shared_lyrics: Optional[dict] = None,   # {section_type -> str} shared pool
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("SectionPill")
        self.setFrameShape(QFrame.StyledPanel)
        self._shared_lyrics = shared_lyrics or {}

        self._build_ui(section_type, energy, mood)

    # -----------------------------------------------------------------------

    def _build_ui(self, section_type: str, energy: str, mood: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # --- Top row: type + remove ---
        top = QHBoxLayout()
        top.setSpacing(4)

        self._type_combo = QComboBox()
        self._type_combo.setObjectName("PillTypeCombo")
        self._type_combo.addItems(SECTION_TYPES)
        self._type_combo.setCurrentText(section_type)
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        top.addWidget(self._type_combo, 1)

        remove_btn = QPushButton("×")
        remove_btn.setObjectName("PillRemoveBtn")
        remove_btn.setFixedSize(16, 16)
        remove_btn.setFlat(True)
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        top.addWidget(remove_btn)

        root.addLayout(top)

        # --- Energy badge ---
        self._energy_combo = QComboBox()
        self._energy_combo.setObjectName("PillEnergyCombo")
        self._energy_combo.addItems(ENERGY_OPTIONS)
        self._energy_combo.setCurrentText(energy)
        self._energy_combo.currentTextChanged.connect(lambda _: self.changed.emit())
        root.addWidget(self._energy_combo)

        # --- Mood field (Phase 2.6) ---
        self._mood_combo = QComboBox()
        self._mood_combo.setObjectName("PillMoodCombo")
        self._mood_combo.setEditable(True)
        self._mood_combo.addItems(MOOD_PRESETS)
        self._mood_combo.setCurrentText(mood)
        self._mood_combo.setPlaceholderText("mood…")
        self._mood_combo.currentTextChanged.connect(lambda _: self.changed.emit())
        root.addWidget(self._mood_combo)

        # --- Lyrics expand toggle ---
        self._lyrics_toggle = QPushButton("▸ lyrics")
        self._lyrics_toggle.setObjectName("PillLyricsToggle")
        self._lyrics_toggle.setCheckable(True)
        self._lyrics_toggle.setChecked(False)
        self._lyrics_toggle.toggled.connect(self._toggle_lyrics)
        root.addWidget(self._lyrics_toggle)

        # --- Lyrics box (hidden by default) ---
        self._lyrics_box = QPlainTextEdit()
        self._lyrics_box.setObjectName("PillLyricsBox")
        self._lyrics_box.setFixedHeight(70)
        self._lyrics_box.setVisible(False)
        self._lyrics_box.setPlaceholderText("lyrics for this section…")
        self._lyrics_box.textChanged.connect(self._on_lyrics_changed)
        root.addWidget(self._lyrics_box)

        self._sync_shared_lyrics()

    # -----------------------------------------------------------------------

    @property
    def section_type(self) -> str:
        return self._type_combo.currentText()

    @property
    def energy(self) -> str:
        return self._energy_combo.currentText()

    @property
    def mood(self) -> str:
        return self._mood_combo.currentText().strip()

    @property
    def lyrics(self) -> str:
        stype = self.section_type
        if stype in SHARED_LYRIC_SECTIONS:
            return self._shared_lyrics.get(stype, "")
        return self._lyrics_box.toPlainText()

    # -----------------------------------------------------------------------

    def _on_type_changed(self, new_type: str) -> None:
        self._sync_shared_lyrics()
        self.changed.emit()

    def _toggle_lyrics(self, checked: bool) -> None:
        self._lyrics_toggle.setText(("▾" if checked else "▸") + " lyrics")
        self._lyrics_box.setVisible(checked)

    def _sync_shared_lyrics(self) -> None:
        """For shared sections (Chorus, Pre-Chorus), pull text from shared pool."""
        stype = self.section_type
        if stype in SHARED_LYRIC_SECTIONS:
            shared_text = self._shared_lyrics.get(stype, "")
            self._lyrics_box.blockSignals(True)
            self._lyrics_box.setPlainText(shared_text)
            self._lyrics_box.blockSignals(False)
            self._lyrics_box.setPlaceholderText(
                f"shared {stype.lower()} lyrics (used for all {stype} sections)…"
            )
        else:
            self._lyrics_box.setPlaceholderText("lyrics for this section…")

    def _on_lyrics_changed(self) -> None:
        stype = self.section_type
        if stype in SHARED_LYRIC_SECTIONS:
            # Write back to shared pool so all pills of this type stay in sync
            self._shared_lyrics[stype] = self._lyrics_box.toPlainText()
        self.changed.emit()


# ---------------------------------------------------------------------------
# Structure builder panel
# ---------------------------------------------------------------------------

class StructureBuilderWidget(QWidget):
    """
    Horizontal pill row + preset strip.

    Call serialise() to get (lyrics_text, caption_hint) ready for generation.
    """

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pills:         list[SectionPill] = []
        self._shared_lyrics: dict[str, str]    = {}   # shared pool for Chorus etc.
        self._verse_counter: int               = 0
        self._build_ui()

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # --- Preset strip ---
        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        preset_lbl = QLabel("Template:")
        preset_lbl.setObjectName("FieldLabel")
        preset_row.addWidget(preset_lbl)

        for name in STRUCTURE_PRESETS:
            btn = QPushButton(name)
            btn.setObjectName("PresetChipBtn")
            btn.clicked.connect(lambda _=False, n=name: self._apply_preset(n))
            preset_row.addWidget(btn)

        preset_row.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("ActionBtn")
        clear_btn.clicked.connect(self._clear_all)
        preset_row.addWidget(clear_btn)

        root.addLayout(preset_row)

        # --- Scrollable pill row ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedHeight(190)

        self._pill_container = QWidget()
        self._pill_layout = QHBoxLayout(self._pill_container)
        self._pill_layout.setContentsMargins(0, 0, 0, 0)
        self._pill_layout.setSpacing(6)
        self._pill_layout.setAlignment(Qt.AlignLeft)

        # "+" add button always at the end
        self._add_btn = QPushButton("+")
        self._add_btn.setObjectName("PillAddBtn")
        self._add_btn.setFixedSize(28, 28)
        self._add_btn.setToolTip("Add section")
        self._add_btn.clicked.connect(lambda: self._add_pill("Verse"))
        self._pill_layout.addWidget(self._add_btn)

        scroll.setWidget(self._pill_container)
        root.addWidget(scroll)

    # -----------------------------------------------------------------------
    # Pill management
    # -----------------------------------------------------------------------

    def _add_pill(
        self,
        section_type: str = "Verse",
        energy: str = "",
        mood: str = "",
        lyrics: str = "",
    ) -> SectionPill:
        pill = SectionPill(
            section_type   = section_type,
            energy         = energy,
            mood           = mood,
            shared_lyrics  = self._shared_lyrics,
        )
        pill.setFixedWidth(130)
        pill.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        pill.remove_requested.connect(self._remove_pill)
        pill.changed.connect(self.changed.emit)

        # If private lyrics supplied (non-shared section), set them
        if lyrics and section_type not in SHARED_LYRIC_SECTIONS:
            pill._lyrics_box.setPlainText(lyrics)
        elif lyrics and section_type in SHARED_LYRIC_SECTIONS:
            self._shared_lyrics[section_type] = lyrics

        # Insert before the + button
        pos = self._pill_layout.count() - 1
        self._pill_layout.insertWidget(pos, pill)
        self._pills.append(pill)
        self.changed.emit()
        return pill

    def _remove_pill(self, pill: SectionPill) -> None:
        if pill in self._pills:
            self._pills.remove(pill)
            self._pill_layout.removeWidget(pill)
            pill.deleteLater()
            self.changed.emit()

    def _clear_all(self) -> None:
        for pill in list(self._pills):
            self._pill_layout.removeWidget(pill)
            pill.deleteLater()
        self._pills.clear()
        self._shared_lyrics.clear()
        self.changed.emit()

    def _apply_preset(self, name: str) -> None:
        self._clear_all()
        for stype, energy, mood in STRUCTURE_PRESETS[name]:
            self._add_pill(stype, energy, mood)

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def serialise(self) -> tuple[str, str]:
        """
        Returns (lyrics_text, caption_hint).

        lyrics_text — the full [Section]\nlyrics\n... string to pass as
                      the lyrics field in TextGenerationRequest.
        caption_hint — energy/mood annotations to append to the style prompt,
                       e.g. "[Verse: Low energy, melancholy] [Chorus: High energy, euphoric]"
        """
        if not self._pills:
            return "", ""

        # Count section type occurrences to number them (Verse 1, Verse 2…)
        type_counts: dict[str, int] = {}
        lyrics_parts: list[str] = []
        caption_parts: list[str] = []

        for pill in self._pills:
            stype  = pill.section_type
            energy = pill.energy
            mood   = pill.mood
            lyr    = pill.lyrics.strip()

            # Numbered label for sections that repeat
            count = type_counts.get(stype, 0) + 1
            type_counts[stype] = count
            if count > 1 and stype not in SHARED_LYRIC_SECTIONS:
                label = f"{stype} {count}"
            else:
                label = stype

            # Lyrics block
            lyrics_parts.append(f"[{label}]")
            if lyr:
                lyrics_parts.append(lyr)

            # Caption hint
            hints = []
            if energy:
                hints.append(f"{energy} energy")
            if mood:
                hints.append(mood)
            if hints:
                caption_parts.append(f"[{label}: {', '.join(hints)}]")

        return "\n".join(lyrics_parts), " ".join(caption_parts)

    def is_empty(self) -> bool:
        return len(self._pills) == 0
