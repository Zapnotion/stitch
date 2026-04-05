"""
ui/widgets/caption_builder.py — Structured caption pre-builder (Phase 7.3 / quality improvement).

Lets the user pick genre, instruments, vocal style, mood, energy, and BPM
from searchable dropdowns populated from vocabulary.json.  Assembles them
into the exact comma-separated tag format ACEStep responds to, and writes
the result directly into the style prompt field.

Also provides one-click preset prompts for common styles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QCompleter, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

# ---------------------------------------------------------------------------
# Load vocabulary
# ---------------------------------------------------------------------------

def _load_vocab() -> dict:
    vocab_path = Path(__file__).parent.parent.parent / "data" / "vocabulary.json"
    try:
        return json.loads(vocab_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Searchable combo (QComboBox + QCompleter)
# ---------------------------------------------------------------------------

class SearchableCombo(QWidget):
    """A QLineEdit with autocomplete from a fixed item list. Acts like a combo."""

    value_changed = Signal(str)

    def __init__(self, items: list[str], placeholder: str = "type to search…",
                 parent=None) -> None:
        super().__init__(parent)
        self._items = items

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText(placeholder)
        self._edit.setObjectName("LineEdit")

        completer = QCompleter(items, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        self._edit.setCompleter(completer)
        self._edit.textChanged.connect(self.value_changed.emit)

        lay.addWidget(self._edit)

    def value(self) -> str:
        return self._edit.text().strip()

    def clear(self) -> None:
        self._edit.clear()

    def set_value(self, v: str) -> None:
        self._edit.setText(v)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class CaptionBuilderWidget(QWidget):
    """
    Structured caption builder.

    Signals:
        caption_ready(str) — emitted when the user clicks "Apply to prompt".
                             The receiver should set the style prompt field.
    """
    caption_ready = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._vocab = _load_vocab()
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 6, 0, 6)
        root.setSpacing(8)

        # --- Quick presets ---
        preset_hdr = QHBoxLayout()
        preset_lbl = QLabel("QUICK PRESETS")
        preset_lbl.setObjectName("SectionHeader")
        preset_hdr.addWidget(preset_lbl)
        preset_hdr.addStretch()
        root.addLayout(preset_hdr)

        presets_scroll = QScrollArea()
        presets_scroll.setWidgetResizable(True)
        presets_scroll.setFrameShape(QFrame.NoFrame)
        presets_scroll.setFixedHeight(38)
        presets_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        presets_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        presets_inner = QWidget()
        presets_lay = QHBoxLayout(presets_inner)
        presets_lay.setContentsMargins(0, 0, 0, 0)
        presets_lay.setSpacing(6)

        for p in self._vocab.get("preset_prompts", []):
            btn = QPushButton(p["name"])
            btn.setObjectName("PresetChipBtn")
            btn.setFixedHeight(26)
            _prompt = p["prompt"]
            btn.clicked.connect(lambda _, pr=_prompt: self._apply_preset(pr))
            presets_lay.addWidget(btn)
        presets_lay.addStretch()
        presets_scroll.setWidget(presets_inner)
        root.addWidget(presets_scroll)

        root.addWidget(self._divider())

        # --- Structured fields ---
        fields_lbl = QLabel("BUILD FROM PARTS")
        fields_lbl.setObjectName("SectionHeader")
        root.addWidget(fields_lbl)

        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        def field_label(text):
            l = QLabel(text)
            l.setObjectName("FieldLabel")
            l.setFixedWidth(80)
            return l

        row = 0
        # Genre
        self._genre = SearchableCombo(self._vocab.get("genres", []), "e.g. thrash metal")
        grid.addWidget(field_label("Genre"), row, 0)
        grid.addWidget(self._genre, row, 1)

        # Subgenre / second genre
        self._subgenre = SearchableCombo(self._vocab.get("genres", []), "optional blend…")
        grid.addWidget(field_label("+ Genre"), row, 2)
        grid.addWidget(self._subgenre, row, 3)
        row += 1

        # Instruments
        self._instruments = SearchableCombo(self._vocab.get("instruments", []), "e.g. distorted guitar")
        grid.addWidget(field_label("Instruments"), row, 0)
        grid.addWidget(self._instruments, row, 1)

        self._instruments2 = SearchableCombo(self._vocab.get("instruments", []), "add more…")
        grid.addWidget(field_label("+ More"), row, 2)
        grid.addWidget(self._instruments2, row, 3)
        row += 1

        # Vocals
        self._vocals = SearchableCombo(self._vocab.get("vocal_styles", []), "e.g. raspy male vocals")
        grid.addWidget(field_label("Vocals"), row, 0)
        grid.addWidget(self._vocals, row, 1)

        # Era
        self._era = SearchableCombo(self._vocab.get("eras", []), "e.g. 1980s")
        grid.addWidget(field_label("Era"), row, 2)
        grid.addWidget(self._era, row, 3)
        row += 1

        # Mood
        self._mood = SearchableCombo(self._vocab.get("moods", []), "e.g. aggressive, dark")
        grid.addWidget(field_label("Mood"), row, 0)
        grid.addWidget(self._mood, row, 1)

        # Energy
        self._energy_combo = SearchableCombo(self._vocab.get("energy", []), "e.g. driving, fast")
        grid.addWidget(field_label("Energy"), row, 2)
        grid.addWidget(self._energy_combo, row, 3)
        row += 1

        # Production
        self._production = SearchableCombo(self._vocab.get("production", []), "e.g. lo-fi, vintage")
        grid.addWidget(field_label("Production"), row, 0)
        grid.addWidget(self._production, row, 1)

        # BPM (numeric)
        bpm_row = QHBoxLayout()
        self._bpm_input = QLineEdit()
        self._bpm_input.setObjectName("LineEdit")
        self._bpm_input.setPlaceholderText("e.g. 140")
        self._bpm_input.setFixedWidth(70)
        bpm_row.addWidget(self._bpm_input)
        bpm_row.addWidget(QLabel("BPM"))
        bpm_row.addStretch()
        bpm_container = QWidget()
        bpm_container.setLayout(bpm_row)
        grid.addWidget(field_label("BPM"), row, 2)
        grid.addWidget(bpm_container, row, 3)
        row += 1

        root.addLayout(grid)

        # --- Preview ---
        root.addWidget(self._divider())

        prev_hdr = QHBoxLayout()
        prev_lbl = QLabel("ASSEMBLED PROMPT")
        prev_lbl.setObjectName("SectionHeader")
        prev_hdr.addWidget(prev_lbl)
        prev_hdr.addStretch()

        self._preview_lbl = QLabel("Fill in fields above…")
        self._preview_lbl.setObjectName("CaptionPreview")
        self._preview_lbl.setWordWrap(True)
        self._preview_lbl.setMinimumHeight(36)

        self._clear_btn = QPushButton("Clear all")
        self._clear_btn.setObjectName("ActionBtn")
        self._clear_btn.clicked.connect(self._clear_all)

        self._apply_btn = QPushButton("Apply to prompt ▸")
        self._apply_btn.setObjectName("GenBtn")
        self._apply_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._apply_btn.clicked.connect(self._on_apply)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addWidget(self._clear_btn)
        btn_row.addWidget(self._apply_btn)

        root.addLayout(prev_hdr)
        root.addWidget(self._preview_lbl)
        root.addLayout(btn_row)

        # Wire live preview
        for field in (self._genre, self._subgenre, self._instruments,
                      self._instruments2, self._vocals, self._era,
                      self._mood, self._energy_combo, self._production):
            field.value_changed.connect(self._update_preview)
        self._bpm_input.textChanged.connect(self._update_preview)

    # ---- Helpers -----------------------------------------------------------

    def _divider(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setObjectName("Divider")
        return f

    def _assemble(self) -> str:
        parts = []
        for field in (self._genre, self._subgenre, self._instruments,
                      self._instruments2, self._vocals, self._era,
                      self._mood, self._energy_combo, self._production):
            v = field.value()
            if v:
                parts.append(v)
        bpm = self._bpm_input.text().strip()
        if bpm.isdigit():
            parts.append(f"{bpm} BPM")
        return ", ".join(parts)

    # ---- Slots -------------------------------------------------------------

    def _update_preview(self) -> None:
        caption = self._assemble()
        self._preview_lbl.setText(caption if caption else "Fill in fields above…")

    def _on_apply(self) -> None:
        caption = self._assemble()
        if caption:
            self.caption_ready.emit(caption)

    def _apply_preset(self, prompt: str) -> None:
        self.caption_ready.emit(prompt)

    def _clear_all(self) -> None:
        for field in (self._genre, self._subgenre, self._instruments,
                      self._instruments2, self._vocals, self._era,
                      self._mood, self._energy_combo, self._production):
            field.clear()
        self._bpm_input.clear()
        self._update_preview()
