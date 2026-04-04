"""
ui/widgets/lyric_timeline.py — Lyric timeline widget for Phase 1.

Displays word-level lyric alignment with:
  • Colour coding by confidence (white / amber / red)
  • Playback cursor that highlights the current word in real time
  • Clickable words that seek the audio player
  • Section dividers ([Verse 1], [Chorus] etc.)
  • Confidence flagging banner (1.3)
  • Right-click context menu per word: Repair / Edit pronunciation (1.4, 1.7)
  • Right-click context menu per section: Regenerate / Lock (1.5, 1.6)

Attached below the waveform inside ResultCard when an alignment sidecar exists.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor, QFont, QTextCharFormat, QTextCursor, QTextOption,
)
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QInputDialog, QLabel,
    QMenu, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from app.config import cfg

# ---------------------------------------------------------------------------
# Colour constants (stay in sync with stylesheet palette)
# ---------------------------------------------------------------------------

_COL_HIGH       = QColor("#DEDEDE")   # confident word
_COL_AMBER      = QColor("#E8A84B")   # borderline
_COL_RED        = QColor("#E85D5D")   # unclear
_COL_CURRENT    = QColor("#FFFFFF")   # currently playing word (bold)
_COL_SECTION_BG = QColor("#1E1E1E")   # section divider background
_COL_SECTION_FG = QColor("#555555")   # section divider text
_COL_LOCKED_FG  = QColor("#4A7A4A")   # locked section indicator


# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

class _Word:
    __slots__ = ("word", "start", "end", "confidence", "char_start", "char_end",
                 "locked", "phoneme_hint")

    def __init__(self, word: str, start: float, end: float, confidence: float) -> None:
        self.word         = word
        self.start        = start
        self.end          = end
        self.confidence   = confidence
        self.char_start   = 0   # position in QTextEdit document (set during render)
        self.char_end     = 0
        self.locked       = False
        self.phoneme_hint = ""  # e.g. "MAO-ten" for "mountain"


class _Section:
    __slots__ = ("label", "start", "end", "char_pos", "locked")

    def __init__(self, label: str, start: float, end: float) -> None:
        self.label    = label
        self.start    = start
        self.end      = end
        self.char_pos = 0   # char position of divider in document
        self.locked   = False


# ---------------------------------------------------------------------------
# Confidence flagging banner (Feature 1.3)
# ---------------------------------------------------------------------------

class _ConfidenceBanner(QWidget):
    """
    Thin info bar: "3 words may be unclear — click to review"
    Clicking jumps to first flagged word. Has a × dismiss button.
    """
    review_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ConfidenceBanner")
        self.setVisible(False)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)

        self._lbl = QLabel()
        self._lbl.setObjectName("BannerLabel")
        self._lbl.setCursor(Qt.PointingHandCursor)
        self._lbl.mousePressEvent = lambda _e: self.review_requested.emit()

        self._dismiss = QPushButton("×")
        self._dismiss.setObjectName("BannerDismiss")
        self._dismiss.setFixedSize(18, 18)
        self._dismiss.setFlat(True)
        self._dismiss.clicked.connect(self.hide)

        lay.addWidget(self._lbl, 1)
        lay.addWidget(self._dismiss)

    def set_count(self, count: int) -> None:
        if count == 0:
            self.hide()
            return
        noun = "word" if count == 1 else "words"
        self._lbl.setText(f"{count} {noun} may be unclear — click to review")
        self.show()


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class LyricTimelineWidget(QWidget):
    """
    Full lyric timeline display. Attach to a ResultCard when an alignment
    sidecar is available.

    Signals:
        seek_requested(float)  — emitted when the user clicks a word;
                                 the float is the word's start time in seconds.
        repair_word_requested(float, float, str)
                               — (region_start, region_end, intended_word)
                                  raised when the user right-clicks → Repair
        regenerate_section_requested(float, float, str)
                               — (section_start, section_end, section_label)
    """

    seek_requested               = Signal(float)
    repair_word_requested        = Signal(float, float, str)   # start, end, word
    regenerate_section_requested = Signal(float, float, str)   # start, end, label

    # Minimum repair window in seconds (per brief §1.4)
    _MIN_REPAIR_WINDOW = 0.3
    _REPAIR_PADDING    = 0.05

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._words:    list[_Word]    = []
        self._sections: list[_Section] = []
        self._current_word_idx: int    = -1

        self._position_timer = QTimer(self)
        self._position_timer.setInterval(50)   # 50ms — matches brief §1.2
        self._position_timer.timeout.connect(self._tick)
        self._current_sec: float = 0.0

        self._build_ui()

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._banner = _ConfidenceBanner()
        self._banner.review_requested.connect(self._jump_to_first_flagged)
        root.addWidget(self._banner)

        self._text = QTextEdit()
        self._text.setObjectName("LyricTimeline")
        self._text.setReadOnly(True)
        self._text.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self._text.setMinimumHeight(80)
        self._text.setMaximumHeight(220)
        self._text.setContextMenuPolicy(Qt.CustomContextMenu)
        self._text.customContextMenuRequested.connect(self._on_context_menu)
        self._text.mousePressEvent = self._on_text_click
        root.addWidget(self._text)

    # -----------------------------------------------------------------------
    # Public API — load sidecar
    # -----------------------------------------------------------------------

    def load_alignment(self, sidecar: dict) -> None:
        """
        Populate the widget from a sidecar dict.
        Call after creating or refreshing a result card.
        """
        self._words    = []
        self._sections = []
        self._current_word_idx = -1

        for w in sidecar.get("words", []):
            self._words.append(_Word(
                word       = w.get("word", ""),
                start      = float(w.get("start", 0.0)),
                end        = float(w.get("end",   0.0)),
                confidence = float(w.get("confidence", 1.0)),
            ))

        for s in sidecar.get("sections", []):
            self._sections.append(_Section(
                label = s.get("label", ""),
                start = float(s.get("start", 0.0)),
                end   = float(s.get("end",   0.0)),
            ))

        self._render()
        self._update_banner()

    # -----------------------------------------------------------------------
    # Playback position hook
    # -----------------------------------------------------------------------

    def set_position(self, sec: float) -> None:
        """Called every ~50ms from AudioPlayer. Updates highlighted word."""
        self._current_sec = sec
        self._tick()

    def start_tracking(self) -> None:
        self._position_timer.start()

    def stop_tracking(self) -> None:
        self._position_timer.stop()
        # Remove current-word highlight
        self._highlight_word(-1)

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def _render(self) -> None:
        """Build the full document from words + section dividers."""
        threshold = cfg.get("alignment_confidence_threshold", 0.6)

        doc = self._text
        doc.clear()

        cursor = doc.textCursor()
        cursor.movePosition(QTextCursor.Start)

        # We'll interleave sections and words by time order.
        # Build a flat event list: ("word", _Word) or ("section", _Section)
        events: list = []
        w_idx = 0
        for sec in self._sections:
            # Add all words that start before this section's start time
            while w_idx < len(self._words) and self._words[w_idx].start < sec.start:
                events.append(("word", self._words[w_idx]))
                w_idx += 1
            events.append(("section", sec))
        while w_idx < len(self._words):
            events.append(("word", self._words[w_idx]))
            w_idx += 1

        for kind, item in events:
            if kind == "section":
                self._insert_section_divider(cursor, item)
            else:
                self._insert_word(cursor, item, threshold)

        doc.setTextCursor(cursor)

    def _insert_word(self, cursor: QTextCursor, word: _Word, threshold: float) -> None:
        fmt = QTextCharFormat()
        if word.confidence >= threshold:
            fmt.setForeground(_COL_HIGH)
        elif word.confidence >= 0.4:
            fmt.setForeground(_COL_AMBER)
        else:
            fmt.setForeground(_COL_RED)

        word.char_start = cursor.position()
        cursor.insertText(word.word + " ", fmt)
        word.char_end = cursor.position() - 1   # exclude trailing space

    def _insert_section_divider(self, cursor: QTextCursor, section: _Section) -> None:
        # Newline before
        nl_fmt = QTextCharFormat()
        nl_fmt.setBackground(_COL_SECTION_BG)
        cursor.insertBlock()

        sec_fmt = QTextCharFormat()
        sec_fmt.setBackground(_COL_SECTION_BG)
        sec_fmt.setForeground(_COL_LOCKED_FG if section.locked else _COL_SECTION_FG)
        sec_fmt.setFontWeight(QFont.Bold)
        sec_fmt.setFontPointSize(10)

        lock_icon = " 🔒" if section.locked else ""
        section.char_pos = cursor.position()
        cursor.insertText(f"[ {section.label}{lock_icon} ]", sec_fmt)
        cursor.insertBlock()

    # -----------------------------------------------------------------------
    # Playback cursor
    # -----------------------------------------------------------------------

    def _tick(self) -> None:
        sec = self._current_sec
        new_idx = -1
        for i, w in enumerate(self._words):
            if w.start <= sec <= w.end:
                new_idx = i
                break
            if w.start > sec:
                break
        if new_idx != self._current_word_idx:
            self._highlight_word(new_idx)

    def _highlight_word(self, idx: int) -> None:
        # Restore previous word's normal colour
        if 0 <= self._current_word_idx < len(self._words):
            prev = self._words[self._current_word_idx]
            self._apply_format(prev, is_current=False)

        self._current_word_idx = idx

        if 0 <= idx < len(self._words):
            curr = self._words[idx]
            self._apply_format(curr, is_current=True)
            # Scroll to keep current word visible
            cursor = self._text.textCursor()
            cursor.setPosition(curr.char_start)
            self._text.setTextCursor(cursor)
            self._text.ensureCursorVisible()

    def _apply_format(self, word: _Word, is_current: bool) -> None:
        threshold = cfg.get("alignment_confidence_threshold", 0.6)
        cursor = self._text.textCursor()
        cursor.setPosition(word.char_start)
        cursor.setPosition(word.char_end, QTextCursor.KeepAnchor)

        fmt = QTextCharFormat()
        if is_current:
            fmt.setForeground(_COL_CURRENT)
            fmt.setFontWeight(QFont.Bold)
        else:
            if word.confidence >= threshold:
                fmt.setForeground(_COL_HIGH)
            elif word.confidence >= 0.4:
                fmt.setForeground(_COL_AMBER)
            else:
                fmt.setForeground(_COL_RED)
            fmt.setFontWeight(QFont.Normal)

        cursor.mergeCharFormat(fmt)

    # -----------------------------------------------------------------------
    # Click → seek
    # -----------------------------------------------------------------------

    def _on_text_click(self, event) -> None:
        # Pass to default handler first so cursor moves
        QTextEdit.mousePressEvent(self._text, event)
        if event.button() != Qt.LeftButton:
            return
        pos = self._text.textCursor().position()
        word = self._word_at_char(pos)
        if word:
            self.seek_requested.emit(word.start)

    def _word_at_char(self, pos: int) -> Optional[_Word]:
        for w in self._words:
            if w.char_start <= pos <= w.char_end:
                return w
        return None

    def _section_at_char(self, pos: int) -> Optional[_Section]:
        for s in self._sections:
            if abs(s.char_pos - pos) < 30:   # rough proximity
                return s
        return None

    # -----------------------------------------------------------------------
    # Context menu (Features 1.4, 1.5, 1.6, 1.7)
    # -----------------------------------------------------------------------

    def _on_context_menu(self, point) -> None:
        cursor = self._text.cursorForPosition(point)
        char_pos = cursor.position()
        word    = self._word_at_char(char_pos)
        section = self._section_at_char(char_pos)

        menu = QMenu(self)

        if word:
            self._build_word_menu(menu, word)

        if section:
            if word:
                menu.addSeparator()
            self._build_section_menu(menu, section)

        if not word and not section:
            menu.addAction("No word at cursor").setEnabled(False)

        menu.exec(self._text.mapToGlobal(point))

    def _build_word_menu(self, menu: QMenu, word: _Word) -> None:
        # --- Repair this word (1.4) ---
        repair_act = menu.addAction(f'Repair "{word.word}"')
        repair_act.triggered.connect(lambda: self._repair_word(word))

        # --- Edit pronunciation (1.7) ---
        phoneme_act = menu.addAction("Edit pronunciation…")
        phoneme_act.triggered.connect(lambda: self._edit_phoneme(word))
        if word.phoneme_hint:
            phoneme_act.setText(f"Edit pronunciation  [{word.phoneme_hint}]")

        # --- Clear phoneme hint ---
        if word.phoneme_hint:
            clear_act = menu.addAction("Clear pronunciation hint")
            clear_act.triggered.connect(lambda: self._clear_phoneme(word))

        # --- Lock this word (1.6) ---
        lock_act = menu.addAction("Lock word" if not word.locked else "Unlock word")
        lock_act.triggered.connect(lambda: self._toggle_word_lock(word))

    def _build_section_menu(self, menu: QMenu, section: _Section) -> None:
        # --- Regenerate this section (1.5) ---
        regen_label = f'Regenerate "{section.label}"'
        if section.locked:
            regen_label += "  (locked — unlock first)"
        regen_act = menu.addAction(regen_label)
        regen_act.setEnabled(not section.locked)
        regen_act.triggered.connect(lambda: self._regenerate_section(section))

        # --- Lock / unlock section (1.5, 1.6) ---
        lock_label = f'Unlock "{section.label}"' if section.locked else f'Lock "{section.label}"'
        lock_act = menu.addAction(lock_label)
        lock_act.triggered.connect(lambda: self._toggle_section_lock(section))

    # -----------------------------------------------------------------------
    # Word repair (1.4)
    # -----------------------------------------------------------------------

    def _repair_word(self, word: _Word) -> None:
        # Pad the region by ±50ms, enforce minimum 300ms window (per brief)
        start = max(0.0, word.start - self._REPAIR_PADDING)
        end   = word.end + self._REPAIR_PADDING
        if (end - start) < self._MIN_REPAIR_WINDOW:
            mid = (start + end) / 2
            start = max(0.0, mid - self._MIN_REPAIR_WINDOW / 2)
            end   = start + self._MIN_REPAIR_WINDOW

        # Include phoneme hint in the intended word string if set (1.7)
        intended = f"[{word.phoneme_hint}]{word.word}" if word.phoneme_hint else word.word

        self.repair_word_requested.emit(start, end, intended)

    # -----------------------------------------------------------------------
    # Phoneme edit (1.7)
    # -----------------------------------------------------------------------

    def _edit_phoneme(self, word: _Word) -> None:
        current = word.phoneme_hint
        text, ok = QInputDialog.getText(
            self,
            "Edit pronunciation",
            f'Enter a phonetic respelling for "{word.word}"\n'
            f'e.g. "MAO-ten" for "mountain"',
            text=current,
        )
        if ok:
            word.phoneme_hint = text.strip()

    def _clear_phoneme(self, word: _Word) -> None:
        word.phoneme_hint = ""

    # -----------------------------------------------------------------------
    # Word lock (1.6)
    # -----------------------------------------------------------------------

    def _toggle_word_lock(self, word: _Word) -> None:
        word.locked = not word.locked
        # Re-render to reflect lock state in tooltip / visual (future: show lock icon)

    # -----------------------------------------------------------------------
    # Section regeneration (1.5)
    # -----------------------------------------------------------------------

    def _regenerate_section(self, section: _Section) -> None:
        self.regenerate_section_requested.emit(section.start, section.end, section.label)

    # -----------------------------------------------------------------------
    # Section lock (1.5, 1.6)
    # -----------------------------------------------------------------------

    def _toggle_section_lock(self, section: _Section) -> None:
        section.locked = not section.locked
        self._render()   # re-render to show/hide lock icon

    def locked_sections(self) -> list[dict]:
        """Return list of locked section dicts — used by caller to build [LOCKED] tags."""
        return [
            {"label": s.label, "start": s.start, "end": s.end}
            for s in self._sections if s.locked
        ]

    def locked_words(self) -> list[str]:
        """Return list of locked word strings."""
        return [w.word for w in self._words if w.locked]

    # -----------------------------------------------------------------------
    # Confidence flagging banner (1.3)
    # -----------------------------------------------------------------------

    def _update_banner(self) -> None:
        threshold = cfg.get("alignment_confidence_threshold", 0.6)
        flagged = sum(1 for w in self._words if w.confidence < threshold)
        self._banner.set_count(flagged)

    def _jump_to_first_flagged(self) -> None:
        threshold = cfg.get("alignment_confidence_threshold", 0.6)
        for w in self._words:
            if w.confidence < threshold:
                self.seek_requested.emit(w.start)
                # Scroll to word
                cursor = self._text.textCursor()
                cursor.setPosition(w.char_start)
                self._text.setTextCursor(cursor)
                self._text.ensureCursorVisible()
                return
