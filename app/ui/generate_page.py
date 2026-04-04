"""
ui/generate_page.py — Generate tab: Text prompt / Cover / Vocal backing modes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSlider, QSpinBox, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget,
)

from app.backend.ace_step_v15 import ACEStepV15 as ACEStepPipeline
from app.backend.separator import StemSeparator as DemucsSeparator
from app.backend.worker import (
    CoverWorker, StemWorker, TextGenerationWorker,
    VocalBackingWorker,
)
from app.backend.exporter import ExportWorker
from app.backend.presets import presets
from app.backend.session import session
from app.config import cfg
from app.models.generation import (
    CoverRequest, GenerationResult,
    LyricsMode, OutputType,
    TextGenerationRequest, VocalBackingRequest,
)
from app.ui.widgets.preset_panel import PresetPanel
from app.ui.widgets.result_card import ResultCard
from app.ui.widgets.structure_builder import StructureBuilderWidget
from app.ui.widgets.upload_zone import UploadZone


class GeneratePage(QWidget):
    status_message    = Signal(str)
    vram_updated      = Signal(float)
    open_in_repair    = Signal(str)   # audio_path — tells main window to switch to repair tab with this file

    def __init__(
        self,
        pipeline: ACEStepPipeline,
        separator: DemucsSeparator,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline  = pipeline
        self._separator = separator
        self._workers: list = []
        self._result_cards: list[ResultCard] = []
        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)

        # --- Left panel ---
        left = QWidget()
        left.setObjectName("LeftPanel")
        left.setMinimumWidth(300)
        left.setMaximumWidth(600)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # Mode tabs
        tab_row = QWidget()
        tab_row.setObjectName("ModeTabs")
        tab_layout = QHBoxLayout(tab_row)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        self._tab_btns: dict[str, QPushButton] = {}
        for mode, label in [("text", "Text prompt"), ("cover", "Cover / restyle"), ("vox", "Vocal + backing")]:
            btn = QPushButton(label)
            btn.setObjectName("ModeTab")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, m=mode: self._switch_mode(m))
            tab_layout.addWidget(btn)
            self._tab_btns[mode] = btn
        self._tab_btns["text"].setChecked(True)

        left_layout.addWidget(tab_row)

        # Stacked panels — wrapped in a scroll area so the Generate button stays
        # pinned at the bottom regardless of how much form content is visible.
        self._stack = QStackedWidget()
        self._panels = {
            "text":  self._build_text_panel(),
            "cover": self._build_cover_panel(),
            "vox":   self._build_vox_panel(),
        }
        for w in self._panels.values():
            self._stack.addWidget(w)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(self._stack)
        left_layout.addWidget(left_scroll, 1)

        # --- Preset panel (collapsible strip) ---
        preset_toggle = QPushButton("▸  Presets")
        preset_toggle.setObjectName("PresetToggle")
        preset_toggle.setCheckable(True)
        preset_toggle.setChecked(False)
        left_layout.addWidget(preset_toggle)

        self._preset_panel = PresetPanel()
        self._preset_panel.setVisible(False)
        self._preset_panel.setMaximumHeight(200)
        self._preset_panel.setContentsMargins(12, 6, 12, 6)
        self._preset_panel.set_mode_getter(
            lambda: next(m for m, b in self._tab_btns.items() if b.isChecked())
        )
        self._preset_panel.set_params_getter(self._collect_current_params)
        self._preset_panel.preset_loaded.connect(self._on_preset_loaded)
        self._preset_panel.preset_saved.connect(
            lambda name: self.status_message.emit(f"Preset saved: {name}")
        )
        left_layout.addWidget(self._preset_panel)

        def _toggle_presets(checked: bool) -> None:
            preset_toggle.setText(("▾" if checked else "▸") + "  Presets")
            self._preset_panel.setVisible(checked)
            if checked:
                self._preset_panel.refresh()

        preset_toggle.toggled.connect(_toggle_presets)

        # Generate button
        self._gen_btn = QPushButton("Generate ×4")
        self._gen_btn.setObjectName("GenBtn")
        self._gen_btn.clicked.connect(self._on_generate)
        left_layout.addWidget(self._gen_btn)

        self._hint_lbl = QLabel(f"ACE-Step · ~10 s each on RTX 3090")
        self._hint_lbl.setObjectName("HintLabel")
        self._hint_lbl.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self._hint_lbl)

        splitter.addWidget(left)

        # --- Right panel ---
        right = QWidget()
        right.setObjectName("RightPanel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        hdr = QWidget()
        hdr.setObjectName("RightHeader")
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(16, 12, 16, 12)
        self._results_title = QLabel("Results")
        self._results_title.setObjectName("RightTitle")
        self._results_sub = QLabel("text prompt · 4 variations")
        self._results_sub.setObjectName("RightSub")
        hdr_layout.addWidget(self._results_title)
        hdr_layout.addWidget(self._results_sub)
        hdr_layout.addStretch()

        # Sort control
        sort_lbl = QLabel("Sort:")
        sort_lbl.setObjectName("FieldLabel")
        self._sort_combo = QComboBox()
        self._sort_combo.setObjectName("SortCombo")
        self._sort_combo.addItems(["Creation order", "Starred first", "Duration ↑", "Duration ↓"])
        self._sort_combo.setFixedWidth(130)
        self._sort_combo.currentIndexChanged.connect(self._apply_sort)
        hdr_layout.addWidget(sort_lbl)
        hdr_layout.addWidget(self._sort_combo)

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("ActionBtn")
        clear_btn.clicked.connect(self._clear_results)
        hdr_layout.addWidget(clear_btn)

        right_layout.addWidget(hdr)

        # Progress bar (hidden when idle)
        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("GenProgress")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(3)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setVisible(False)
        right_layout.addWidget(self._progress_bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self._cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_widget)
        self._cards_layout.setContentsMargins(14, 14, 14, 14)
        self._cards_layout.setSpacing(10)
        self._cards_layout.addStretch()

        # Empty state
        self._empty_lbl = QLabel("Generate something to see results here.")
        self._empty_lbl.setObjectName("EmptyState")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._cards_layout.insertWidget(0, self._empty_lbl)

        scroll.setWidget(self._cards_widget)
        right_layout.addWidget(scroll, 1)

        splitter.addWidget(right)
        splitter.setSizes([340, 720])
        splitter.setStretchFactor(0, 0)   # left: don't auto-stretch
        splitter.setStretchFactor(1, 1)   # right: takes all extra space
        splitter.setHandleWidth(4)         # visible drag handle
        splitter.setChildrenCollapsible(False)

        root.addWidget(splitter)

    # --- Text panel ---------------------------------------------------------

    def _build_text_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)

        lay.addWidget(self._field_label("Output type"))
        self._output_type_row = self._pill_row(["With vocals", "Instrumental"])
        lay.addWidget(self._output_type_row)

        lay.addWidget(self._field_label("Lyrics"))
        self._lyrics_mode_row = self._pill_row(["AI writes", "I provide", "Instrumental"])
        # AI writes = v1.5 LLM plans lyrics via CoT; I provide = pass directly to DiT
        lay.addWidget(self._lyrics_mode_row)

        # AI lyrics prompt box — shown when "AI writes" is selected.
        # Tells the LM what to write about (theme, subject, mood, language).
        # If left blank, the style prompt is used as the query instead.
        self._ai_lyrics_prompt = QPlainTextEdit()
        self._ai_lyrics_prompt.setObjectName("PromptBox")
        self._ai_lyrics_prompt.setPlaceholderText(
            "What should the AI sing about?\n"
            "e.g. longing for someone far away, sung in English\n"
            "e.g. a triumphant anthem about overcoming doubt\n"
            "Leave blank to let the style prompt guide the lyrics."
        )
        self._ai_lyrics_prompt.setFixedHeight(90)
        self._ai_lyrics_prompt.setVisible(True)   # AI writes is default
        lay.addWidget(self._ai_lyrics_prompt)

        # --- AI writes controls (creativity + adherence sliders) ---
        # Shown only when "AI writes" pill is active, collapsed by default.
        self._ai_controls = QWidget()
        self._ai_controls.setObjectName("AIControlsBox")
        ai_lay = QVBoxLayout(self._ai_controls)
        ai_lay.setContentsMargins(0, 0, 0, 0)
        ai_lay.setSpacing(4)

        def _slider_row(label_left: str, label_right: str,
                        lo: int, hi: int, val: int) -> tuple:
            """Return (container_widget, QSlider, value_label)."""
            row_w = QWidget()
            row_l = QVBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(2)

            # Top row: left label + value + right label
            hdr = QHBoxLayout()
            lbl_l = QLabel(label_left)
            lbl_l.setObjectName("FieldLabel")
            lbl_r = QLabel(label_right)
            lbl_r.setObjectName("FieldLabelDim")
            val_lbl = QLabel(f"{val / 10:.1f}")
            val_lbl.setObjectName("SliderValue")
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            hdr.addWidget(lbl_l)
            hdr.addStretch()
            hdr.addWidget(val_lbl)
            row_l.addLayout(hdr)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(val)
            slider.setObjectName("LyricsSlider")
            slider.valueChanged.connect(
                lambda v, vl=val_lbl: vl.setText(f"{v / 10:.1f}")
            )
            row_l.addWidget(slider)

            # Bottom hint row
            hint_row = QHBoxLayout()
            hl = QLabel(label_left.split(" ")[0])
            hl.setObjectName("SliderHint")
            hr = QLabel(label_right.split(" ")[0])
            hr.setObjectName("SliderHint")
            hint_row.addWidget(hl)
            hint_row.addStretch()
            hint_row.addWidget(hr)
            row_l.addLayout(hint_row)

            return row_w, slider, val_lbl

        # "Advanced" toggle — reveals the two AI sliders
        self._ai_advanced_toggle = QPushButton("▸  Advanced")
        self._ai_advanced_toggle.setObjectName("AdvancedToggle")
        self._ai_advanced_toggle.setCheckable(True)
        self._ai_advanced_toggle.setChecked(False)

        self._ai_sliders_box = QWidget()
        self._ai_sliders_box.setVisible(False)
        sliders_lay = QVBoxLayout(self._ai_sliders_box)
        sliders_lay.setContentsMargins(0, 4, 0, 0)
        sliders_lay.setSpacing(6)

        creativity_w, self._creativity_slider, self._creativity_val = _slider_row(
            "Safe", "Wild", 0, 10, 5   # default 0.5
        )
        creativity_label = QLabel("Creativity")
        creativity_label.setObjectName("FieldLabel")
        sliders_lay.addWidget(creativity_label)
        sliders_lay.addWidget(creativity_w)

        adherence_w, self._adherence_slider, self._adherence_val = _slider_row(
            "Loose", "Strict", 0, 10, 7   # default 0.7
        )
        adherence_label = QLabel("Topic adherence")
        adherence_label.setObjectName("FieldLabel")
        sliders_lay.addWidget(adherence_label)
        sliders_lay.addWidget(adherence_w)

        def _toggle_ai_advanced(checked: bool) -> None:
            self._ai_advanced_toggle.setText(("▾" if checked else "▸") + "  Advanced")
            self._ai_sliders_box.setVisible(checked)

        self._ai_advanced_toggle.toggled.connect(_toggle_ai_advanced)

        ai_lay.addWidget(self._ai_advanced_toggle)
        ai_lay.addWidget(self._ai_sliders_box)

        lay.addWidget(self._ai_controls)

        # User lyrics box — shown only when "I provide" is selected
        self._user_lyrics = QPlainTextEdit()
        self._user_lyrics.setObjectName("PromptBox")
        self._user_lyrics.setPlaceholderText(
            "[Verse 1]\nYour lyrics here…\n\n[Chorus]\n…"
        )
        self._user_lyrics.setFixedHeight(110)
        self._user_lyrics.setVisible(False)
        lay.addWidget(self._user_lyrics)

        # Toggle lyrics boxes + AI controls based on pill selection:
        #   pill[0] = AI writes  → show ai_lyrics_prompt + ai_controls, hide user_lyrics
        #   pill[1] = I provide  → hide ai_lyrics_prompt + ai_controls, show user_lyrics
        #   pill[2] = Instrumental → hide all
        def _on_lyrics_pill():
            pills = self._lyrics_mode_row.findChildren(QPushButton)
            ai_checked   = len(pills) > 0 and pills[0].isChecked()
            user_checked = len(pills) > 1 and pills[1].isChecked()
            self._ai_lyrics_prompt.setVisible(ai_checked)
            self._ai_controls.setVisible(ai_checked)
            self._user_lyrics.setVisible(user_checked)
        for btn in self._lyrics_mode_row.findChildren(QPushButton):
            btn.clicked.connect(_on_lyrics_pill)

        lay.addWidget(self._field_label("Style prompt"))
        self._style_prompt = QPlainTextEdit()
        self._style_prompt.setObjectName("PromptBox")
        self._style_prompt.setPlaceholderText("genre, mood, instruments, vocal style, era…")
        self._style_prompt.setFixedHeight(90)
        lay.addWidget(self._style_prompt)

        # --- Phase 2.4: Micro-genre blend (toggle between blend and manual) ---
        genre_hdr_row = QHBoxLayout()
        genre_lbl = self._field_label("Genre blend")
        self._genre_manual_btn = QPushButton("Manual ▸")
        self._genre_manual_btn.setObjectName("AdvancedToggle")
        self._genre_manual_btn.setCheckable(True)
        self._genre_manual_btn.setChecked(False)
        genre_hdr_row.addWidget(genre_lbl)
        genre_hdr_row.addStretch()
        genre_hdr_row.addWidget(self._genre_manual_btn)
        lay.addLayout(genre_hdr_row)

        self._genre_blend_box = self._build_genre_blend_widget()
        lay.addWidget(self._genre_blend_box)

        # --- Phase 2.3: Exclusions (negative prompt) ---
        self._exclusions_toggle = QPushButton("▸  Exclusions")
        self._exclusions_toggle.setObjectName("AdvancedToggle")
        self._exclusions_toggle.setCheckable(True)
        self._exclusions_toggle.setChecked(False)
        lay.addWidget(self._exclusions_toggle)

        self._exclusions_box = QWidget()
        excl_lay = QVBoxLayout(self._exclusions_box)
        excl_lay.setContentsMargins(0, 0, 0, 0)
        excl_lay.setSpacing(2)
        self._exclusions = QPlainTextEdit()
        self._exclusions.setObjectName("PromptBox")
        self._exclusions.setPlaceholderText("no piano, no autotune, no trap beat")
        self._exclusions.setFixedHeight(52)
        excl_lay.addWidget(self._exclusions)
        self._exclusions_box.setVisible(False)
        lay.addWidget(self._exclusions_box)

        def _toggle_exclusions(checked: bool) -> None:
            self._exclusions_toggle.setText(("▾" if checked else "▸") + "  Exclusions")
            self._exclusions_box.setVisible(checked)
        self._exclusions_toggle.toggled.connect(_toggle_exclusions)

        # --- Phase 2.2: Song structure builder ---
        self._structure_toggle = QPushButton("▸  Song structure")
        self._structure_toggle.setObjectName("AdvancedToggle")
        self._structure_toggle.setCheckable(True)
        self._structure_toggle.setChecked(False)
        lay.addWidget(self._structure_toggle)

        self._structure_builder = StructureBuilderWidget()
        self._structure_builder.setVisible(False)
        lay.addWidget(self._structure_builder)

        def _toggle_structure(checked: bool) -> None:
            self._structure_toggle.setText(("▾" if checked else "▸") + "  Song structure")
            self._structure_builder.setVisible(checked)
        self._structure_toggle.toggled.connect(_toggle_structure)

        # --- Phase 2.1: Musical parameters (BPM, key, time signature, ref track) ---
        self._musical_toggle = QPushButton("▸  Musical parameters")
        self._musical_toggle.setObjectName("AdvancedToggle")
        self._musical_toggle.setCheckable(True)
        self._musical_toggle.setChecked(False)
        lay.addWidget(self._musical_toggle)

        self._musical_box = self._build_musical_params_widget()
        self._musical_box.setVisible(False)
        lay.addWidget(self._musical_box)

        def _toggle_musical(checked: bool) -> None:
            self._musical_toggle.setText(("▾" if checked else "▸") + "  Musical parameters")
            self._musical_box.setVisible(checked)
        self._musical_toggle.toggled.connect(_toggle_musical)

        lay.addWidget(self._divider())

        grid = QHBoxLayout()
        dur_box = QVBoxLayout()
        dur_box.addWidget(self._field_label("Duration (s)"))
        self._duration_spin = QSpinBox()
        self._duration_spin.setRange(5, 300)
        self._duration_spin.setValue(cfg.default_duration)
        self._duration_spin.setObjectName("SpinBox")
        dur_box.addWidget(self._duration_spin)
        grid.addLayout(dur_box)

        var_box = QVBoxLayout()
        var_box.addWidget(self._field_label("Variations"))
        self._variations_spin = QSpinBox()
        self._variations_spin.setRange(1, 16)
        self._variations_spin.setValue(cfg.default_variations)
        self._variations_spin.setObjectName("SpinBox")
        self._variations_spin.valueChanged.connect(
            lambda v: self._gen_btn.setText(f"Generate ×{v}")
            if hasattr(self, '_gen_btn') else None
        )
        var_box.addWidget(self._variations_spin)
        grid.addLayout(var_box)

        lay.addLayout(grid)

        seed_lora = QHBoxLayout()

        seed_box = QVBoxLayout()
        seed_box.addWidget(self._field_label("Seed (blank = random)"))
        self._seed_input = QLineEdit()
        self._seed_input.setPlaceholderText("random")
        self._seed_input.setObjectName("LineEdit")
        seed_box.addWidget(self._seed_input)
        seed_lora.addLayout(seed_box)

        lora_box = QVBoxLayout()
        lora_box.addWidget(self._field_label("LoRA"))
        self._lora_combo = QComboBox()
        self._lora_combo.setObjectName("SortCombo")
        self._lora_combo.addItem("None")
        self._lora_combo.addItems(self._scan_loras())
        refresh_btn = QPushButton("↻")
        refresh_btn.setObjectName("ActionBtn")
        refresh_btn.setFixedWidth(26)
        refresh_btn.setToolTip("Rescan LoRA folder")
        refresh_btn.clicked.connect(self._refresh_loras)
        lora_row = QHBoxLayout()
        lora_row.setSpacing(4)
        lora_row.addWidget(self._lora_combo, 1)
        lora_row.addWidget(refresh_btn)
        lora_box.addLayout(lora_row)
        seed_lora.addLayout(lora_box)

        lay.addLayout(seed_lora)

        lay.addStretch()
        return w

    # --- Genre blend widget (Phase 2.4) -------------------------------------

    def _build_genre_blend_widget(self) -> QWidget:
        """Two genre selectors + proportion slider + live preview."""
        import json as _json
        genres_file = Path(__file__).parent.parent.parent / "app" / "data" / "genres.json"
        try:
            genres = _json.loads(genres_file.read_text(encoding="utf-8"))["genres"]
        except Exception:
            genres = ["pop", "rock", "jazz", "electronic", "classical"]

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # Row 1: Genre A + slider + Genre B
        blend_row = QHBoxLayout()
        blend_row.setSpacing(6)

        self._genre_a = QComboBox()
        self._genre_a.setObjectName("SortCombo")
        self._genre_a.addItems(genres)
        self._genre_a.setCurrentText("indie pop")
        blend_row.addWidget(self._genre_a, 2)

        self._genre_slider = QSlider(Qt.Horizontal)
        self._genre_slider.setRange(0, 100)
        self._genre_slider.setValue(50)
        self._genre_slider.setObjectName("LyricsSlider")
        blend_row.addWidget(self._genre_slider, 3)

        self._genre_b = QComboBox()
        self._genre_b.setObjectName("SortCombo")
        self._genre_b.addItems(genres)
        self._genre_b.setCurrentText("cinematic orchestral")
        blend_row.addWidget(self._genre_b, 2)
        lay.addLayout(blend_row)

        # Row 2: optional third genre
        third_row = QHBoxLayout()
        third_row.setSpacing(6)
        add_third_btn = QPushButton("+ third genre")
        add_third_btn.setObjectName("AdvancedToggle")
        add_third_btn.setCheckable(True)
        third_row.addWidget(add_third_btn)

        self._genre_c_box = QWidget()
        gc_lay = QHBoxLayout(self._genre_c_box)
        gc_lay.setContentsMargins(0, 0, 0, 0)
        gc_lay.setSpacing(4)
        gc_lbl = QLabel("+ ")
        gc_lbl.setObjectName("FieldLabel")
        self._genre_c = QComboBox()
        self._genre_c.setObjectName("SortCombo")
        self._genre_c.addItems(genres)
        self._genre_c_pct = QSpinBox()
        self._genre_c_pct.setRange(5, 50)
        self._genre_c_pct.setValue(20)
        self._genre_c_pct.setSuffix("%")
        self._genre_c_pct.setObjectName("SpinBox")
        self._genre_c_pct.setFixedWidth(60)
        gc_lay.addWidget(gc_lbl)
        gc_lay.addWidget(self._genre_c, 1)
        gc_lay.addWidget(self._genre_c_pct)
        self._genre_c_box.setVisible(False)
        third_row.addWidget(self._genre_c_box)
        third_row.addStretch()
        add_third_btn.toggled.connect(self._genre_c_box.setVisible)
        lay.addLayout(third_row)

        # Row 3: live preview label
        self._genre_preview = QLabel()
        self._genre_preview.setObjectName("HintLabel")
        self._genre_preview.setWordWrap(True)
        lay.addWidget(self._genre_preview)

        # Wire up live preview and manual toggle
        def _update_preview():
            prompt = self._build_genre_prompt()
            self._genre_preview.setText(f"→ {prompt}" if prompt else "")
            # Sync into style prompt only when blend mode is active
            if not self._genre_manual_btn.isChecked():
                self._style_prompt.setPlainText(prompt)

        self._genre_a.currentTextChanged.connect(lambda _: _update_preview())
        self._genre_b.currentTextChanged.connect(lambda _: _update_preview())
        self._genre_c.currentTextChanged.connect(lambda _: _update_preview())
        self._genre_slider.valueChanged.connect(lambda _: _update_preview())
        self._genre_c_pct.valueChanged.connect(lambda _: _update_preview())
        add_third_btn.toggled.connect(lambda _: _update_preview())

        def _toggle_manual(checked: bool):
            self._genre_manual_btn.setText("Manual ▸" if not checked else "◂ Blend")
            w.setVisible(not checked)
            if checked:
                self._style_prompt.setPlaceholderText("genre, mood, instruments, vocal style, era…")
        self._genre_manual_btn.toggled.connect(_toggle_manual)

        _update_preview()
        return w

    def _build_genre_prompt(self) -> str:
        """Construct the blended style prompt string from genre controls."""
        a    = self._genre_a.currentText().strip()
        b    = self._genre_b.currentText().strip()
        pct  = self._genre_slider.value()   # 0=all-a, 100=all-b

        use_c = hasattr(self, '_genre_c_box') and self._genre_c_box.isVisible()
        c     = self._genre_c.currentText().strip() if use_c else ""
        c_pct = self._genre_c_pct.value() if use_c else 0

        if pct == 0:
            base = a
        elif pct == 100:
            base = b
        elif pct <= 30:
            base = f"{a}-influenced {b}"
        elif pct >= 70:
            base = f"{b}-influenced {a}"
        else:
            base = f"{a} and {b} blend"

        if c:
            base = f"{base} with {c} elements ({c_pct}%)"
        return base

    # --- Musical params widget (Phase 2.1 + 2.5) ----------------------------

    def _build_musical_params_widget(self) -> QWidget:
        """BPM slider + tap, key dropdown, time signature, reference track extractor."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(8)

        # BPM row
        bpm_row = QHBoxLayout()
        bpm_row.setSpacing(6)
        bpm_lbl = self._field_label("BPM")
        bpm_row.addWidget(bpm_lbl)

        self._bpm_slider = QSlider(Qt.Horizontal)
        self._bpm_slider.setRange(0, 200)   # 0 = free; 1–39 clamped to 40 on release
        self._bpm_slider.setValue(0)
        self._bpm_slider.setObjectName("LyricsSlider")
        bpm_row.addWidget(self._bpm_slider, 1)

        self._bpm_lbl = QLabel("Free")
        self._bpm_lbl.setObjectName("SliderValue")
        self._bpm_lbl.setFixedWidth(34)
        bpm_row.addWidget(self._bpm_lbl)

        self._bpm_tap_btn = QPushButton("Tap")
        self._bpm_tap_btn.setObjectName("ActionBtn")
        self._bpm_tap_btn.setFixedWidth(36)
        self._bpm_tap_btn.setToolTip("Tap to set BPM")
        bpm_row.addWidget(self._bpm_tap_btn)
        lay.addLayout(bpm_row)

        # Snap markers hint
        snap_lbl = QLabel("Snap: 60  80  90  100  120  140  160")
        snap_lbl.setObjectName("HintLabel")
        lay.addWidget(snap_lbl)

        # Wire BPM slider
        self._bpm_slider.valueChanged.connect(self._on_bpm_changed)
        self._bpm_slider.sliderReleased.connect(self._snap_bpm)

        # Tap BPM state
        self._tap_times: list[float] = []
        self._tap_reset_timer = QTimer(self)
        self._tap_reset_timer.setSingleShot(True)
        self._tap_reset_timer.setInterval(2000)
        self._tap_reset_timer.timeout.connect(lambda: self._tap_times.clear())
        self._bpm_tap_btn.clicked.connect(self._on_bpm_tap)

        # Key selector
        key_row = QHBoxLayout()
        key_row.setSpacing(6)
        key_row.addWidget(self._field_label("Key"))
        self._key_combo = QComboBox()
        self._key_combo.setObjectName("SortCombo")
        keys = ["Free"] + [
            f"{n} {q}"
            for q in ("major", "minor")
            for n in ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
        ]
        self._key_combo.addItems(keys)
        key_row.addWidget(self._key_combo, 1)
        lay.addLayout(key_row)

        # Time signature
        ts_row = QHBoxLayout()
        ts_row.setSpacing(6)
        ts_row.addWidget(self._field_label("Time"))
        self._time_sig_combo = QComboBox()
        self._time_sig_combo.setObjectName("SortCombo")
        self._time_sig_combo.addItems(["Free", "4/4", "3/4", "6/8", "5/4", "7/8"])
        ts_row.addWidget(self._time_sig_combo, 1)
        lay.addLayout(ts_row)

        # Phase 2.5: Reference track extractor
        lay.addWidget(self._divider())
        ref_hdr = QLabel("Extract from reference track")
        ref_hdr.setObjectName("FieldLabel")
        lay.addWidget(ref_hdr)

        ref_row = QHBoxLayout()
        ref_row.setSpacing(6)
        self._ref_path_lbl = QLabel("No file selected")
        self._ref_path_lbl.setObjectName("HintLabel")
        self._ref_path_lbl.setSizePolicy(
            self._ref_path_lbl.sizePolicy().horizontalPolicy(),
            self._ref_path_lbl.sizePolicy().verticalPolicy(),
        )
        ref_browse = QPushButton("Browse…")
        ref_browse.setObjectName("ActionBtn")
        ref_browse.clicked.connect(self._on_ref_browse)
        ref_row.addWidget(self._ref_path_lbl, 1)
        ref_row.addWidget(ref_browse)
        lay.addLayout(ref_row)

        self._ref_extract_btn = QPushButton("Extract style")
        self._ref_extract_btn.setObjectName("ActionBtn")
        self._ref_extract_btn.setEnabled(False)
        self._ref_extract_btn.clicked.connect(self._on_ref_extract)
        lay.addWidget(self._ref_extract_btn)

        self._ref_result_lbl = QLabel()
        self._ref_result_lbl.setObjectName("HintLabel")
        self._ref_result_lbl.setWordWrap(True)
        self._ref_result_lbl.setVisible(False)
        lay.addWidget(self._ref_result_lbl)

        self._ref_audio_path: str = ""

        return w

    # --- BPM tap and snap ---------------------------------------------------

    _BPM_SNAPS = [60, 80, 90, 100, 120, 140, 160]

    def _on_bpm_changed(self, val: int) -> None:
        self._bpm_lbl.setText("Free" if val == 0 else str(max(40, val)))

    def _snap_bpm(self) -> None:
        val = self._bpm_slider.value()
        if val == 0:
            return
        # Clamp to valid BPM floor (brief says 40–200; values 1–39 are nonsensical)
        if val < 40:
            self._bpm_slider.setValue(40)
            return
        closest = min(self._BPM_SNAPS, key=lambda s: abs(s - val))
        if abs(closest - val) <= 6:
            self._bpm_slider.setValue(closest)

    def _on_bpm_tap(self) -> None:
        import time
        now = time.monotonic()
        self._tap_times.append(now)
        self._tap_reset_timer.start()
        if len(self._tap_times) >= 2:
            intervals = [
                self._tap_times[i] - self._tap_times[i - 1]
                for i in range(1, len(self._tap_times))
            ]
            avg_interval = sum(intervals) / len(intervals)
            bpm = int(round(60.0 / avg_interval))
            bpm = max(40, min(200, bpm))
            self._bpm_slider.setValue(bpm)

    # --- Reference track extraction (Phase 2.5) -----------------------------

    def _on_ref_browse(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select reference audio",
            str(cfg.inputs_dir),
            "Audio files (*.wav *.mp3 *.flac *.ogg *.aif *.aiff)",
        )
        if path:
            self._ref_audio_path = path
            from pathlib import Path as _P
            self._ref_path_lbl.setText(_P(path).name)
            self._ref_extract_btn.setEnabled(True)

    def _on_ref_extract(self) -> None:
        if not self._ref_audio_path:
            return
        self._ref_extract_btn.setEnabled(False)
        self._ref_extract_btn.setText("Analysing…")
        self._ref_result_lbl.setVisible(False)

        path = self._ref_audio_path

        from PySide6.QtCore import QThread, Signal as _Signal

        class _ExtractWorker(QThread):
            done  = _Signal(dict)
            error = _Signal(str)
            def __init__(self, p):
                super().__init__()
                self._p = p
            def run(self):
                try:
                    import librosa
                    import numpy as np
                    y, sr = librosa.load(self._p, sr=None, mono=True, duration=60.0)
                    # BPM
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    bpm = int(round(float(tempo)))
                    # Key via chroma
                    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                    chroma_mean = chroma.mean(axis=1)
                    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                    root_idx = int(np.argmax(chroma_mean))
                    root = note_names[root_idx]
                    # Major vs minor heuristic: compare 3rd and b3rd strengths
                    minor_3rd = chroma_mean[(root_idx + 3) % 12]
                    major_3rd = chroma_mean[(root_idx + 4) % 12]
                    quality = "minor" if minor_3rd > major_3rd else "major"
                    key_str = f"{root} {quality}"
                    # Spectral descriptors
                    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
                    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
                    zcr      = librosa.feature.zero_crossing_rate(y).mean()
                    rms      = librosa.feature.rms(y=y).mean()
                    # Map to adjectives
                    brightness = "bright" if centroid > 3000 else "dark"
                    density    = "dense"  if rolloff  > 4000 else "sparse"
                    energy_adj = "energetic" if rms > 0.05 else "calm"
                    self.done.emit({
                        "bpm": bpm, "key": key_str,
                        "brightness": brightness, "density": density,
                        "energy": energy_adj,
                    })
                except Exception as exc:
                    self.error.emit(str(exc))

        worker = _ExtractWorker(path)

        def _on_done(info: dict):
            bpm = max(40, min(200, info["bpm"]))
            self._bpm_slider.setValue(bpm)
            key_text = info["key"]
            idx = self._key_combo.findText(key_text)
            if idx >= 0:
                self._key_combo.setCurrentIndex(idx)
            # Build descriptor string and append to style prompt
            descriptors = f"{info['brightness']}, {info['density']}, {info['energy']}"
            existing = self._style_prompt.toPlainText().strip()
            if existing:
                self._style_prompt.setPlainText(f"{existing}, {descriptors}")
            else:
                self._style_prompt.setPlainText(descriptors)
            self._ref_result_lbl.setText(
                f"Detected: {bpm} BPM · {key_text} · {descriptors}"
            )
            self._ref_result_lbl.setVisible(True)
            self._ref_extract_btn.setEnabled(True)
            self._ref_extract_btn.setText("Extract style")

        def _on_error(msg: str):
            self._ref_result_lbl.setText(f"Analysis failed: {msg}")
            self._ref_result_lbl.setVisible(True)
            self._ref_extract_btn.setEnabled(True)
            self._ref_extract_btn.setText("Extract style")

        worker.done.connect(_on_done)
        worker.error.connect(_on_error)
        worker.start()
        self._workers.append(worker)

    # --- Cover panel --------------------------------------------------------

    def _build_cover_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)

        info = QLabel("Provide an audio file and a new style prompt. ACE-Step will transform the music while preserving its structure.")
        info.setObjectName("InfoBox")
        info.setWordWrap(True)
        lay.addWidget(info)

        lay.addWidget(self._field_label("Source audio"))
        self._cover_upload = UploadZone("Drop source audio here")
        lay.addWidget(self._cover_upload)

        lay.addWidget(self._field_label("Target style prompt"))
        self._cover_prompt = QPlainTextEdit()
        self._cover_prompt.setObjectName("PromptBox")
        self._cover_prompt.setPlaceholderText("describe the new style, not the source…")
        self._cover_prompt.setFixedHeight(70)
        lay.addWidget(self._cover_prompt)

        lay.addWidget(self._field_label("New lyrics (optional)"))
        self._cover_lyrics = QPlainTextEdit()
        self._cover_lyrics.setObjectName("PromptBox")
        self._cover_lyrics.setPlaceholderText("leave blank to adapt original vocal structure…")
        self._cover_lyrics.setFixedHeight(50)
        lay.addWidget(self._cover_lyrics)

        lay.addWidget(self._field_label("How closely to follow source"))
        strength_row = QHBoxLayout()
        self._cover_strength = QSlider(Qt.Horizontal)
        self._cover_strength.setRange(0, 10)
        self._cover_strength.setValue(7)
        self._cover_strength_lbl = QLabel("0.7")
        self._cover_strength_lbl.setFixedWidth(28)
        self._cover_strength.valueChanged.connect(
            lambda v: self._cover_strength_lbl.setText(f"{v/10:.1f}")
        )
        strength_row.addWidget(self._cover_strength)
        strength_row.addWidget(self._cover_strength_lbl)
        lay.addLayout(strength_row)

        lay.addWidget(self._divider())

        var_box = QVBoxLayout()
        var_box.addWidget(self._field_label("Variations"))
        self._cover_variations = QSpinBox()
        self._cover_variations.setRange(1, 16)
        self._cover_variations.setValue(4)
        self._cover_variations.setObjectName("SpinBox")
        self._cover_variations.valueChanged.connect(
            lambda v: self._gen_btn.setText(f"Generate cover ×{v}")
            if hasattr(self, '_gen_btn') else None
        )
        var_box.addWidget(self._cover_variations)
        lay.addLayout(var_box)

        seed_box2 = QVBoxLayout()
        seed_box2.addWidget(self._field_label("Seed (blank = random)"))
        self._cover_seed = QLineEdit()
        self._cover_seed.setPlaceholderText("random")
        self._cover_seed.setObjectName("LineEdit")
        seed_box2.addWidget(self._cover_seed)
        lay.addLayout(seed_box2)

        lay.addStretch()
        return w

    # --- Vocal panel --------------------------------------------------------

    def _build_vox_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)

        info = QLabel("Upload your vocal recording. ACE-Step will generate full backing instrumentation to match it.")
        info.setObjectName("InfoBox")
        info.setWordWrap(True)
        lay.addWidget(info)

        warn = QLabel("Note: this uses your voice as a melodic reference only — it does not clone or reproduce your voice in the output.")
        warn.setObjectName("WarnBox")
        warn.setWordWrap(True)
        lay.addWidget(warn)

        lay.addWidget(self._field_label("Your vocal track"))
        self._vox_upload = UploadZone("Drop vocal WAV / MP3 here")
        lay.addWidget(self._vox_upload)

        lay.addWidget(self._field_label("Backing style prompt"))
        self._vox_prompt = QPlainTextEdit()
        self._vox_prompt.setObjectName("PromptBox")
        self._vox_prompt.setPlaceholderText("describe the instrumentation you want behind your vocal…")
        self._vox_prompt.setFixedHeight(70)
        lay.addWidget(self._vox_prompt)

        lay.addWidget(self._divider())

        var_box = QVBoxLayout()
        var_box.addWidget(self._field_label("Variations"))
        self._vox_variations = QSpinBox()
        self._vox_variations.setRange(1, 16)
        self._vox_variations.setValue(4)
        self._vox_variations.setObjectName("SpinBox")
        self._vox_variations.valueChanged.connect(
            lambda v: self._gen_btn.setText(f"Generate backing ×{v}")
            if hasattr(self, '_gen_btn') else None
        )
        var_box.addWidget(self._vox_variations)
        lay.addLayout(var_box)

        seed_box3 = QVBoxLayout()
        seed_box3.addWidget(self._field_label("Seed (blank = random)"))
        self._vox_seed = QLineEdit()
        self._vox_seed.setPlaceholderText("random")
        self._vox_seed.setObjectName("LineEdit")
        seed_box3.addWidget(self._vox_seed)
        lay.addLayout(seed_box3)

        lay.addStretch()
        return w

    # -----------------------------------------------------------------------
    # Mode switching
    # -----------------------------------------------------------------------

    def _switch_mode(self, mode: str) -> None:
        for m, btn in self._tab_btns.items():
            btn.setChecked(m == mode)
        self._stack.setCurrentWidget(self._panels[mode])
        labels = {
            "text":  "Generate",
            "cover": "Generate cover",
            "vox":   "Generate backing",
        }
        counts = {
            "text":  self._variations_spin.value(),
            "cover": self._cover_variations.value(),
            "vox":   self._vox_variations.value(),
        }
        self._gen_btn.setText(f"{labels[mode]} ×{counts[mode]}")
        self._results_sub.setText(f"{mode} prompt · {counts[mode]} variations")

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------

    def _on_generate(self) -> None:
        mode = next(m for m, btn in self._tab_btns.items() if btn.isChecked())
        self._clear_results()

        if mode == "text":
            self._run_text()
        elif mode == "cover":
            self._run_cover()
        elif mode == "vox":
            self._run_vox()

    def _run_text(self) -> None:
        seed_txt = self._seed_input.text().strip()
        seed     = int(seed_txt) if seed_txt.isdigit() else None
        lora_val = self._lora_combo.currentText()

        # Determine lyrics mode from pill selection
        mode_pills = self._lyrics_mode_row.findChildren(QPushButton)
        lyrics_mode = LyricsMode.USER if (
            len(mode_pills) > 1 and mode_pills[1].isChecked()
        ) else LyricsMode.AI_WRITES

        # Output type (vocals vs instrumental)
        out_pills = self._output_type_row.findChildren(QPushButton)
        out_type = OutputType.INSTRUMENTAL if (
            len(out_pills) > 1 and out_pills[1].isChecked()
        ) else OutputType.WITH_VOCALS

        # Phase 2.1: Musical parameters
        _raw_bpm = self._bpm_slider.value() if hasattr(self, '_bpm_slider') else 0
        bpm_val = _raw_bpm if _raw_bpm == 0 else max(40, _raw_bpm)
        key_val = self._key_combo.currentText() if hasattr(self, '_key_combo') else "Free"
        ts_val  = self._time_sig_combo.currentText() if hasattr(self, '_time_sig_combo') else "Free"

        # Phase 2.3: Exclusions
        exclusions = self._exclusions.toPlainText().strip() if hasattr(self, '_exclusions') else ""

        # Phase 2.2: Structure builder — merge lyrics if builder is populated
        style_prompt = self._style_prompt.toPlainText().strip()
        user_lyrics  = self._user_lyrics.toPlainText().strip()

        if (hasattr(self, '_structure_builder')
                and self._structure_toggle.isChecked()
                and not self._structure_builder.is_empty()):
            struct_lyrics, caption_hint = self._structure_builder.serialise()
            # Prepend structure lyrics to any user-typed lyrics
            if struct_lyrics:
                user_lyrics = struct_lyrics
                lyrics_mode = LyricsMode.USER
            # Append energy/mood hints to style prompt
            if caption_hint:
                style_prompt = f"{style_prompt} {caption_hint}".strip()

        req = TextGenerationRequest(
            style_prompt       = style_prompt,
            lyrics_mode        = lyrics_mode,
            user_lyrics        = user_lyrics,
            lyrics_prompt      = self._ai_lyrics_prompt.toPlainText().strip(),
            lyrics_creativity  = self._creativity_slider.value() / 10.0,
            lyrics_adherence   = self._adherence_slider.value()  / 10.0,
            lyrics_model       = cfg.get("lyrics_model", ""),
            output_type        = out_type,
            variations         = self._variations_spin.value(),
            duration_secs      = self._duration_spin.value(),
            seed               = seed,
            lora               = None if lora_val == "None" else str(cfg.models_dir / "loras" / lora_val),
            output_dir         = str(cfg.outputs_dir),
            # Phase 2
            bpm                = bpm_val if bpm_val > 0 else None,
            key                = "" if key_val == "Free" else key_val,
            time_signature     = "" if ts_val == "Free" else ts_val,
            exclusions         = exclusions,
        )
        self._gen_btn.setEnabled(False)
        self._gen_btn.setText("Generating…")
        w = TextGenerationWorker(self._pipeline, req)
        self._connect_worker(w)
        w.start()
        self._workers.append(w)

    def _run_cover(self) -> None:
        if not self._cover_upload.loaded_path:
            self.status_message.emit("Please select a source audio file first.")
            return
        cover_seed_txt = self._cover_seed.text().strip()
        req = CoverRequest(
            source_audio_path = self._cover_upload.loaded_path,
            target_style      = self._cover_prompt.toPlainText().strip(),
            new_lyrics        = self._cover_lyrics.toPlainText().strip(),
            follow_strength   = self._cover_strength.value() / 10.0,
            variations        = self._cover_variations.value(),
            seed              = int(cover_seed_txt) if cover_seed_txt.isdigit() else None,
            output_dir        = str(cfg.outputs_dir),
        )
        self._gen_btn.setEnabled(False)
        self._gen_btn.setText("Generating…")
        w = CoverWorker(self._pipeline, req)
        self._connect_worker(w)
        w.start()
        self._workers.append(w)

    def _run_vox(self) -> None:
        if not self._vox_upload.loaded_path:
            self.status_message.emit("Please upload a vocal track first.")
            return
        vox_seed_txt = self._vox_seed.text().strip()
        req = VocalBackingRequest(
            vocal_audio_path = self._vox_upload.loaded_path,
            backing_style    = self._vox_prompt.toPlainText().strip(),
            variations       = self._vox_variations.value(),
            seed             = int(vox_seed_txt) if vox_seed_txt.isdigit() else None,
            output_dir       = str(cfg.outputs_dir),
        )
        self._gen_btn.setEnabled(False)
        self._gen_btn.setText("Generating…")
        w = VocalBackingWorker(self._pipeline, req)
        self._connect_worker(w)
        w.start()
        self._workers.append(w)

    def _connect_worker(self, w) -> None:
        w.progress.connect(self._on_progress)
        w.result.connect(self._on_results)
        w.error.connect(self._on_error)
        w.done.connect(self._on_done)

    def _on_progress(self, p) -> None:
        self.status_message.emit(p.message)
        pct = int(p.overall_pct)
        self._progress_bar.setValue(pct)
        if not self._progress_bar.isVisible():
            self._progress_bar.setVisible(True)

    def _on_results(self, results: list[GenerationResult]) -> None:
        from app.backend.logger import log
        log.info(f"UI received {len(results) if results else 0} result(s)")
        if not results:
            log.warning("_on_results called with empty list")
            return
        self._empty_lbl.setVisible(False)
        for r in results:
            # Record to session history
            session.record(
                mode            = r.mode.value,
                audio_path      = r.audio_path,
                duration_secs   = r.duration_secs,
                seed            = r.seed,
                variation_index = r.variation_index,
                style_prompt    = r.style_prompt,
                alignment_path  = r.alignment_path or "",
                starred         = r.starred,
            )
            card = ResultCard(r)
            card.stems_requested.connect(self._on_stems_requested)
            card.download_wav.connect(self._on_download_wav)
            card.download_mp3.connect(self._on_download_mp3)
            card.repaint_requested.connect(
                lambda res: self.open_in_repair.emit(res.audio_path)
            )
            card.star_toggled.connect(lambda _: self._apply_sort())
            # Phase 1: word repair and section regeneration from lyric timeline
            card.repair_word_requested.connect(self._on_repair_word)
            card.regenerate_section_requested.connect(self._on_regenerate_section)
            # Phase 3: wire mix_rendered so rendered mixes appear as new result cards
            card.mix_rendered.connect(self._on_mix_rendered)
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
            self._result_cards.append(card)

    def _on_mix_rendered(self, source: GenerationResult, mix_path: str) -> None:
        """Create a new result card for a custom stem mix (Phase 3)."""
        from app.models.generation import GenerationMode
        import time as _time
        mix_result = GenerationResult(
            variation_index = len(self._result_cards) + 1,
            audio_path      = mix_path,
            seed            = source.seed,
            duration_secs   = source.duration_secs,
            mode            = source.mode,
            style_prompt    = source.style_prompt,
            lyrics          = source.lyrics,
        )
        session.record(
            mode            = mix_result.mode.value,
            audio_path      = mix_result.audio_path,
            duration_secs   = mix_result.duration_secs,
            seed            = mix_result.seed,
            variation_index = mix_result.variation_index,
            style_prompt    = f"[Custom mix] {mix_result.style_prompt}",
        )
        self._empty_lbl.setVisible(False)
        card = ResultCard(mix_result)
        card.stems_requested.connect(self._on_stems_requested)
        card.download_wav.connect(self._on_download_wav)
        card.download_mp3.connect(self._on_download_mp3)
        card.repaint_requested.connect(lambda res: self.open_in_repair.emit(res.audio_path))
        card.star_toggled.connect(lambda _: self._apply_sort())
        card.repair_word_requested.connect(self._on_repair_word)
        card.regenerate_section_requested.connect(self._on_regenerate_section)
        card.mix_rendered.connect(self._on_mix_rendered)
        self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
        self._result_cards.append(card)
        self.status_message.emit(f"Custom mix saved → {Path(mix_path).name}")

    def _on_repair_word(
        self,
        result: GenerationResult,
        region_start: float,
        region_end: float,
        intended_word: str,
    ) -> None:
        """
        Route a single-word repair request from the lyric timeline (Phase 1.4).
        Uses the existing RepairRequest/RepairWorker machinery — region mode
        with the intended word as the hint prompt. On success, splices the
        repaired audio back and re-runs alignment on the new variation.
        """
        from app.backend.worker import RepairWorker, AlignmentWorker
        from app.models.generation import RepairMode, RepairRequest

        req = RepairRequest(
            source_audio_path = result.audio_path,
            mode              = RepairMode.REGION,
            region_start_sec  = region_start,
            region_end_sec    = region_end,
            hint_prompt       = intended_word,
            output_dir        = str(cfg.outputs_dir),
        )
        w = RepairWorker(self._pipeline, req)
        w.result.connect(lambda repaired, rs=region_start, re=region_end:
                         self._on_word_repair_done(repaired, result, rs, re))
        w.error.connect(lambda msg: self.status_message.emit(f"Word repair failed: {msg}"))
        w.start()
        self._workers.append(w)
        self.status_message.emit(f'Repairing "{intended_word}"…')

    @staticmethod
    def _splice_word_repair(
        source: GenerationResult,
        repaired_path: str,
        region_start: float,
        region_end: float,
    ) -> str:
        """
        Splice the repaired region back into the source audio (Bug 5 fix).
        Returns the path of the spliced file, or repaired_path on failure.
        Mirrors the splice logic in _ace_worker_v15._splice_repair but runs
        on the UI side for word/section repairs that go through RepairWorker.
        """
        import uuid
        try:
            import soundfile as sf
            import numpy as np
            from pathlib import Path as _P

            orig,    sr_o = sf.read(source.audio_path, dtype="float32", always_2d=True)
            repaired_arr, sr_r = sf.read(repaired_path, dtype="float32", always_2d=True)

            if sr_o != sr_r:
                return repaired_path

            start_s = max(0, min(int(round(region_start * sr_o)), len(orig)))
            end_s   = max(start_s, min(int(round(region_end   * sr_o)), len(orig)))
            region_len = end_s - start_s
            if region_len <= 0:
                return repaired_path

            # Match channel count
            if orig.shape[1] != repaired_arr.shape[1]:
                n = max(orig.shape[1], repaired_arr.shape[1])
                if orig.shape[1] < n:
                    orig = np.repeat(orig, n, axis=1)
                if repaired_arr.shape[1] < n:
                    repaired_arr = np.repeat(repaired_arr, n, axis=1)

            rep_start = min(start_s, len(repaired_arr))
            rep_end   = min(end_s,   len(repaired_arr))
            patch = repaired_arr[rep_start:rep_end]
            if len(patch) < region_len:
                patch = np.concatenate([
                    patch,
                    np.zeros((region_len - len(patch), orig.shape[1]), dtype=np.float32)
                ])

            spliced = orig.copy()
            spliced[start_s:end_s] = patch[:region_len]

            out_dir  = _P(source.audio_path).parent
            out_name = f"{_P(source.audio_path).stem}_patched_{uuid.uuid4().hex[:6]}.wav"
            out_path = str(out_dir / out_name)
            sf.write(out_path, spliced, sr_o)
            return out_path
        except Exception as exc:
            from app.backend.logger import log
            log.warning(f"[word_repair_splice] {exc} — using raw repaired file")
            return repaired_path

    def _on_word_repair_done(
        self,
        repaired: list[GenerationResult],
        source: GenerationResult,
        region_start: float = 0.0,
        region_end: float = 0.0,
    ) -> None:
        """After a word repair: splice region back, add new card, re-align."""
        from app.backend.worker import AlignmentWorker
        if not repaired:
            return
        new_result = repaired[0]

        # Splice the repaired region into the source audio (Bug 5).
        # Only splice for region repairs where we have valid timestamps.
        if region_end > region_start and source.audio_path:
            spliced_path = self._splice_word_repair(
                source, new_result.audio_path, region_start, region_end
            )
            new_result.audio_path = spliced_path

        # Carry the original lyrics forward so alignment can run
        new_result.lyrics = source.lyrics
        new_result.style_prompt = source.style_prompt

        self._empty_lbl.setVisible(False)
        card = ResultCard(new_result)
        card.stems_requested.connect(self._on_stems_requested)
        card.download_wav.connect(self._on_download_wav)
        card.download_mp3.connect(self._on_download_mp3)
        card.repaint_requested.connect(
            lambda res: self.open_in_repair.emit(res.audio_path)
        )
        card.repair_word_requested.connect(self._on_repair_word)
        card.regenerate_section_requested.connect(self._on_regenerate_section)
        self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
        self._result_cards.append(card)

        # Re-align the new variation in the background
        aw = AlignmentWorker(new_result)
        aw.aligned.connect(lambda r: self._on_alignment_done(r))
        aw.start()
        self._workers.append(aw)
        self.status_message.emit("Word repaired — re-aligning…")

    def _on_alignment_done(self, result: GenerationResult) -> None:
        """Find the card for this result and refresh its timeline."""
        card = next((c for c in self._result_cards if c.result is result), None)
        if card:
            card.refresh_alignment()
        self.status_message.emit("Alignment updated.")

    def _on_regenerate_section(
        self,
        result: GenerationResult,
        section_start: float,
        section_end: float,
        section_label: str,
    ) -> None:
        """
        Regenerate a single section from the lyric timeline (Phase 1.5).
        Same region-repair route, with a 100ms crossfade at both boundaries
        applied post-splice via scipy (handled in the worker's result callback).
        """
        from app.backend.worker import RepairWorker
        from app.models.generation import RepairMode, RepairRequest

        # Add 100ms padding on each side for the crossfade splice (§1.5)
        CROSSFADE = 0.1
        req = RepairRequest(
            source_audio_path = result.audio_path,
            mode              = RepairMode.REGION,
            region_start_sec  = max(0.0, section_start - CROSSFADE),
            region_end_sec    = section_end + CROSSFADE,
            hint_prompt       = f"Regenerate {section_label}",
            output_dir        = str(cfg.outputs_dir),
        )
        w = RepairWorker(self._pipeline, req)
        # Pass the padded region coords so _on_word_repair_done can splice correctly
        regen_start = max(0.0, section_start - CROSSFADE)
        regen_end   = section_end + CROSSFADE
        w.result.connect(lambda repaired, rs=regen_start, re=regen_end:
                         self._on_word_repair_done(repaired, result, rs, re))
        w.error.connect(lambda msg: self.status_message.emit(f"Section regen failed: {msg}"))
        w.start()
        self._workers.append(w)
        self.status_message.emit(f"Regenerating {section_label}…")

    def _on_error(self, msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._progress_bar.setValue(0)
        self.status_message.emit(f"Error: {msg}")

    def _on_done(self) -> None:
        self._cleanup_workers()
        self._progress_bar.setVisible(False)
        self._progress_bar.setValue(0)
        self._gen_btn.setEnabled(True)
        mode = next(m for m, btn in self._tab_btns.items() if btn.isChecked())
        self._switch_mode(mode)

    def _clear_results(self) -> None:
        for card in self._result_cards:
            card.deleteLater()
        self._result_cards.clear()
        self._empty_lbl.setVisible(True)

    # --- Sort ---------------------------------------------------------------

    def _apply_sort(self) -> None:
        """Re-order result cards according to the sort combo selection."""
        if not self._result_cards:
            return
        mode = self._sort_combo.currentText()

        if mode == "Starred first":
            ordered = sorted(self._result_cards, key=lambda c: (0 if c.result.starred else 1))
        elif mode == "Duration ↑":
            ordered = sorted(self._result_cards, key=lambda c: c.result.duration_secs)
        elif mode == "Duration ↓":
            ordered = sorted(self._result_cards, key=lambda c: -c.result.duration_secs)
        else:
            ordered = sorted(self._result_cards, key=lambda c: c.result.variation_index)

        # Remove and re-insert in new order (preserve stretch at end)
        stretch = self._cards_layout.takeAt(self._cards_layout.count() - 1)
        for card in self._result_cards:
            self._cards_layout.removeWidget(card)
        for card in ordered:
            self._cards_layout.addWidget(card)
        if stretch:
            self._cards_layout.addItem(stretch)
        self._result_cards = ordered

    # --- Stems --------------------------------------------------------------

    def _on_stems_requested(self, result: GenerationResult) -> None:
        # Find the card that emitted this signal by matching audio_path
        card = next((c for c in self._result_cards if c.result is result), None)
        if card is None:
            return
        w = StemWorker(self._separator, result.audio_path, f"stems_{result.variation_index}")
        _card = card  # capture for lambdas
        w.stems_ready.connect(lambda path, stems, c=_card: self._on_stems_ready(c, path, stems))
        w.done.connect(lambda c=_card: c.enable_stems_btn())
        w.start()
        self._workers.append(w)
        self.status_message.emit("Separating stems…")

    def _on_stems_ready(self, card: ResultCard, path: str, stems: dict) -> None:
        card.populate_stems(stems)
        self.status_message.emit("Stems ready")

    # --- Downloads ----------------------------------------------------------

    def _on_download_wav(self, result: GenerationResult) -> None:
        from PySide6.QtWidgets import QFileDialog
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save WAV", str(cfg.outputs_dir / result.filename),
            "WAV files (*.wav)"
        )
        if dest:
            shutil.copy2(result.audio_path, dest)

    def _on_download_mp3(self, result: GenerationResult) -> None:
        from PySide6.QtWidgets import QFileDialog
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save MP3",
            str(cfg.outputs_dir / result.filename.replace(".wav", ".mp3")),
            "MP3 files (*.mp3)"
        )
        if dest:
            w = ExportWorker(result.audio_path, dest, fmt="mp3", bitrate=cfg.mp3_bitrate)
            w.finished.connect(lambda p: self.status_message.emit(f"Saved: {Path(p).name}"))
            w.error.connect(lambda e: self.status_message.emit(f"Export error: {e}"))
            w.start()
            self._workers.append(w)
            self.status_message.emit("Exporting MP3…")

    # -----------------------------------------------------------------------
    # LoRA helpers
    # -----------------------------------------------------------------------

    def _scan_loras(self) -> list[str]:
        """Return .safetensors / .pt filenames found in models_dir/loras/."""
        lora_dir = cfg.models_dir / "loras"
        if not lora_dir.exists():
            return []
        exts = {".safetensors", ".pt", ".pth", ".bin"}
        return sorted(p.name for p in lora_dir.iterdir() if p.suffix.lower() in exts)

    def _refresh_loras(self) -> None:
        current = self._lora_combo.currentText()
        self._lora_combo.clear()
        self._lora_combo.addItem("None")
        self._lora_combo.addItems(self._scan_loras())
        idx = self._lora_combo.findText(current)
        if idx >= 0:
            self._lora_combo.setCurrentIndex(idx)
        self.status_message.emit("LoRA list refreshed")

    # -----------------------------------------------------------------------
    # Presets
    # -----------------------------------------------------------------------

    def _collect_current_params(self) -> dict:
        """Snapshot all current form values for preset saving."""
        mode = next(m for m, b in self._tab_btns.items() if b.isChecked())
        if mode == "text":
            params = {
                "style_prompt":  self._style_prompt.toPlainText(),
                "duration_secs": self._duration_spin.value(),
                "variations":    self._variations_spin.value(),
                "seed":          self._seed_input.text(),
                "lora":          self._lora_combo.currentText(),
                # Phase 2
                "bpm":           (lambda v: 0 if v < 40 else v)(self._bpm_slider.value()) if hasattr(self, '_bpm_slider') else 0,
                "key":           self._key_combo.currentText() if hasattr(self, '_key_combo') else "Free",
                "time_signature": self._time_sig_combo.currentText() if hasattr(self, '_time_sig_combo') else "Free",
                "exclusions":    self._exclusions.toPlainText() if hasattr(self, '_exclusions') else "",
            }
            return params
        elif mode == "cover":
            return {
                "target_style":    self._cover_prompt.toPlainText(),
                "new_lyrics":      self._cover_lyrics.toPlainText(),
                "follow_strength": self._cover_strength.value() / 10.0,
                "variations":      self._cover_variations.value(),
                "seed":            self._cover_seed.text(),
            }
        else:  # vox
            return {
                "backing_style": self._vox_prompt.toPlainText(),
                "variations":    self._vox_variations.value(),
                "seed":          self._vox_seed.text(),
            }

    def _on_preset_loaded(self, preset_data: dict) -> None:
        mode   = preset_data.get("mode", "text")
        params = preset_data.get("params", {})

        # Switch to the right mode
        self._switch_mode(mode)
        for m, btn in self._tab_btns.items():
            btn.setChecked(m == mode)
        self._stack.setCurrentWidget(self._panels[mode])

        if mode == "text":
            if "style_prompt"  in params: self._style_prompt.setPlainText(params["style_prompt"])
            if "duration_secs" in params: self._duration_spin.setValue(int(params["duration_secs"]))
            if "variations"    in params: self._variations_spin.setValue(int(params["variations"]))
            if "seed"          in params: self._seed_input.setText(str(params["seed"]))
            if "lora"          in params:
                idx = self._lora_combo.findText(params["lora"])
                if idx >= 0: self._lora_combo.setCurrentIndex(idx)
            # Phase 2
            if "bpm" in params and hasattr(self, '_bpm_slider'):
                self._bpm_slider.setValue(int(params.get("bpm") or 0))
            if "key" in params and hasattr(self, '_key_combo'):
                idx = self._key_combo.findText(params["key"])
                if idx >= 0: self._key_combo.setCurrentIndex(idx)
            if "time_signature" in params and hasattr(self, '_time_sig_combo'):
                idx = self._time_sig_combo.findText(params["time_signature"])
                if idx >= 0: self._time_sig_combo.setCurrentIndex(idx)
            if "exclusions" in params and hasattr(self, '_exclusions'):
                self._exclusions.setPlainText(params["exclusions"])
                if params["exclusions"].strip():
                    self._exclusions_toggle.setChecked(True)
        elif mode == "cover":
            if "target_style"    in params: self._cover_prompt.setPlainText(params["target_style"])
            if "new_lyrics"      in params: self._cover_lyrics.setPlainText(params["new_lyrics"])
            if "follow_strength" in params:
                self._cover_strength.setValue(int(float(params["follow_strength"]) * 10))
            if "variations" in params: self._cover_variations.setValue(int(params["variations"]))
            if "seed"       in params: self._cover_seed.setText(str(params["seed"]))
        elif mode == "vox":
            if "backing_style" in params: self._vox_prompt.setPlainText(params["backing_style"])
            if "variations"    in params: self._vox_variations.setValue(int(params["variations"]))
            if "seed"          in params: self._vox_seed.setText(str(params["seed"]))

        self.status_message.emit(f"Preset loaded: {preset_data.get('name', '')}")

    # -----------------------------------------------------------------------
    # Helpers

    def _cleanup_workers(self) -> None:
        """Remove finished workers to avoid accumulation."""
        self._workers = [w for w in self._workers if not w.isFinished()]

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("FieldLabel")
        return lbl

    def _divider(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setObjectName("Divider")
        return f

    def _pill_row(self, options: list[str]) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        btns: list[QPushButton] = []

        def _select(selected_btn):
            for b in btns:
                b.setChecked(b is selected_btn)

        for opt in options:
            btn = QPushButton(opt)
            btn.setObjectName("PillBtn")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, b=btn: _select(b))
            lay.addWidget(btn)
            btns.append(btn)
        if btns:
            btns[0].setChecked(True)
        lay.addStretch()
        return row
