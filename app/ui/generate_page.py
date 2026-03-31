"""
ui/generate_page.py — Generate tab: Text prompt / Cover / Vocal backing modes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSlider, QSpinBox, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget,
)

from app.backend.ace_step import ACEStepPipeline
from app.backend.demucs import DemucsSeparator
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
        left.setFixedWidth(320)
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

        # Stacked panels
        self._stack = QStackedWidget()
        self._panels = {
            "text":  self._build_text_panel(),
            "cover": self._build_cover_panel(),
            "vox":   self._build_vox_panel(),
        }
        for w in self._panels.values():
            self._stack.addWidget(w)
        left_layout.addWidget(self._stack, 1)

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
        hdr_layout.setContentsMargins(14, 10, 14, 10)
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
        self._cards_layout.setContentsMargins(12, 12, 12, 12)
        self._cards_layout.setSpacing(8)
        self._cards_layout.addStretch()

        # Empty state
        self._empty_lbl = QLabel("Generate something to see results here.")
        self._empty_lbl.setObjectName("EmptyState")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        self._cards_layout.insertWidget(0, self._empty_lbl)

        scroll.setWidget(self._cards_widget)
        right_layout.addWidget(scroll, 1)

        splitter.addWidget(right)
        splitter.setSizes([320, 600])

        root.addWidget(splitter)

    # --- Text panel ---------------------------------------------------------

    def _build_text_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)

        lay.addWidget(self._field_label("Output type"))
        self._output_type_row = self._pill_row(["With vocals", "Instrumental"])
        lay.addWidget(self._output_type_row)

        lay.addWidget(self._field_label("Lyrics"))
        self._lyrics_mode_row = self._pill_row(["AI writes", "I provide"])
        lay.addWidget(self._lyrics_mode_row)

        lay.addWidget(self._field_label("Style prompt"))
        self._style_prompt = QPlainTextEdit()
        self._style_prompt.setObjectName("PromptBox")
        self._style_prompt.setPlaceholderText("genre, mood, instruments, vocal style, era…")
        self._style_prompt.setFixedHeight(90)
        lay.addWidget(self._style_prompt)

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
        req = TextGenerationRequest(
            style_prompt  = self._style_prompt.toPlainText().strip(),
            variations    = self._variations_spin.value(),
            duration_secs = self._duration_spin.value(),
            seed          = seed,
            lora          = None if lora_val == "None" else str(cfg.models_dir / "loras" / lora_val),
            output_dir    = str(cfg.outputs_dir),
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
            )
            card = ResultCard(r)
            card.stems_requested.connect(self._on_stems_requested)
            card.download_wav.connect(self._on_download_wav)
            card.download_mp3.connect(self._on_download_mp3)
            card.repaint_requested.connect(
                lambda res: self.open_in_repair.emit(res.audio_path)
            )
            card.star_toggled.connect(lambda _: self._apply_sort())
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
            self._result_cards.append(card)

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
            return {
                "style_prompt":  self._style_prompt.toPlainText(),
                "duration_secs": self._duration_spin.value(),
                "variations":    self._variations_spin.value(),
                "seed":          self._seed_input.text(),
                "lora":          self._lora_combo.currentText(),
            }
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
