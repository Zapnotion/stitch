"""
ui/settings_dialog.py — Settings dialog.

Exposes all config.json keys as editable form fields.
Changes are written back to config.json on OK.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QSlider, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from app.config import cfg
from app.backend.logger import log


class SettingsDialog(QDialog):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stitch — Settings")
        self.setMinimumWidth(520)
        self.setModal(True)
        self._build_ui()
        self._load_values()

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()
        tabs.setObjectName("SettingsTabs")
        tabs.addTab(self._build_generation_tab(), "Generation")
        tabs.addTab(self._build_paths_tab(),      "Paths")
        tabs.addTab(self._build_model_tab(),       "Models")
        tabs.addTab(self._build_export_tab(),      "Export")
        root.addWidget(tabs, 1)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self._on_reset)
        btns.setContentsMargins(12, 8, 12, 12)
        root.addWidget(btns)

    # --- Tabs ---------------------------------------------------------------

    def _build_generation_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(10)

        self._default_variations = QSpinBox()
        self._default_variations.setRange(1, 16)
        form.addRow("Default variations:", self._default_variations)

        self._default_duration = QSpinBox()
        self._default_duration.setRange(5, 300)
        self._default_duration.setSuffix(" s")
        form.addRow("Default duration:", self._default_duration)

        self._device = QComboBox()
        self._device.addItems(["auto", "cuda", "cpu"])
        form.addRow("Compute device:", self._device)

        self._sample_rate = QComboBox()
        self._sample_rate.addItems(["44100", "48000", "22050"])
        form.addRow("Sample rate:", self._sample_rate)

        # --- Lyric alignment (Phase 1) ---
        form.addRow(self._divider())

        from app.backend.aligner import ALIGNMENT_AVAILABLE
        if not ALIGNMENT_AVAILABLE:
            from PySide6.QtWidgets import QLabel as _QLabel
            warn = _QLabel(
                "whisper-timestamped not installed.\n"
                "Lyric alignment and timeline features are unavailable.\n"
                "Run run.bat to install all dependencies."
            )
            warn.setObjectName("WarnBox")
            warn.setWordWrap(True)
            form.addRow(warn)

        align_hdr = QLabel("Lyric alignment")
        align_hdr.setObjectName("GroupLabel")
        form.addRow(align_hdr)

        # Confidence threshold slider
        threshold_row = QHBoxLayout()
        self._confidence_threshold = QSlider(Qt.Horizontal)
        self._confidence_threshold.setRange(0, 10)   # mapped: value/10 → 0.0–1.0
        self._confidence_threshold.setValue(6)         # default 0.6
        self._confidence_threshold.setObjectName("LyricsSlider")
        self._confidence_threshold_lbl = QLabel("0.6")
        self._confidence_threshold_lbl.setFixedWidth(28)
        self._confidence_threshold.valueChanged.connect(
            lambda v: self._confidence_threshold_lbl.setText(f"{v/10:.1f}")
        )
        threshold_row.addWidget(self._confidence_threshold)
        threshold_row.addWidget(self._confidence_threshold_lbl)
        form.addRow("Flag words below confidence:", threshold_row)

        # Whisper model size
        self._whisper_model = QComboBox()
        self._whisper_model.addItems(["tiny", "base", "small", "medium"])
        self._whisper_model.setToolTip(
            "Whisper model used for lyric alignment.\n"
            "tiny: ~39 MB, fastest, lower accuracy.\n"
            "base: ~150 MB, recommended balance.\n"
            "small: ~490 MB, more accurate, slower.\n"
            "medium: ~1.5 GB, best accuracy, slow on CPU."
        )
        form.addRow("Alignment model:", self._whisper_model)

        return w

    def _build_paths_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        self._outputs_dir = self._path_row(lay, "Outputs directory:")
        self._inputs_dir  = self._path_row(lay, "Inputs directory:")
        self._models_dir  = self._path_row(lay, "Models cache directory:")
        self._stems_dir   = self._path_row(lay, "Stems directory:")

        lay.addWidget(self._divider())

        open_btn = QPushButton("Open App Data Folder")
        open_btn.setObjectName("ActionBtn")
        open_btn.clicked.connect(self._open_appdata)
        lay.addWidget(open_btn)

        lay.addStretch()
        return w

    def _build_model_tab(self) -> QWidget:
        w = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        form = QFormLayout(inner)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(10)

        # --- ACE-Step LM ---
        self._lm_model = QComboBox()
        self._lm_model.addItems([
            "acestep-5Hz-lm-0.6B",
            "acestep-5Hz-lm-1.7B",
            "acestep-5Hz-lm-4B",
        ])
        self._lm_model.setToolTip(
            "LM model used inside ACE-Step for caption/lyric planning.\n"
            "0.6B: Recommended for Windows — fast and reliable.\n"
            "1.7B: Better quality but can hang on Windows without Triton.\n"
            "4B: Highest quality, needs 24+ GB VRAM."
        )
        form.addRow("ACE-Step LM model:", self._lm_model)

        self._demucs_model = QComboBox()
        self._demucs_model.addItems(["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"])
        form.addRow("Demucs model (stem fallback):", self._demucs_model)

        form.addRow(self._divider())

        # --- Lyrics Model (offline GGUF) ---
        lyr_hdr = QLabel("Lyrics Generation Model")
        lyr_hdr.setObjectName("GroupLabel")
        form.addRow(lyr_hdr)

        from app.backend.lyrics_gen import LYRICS_MODELS, LYRICS_MODEL_NAMES, DEFAULT_LYRICS_MODEL
        from app.config import cfg as _cfg

        self._lyrics_model = QComboBox()
        self._lyrics_model.addItems(LYRICS_MODEL_NAMES)
        self._lyrics_model.setToolTip(
            "Offline GGUF model used for AI lyric writing.\n\n"
            "CPU-friendly (1.5B): ~1 GB download. Zero VRAM. Good for low-end PCs.\n"
            "Balanced (3B): ~2 GB. Optional partial GPU offload (~1 GB VRAM).\n"
            "GPU-assisted (7B): ~4.5 GB. ~3–4 GB VRAM. Best lyric quality.\n\n"
            "Downloads once into your Models cache folder. Falls back to built-in\n"
            "templates until the model has been downloaded."
        )
        form.addRow("Lyrics model:", self._lyrics_model)

        # Download status + button row
        self._lyr_status = QLabel("Checking…")
        self._lyr_status.setObjectName("FieldLabelDim")
        self._lyr_status.setWordWrap(True)

        self._lyr_download_btn = QPushButton("Download selected model")
        self._lyr_download_btn.setObjectName("ActionBtn")
        self._lyr_download_btn.clicked.connect(self._on_download_lyrics_model)

        lyr_row = QHBoxLayout()
        lyr_row.addWidget(self._lyr_status, 1)
        lyr_row.addWidget(self._lyr_download_btn)
        form.addRow(lyr_row)

        lyr_info = QLabel(
            "Models download from HuggingFace on first use (or via the button above).\n"
            "Stored in your Models cache folder. All inference runs fully offline.\n"
            "The Creativity and Topic adherence sliders appear on the Generate tab\n"
            "when 'AI writes' lyrics mode is selected."
        )
        lyr_info.setObjectName("InfoBox")
        lyr_info.setWordWrap(True)
        form.addRow(lyr_info)

        form.addRow(self._divider())

        ace_info = QLabel(
            "ACE-Step 1.5 uses a hybrid LM+DiT architecture.\n"
            "Models auto-download on first generation (~7 GB total).\n"
            "Changes to ACE-Step LM take effect after restarting Stitch."
        )
        ace_info.setObjectName("InfoBox")
        ace_info.setWordWrap(True)
        form.addRow(ace_info)

        scroll.setWidget(inner)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # Refresh status whenever the combo changes
        self._lyrics_model.currentTextChanged.connect(self._refresh_lyr_status)
        return w

    def _refresh_lyr_status(self, model_name: str = "") -> None:
        """Update the download-status label for the currently selected lyrics model."""
        try:
            from app.backend.lyrics_gen import LYRICS_MODELS, _model_path
            name = model_name or self._lyrics_model.currentText()
            info = LYRICS_MODELS.get(name, {})
            local = _model_path(name, cfg.models_dir)
            if local is not None:
                size_mb = local.stat().st_size // (1024 * 1024)
                self._lyr_status.setText(f"✓ Downloaded ({size_mb} MB)")
                self._lyr_download_btn.setText("Re-download")
            else:
                repo = info.get("repo", "")
                self._lyr_status.setText(f"Not downloaded · {repo}")
                self._lyr_download_btn.setText("Download selected model")
        except Exception:
            self._lyr_status.setText("Status unavailable")

    def _on_download_lyrics_model(self) -> None:
        """Kick off a lyrics model download in a background thread."""
        from PySide6.QtCore import QThread, Signal as Sig

        name = self._lyrics_model.currentText()
        self._lyr_download_btn.setEnabled(False)
        self._lyr_status.setText("Downloading…")

        class _DownloadThread(QThread):
            finished = Sig(bool, str)   # (success, message)
            def __init__(self, n, d):
                super().__init__()
                self._n = n; self._d = d
            def run(self):
                try:
                    from app.backend.lyrics_gen import download_lyrics_model
                    path = download_lyrics_model(self._n, self._d)
                    if path:
                        self.finished.emit(True, f"✓ Downloaded to {path.name}")
                    else:
                        self.finished.emit(False, "Download failed — check logs")
                except Exception as exc:
                    self.finished.emit(False, str(exc))

        def _done(ok: bool, msg: str):
            self._lyr_status.setText(msg)
            self._lyr_download_btn.setEnabled(True)
            self._lyr_download_btn.setText("Download selected model")
            self._refresh_lyr_status()

        self._dl_thread = _DownloadThread(name, cfg.models_dir)
        self._dl_thread.finished.connect(_done)
        self._dl_thread.start()

    def _build_export_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(10)

        self._output_format = QComboBox()
        self._output_format.addItems(["wav", "mp3"])
        form.addRow("Default output format:", self._output_format)

        self._mp3_bitrate = QComboBox()
        self._mp3_bitrate.addItems(["128", "192", "256", "320"])
        form.addRow("MP3 bitrate (kbps):", self._mp3_bitrate)

        warn = QLabel(
            "MP3 export requires ffmpeg to be installed and on your PATH.\n"
            "Download: https://ffmpeg.org/download.html"
        )
        warn.setObjectName("WarnBox")
        warn.setWordWrap(True)
        form.addRow(warn)

        return w

    # -----------------------------------------------------------------------
    # Path row helper
    # -----------------------------------------------------------------------

    def _path_row(self, parent_layout: QVBoxLayout, label: str) -> QLineEdit:
        lbl = QLabel(label)
        lbl.setObjectName("FieldLabel")
        parent_layout.addWidget(lbl)

        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setObjectName("LineEdit")
        btn = QPushButton("Browse…")
        btn.setObjectName("ActionBtn")
        btn.setFixedWidth(74)
        btn.clicked.connect(lambda _, e=edit: self._browse_dir(e))
        row.addWidget(edit, 1)
        row.addWidget(btn)
        parent_layout.addLayout(row)
        return edit

    def _browse_dir(self, edit: QLineEdit) -> None:
        current = edit.text() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Select folder", current)
        if chosen:
            edit.setText(chosen)

    # -----------------------------------------------------------------------
    # Load / save values
    # -----------------------------------------------------------------------

    def _load_values(self) -> None:
        self._default_variations.setValue(cfg.default_variations)
        self._default_duration.setValue(cfg.default_duration)
        self._device.setCurrentText(cfg.get("device", "auto"))
        self._sample_rate.setCurrentText(str(cfg.get("default_sample_rate", "44100")))

        # Alignment (Phase 1)
        threshold = cfg.get("alignment_confidence_threshold", 0.6)
        self._confidence_threshold.setValue(int(round(threshold * 10)))
        self._confidence_threshold_lbl.setText(f"{threshold:.1f}")
        whisper_idx = self._whisper_model.findText(cfg.get("whisper_model", "base"))
        if whisper_idx >= 0:
            self._whisper_model.setCurrentIndex(whisper_idx)

        self._outputs_dir.setText(str(cfg.outputs_dir))
        self._inputs_dir.setText(str(cfg.inputs_dir))
        self._models_dir.setText(str(cfg.models_dir))
        self._stems_dir.setText(str(cfg.stems_dir))

        lm_idx = self._lm_model.findText(cfg.get('lm_model', 'acestep-5Hz-lm-0.6B'))
        if lm_idx >= 0: self._lm_model.setCurrentIndex(lm_idx)
        idx = self._demucs_model.findText(cfg.demucs_model)
        if idx >= 0:
            self._demucs_model.setCurrentIndex(idx)

        lyr_idx = self._lyrics_model.findText(cfg.get("lyrics_model", ""))
        if lyr_idx >= 0:
            self._lyrics_model.setCurrentIndex(lyr_idx)
        self._refresh_lyr_status()

        fmt_idx = self._output_format.findText(cfg.output_format)
        if fmt_idx >= 0:
            self._output_format.setCurrentIndex(fmt_idx)
        br_idx = self._mp3_bitrate.findText(str(cfg.mp3_bitrate))
        if br_idx >= 0:
            self._mp3_bitrate.setCurrentIndex(br_idx)

    def _on_ok(self) -> None:
        updates = {
            "default_variations":  self._default_variations.value(),
            "default_duration":    self._default_duration.value(),
            "device":              self._device.currentText(),
            "default_sample_rate": int(self._sample_rate.currentText()),
            # Alignment (Phase 1)
            "alignment_confidence_threshold": self._confidence_threshold.value() / 10.0,
            "whisper_model":       self._whisper_model.currentText(),
            "outputs_dir":         self._outputs_dir.text(),
            "inputs_dir":          self._inputs_dir.text(),
            "models_dir":          self._models_dir.text(),
            "stems_dir":           self._stems_dir.text(),
            "lm_model":            self._lm_model.currentText(),
            "demucs_model":        self._demucs_model.currentText(),
            "lyrics_model":        self._lyrics_model.currentText(),
            "output_format":       self._output_format.currentText(),
            "mp3_bitrate":         int(self._mp3_bitrate.currentText()),
        }
        # Ensure new dirs exist
        for key in ("outputs_dir", "inputs_dir", "models_dir", "stems_dir"):
            try:
                Path(updates[key]).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

        cfg.update(updates)
        log.info("Settings saved")
        self.accept()

    def _on_reset(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Reset Defaults",
            "Reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            cfg.update({})   # will be repopulated from _DEFAULTS on next load
            self._load_values()

    def _open_appdata(self) -> None:
        import subprocess, sys
        from app.config import APP_DATA_DIR
        folder = str(APP_DATA_DIR)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", folder])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    # -----------------------------------------------------------------------

    def _divider(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setObjectName("Divider")
        return f
