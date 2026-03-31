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
    QPushButton, QScrollArea, QSpinBox,
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
        form = QFormLayout(w)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(10)

        self._ace_model = QLineEdit()
        self._ace_model.setPlaceholderText("ACE-Step/ACE-Step-v1-3.5B")
        form.addRow("ACE-Step model ID:", self._ace_model)

        self._demucs_model = QComboBox()
        self._demucs_model.addItems(["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"])
        form.addRow("Demucs model:", self._demucs_model)

        info = QLabel(
            "ACE-Step model IDs are HuggingFace repo IDs (e.g. ACE-Step/ACE-Step-v1-3.5B).\n"
            "Models are cached in the Models cache directory above.\n"
            "Changing the model ID takes effect after restarting Stitch."
        )
        info.setObjectName("InfoBox")
        info.setWordWrap(True)
        form.addRow(info)

        return w

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

        self._outputs_dir.setText(str(cfg.outputs_dir))
        self._inputs_dir.setText(str(cfg.inputs_dir))
        self._models_dir.setText(str(cfg.models_dir))
        self._stems_dir.setText(str(cfg.stems_dir))

        self._ace_model.setText(cfg.ace_step_model)
        idx = self._demucs_model.findText(cfg.demucs_model)
        if idx >= 0:
            self._demucs_model.setCurrentIndex(idx)

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
            "outputs_dir":         self._outputs_dir.text(),
            "inputs_dir":          self._inputs_dir.text(),
            "models_dir":          self._models_dir.text(),
            "stems_dir":           self._stems_dir.text(),
            "ace_step_model":      self._ace_model.text().strip(),
            "demucs_model":        self._demucs_model.currentText(),
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
