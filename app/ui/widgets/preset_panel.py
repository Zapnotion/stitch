"""
ui/widgets/preset_panel.py — Preset picker and saver widget.

Displays a list of saved presets with load / delete actions.
A "Save current as preset" button at the top triggers a name dialog.

Signals:
    preset_loaded(dict)  — emitted when user clicks a preset row
    preset_saved(str)    — emitted after save completes (name)
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFrame,
    QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from app.backend.presets import presets


_DOT_COLORS = ["#378ADD", "#1D9E75", "#EF9F27", "#B09ECC", "#E05555", "#70C8F0"]


class PresetPanel(QWidget):
    preset_loaded = Signal(object)   # full preset dict
    preset_saved  = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current_params_fn = None   # callable → dict of current params
        self._current_mode_fn   = None   # callable → str mode
        self._build_ui()
        self.refresh()

    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        hdr = QLabel("PRESETS")
        hdr.setObjectName("SectionHeader")
        hdr.setContentsMargins(0, 0, 0, 6)
        root.addWidget(hdr)

        save_btn = QPushButton("+ Save current as preset")
        save_btn.setObjectName("SavePresetBtn")
        save_btn.clicked.connect(self._on_save)
        root.addWidget(save_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 6, 0, 0)
        self._list_layout.setSpacing(3)
        self._list_layout.addStretch()

        scroll.setWidget(self._list_widget)
        root.addWidget(scroll, 1)

        self._empty_lbl = QLabel("No presets yet.")
        self._empty_lbl.setObjectName("EmptyState")
        self._empty_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self._empty_lbl)

    # -----------------------------------------------------------------------

    def set_params_getter(self, fn) -> None:
        """Pass a callable that returns the current form params dict."""
        self._current_params_fn = fn

    def set_mode_getter(self, fn) -> None:
        """Pass a callable that returns the current mode string."""
        self._current_mode_fn = fn

    def refresh(self) -> None:
        """Re-read presets from disk and rebuild the list."""
        # Clear existing rows
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        all_presets = presets.list_all()
        self._empty_lbl.setVisible(len(all_presets) == 0)

        for i, p in enumerate(all_presets):
            row = self._make_row(p, i)
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)

    # -----------------------------------------------------------------------

    def _make_row(self, preset_data: dict, idx: int) -> QWidget:
        row = QWidget()
        row.setObjectName("PresetRow")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(8)

        dot = QLabel("●")
        dot.setFixedWidth(10)
        dot.setStyleSheet(f"color: {_DOT_COLORS[idx % len(_DOT_COLORS)]};")
        lay.addWidget(dot)

        name = preset_data.get("name", "Unnamed")
        mode = preset_data.get("mode", "text")
        name_lbl = QLabel(name)
        name_lbl.setObjectName("PresetName")
        mode_lbl = QLabel(mode)
        mode_lbl.setObjectName("PresetMode")
        lay.addWidget(name_lbl, 1)
        lay.addWidget(mode_lbl)

        load_btn = QPushButton("Load")
        load_btn.setObjectName("PresetLoadBtn")
        load_btn.setFixedWidth(42)
        _data = preset_data
        load_btn.clicked.connect(lambda _, d=_data: self._on_load(d))
        lay.addWidget(load_btn)

        del_btn = QPushButton("✕")
        del_btn.setObjectName("PresetDelBtn")
        del_btn.setFixedWidth(22)
        _name = name
        del_btn.clicked.connect(lambda _, n=_name: self._on_delete(n))
        lay.addWidget(del_btn)

        return row

    # -----------------------------------------------------------------------

    def _on_save(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Preset name:",
            QLineEdit.Normal, ""
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        mode   = self._current_mode_fn() if self._current_mode_fn else "text"
        params = self._current_params_fn() if self._current_params_fn else {}

        try:
            presets.save(name, mode, params)
            self.refresh()
            self.preset_saved.emit(name)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Save Failed", str(exc))

    def _on_load(self, preset_data: dict) -> None:
        self.preset_loaded.emit(preset_data)

    def _on_delete(self, name: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Delete Preset",
            f"Delete preset '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            presets.delete(name)
            self.refresh()
