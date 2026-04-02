"""
ui/main_window.py — Main application window.
Topbar, nav, VRAM meter, status bar, page switching.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QStackedWidget,
    QStatusBar, QVBoxLayout, QWidget,
)

from app.backend.ace_step_v15 import ACEStepV15 as ACEStepPipeline
from app.backend.separator import StemSeparator
from app.backend.worker import ModelLoaderWorker
from app.config import cfg
from app.ui.generate_page import GeneratePage
from app.ui.history_page import HistoryPage
from app.ui.repair_page import RepairPage
from app.ui.settings_dialog import SettingsDialog
from app.ui.widgets.log_panel import LogPanel


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stitch")
        self.resize(1060, 700)
        self.setMinimumSize(860, 560)

        # Initialise backend
        self._pipeline  = ACEStepPipeline(
            models_dir = str(cfg.models_dir),
            device     = cfg.device,
        )
        self._separator = StemSeparator(
            device          = cfg.device,
            output_base_dir = str(cfg.stems_dir),
            model_cache_dir = str(cfg.models_dir),
        )

        self._build_ui()
        self._apply_stylesheet()
        self._start_vram_timer()
        self._load_model()

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Topbar ---
        topbar = QWidget()
        topbar.setObjectName("Topbar")
        topbar.setFixedHeight(44)
        tb_layout = QHBoxLayout(topbar)
        tb_layout.setContentsMargins(0, 0, 16, 0)
        tb_layout.setSpacing(0)

        logo = QLabel("Stitch")
        logo.setObjectName("Logo")
        logo.setContentsMargins(20, 0, 20, 0)
        tb_layout.addWidget(logo)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setObjectName("TopSep")
        tb_layout.addWidget(sep)

        self._nav_gen = QPushButton("Generate")
        self._nav_gen.setObjectName("NavBtn")
        self._nav_gen.setCheckable(True)
        self._nav_gen.setChecked(True)
        self._nav_gen.clicked.connect(lambda: self._switch_page("gen"))

        self._nav_repair = QPushButton("Repair")
        self._nav_repair.setObjectName("NavBtn")
        self._nav_repair.setCheckable(True)
        self._nav_repair.clicked.connect(lambda: self._switch_page("repair"))

        self._nav_history = QPushButton("History")
        self._nav_history.setObjectName("NavBtn")
        self._nav_history.setCheckable(True)
        self._nav_history.clicked.connect(lambda: self._switch_page("history"))

        tb_layout.addWidget(self._nav_gen)
        tb_layout.addWidget(self._nav_repair)
        tb_layout.addWidget(self._nav_history)
        tb_layout.addStretch()

        # VRAM indicator
        vram_row = QHBoxLayout()
        vram_row.setSpacing(6)
        vram_lbl = QLabel("VRAM")
        vram_lbl.setObjectName("SmLabel")
        self._vram_bar_bg = QLabel()
        self._vram_bar_bg.setObjectName("VramBarBg")
        self._vram_bar_bg.setFixedSize(52, 4)
        self._vram_fill = QLabel()
        self._vram_fill.setObjectName("VramFill")
        self._vram_text = QLabel("– / – GB")
        self._vram_text.setObjectName("SmLabel")

        vram_row.addWidget(vram_lbl)
        vram_row.addWidget(self._vram_bar_bg)
        vram_row.addWidget(self._vram_text)
        tb_layout.addLayout(vram_row)

        # Settings button
        settings_btn = QPushButton("⚙")
        settings_btn.setObjectName("SettingsBtn")
        settings_btn.setFixedSize(28, 28)
        settings_btn.setToolTip("Settings")
        settings_btn.clicked.connect(self._open_settings)
        tb_layout.addWidget(settings_btn)

        root.addWidget(topbar)

        # --- Pages ---
        self._stack = QStackedWidget()

        self._gen_page = GeneratePage(self._pipeline, self._separator)
        self._gen_page.status_message.connect(self._set_status)
        self._gen_page.open_in_repair.connect(self._on_open_in_repair)

        self._repair_page = RepairPage(self._pipeline)
        self._repair_page.status_message.connect(self._set_status)

        self._history_page = HistoryPage()
        self._history_page.load_into_generate.connect(self._on_load_history_entry)

        self._stack.addWidget(self._gen_page)
        self._stack.addWidget(self._repair_page)
        self._stack.addWidget(self._history_page)

        root.addWidget(self._stack, 1)

        # --- Log panel ---
        self._log_panel = LogPanel()
        root.addWidget(self._log_panel)

        # --- Status bar ---
        self._status = QStatusBar()
        self._status.setObjectName("AppStatusBar")
        self.setStatusBar(self._status)
        self._model_lbl = QLabel("Loading model…")
        self._model_lbl.setObjectName("StatusLabel")
        self._status.addWidget(self._model_lbl)

    # -----------------------------------------------------------------------
    # Page switching
    # -----------------------------------------------------------------------

    def _switch_page(self, page: str) -> None:
        self._nav_gen.setChecked(page == "gen")
        self._nav_repair.setChecked(page == "repair")
        self._nav_history.setChecked(page == "history")
        pages = {
            "gen":     self._gen_page,
            "repair":  self._repair_page,
            "history": self._history_page,
        }
        self._stack.setCurrentWidget(pages[page])
        if page == "history":
            self._history_page.refresh()

    def _on_open_in_repair(self, audio_path: str) -> None:
        """Switch to Repair tab and pre-load the given audio file."""
        self._switch_page("repair")
        self._repair_page._upload.set_file(audio_path)

    def _on_load_history_entry(self, entry: dict) -> None:
        """Load a history entry's prompt back into the generate page."""
        self._switch_page("gen")
        # Pass prompt through to generate page if it supports it
        prompt = entry.get("style_prompt", "")
        if hasattr(self._gen_page, "_style_prompt") and prompt:
            self._gen_page._style_prompt.setPlainText(prompt)
        self._set_status(f"Loaded from history: {entry.get('mode', '')} · {prompt[:40]}")

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self)
        dlg.exec()

    def closeEvent(self, event) -> None:
        """Stop all workers and unload models before closing."""
        from app.backend.logger import log
        log.info("Stitch shutting down")

        # Stop any running workers in generate and repair pages
        for page in (self._gen_page, self._repair_page):
            for w in getattr(page, '_workers', []):
                if not w.isFinished():
                    w.quit()
                    w.wait(2000)

        # Unload model to free VRAM
        try:
            self._pipeline.unload()
        except Exception:
            pass

        self._log_panel.shutdown()
        event.accept()

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_model(self) -> None:
        self._loader = ModelLoaderWorker(self._pipeline)
        self._loader.loaded.connect(self._on_model_loaded)
        self._loader.message.connect(self._set_status)
        self._loader.start()

    def _on_model_loaded(self, ok: bool) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                device_str = f"{gpu}  {vram_gb:.0f} GB"
            else:
                device_str = "CPU"
        except Exception:
            device_str = "unknown device"

        demucs_str = "Stems ready" if self._separator.is_available else "No stem separator"

        if ok:
            self._model_lbl.setText(f"● ACE-Step 1.5 ready  ·  {demucs_str}  ·  {device_str}")
        else:
            self._model_lbl.setText(f"⚠ ACE-Step 1.5 not available (stub mode)  ·  {device_str}")

    # -----------------------------------------------------------------------
    # VRAM polling
    # -----------------------------------------------------------------------

    def _start_vram_timer(self) -> None:
        self._vram_timer = QTimer(self)
        self._vram_timer.timeout.connect(self._update_vram)
        self._vram_timer.start(2000)   # every 2 s
        self._update_vram()

    def _update_vram(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                used  = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                pct   = min(used / total, 1.0)
                fill_w = int(52 * pct)
                self._vram_fill.setFixedSize(fill_w, 4)
                self._vram_text.setText(f"{used:.1f} / {total:.0f} GB")
                return
        except Exception:
            pass
        self._vram_text.setText("CPU mode")

    # -----------------------------------------------------------------------
    # Status bar
    # -----------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self._status.showMessage(msg, 8000)

    # -----------------------------------------------------------------------
    # Stylesheet
    # -----------------------------------------------------------------------

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #141414;
                color: #DEDEDE;
                font-family: "Segoe UI", system-ui, sans-serif;
                font-size: 13px;
            }

            /* Topbar */
            #Topbar {
                background: #1A1A1A;
                border-bottom: 1px solid #2A2A2A;
            }
            #Logo {
                font-size: 15px;
                font-weight: 600;
                color: #EFEFEF;
            }
            #TopSep { color: #2A2A2A; }
            #NavBtn {
                border: none;
                background: transparent;
                color: #888;
                padding: 0 18px;
                height: 44px;
                font-size: 13px;
                border-bottom: 2px solid transparent;
            }
            #NavBtn:checked {
                color: #EFEFEF;
                font-weight: 600;
                border-bottom: 2px solid #378ADD;
            }
            #NavBtn:hover { color: #CCC; }

            /* Panels */
            #LeftPanel {
                background: #1A1A1A;
                border-right: 1px solid #2A2A2A;
            }
            #RightPanel { background: #141414; }
            #RightHeader {
                background: #1A1A1A;
                border-bottom: 1px solid #2A2A2A;
            }
            #RightTitle { font-size: 13px; font-weight: 600; color: #EFEFEF; }
            #RightSub   { font-size: 11px; color: #666; margin-left: 8px; }

            /* Mode tabs */
            #ModeTabs {
                background: #1A1A1A;
                border-bottom: 1px solid #2A2A2A;
            }
            #ModeTab {
                border: none;
                background: transparent;
                color: #777;
                padding: 9px 0;
                font-size: 12px;
                border-bottom: 2px solid transparent;
            }
            #ModeTab:checked {
                color: #EFEFEF;
                font-weight: 600;
                border-bottom: 2px solid #378ADD;
            }
            #ModeTab:hover { color: #BBB; }

            /* Form elements */
            #FieldLabel { font-size: 11px; color: #888; }
            QPlainTextEdit, QLineEdit, QSpinBox, QComboBox {
                background: #1E1E1E;
                border: 1px solid #2E2E2E;
                border-radius: 5px;
                color: #DEDEDE;
                padding: 5px 8px;
                font-size: 12px;
            }
            QPlainTextEdit:focus, QLineEdit:focus, QSpinBox:focus {
                border-color: #378ADD;
            }
            QSlider::groove:horizontal {
                background: #2A2A2A;
                height: 4px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #378ADD;
                width: 12px; height: 12px;
                border-radius: 6px;
                margin: -4px 0;
            }
            #Divider { background: #2A2A2A; max-height: 1px; }

            /* Pills */
            #PillBtn {
                border: 1px solid #2E2E2E;
                border-radius: 10px;
                background: transparent;
                color: #888;
                padding: 3px 12px;
                font-size: 11px;
            }
            #PillBtn:checked {
                background: #1B3A5E;
                color: #7EC2F8;
                border-color: #378ADD;
            }
            #PillBtn:hover { border-color: #555; color: #BBB; }

            /* Buttons */
            #GenBtn {
                background: #1B3A5E;
                color: #7EC2F8;
                border: 1px solid #378ADD;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                font-weight: 600;
                margin: 0 14px 8px 14px;
            }
            #GenBtn:hover    { background: #244A78; }
            #GenBtn:disabled { background: #1A1A1A; color: #444; border-color: #333; }

            #GenBtnGreen {
                background: #1D3A1D;
                color: #7EC878;
                border: 1px solid #4A9E4A;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                font-weight: 600;
                margin: 0 14px 8px 14px;
            }
            #GenBtnGreen:hover    { background: #264826; }
            #GenBtnGreen:disabled { background: #1A1A1A; color: #444; border-color: #333; }

            #HintLabel { font-size: 10px; color: #555; margin-bottom: 10px; }

            /* Info / warn boxes */
            #InfoBox {
                background: #0F2A47;
                border: 1px solid #1C4A80;
                border-radius: 6px;
                color: #6EB4F0;
                padding: 8px 10px;
                font-size: 11px;
            }
            #WarnBox {
                background: #3A2800;
                border: 1px solid #7A5800;
                border-radius: 6px;
                color: #D4A84B;
                padding: 8px 10px;
                font-size: 11px;
            }

            /* Upload zone */
            #DropZone {
                border: 1px dashed #333;
                border-radius: 6px;
                background: transparent;
            }
            #DropZone:hover { background: #1E1E1E; border-color: #555; }
            #DropLabel  { font-size: 12px; color: #888; }
            #DropSub    { font-size: 10px; color: #555; }
            #DropBrowse { font-size: 11px; color: #378ADD; }
            #LoadedFile {
                background: #1E1E1E;
                border: 1px solid #2E2E2E;
                border-radius: 6px;
            }
            #LoadedDot  { color: #1D9E75; font-size: 10px; }
            #LoadedName { font-size: 12px; color: #DEDEDE; }
            #LoadedDur  { font-size: 10px; color: #666; }
            #RemoveBtn  { font-size: 10px; color: #555; border: none; background: transparent; }
            #RemoveBtn:hover { color: #E05555; }

            /* Result card */
            #ResultCard {
                background: #1A1A1A;
                border: 1px solid #2A2A2A;
                border-radius: 8px;
            }
            #CardTitle  { font-size: 12px; font-weight: 600; color: #DEDEDE; }
            #SeedLabel  { font-size: 10px; color: #555; }
            #Badge {
                font-size: 10px;
                padding: 2px 7px;
                border-radius: 8px;
                background: #1E1E1E;
                color: #777;
            }
            #StarBtn { font-size: 14px; background: transparent; border: none; color: #EF9F27; }

            #ActionBtn {
                border: 1px solid #2A2A2A;
                border-radius: 5px;
                background: transparent;
                color: #888;
                padding: 4px 8px;
                font-size: 11px;
            }
            #ActionBtn:hover { background: #242424; color: #CCC; }

            #StemsTray  { background: #161616; border-top: 1px solid #222; padding: 8px; }
            #StemsHeader { font-size: 10px; font-weight: 600; color: #555; letter-spacing: 0.05em; }
            #StemLabel  { font-size: 11px; color: #888; }
            #StemDlBtn  { font-size: 10px; border: 1px solid #2A2A2A; border-radius: 4px; background: #1A1A1A; color: #777; }

            /* Repair page */
            #WaveformArea { background: #141414; }
            #WaveFilename { font-size: 10px; color: #555; }
            #RegionLabel  { font-size: 10px; color: #EF9F27; }
            #ComparisonArea { background: #1A1A1A; border-top: 1px solid #2A2A2A; }
            #KeepBtn {
                background: #1D3A1D;
                color: #7EC878;
                border: 1px solid #4A9E4A;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 11px;
            }

            /* Generation progress bar */
            #GenProgress {
                background: #1A1A1A;
                border: none;
                border-radius: 0;
            }
            #GenProgress::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1D9E75, stop:0.5 #378ADD, stop:1 #7B5EA7
                );
                border-radius: 0;
            }

            /* Sort combo in results header */
            #SortCombo {
                background: #1A1A1A;
                border: 1px solid #2A2A2A;
                border-radius: 4px;
                color: #777;
                font-size: 11px;
                padding: 3px 6px;
            }
            #SortCombo::drop-down { border: none; width: 16px; }

            /* History page */
            #HistoryTable {
                background: #141414;
                alternate-background-color: #171717;
                border: none;
                gridline-color: transparent;
                font-size: 12px;
                color: #CCCCCC;
            }
            QHeaderView::section {
                background: #1A1A1A;
                color: #666;
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 0.05em;
                padding: 5px 8px;
                border: none;
                border-bottom: 1px solid #2A2A2A;
                border-right: 1px solid #222;
            }
            QTableWidget::item:selected {
                background: #1B3A5E;
                color: #EFEFEF;
            }
            #HistoryDetail {
                background: #1A1A1A;
                border-left: 1px solid #2A2A2A;
            }

            /* SmLabel / VRAM */
            #SmLabel { font-size: 10px; color: #555; }

            /* Presets toggle strip */
            #PresetToggle {
                background: #161616;
                border: none;
                border-top: 1px solid #242424;
                border-bottom: 1px solid #242424;
                color: #555;
                font-size: 11px;
                padding: 5px 14px;
                text-align: left;
            }
            #PresetToggle:hover   { color: #888; background: #1A1A1A; }
            #PresetToggle:checked { color: #AAA; }

            /* Empty state */
            #EmptyState { font-size: 12px; color: #444; }

            /* Audio player */
            #PlayBtn {
                background: #242424;
                border: 1px solid #333;
                border-radius: 13px;
                color: #AAAAAA;
                font-size: 11px;
            }
            #PlayBtn:hover  { background: #2E2E2E; color: #DEDEDE; border-color: #444; }
            #PlayBtn:pressed { background: #1A1A1A; }
            #ScrubSlider::groove:horizontal {
                background: #2A2A2A;
                height: 3px;
                border-radius: 1px;
            }
            #ScrubSlider::sub-page:horizontal {
                background: #378ADD;
                border-radius: 1px;
            }
            #ScrubSlider::handle:horizontal {
                background: #AAAAAA;
                width: 10px; height: 10px;
                border-radius: 5px;
                margin: -4px 0;
            }
            #ScrubSlider::handle:horizontal:hover { background: #DEDEDE; }
            #TimeLabel { font-size: 10px; color: #555; font-family: "Courier New", monospace; }

            /* Presets panel */
            #SectionHeader {
                font-size: 10px;
                font-weight: 600;
                color: #555;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            #SavePresetBtn {
                background: #1E1E1E;
                border: 1px solid #2E2E2E;
                border-radius: 5px;
                color: #888;
                font-size: 11px;
                padding: 5px 8px;
                text-align: left;
            }
            #SavePresetBtn:hover { background: #252525; color: #BBB; border-color: #444; }
            #PresetRow {
                background: transparent;
                border: 1px solid #242424;
                border-radius: 5px;
            }
            #PresetRow:hover { background: #1E1E1E; }
            #PresetName { font-size: 12px; color: #CCCCCC; }
            #PresetMode { font-size: 10px; color: #555; }
            #PresetLoadBtn {
                background: #1B3A5E;
                color: #7EC2F8;
                border: 1px solid #2A5A8E;
                border-radius: 4px;
                font-size: 10px;
                padding: 2px 6px;
            }
            #PresetLoadBtn:hover { background: #244A78; }
            #PresetDelBtn {
                background: transparent;
                border: none;
                color: #444;
                font-size: 11px;
            }
            #PresetDelBtn:hover { color: #E05555; }

            /* Settings button */
            #SettingsBtn {
                background: transparent;
                border: none;
                color: #555;
                font-size: 15px;
                border-radius: 4px;
            }
            #SettingsBtn:hover { color: #AAA; background: #242424; }

            /* Settings dialog */
            QDialog { background: #1A1A1A; }
            QTabWidget::pane {
                border: 1px solid #2A2A2A;
                background: #141414;
            }
            QTabBar::tab {
                background: #1A1A1A;
                color: #777;
                padding: 7px 16px;
                border: 1px solid #2A2A2A;
                border-bottom: none;
                font-size: 12px;
            }
            QTabBar::tab:selected { color: #EFEFEF; background: #141414; font-weight: 600; }
            QTabBar::tab:hover    { color: #BBB; }
            QGroupBox {
                border: 1px solid #2A2A2A;
                border-radius: 5px;
                margin-top: 8px;
                font-size: 11px;
                color: #666;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QDialogButtonBox QPushButton {
                background: #1E1E1E;
                border: 1px solid #2E2E2E;
                border-radius: 5px;
                color: #AAAAAA;
                padding: 5px 16px;
                font-size: 12px;
                min-width: 70px;
            }
            QDialogButtonBox QPushButton:hover    { background: #252525; color: #DDD; }
            QDialogButtonBox QPushButton:default  { background: #1B3A5E; color: #7EC2F8; border-color: #378ADD; }
            QDialogButtonBox QPushButton:default:hover { background: #244A78; }
            QFormLayout QLabel { font-size: 12px; color: #888; }
            QComboBox {
                background: #1E1E1E;
                border: 1px solid #2E2E2E;
                border-radius: 5px;
                color: #DEDEDE;
                padding: 5px 8px;
                font-size: 12px;
            }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background: #1E1E1E;
                border: 1px solid #2E2E2E;
                selection-background-color: #1B3A5E;
                color: #DEDEDE;
            }
            #SettingsTabs QTabWidget::pane { border-top: 1px solid #2A2A2A; }
            #LineEdit {
                background: #1E1E1E;
                border: 1px solid #2E2E2E;
                border-radius: 5px;
                color: #DEDEDE;
                padding: 5px 8px;
                font-size: 12px;
            }

            /* Log panel */
            #LogBar {
                background: #161616;
                border-top: 1px solid #242424;
            }
            #LogToggle {
                background: transparent;
                border: none;
                color: #555;
                font-size: 10px;
                text-align: left;
                padding: 0;
            }
            #LogToggle:hover   { color: #888; }
            #LogToggle:checked { color: #AAA; }
            #LogClear {
                background: transparent;
                border: none;
                color: #444;
                font-size: 10px;
            }
            #LogClear:hover { color: #888; }
            #LogView {
                background: #0E0E0E;
                border: none;
                border-top: 1px solid #222;
                color: #888;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 11px;
            }

            /* Status bar */
            QStatusBar { background: #1A1A1A; border-top: 1px solid #2A2A2A; font-size: 10px; color: #555; }
            #StatusLabel { font-size: 10px; color: #555; }

            /* Scrollbars */
            QScrollBar:vertical {
                background: #141414;
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #333;
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal {
                background: #141414;
                height: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:horizontal {
                background: #333;
                border-radius: 3px;
                min-width: 20px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        """)
