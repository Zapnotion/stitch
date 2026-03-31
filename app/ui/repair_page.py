"""
ui/repair_page.py — Repair / inpainting tab.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from app.backend.ace_step import ACEStepPipeline
from app.backend.exporter import ExportWorker
from app.backend.session import session
from app.backend.worker import RepairWorker
from app.config import cfg
from app.models.generation import GenerationResult, RepairMode, RepairRequest
from app.ui.widgets.audio_player import AudioPlayer
from app.ui.widgets.upload_zone import UploadZone
from app.ui.widgets.waveform import WaveformWidget


class RepairPage(QWidget):
    status_message = Signal(str)

    def __init__(self, pipeline: ACEStepPipeline, parent=None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._workers: list = []
        self._result_cards: list[ResultCard] = []
        self._region_start = 0.0
        self._region_end   = 0.0
        self._current_repair_result = None
        self._build_ui()

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Left ---
        left = QWidget()
        left.setObjectName("LeftPanel")
        left.setFixedWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        panel = QWidget()
        panel_lay = QVBoxLayout(panel)
        panel_lay.setContentsMargins(14, 12, 14, 12)
        panel_lay.setSpacing(10)

        panel_lay.addWidget(self._field_label("Audio file"))
        self._upload = UploadZone("Drop audio file here")
        self._upload.file_selected.connect(self._on_file_selected)
        self._upload.file_cleared.connect(self._on_file_cleared)
        panel_lay.addWidget(self._upload)

        panel_lay.addWidget(self._divider())

        panel_lay.addWidget(self._field_label("Repair mode"))
        self._mode_region = QPushButton("Selected region")
        self._mode_region.setObjectName("PillBtn")
        self._mode_region.setCheckable(True)
        self._mode_region.setChecked(True)
        self._mode_full = QPushButton("Full file")
        self._mode_full.setObjectName("PillBtn")
        self._mode_full.setCheckable(True)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        mode_row.addWidget(self._mode_region)
        mode_row.addWidget(self._mode_full)
        mode_row.addStretch()

        def _pick_mode_region():
            self._mode_region.setChecked(True)
            self._mode_full.setChecked(False)

        def _pick_mode_full():
            self._mode_full.setChecked(True)
            self._mode_region.setChecked(False)

        self._mode_region.clicked.connect(_pick_mode_region)
        self._mode_full.clicked.connect(_pick_mode_full)
        panel_lay.addLayout(mode_row)

        panel_lay.addWidget(self._field_label("Hint prompt (optional)"))
        self._hint = QPlainTextEdit()
        self._hint.setObjectName("PromptBox")
        self._hint.setPlaceholderText("describe what's in this audio to help the model…")
        self._hint.setFixedHeight(60)
        panel_lay.addWidget(self._hint)

        panel_lay.addStretch()
        left_layout.addWidget(panel, 1)

        self._repair_btn = QPushButton("Run repair pass")
        self._repair_btn.setObjectName("GenBtnGreen")
        self._repair_btn.clicked.connect(self._on_repair)
        left_layout.addWidget(self._repair_btn)

        self._hint_lbl = QLabel("select a region on the waveform first")
        self._hint_lbl.setObjectName("HintLabel")
        self._hint_lbl.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self._hint_lbl)

        root.addWidget(left)

        # --- Right ---
        right = QWidget()
        right.setObjectName("RightPanel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        hdr = QWidget()
        hdr.setObjectName("RightHeader")
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(14, 10, 14, 10)
        hdr_lay.addWidget(self._lbl("Waveform", "RightTitle"))
        hdr_lay.addWidget(self._lbl("drag to select a region to repair", "RightSub"))
        hdr_lay.addStretch()
        right_layout.addWidget(hdr)

        waveform_container = QWidget()
        waveform_container.setObjectName("WaveformArea")
        wc_lay = QVBoxLayout(waveform_container)
        wc_lay.setContentsMargins(16, 14, 16, 0)
        wc_lay.setSpacing(4)

        self._filename_lbl = QLabel("")
        self._filename_lbl.setObjectName("WaveFilename")
        wc_lay.addWidget(self._filename_lbl)

        self._waveform = WaveformWidget(selectable=True)
        self._waveform.setFixedHeight(72)
        self._waveform.region_selected.connect(self._on_region_selected)
        wc_lay.addWidget(self._waveform)

        self._region_lbl = QLabel("")
        self._region_lbl.setObjectName("RegionLabel")
        self._region_lbl.setAlignment(Qt.AlignCenter)
        wc_lay.addWidget(self._region_lbl)

        # Source file player
        self._source_player = AudioPlayer()
        self._source_player._scrub.valueChanged.connect(
            lambda v: self._waveform.set_playhead(v / 1000.0)
        )
        self._source_player.playback_stopped.connect(
            lambda _: self._waveform.set_playhead(-1.0)
        )
        wc_lay.addWidget(self._source_player)

        right_layout.addWidget(waveform_container)

        # Comparison area (shown after repair)
        self._comparison = QWidget()
        self._comparison.setObjectName("ComparisonArea")
        self._comparison.setVisible(False)
        comp_lay = QVBoxLayout(self._comparison)
        comp_lay.setContentsMargins(16, 12, 16, 12)
        comp_lay.setSpacing(8)

        comp_hdr = QLabel("AFTER REPAIR — COMPARISON")
        comp_hdr.setObjectName("StemsHeader")
        comp_lay.addWidget(comp_hdr)

        waves_row = QHBoxLayout()
        bef_box = QVBoxLayout()
        bef_box.addWidget(self._lbl("Before", "FieldLabel"))
        self._before_wave = WaveformWidget(selectable=False)
        self._before_wave.setFixedHeight(36)
        bef_box.addWidget(self._before_wave)
        self._before_player = AudioPlayer()
        bef_box.addWidget(self._before_player)
        waves_row.addLayout(bef_box)

        aft_box = QVBoxLayout()
        aft_box.addWidget(self._lbl("After", "FieldLabel"))
        self._after_wave = WaveformWidget(selectable=False)
        self._after_wave.setFixedHeight(36)
        aft_box.addWidget(self._after_wave)
        self._after_player = AudioPlayer()
        aft_box.addWidget(self._after_player)
        waves_row.addLayout(aft_box)
        comp_lay.addLayout(waves_row)

        action_row = QHBoxLayout()
        self._keep_repaired_btn = QPushButton("Keep repaired")
        self._keep_repaired_btn.setObjectName("KeepBtn")
        self._keep_original_btn = QPushButton("Keep original")
        self._keep_original_btn.setObjectName("ActionBtn")
        self._save_repaired_btn = QPushButton("⬇ Save as…")
        self._save_repaired_btn.setObjectName("ActionBtn")
        self._save_repaired_btn.setEnabled(False)
        action_row.addWidget(self._keep_repaired_btn)
        action_row.addWidget(self._keep_original_btn)
        action_row.addWidget(self._save_repaired_btn)
        comp_lay.addLayout(action_row)

        right_layout.addWidget(self._comparison)
        right_layout.addStretch()

        root.addWidget(right, 1)

    # -----------------------------------------------------------------------
    # Handlers
    # -----------------------------------------------------------------------

    def _on_file_selected(self, path: str) -> None:
        self._waveform.load_audio(path)
        self._filename_lbl.setText(f"{Path(path).name}")
        self._before_wave.load_audio(path)
        self._source_player.load(path)
        self._before_player.load(path)

    def _on_file_cleared(self) -> None:
        self._waveform.clear()
        self._filename_lbl.setText("")
        self._comparison.setVisible(False)

    def _on_region_selected(self, start: float, end: float) -> None:
        self._region_start = start
        self._region_end   = end
        fmt = lambda s: f"{int(s)//60}:{int(s)%60:02d}"
        self._region_lbl.setText(f"selected: {fmt(start)} – {fmt(end)}")

    def _on_repair(self) -> None:
        if not self._upload.loaded_path:
            self.status_message.emit("Please load an audio file first.")
            return

        mode = RepairMode.REGION if self._mode_region.isChecked() else RepairMode.FULL_FILE
        if mode == RepairMode.REGION and self._region_end <= self._region_start:
            self.status_message.emit("Please drag to select a region on the waveform first.")
            return

        req = RepairRequest(
            source_audio_path = self._upload.loaded_path,
            mode              = mode,
            region_start_sec  = self._region_start,
            region_end_sec    = self._region_end,
            hint_prompt       = self._hint.toPlainText().strip(),
            output_dir        = str(cfg.outputs_dir),
        )

        self._repair_btn.setEnabled(False)
        self._repair_btn.setText("Repairing…")
        self.status_message.emit("Running repair pass…")

        w = RepairWorker(self._pipeline, req)
        w.result.connect(self._on_repair_done)
        w.error.connect(self._on_error)
        w.done.connect(self._on_worker_done)
        w.start()
        self._workers.append(w)

    def _on_repair_done(self, results: list[GenerationResult]) -> None:
        if not results:
            return
        r = results[0]
        self._current_repair_result = r
        self._after_wave.load_audio(r.audio_path)
        self._after_player.load(r.audio_path)
        self._save_repaired_btn.setEnabled(True)

        # Disconnect previous connections to avoid stacking signals across repeated repairs
        try: self._keep_repaired_btn.clicked.disconnect()
        except RuntimeError: pass
        try: self._keep_original_btn.clicked.disconnect()
        except RuntimeError: pass
        try: self._save_repaired_btn.clicked.disconnect()
        except RuntimeError: pass

        self._keep_repaired_btn.clicked.connect(lambda: self._accept_repair(r))
        self._keep_original_btn.clicked.connect(self._reject_repair)
        self._save_repaired_btn.clicked.connect(lambda: self._on_save_repaired(r))

        # Record to session history
        session.record(
            mode            = "repair",
            audio_path      = r.audio_path,
            duration_secs   = r.duration_secs,
            seed            = r.seed,
            variation_index = 1,
            style_prompt    = r.style_prompt,
        )

        self._comparison.setVisible(True)
        self.status_message.emit("Repair complete — compare before/after above")

    def _accept_repair(self, result: GenerationResult) -> None:
        self._comparison.setVisible(False)
        self.status_message.emit(f"Kept repaired version: {result.filename}")

    def _reject_repair(self) -> None:
        self._comparison.setVisible(False)
        self.status_message.emit("Kept original.")

    def _on_save_repaired(self, result: GenerationResult) -> None:
        from PySide6.QtWidgets import QFileDialog
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save repaired audio",
            str(cfg.outputs_dir / result.filename),
            "Audio files (*.wav *.mp3)"
        )
        if not dest:
            return
        fmt = "mp3" if dest.lower().endswith(".mp3") else "wav"
        w = ExportWorker(result.audio_path, dest, fmt=fmt, bitrate=cfg.mp3_bitrate)
        w.finished.connect(lambda p: self.status_message.emit(f"Saved: {Path(p).name}"))
        w.error.connect(lambda e: self.status_message.emit(f"Save error: {e}"))
        w.start()
        self._workers.append(w)

    def _on_error(self, msg: str) -> None:
        self.status_message.emit(f"Repair error: {msg}")

    def _on_worker_done(self) -> None:
        self._cleanup_workers()
        self._repair_btn.setEnabled(True)
        self._repair_btn.setText("Run repair pass")

    # --- Helpers ------------------------------------------------------------

    def _cleanup_workers(self) -> None:
        """Remove finished workers to avoid accumulation."""
        self._workers = [w for w in self._workers if not w.isFinished()]

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("FieldLabel")
        return lbl

    def _lbl(self, text: str, obj: str = "") -> QLabel:
        lbl = QLabel(text)
        if obj:
            lbl.setObjectName(obj)
        return lbl

    def _divider(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setObjectName("Divider")
        return f
