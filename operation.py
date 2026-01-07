import sys
from dataclasses import dataclass
from typing import Optional, Dict

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QStackedWidget, QSlider, QSpinBox,
    QGroupBox, QGridLayout, QSizePolicy, QFileDialog
)


@dataclass
class UIParams:
    brightness: int = 0
    sharpness: int = 0
    denoise: int = 0
    red: int = 0
    green: int = 0
    blue: int = 0
    mono: bool = False
    preset: str = "none"
    crop_ratio: str = "Original"


class OperationWindow(QMainWindow):
    # ---- Signals (so your processing file can connect later) ----
    sig_cancel = Signal()
    sig_done = Signal(str)                  # path
    sig_apply = Signal(str, UIParams)       # tab_name, params
    sig_params_changed = Signal(UIParams)   # live preview
    sig_tab_changed = Signal(str)           # adjust/filter/removebg/crop
    sig_removebg_start = Signal()
    sig_removebg_cancel = Signal()
    sig_undo = Signal()
    sig_redo = Signal()

    def __init__(self, initial_pixmap: Optional[QPixmap] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("V.A.L.O. — Operation")

        self.params = UIParams()
        self._current_tab = "adjust"

        # Root
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Topbar
        topbar = QFrame()
        topbar.setObjectName("TopBar")
        topbar.setFixedHeight(58)
        top_l = QHBoxLayout(topbar)
        top_l.setContentsMargins(0, 0, 0, 0)

        title = QLabel("V.A.L.O.")
        title.setObjectName("TopTitle")
        title.setAlignment(Qt.AlignCenter)

        top_l.addStretch(1)
        top_l.addWidget(title)
        top_l.addStretch(1)

        root_layout.addWidget(topbar)

        # Shell
        shell = QWidget()
        shell_l = QVBoxLayout(shell)
        shell_l.setContentsMargins(18, 12, 18, 18)
        shell_l.setSpacing(12)
        root_layout.addWidget(shell, 1)

        # Toolbar
        toolbar = QWidget()
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.setSpacing(10)

        self.undo_btn = QPushButton("↩")
        self.undo_btn.setObjectName("IconBtn")
        self.undo_btn.setFixedSize(34, 34)
        self.undo_btn.clicked.connect(self.sig_undo.emit)

        self.redo_btn = QPushButton("↪")
        self.redo_btn.setObjectName("IconBtn")
        self.redo_btn.setFixedSize(34, 34)
        self.redo_btn.clicked.connect(self.sig_redo.emit)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("GhostBtn")
        self.cancel_btn.setFixedHeight(34)
        self.cancel_btn.clicked.connect(self.sig_cancel.emit)

        left_tools = QWidget()
        lt = QHBoxLayout(left_tools)
        lt.setContentsMargins(0, 0, 0, 0)
        lt.setSpacing(10)
        lt.addWidget(self.undo_btn)
        lt.addWidget(self.redo_btn)
        lt.addWidget(self.cancel_btn)

        self.done_btn = QPushButton("Done")
        self.done_btn.setObjectName("PrimaryBtn")
        self.done_btn.setFixedHeight(34)
        self.done_btn.clicked.connect(self._on_done_clicked)

        tb.addWidget(left_tools)
        tb.addStretch(1)
        tb.addWidget(self.done_btn)

        shell_l.addWidget(toolbar)

        # Body layout: panel + stage
        body = QWidget()
        body_l = QHBoxLayout(body)
        body_l.setContentsMargins(0, 0, 0, 0)
        body_l.setSpacing(18)
        shell_l.addWidget(body, 1)

        # Left panel
        panel = QWidget()
        panel.setFixedWidth(420)
        panel_l = QVBoxLayout(panel)
        panel_l.setContentsMargins(0, 0, 0, 0)
        panel_l.setSpacing(10)

        tabs = QWidget()
        tabs.setObjectName("Tabs")
        tabs_l = QHBoxLayout(tabs)
        tabs_l.setContentsMargins(0, 0, 0, 0)
        tabs_l.setSpacing(0)

        self.tab_buttons: Dict[str, QPushButton] = {}
        self.stack = QStackedWidget()

        def add_tab(label: str, name: str):
            b = QPushButton(label)
            b.setObjectName("TabBtn")
            b.setCheckable(True)
            b.clicked.connect(lambda: self.open_tab(name))
            self.tab_buttons[name] = b
            tabs_l.addWidget(b)

        add_tab("Adjust", "adjust")
        add_tab("Filter", "filter")
        add_tab("Remove BG", "removebg")
        add_tab("Crop", "crop")

        panel_l.addWidget(tabs)
        panel_l.addWidget(self.stack, 1)

        # ---- Adjust page ----
        adjust_page = QWidget()
        adj = QVBoxLayout(adjust_page)
        adj.setContentsMargins(0, 0, 0, 0)
        adj.setSpacing(10)

        self._make_slider_block(adj, "Brightness", -100, 100, "brightness")
        self._make_slider_block(adj, "Sharpness", -100, 100, "sharpness")
        self._make_slider_block(adj, "Noise Reduction", 0, 100, "denoise")
        adj.addStretch(1)

        # ---- Filter page ----
        filter_page = QWidget()
        fil = QVBoxLayout(filter_page)
        fil.setContentsMargins(0, 0, 0, 0)
        fil.setSpacing(10)

        presets = QGroupBox("Presets")
        presets.setObjectName("Block")
        g = QGridLayout(presets)
        g.setContentsMargins(12, 12, 12, 12)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(10)

        preset_names = ["none", "mono", "dramatic-warm", "noir", "dramatic-cool"]
        self.preset_btns: Dict[str, QPushButton] = {}
        for i, p in enumerate(preset_names):
            btn = QPushButton(p.replace("-", " ").title())
            btn.setObjectName("CropBtn")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, name=p: self._on_preset_clicked(name))
            self.preset_btns[p] = btn
            g.addWidget(btn, i // 3, i % 3)

        fil.addWidget(presets)

        self._make_slider_block(fil, "Red", -100, 100, "red")
        self._make_slider_block(fil, "Green", -100, 100, "green")
        self._make_slider_block(fil, "Blue", -100, 100, "blue")

        self.mono_toggle = QPushButton("Mono (toggle)")
        self.mono_toggle.setObjectName("GhostBtn")
        self.mono_toggle.setCheckable(True)
        self.mono_toggle.clicked.connect(self._on_mono_toggle)
        fil.addWidget(self.mono_toggle)

        fil.addStretch(1)

        # ---- Remove BG page ----
        remove_page = QWidget()
        rm = QVBoxLayout(remove_page)
        rm.setContentsMargins(0, 0, 0, 0)
        rm.setSpacing(10)

        remove_box = QGroupBox("Remove Background")
        remove_box.setObjectName("Block")
        rb = QVBoxLayout(remove_box)

        self.remove_status = QLabel("Ready")
        self.remove_status.setObjectName("RemoveTitle")
        self.remove_status.setAlignment(Qt.AlignCenter)

        btn_row = QWidget()
        br = QHBoxLayout(btn_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(10)

        self.remove_start_btn = QPushButton("Start")
        self.remove_start_btn.setObjectName("PrimaryBtn")
        self.remove_start_btn.setFixedHeight(34)
        self.remove_start_btn.clicked.connect(self.sig_removebg_start.emit)

        self.remove_cancel_btn = QPushButton("Cancel")
        self.remove_cancel_btn.setObjectName("GhostBtn")
        self.remove_cancel_btn.setFixedHeight(34)
        self.remove_cancel_btn.setEnabled(False)
        self.remove_cancel_btn.clicked.connect(self.sig_removebg_cancel.emit)

        br.addStretch(1)
        br.addWidget(self.remove_start_btn)
        br.addWidget(self.remove_cancel_btn)
        br.addStretch(1)

        hint = QLabel("Start makes a preview. Press ✓ to apply.")
        hint.setObjectName("HintText")
        hint.setAlignment(Qt.AlignCenter)

        rb.addWidget(self.remove_status)
        rb.addWidget(btn_row)
        rb.addWidget(hint)

        rm.addWidget(remove_box)
        rm.addStretch(1)

        # ---- Crop page ----
        crop_page = QWidget()
        cp = QVBoxLayout(crop_page)
        cp.setContentsMargins(0, 0, 0, 0)
        cp.setSpacing(10)

        crop_box = QGroupBox("Crop Ratios")
        crop_box.setObjectName("Block")
        cg = QGridLayout(crop_box)
        cg.setContentsMargins(12, 12, 12, 12)
        cg.setHorizontalSpacing(10)
        cg.setVerticalSpacing(10)

        crop_labels = ["Original", "Custom", "Square", "9:16", "4:5", "5:7", "3:4", "3:5", "2:3"]
        self.crop_btns: Dict[str, QPushButton] = {}
        for i, lab in enumerate(crop_labels):
            b = QPushButton(lab)
            b.setObjectName("CropBtn")
            b.setCheckable(True)
            b.clicked.connect(lambda _, t=lab: self._on_crop_ratio(t))
            self.crop_btns[lab] = b
            cg.addWidget(b, i // 3, i % 3)

        crop_hint = QLabel("Pick ratio, then crop in your processing file. Press ✓ to apply.")
        crop_hint.setObjectName("HintText")

        cp.addWidget(crop_box)
        cp.addWidget(crop_hint)
        cp.addStretch(1)

        # Add pages
        self.stack.addWidget(adjust_page)
        self.stack.addWidget(filter_page)
        self.stack.addWidget(remove_page)
        self.stack.addWidget(crop_page)

        body_l.addWidget(panel, 0)

        # Stage area
        stage_col = QWidget()
        stage_l = QVBoxLayout(stage_col)
        stage_l.setContentsMargins(0, 0, 0, 0)
        stage_l.setSpacing(12)

        stage_frame = QFrame()
        stage_frame.setObjectName("StageFrame")
        sf = QVBoxLayout(stage_frame)
        sf.setContentsMargins(18, 18, 18, 18)

        self.stage_label = QLabel()
        self.stage_label.setAlignment(Qt.AlignCenter)
        self.stage_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stage_label.setMinimumSize(520, 420)

        if initial_pixmap is not None:
            self.set_stage_pixmap(initial_pixmap)

        sf.addWidget(self.stage_label)
        stage_l.addWidget(stage_frame, 1)

        actions = QWidget()
        al = QHBoxLayout(actions)
        al.setContentsMargins(0, 0, 0, 0)

        self.apply_btn = QPushButton("✓")
        self.apply_btn.setObjectName("PrimaryBtn")
        self.apply_btn.setFixedSize(44, 34)
        self.apply_btn.clicked.connect(self._on_apply)

        al.addStretch(1)
        al.addWidget(self.apply_btn)
        al.addStretch(1)

        stage_l.addWidget(actions, 0)

        body_l.addWidget(stage_col, 1)

        # Styles
        self.setStyleSheet("""
            QWidget { background:#e9e6e3; color:#111; font-family: system-ui, Segoe UI, Arial; }
            #TopBar { background:#d9d9d9; }
            #TopTitle { font-weight:900; letter-spacing:0.22em; font-size:28px; }

            #Tabs { background:#e7e7e7; border:1px solid #cfcfcf; border-radius:10px; }
            QPushButton#TabBtn { background:#e7e7e7; border:none; padding:10px 8px; font-weight:800; }
            QPushButton#TabBtn:checked { background:#d6d6d6; }

            QPushButton#IconBtn { background:#f6f6f6; border:1px solid #cfcfcf; border-radius:8px; }
            QPushButton#GhostBtn { background:#f6f6f6; border:1px solid #cfcfcf; border-radius:8px; padding:0 12px; font-weight:700; }
            QPushButton#PrimaryBtn { background:#1f74ff; color:white; border:none; border-radius:8px; padding:0 16px; font-weight:800; }
            QPushButton:disabled { opacity:0.45; }

            QGroupBox#Block { border:1px solid #cfcfcf; border-radius:10px; margin-top:10px; background:#e7e7e7; }
            QGroupBox#Block::title { subcontrol-origin: margin; left:12px; padding:0 6px; font-weight:900; }

            QLabel#RemoveTitle { font-weight:900; }
            QLabel#HintText { color:#6b6b6b; font-size:12px; }

            QPushButton#CropBtn { background:#f3f3f3; border:1px solid #cfcfcf; border-radius:8px; padding:10px 8px; font-weight:800; }
            QPushButton#CropBtn:checked { background:#d6d6d6; }

            #StageFrame { background:#efefef; border:2px solid #cfcfcf; }
        """)

        # Default selections
        self.open_tab("adjust")
        self._set_active_preset("none")
        self._set_active_crop("Original")

        self.showMaximized()

    # ---------------- UI helpers ----------------
    def set_stage_pixmap(self, pix: QPixmap):
        if pix is None:
            self.stage_label.clear()
            return
        self.stage_label.setPixmap(pix.scaled(
            self.stage_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        pm = self.stage_label.pixmap()
        if pm:
            self.set_stage_pixmap(pm)

    def _make_slider_block(self, parent: QVBoxLayout, title: str, lo: int, hi: int, key: str):
        box = QGroupBox(title)
        box.setObjectName("Block")
        l = QVBoxLayout(box)
        l.setContentsMargins(12, 12, 12, 12)
        l.setSpacing(8)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(0)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)

        label = QLabel("value")
        label.setFixedWidth(42)

        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(0)

        reset = QPushButton("⟲")
        reset.setObjectName("IconBtn")
        reset.setFixedSize(34, 34)

        def set_value(v: int):
            setattr(self.params, key, int(v))
            self.sig_params_changed.emit(self.params)

        slider.valueChanged.connect(lambda v: (spin.blockSignals(True), spin.setValue(v), spin.blockSignals(False), set_value(v)))
        spin.valueChanged.connect(lambda v: (slider.blockSignals(True), slider.setValue(v), slider.blockSignals(False), set_value(v)))
        reset.clicked.connect(lambda: (slider.setValue(0), spin.setValue(0)))

        rl.addWidget(label)
        rl.addWidget(spin)
        rl.addStretch(1)
        rl.addWidget(reset)

        l.addWidget(slider)
        l.addWidget(row)

        parent.addWidget(box)

    # ---------------- Tabs ----------------
    def open_tab(self, name: str):
        self._current_tab = name
        order = ["adjust", "filter", "removebg", "crop"]
        for n, btn in self.tab_buttons.items():
            btn.setChecked(n == name)

        self.stack.setCurrentIndex(order.index(name))
        self.sig_tab_changed.emit(name)

    # ---------------- Filter preset / mono ----------------
    def _set_active_preset(self, preset: str):
        self.params.preset = preset
        for n, b in self.preset_btns.items():
            b.blockSignals(True)
            b.setChecked(n == preset)
            b.blockSignals(False)

    def _on_preset_clicked(self, preset: str):
        self._set_active_preset(preset)
        self.sig_params_changed.emit(self.params)

    def _on_mono_toggle(self):
        self.params.mono = bool(self.mono_toggle.isChecked())
        self.sig_params_changed.emit(self.params)

    # ---------------- Crop ----------------
    def _set_active_crop(self, label: str):
        self.params.crop_ratio = label
        for n, b in self.crop_btns.items():
            b.blockSignals(True)
            b.setChecked(n == label)
            b.blockSignals(False)

    def _on_crop_ratio(self, label: str):
        self._set_active_crop(label)
        self.sig_params_changed.emit(self.params)

    # ---------------- Apply / Done ----------------
    def _on_apply(self):
        # UI-only: just emit signal; your processing file will commit
        self.sig_apply.emit(self._current_tab, self.params)

    def _on_done_clicked(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "download/image_01.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;WEBP (*.webp)"
        )
        if not path:
            return
        self.sig_done.emit(path)


# ---- test run ----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OperationWindow()
    win.show()
    sys.exit(app.exec())
