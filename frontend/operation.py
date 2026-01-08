import sys
import cv2
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Tuple, List
from PySide6.QtCore import Qt, Signal, QObject, QThread, QRect, QPoint
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QStackedWidget, QSlider, QSpinBox, QGroupBox, QGridLayout, QSizePolicy, QFileDialog)
from backend.main import process_image, remove_background


# =========================
# Params
# =========================
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
    crop: dict = field(default_factory=lambda: {"enabled": False, "x": 0, "y": 0, "w": 1, "h": 1})


# =========================
# Undo/Redo state snapshot
# =========================
@dataclass
class HistoryState:
    base_idx: int
    params: dict


# =========================
# Helpers
# =========================
def qpixmap_to_bgr(pixmap: QPixmap) -> np.ndarray:
    if pixmap is None or pixmap.isNull():
        raise ValueError("Invalid pixmap")

    img = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()

    buf = np.frombuffer(img.constBits(), dtype=np.uint8).reshape((h, bpl))
    rgba = buf[:, : w * 4].reshape((h, w, 4))
    rgb = rgba[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def bgra_to_qpixmap(bgra: np.ndarray) -> QPixmap:
    rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
    h, w = rgba.shape[:2]
    qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def split_bgra(img: np.ndarray):
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3].copy(), img[:, :, 3].copy()
    return img, None


def merge_bgra(bgr: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    if alpha is None:
        return bgr
    return cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])


def apply_crop_local(img: np.ndarray, crop: dict) -> np.ndarray:
    if img is None:
        return img
    if not crop or not crop.get("enabled", False):
        return img

    H, W = img.shape[:2]
    x = int(np.clip(crop.get("x", 0), 0, 1) * W)
    y = int(np.clip(crop.get("y", 0), 0, 1) * H)
    cw = int(np.clip(crop.get("w", 1), 0, 1) * W)
    ch = int(np.clip(crop.get("h", 1), 0, 1) * H)

    cw = max(1, min(cw, W - x))
    ch = max(1, min(ch, H - y))
    return img[y:y + ch, x:x + cw].copy()


# =========================
# Crop Stage Label
# =========================
class CropStageLabel(QLabel):
    sig_became_custom = Signal()

    HANDLE_SIZE = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        self._overlay_enabled = False

        self._mask_color = QColor(0, 0, 0, 120)
        self._rect_pen = QPen(QColor(255, 255, 255), 2)
        self._rect_pen.setCosmetic(True)
        self._handle_brush = QBrush(QColor(255, 255, 255))

        self._crop_rect = QRect()
        self._dragging = False
        self._drag_mode = None  # "move" or handle name
        self._start_pos = QPoint()
        self._start_rect = QRect()

        self._aspect: Optional[float] = None  # w/h locked ratio
        self._on_crop_changed = None
        self._told_custom = False

    def set_overlay_enabled(self, enabled: bool):
        self._overlay_enabled = enabled
        self.update()

    def set_crop_callback(self, fn):
        self._on_crop_changed = fn

    def set_aspect_ratio(self, aspect: Optional[float]):
        self._aspect = aspect
        self._told_custom = False

    def clear_crop(self):
        self._crop_rect = QRect()
        self._told_custom = False
        self.update()
        if self._on_crop_changed:
            self._on_crop_changed({"enabled": False, "x": 0, "y": 0, "w": 1, "h": 1})

    def ensure_crop_box(self, mode: str):
        draw = self._pixmap_draw_rect()
        if draw.isNull():
            return

        if mode == "custom" or self._aspect is None:
            w = int(draw.width() * 0.80)
            h = int(draw.height() * 0.80)
            cx, cy = draw.center().x(), draw.center().y()
            left = cx - w // 2
            top = cy - h // 2
            self._crop_rect = QRect(left, top, w, h).intersected(draw)
            self._told_custom = False
            self.update()
            self._emit_normalized()
            return

        target_aspect = self._aspect  # w/h
        dw, dh = draw.width(), draw.height()

        w = int(dw * 0.80)
        h = int(w / target_aspect)
        if h > int(dh * 0.80):
            h = int(dh * 0.80)
            w = int(h * target_aspect)

        cx, cy = draw.center().x(), draw.center().y()
        left = cx - w // 2
        top = cy - h // 2
        self._crop_rect = QRect(left, top, w, h).intersected(draw)

        self._told_custom = False
        self.update()
        self._emit_normalized()

    def _pixmap_draw_rect(self) -> QRect:
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return QRect()

        lw, lh = self.width(), self.height()
        pw, ph = pm.width(), pm.height()
        scale = min(lw / pw, lh / ph)
        dw, dh = int(pw * scale), int(ph * scale)
        x = (lw - dw) // 2
        y = (lh - dh) // 2
        return QRect(x, y, dw, dh)

    def _handles(self, r: QRect) -> dict:
        s = self.HANDLE_SIZE
        cx, cy = r.center().x(), r.center().y()

        def box(x, y):
            return QRect(x - s // 2, y - s // 2, s, s)

        return {
            "tl": box(r.left(), r.top()),
            "tr": box(r.right(), r.top()),
            "bl": box(r.left(), r.bottom()),
            "br": box(r.right(), r.bottom()),
            "tm": box(cx, r.top()),
            "bm": box(cx, r.bottom()),
            "ml": box(r.left(), cy),
            "mr": box(r.right(), cy),
        }

    def _hit_test(self, pos: QPoint) -> Optional[str]:
        draw = self._pixmap_draw_rect()
        if draw.isNull() or self._crop_rect.isNull():
            return None

        r = self._crop_rect.intersected(draw)
        if r.isNull():
            return None

        for name, hr in self._handles(r).items():
            if hr.contains(pos):
                return name

        if r.contains(pos):
            return "move"

        return None

    def _clamp_rect(self, r: QRect) -> QRect:
        draw = self._pixmap_draw_rect()
        if draw.isNull():
            return r
        return r.intersected(draw)

    def _emit_normalized(self):
        draw = self._pixmap_draw_rect()
        if draw.isNull() or self._crop_rect.isNull():
            return

        r = self._crop_rect.intersected(draw)
        if r.isNull() or r.width() < 2 or r.height() < 2:
            return

        nx = (r.x() - draw.x()) / draw.width()
        ny = (r.y() - draw.y()) / draw.height()
        nw = r.width() / draw.width()
        nh = r.height() / draw.height()

        if self._on_crop_changed:
            self._on_crop_changed({
                "enabled": True,
                "x": float(max(0, min(1, nx))),
                "y": float(max(0, min(1, ny))),
                "w": float(max(0, min(1, nw))),
                "h": float(max(0, min(1, nh))),
            })

    def _notify_custom_once(self):
        if not self._told_custom:
            self._told_custom = True
            self.sig_became_custom.emit()

    def mousePressEvent(self, e):
        if not self._overlay_enabled:
            return
        if e.button() != Qt.LeftButton:
            return

        draw = self._pixmap_draw_rect()
        if draw.isNull():
            return

        p = e.position().toPoint()
        if not draw.contains(p):
            return

        if self._crop_rect.isNull():
            self.ensure_crop_box("custom" if self._aspect is None else "aspect")

        hit = self._hit_test(p)
        if hit is None:
            return

        self._dragging = True
        self._drag_mode = hit
        self._start_pos = p
        self._start_rect = QRect(self._crop_rect)

    def mouseMoveEvent(self, e):
        if not self._overlay_enabled or not self._dragging:
            return

        draw = self._pixmap_draw_rect()
        if draw.isNull() or self._crop_rect.isNull():
            return

        p = e.position().toPoint()
        dx = p.x() - self._start_pos.x()
        dy = p.y() - self._start_pos.y()

        r0 = QRect(self._start_rect)
        mode = self._drag_mode
        min_size = 20

        if mode == "move":
            r = QRect(r0)
            r.translate(dx, dy)

            if r.left() < draw.left():
                r.moveLeft(draw.left())
            if r.top() < draw.top():
                r.moveTop(draw.top())
            if r.right() > draw.right():
                r.moveRight(draw.right())
            if r.bottom() > draw.bottom():
                r.moveBottom(draw.bottom())

            self._crop_rect = r
            self.update()
            self._emit_normalized()
            return

        # resizing => custom
        if self._aspect is not None:
            self._aspect = None
            self._notify_custom_once()

        r = QRect(r0)

        if mode in ("tl", "ml", "bl"):
            r.setLeft(r0.left() + dx)
        if mode in ("tr", "mr", "br"):
            r.setRight(r0.right() + dx)
        if mode in ("tl", "tm", "tr"):
            r.setTop(r0.top() + dy)
        if mode in ("bl", "bm", "br"):
            r.setBottom(r0.bottom() + dy)

        if r.width() < min_size:
            if mode in ("tl", "ml", "bl"):
                r.setLeft(r.right() - min_size)
            else:
                r.setRight(r.left() + min_size)

        if r.height() < min_size:
            if mode in ("tl", "tm", "tr"):
                r.setTop(r.bottom() - min_size)
            else:
                r.setBottom(r.top() + min_size)

        r = self._clamp_rect(r)
        self._crop_rect = r
        self.update()
        self._emit_normalized()

    def mouseReleaseEvent(self, e):
        if not self._overlay_enabled:
            return
        if e.button() != Qt.LeftButton:
            return
        self._dragging = False
        self._drag_mode = None
        self._emit_normalized()
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)

        if not self._overlay_enabled:
            return

        draw = self._pixmap_draw_rect()
        if draw.isNull() or self._crop_rect.isNull():
            return

        r = self._crop_rect.intersected(draw)
        if r.isNull() or r.width() < 3 or r.height() < 3:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self._mask_color))
        p.drawRect(draw.x(), draw.y(), draw.width(), r.y() - draw.y())
        p.drawRect(draw.x(), r.bottom(), draw.width(), draw.bottom() - r.bottom() + 1)
        p.drawRect(draw.x(), r.y(), r.x() - draw.x(), r.height())
        p.drawRect(r.right(), r.y(), draw.right() - r.right() + 1, r.height())

        p.setBrush(Qt.NoBrush)
        p.setPen(self._rect_pen)
        p.drawRect(r)

        p.setBrush(self._handle_brush)
        p.setPen(Qt.NoPen)
        for hr in self._handles(r).values():
            p.drawRect(hr)

        p.end()


# =========================
# Remove BG Worker
# =========================
class RemoveBgWorker(QObject):
    finished = Signal(object)   # BGRA ndarray
    error = Signal(str)

    def __init__(self, bgr_img: np.ndarray):
        super().__init__()
        self.bgr_img = bgr_img

    def run(self):
        try:
            out = remove_background(self.bgr_img)
            self.finished.emit(out)
        except Exception as e:
            self.error.emit(str(e))


# =========================
# Operation Window
# =========================
class OperationWindow(QMainWindow):
    sig_cancel = Signal()
    sig_done = Signal(str)
    sig_params_changed = Signal(UIParams)

    def __init__(self, initial_pixmap: Optional[QPixmap] = None, parent=None):
        super().__init__(parent)
        title = QLabel("V.A.L.O.")
        title.setObjectName("TopTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setAttribute(Qt.WA_StyledBackground, True)

        self.params = UIParams()
        self._current_tab = "adjust"

        self._base_images: List[np.ndarray] = []
        self._base_idx: int = -1

        self._stage_pixmap_source: Optional[QPixmap] = None

        # remove bg state
        self._remove_running = False
        self._remove_cancelled = False
        self._remove_thread: Optional[QThread] = None
        self._remove_worker: Optional[RemoveBgWorker] = None
        self._remove_preview_bgra: Optional[np.ndarray] = None  # preview result, not committed
        self._remove_dirty = False

        # crop dirty
        self._crop_dirty = False

        # history
        self._history: List[HistoryState] = []
        self._hist_idx: int = -1
        self._suspend_history = False

        self._controls: Dict[str, Tuple[QSlider, QSpinBox]] = {}

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
        self.undo_btn.setFixedSize(44, 34)
        self.undo_btn.clicked.connect(self._undo)

        self.redo_btn = QPushButton("↪")
        self.redo_btn.setObjectName("IconBtn")
        self.redo_btn.setFixedSize(44, 34)
        self.redo_btn.clicked.connect(self._redo)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("GhostBtn")
        self.cancel_btn.setFixedHeight(34)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)

        self.done_btn = QPushButton("Done")
        self.done_btn.setObjectName("PrimaryBtn")
        self.done_btn.setFixedHeight(34)
        self.done_btn.clicked.connect(self._on_done_clicked)

        tb.addWidget(self.undo_btn)
        tb.addWidget(self.redo_btn)
        tb.addWidget(self.cancel_btn)
        tb.addStretch(1)
        tb.addWidget(self.done_btn)
        shell_l.addWidget(toolbar)

        # Body
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
        add_tab("Remove Background", "removebg")
        add_tab("Crop", "crop")

        panel_l.addWidget(tabs)
        panel_l.addWidget(self.stack, 1)

        # --- Adjust page ---
        adjust_page = QWidget()
        adj = QVBoxLayout(adjust_page)
        adj.setContentsMargins(0, 0, 0, 0)
        adj.setSpacing(10)
        self._make_slider_block(adj, "Brightness", -100, 100, "brightness")
        self._make_slider_block(adj, "Sharpness", -100, 100, "sharpness")
        self._make_slider_block(adj, "Noise Reduction", 0, 100, "denoise")
        adj.addStretch(1)

        # --- Filter page ---
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
        fil.addStretch(1)

        # --- Remove BG page ---
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
        self.remove_start_btn.clicked.connect(self._on_removebg_start)

        self.remove_cancel_btn = QPushButton("Cancel")
        self.remove_cancel_btn.setObjectName("GhostBtn")
        self.remove_cancel_btn.setFixedHeight(34)
        self.remove_cancel_btn.setEnabled(False)
        self.remove_cancel_btn.clicked.connect(self._on_removebg_cancel)

        br.addStretch(1)
        br.addWidget(self.remove_start_btn)
        br.addWidget(self.remove_cancel_btn)
        br.addStretch(1)

        hint = QLabel("Press Start to preview background removal.")
        hint.setObjectName("HintText")
        hint.setAlignment(Qt.AlignCenter)

        rb.addWidget(self.remove_status)
        rb.addWidget(btn_row)
        rb.addWidget(hint)

        rm.addWidget(remove_box)

        # Apply button OUTSIDE the panel (your #3)
        self.remove_apply_btn = QPushButton("✓ Apply")
        self.remove_apply_btn.setObjectName("ApplyBtn")
        self.remove_apply_btn.setFixedHeight(36)
        self.remove_apply_btn.setEnabled(False)
        self.remove_apply_btn.clicked.connect(self._apply_removebg)

        apply_out = QWidget()
        ao = QHBoxLayout(apply_out)
        ao.setContentsMargins(0, 0, 0, 0)
        ao.addStretch(1)
        ao.addWidget(self.remove_apply_btn)
        ao.addStretch(1)

        rm.addWidget(apply_out)
        rm.addStretch(1)

        # --- Crop page ---
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

        self.crop_apply_btn = QPushButton("✓ Apply")
        self.crop_apply_btn.setObjectName("ApplyBtn")
        self.crop_apply_btn.setFixedHeight(36)
        self.crop_apply_btn.setEnabled(False)
        self.crop_apply_btn.clicked.connect(self._apply_crop)

        crop_apply_row = QWidget()
        car = QHBoxLayout(crop_apply_row)
        car.setContentsMargins(0, 0, 0, 0)
        car.addStretch(1)
        car.addWidget(self.crop_apply_btn)
        car.addStretch(1)

        cp.addWidget(crop_box)
        cp.addWidget(crop_apply_row)
        cp.addStretch(1)

        # stack
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

        self.stage_label = CropStageLabel()
        self.stage_label.setAlignment(Qt.AlignCenter)
        self.stage_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stage_label.setMinimumSize(520, 420)
        self.stage_label.set_crop_callback(self._on_crop_changed)
        self.stage_label.sig_became_custom.connect(self._force_crop_custom)
        self.stage_label.set_overlay_enabled(False)

        sf.addWidget(self.stage_label)
        stage_l.addWidget(stage_frame, 1)

        body_l.addWidget(stage_col, 1)

        # Styles
        self.setStyleSheet("""
            QWidget { 
                background:#e9e6e3; 
                color:#111;
                font-family: system-ui, Segoe UI, Arial; 
            }
                           
            #TopBar { 
                background:#d9d9d9; 
            }
                           
            #TopTitle {
                background:#d9d9d9;
                font-weight:900;
                letter-spacing:0.22em;
                font-size:28px;
                padding:6px 18px;
                border-radius:6px;
            }

            #Tabs { 
                background:#e7e7e7; 
                border:1px solid #cfcfcf; 
                border-radius:10px; 
            }
                           
            QPushButton#TabBtn { 
                background:#e7e7e7; 
                border:none; 
                padding:10px 8px; 
                font-weight:800;
            }
                           
            QPushButton#TabBtn:checked { 
                background:#d6d6d6; 
            }

            QPushButton#IconBtn { 
                background:#f6f6f6; 
                border:1px solid #cfcfcf; 
                border-radius:8px; 
                font-weight:900; 
            }
                           
            QPushButton#GhostBtn { 
                background:#f6f6f6;
                border:1px solid #cfcfcf; 
                border-radius:8px; 
                padding:0 12px; 
                font-weight:700; 
            }
                           
            QPushButton#PrimaryBtn { 
                background:#1f74ff; 
                color:white; 
                border:none; 
                border-radius:8px; 
                padding:0 16px; 
                font-weight:800; 
            }

            QPushButton#ApplyBtn {
                border-radius: 8px;
                padding: 0 18px;
                font-weight: 900;
                border: 1px solid #cfcfcf;
                background: white;
                color: #9a9a9a;
            }
                           
            QPushButton#ApplyBtn:enabled {
                background: #1f74ff;
                color: white;
                border: none;
            }

            QPushButton:disabled { 
                color:#9a9a9a; 
                border-color:#d6d6d6; 
                background:#f0f0f0; 
            }

            QGroupBox#Block {
                border:1px solid #cfcfcf; 
                border-radius:10px; 
                margin-top:10px; 
                background:#e7e7e7; 
            }
                           
            QGroupBox#Block::title { 
                subcontrol-origin: margin; 
                left:12px;
                padding:0 6px; 
                font-weight:900; 
            }

            QLabel#RemoveTitle { 
                font-weight:900; 
            }
                           
            QLabel#HintText { 
                color:#6b6b6b; 
                font-size:12px; 
            }

            QPushButton#CropBtn { 
                background:#f3f3f3; 
                border:1px solid #cfcfcf; 
                border-radius:8px; 
                padding:10px 8px; 
                font-weight:800; 
            }
                           
            QPushButton#CropBtn:checked { 
                background:#d6d6d6; 
            }

            #StageFrame { 
                background:#efefef; 
                border:2px solid #cfcfcf; 
            }
                           
        """)

        # signals
        self.sig_params_changed.connect(lambda _: self.update_preview())

        # defaults
        self.open_tab("adjust")
        self._set_active_preset("none")
        self._set_active_crop("Original")

        # Load image
        if initial_pixmap is not None:
            self.set_original_image(initial_pixmap)

        self._update_undo_redo_buttons()
        self._refresh_apply_buttons()
        self.showMaximized()

    # =========================
    # Base + Render
    # =========================
    def set_original_image(self, pixmap: QPixmap):
        base = qpixmap_to_bgr(pixmap)
        self._base_images = [base]
        self._base_idx = 0

        self.params = UIParams()

        self._history = []
        self._hist_idx = -1
        self._push_history()

        self.update_preview()

    def _current_base(self) -> np.ndarray:
        return self._base_images[self._base_idx]

    def _render_current_preview(self) -> np.ndarray:
        base = self._current_base()
        bgr, alpha = split_bgra(base)

        p = asdict(self.params)
        p["crop"] = None  # crop not applied live

        out_bgr = process_image(bgr.copy(), p)
        return merge_bgra(out_bgr, alpha)

    def update_preview(self):
        if self._base_idx < 0:
            return

        out = self._render_current_preview()
        if out.ndim == 3 and out.shape[2] == 4:
            self.set_stage_pixmap(bgra_to_qpixmap(out))
        else:
            self.set_stage_pixmap(bgr_to_qpixmap(out))

    def set_stage_pixmap(self, pix: QPixmap):
        if pix is None or pix.isNull():
            self.stage_label.clear()
            return
        self._stage_pixmap_source = pix
        self.stage_label.setPixmap(
            pix.scaled(self.stage_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._stage_pixmap_source is not None and not self._stage_pixmap_source.isNull():
            self.stage_label.setPixmap(
                self._stage_pixmap_source.scaled(self.stage_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    # =========================
    # History (undo/redo)
    #   FIX #1: undo/redo stays in SAME MODE (no tab switching)
    # =========================
    def _snapshot_params(self) -> dict:
        return asdict(self.params)

    def _push_history(self):
        if self._suspend_history:
            return

        st = HistoryState(
            base_idx=self._base_idx,
            params=self._snapshot_params(),
        )

        if self._history:
            prev = self._history[self._hist_idx]
            if prev.base_idx == st.base_idx and prev.params == st.params:
                return

        if self._hist_idx < len(self._history) - 1:
            self._history = self._history[: self._hist_idx + 1]

        self._history.append(st)
        self._hist_idx = len(self._history) - 1
        self._update_undo_redo_buttons()

    def _restore_history(self, idx: int):
        if idx < 0 or idx >= len(self._history):
            return

        st = self._history[idx]
        self._hist_idx = idx

        # keep current tab (do NOT switch)
        cur_tab = self._current_tab

        self._suspend_history = True
        try:
            self._base_idx = st.base_idx
            p = st.params
            self.params = UIParams(**{k: p[k] for k in p if k in UIParams.__dataclass_fields__})

            # restore preset check + crop check
            self._set_active_preset(self.params.preset)
            self._set_active_crop(self.params.crop_ratio)

            # restore sliders/spins
            for key, (slider, spin) in self._controls.items():
                v = int(getattr(self.params, key))
                slider.blockSignals(True)
                spin.blockSignals(True)
                slider.setValue(v)
                spin.setValue(v)
                slider.blockSignals(False)
                spin.blockSignals(False)

            # stay in same tab, but refresh overlay visibility for crop
            self.open_tab(cur_tab)
        finally:
            self._suspend_history = False

        self._refresh_apply_buttons()
        self.update_preview()
        self._update_undo_redo_buttons()

    def _update_undo_redo_buttons(self):
        self.undo_btn.setEnabled(self._hist_idx > 0)
        self.redo_btn.setEnabled(self._hist_idx < len(self._history) - 1)

    def _undo(self):
        if self._remove_running:
            return
        if self._hist_idx <= 0:
            return
        self._restore_history(self._hist_idx - 1)

    def _redo(self):
        if self._remove_running:
            return
        if self._hist_idx >= len(self._history) - 1:
            return
        self._restore_history(self._hist_idx + 1)

    # =========================
    # Sliders (history on finalize)
    # =========================
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

        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(0)

        reset = QPushButton("⟲")
        reset.setObjectName("IconBtn")
        reset.setFixedSize(44, 34)

        def set_value(v: int):
            setattr(self.params, key, int(v))
            self.sig_params_changed.emit(self.params)

        slider.valueChanged.connect(lambda v: (spin.blockSignals(True), spin.setValue(v), spin.blockSignals(False), set_value(v)))
        spin.valueChanged.connect(lambda v: (slider.blockSignals(True), slider.setValue(v), slider.blockSignals(False), set_value(v)))

        slider.sliderReleased.connect(lambda: self._push_history())
        spin.editingFinished.connect(lambda: self._push_history())
        reset.clicked.connect(lambda: (slider.setValue(0), spin.setValue(0), self._push_history()))

        rl.addWidget(spin)
        rl.addStretch(1)
        rl.addWidget(reset)

        l.addWidget(slider)
        l.addWidget(row)
        parent.addWidget(box)

        self._controls[key] = (slider, spin)

    # =========================
    # Tabs + crop overlay visibility
    # =========================
    def open_tab(self, name: str):
        if self._remove_running:
            return

        self._current_tab = name
        order = ["adjust", "filter", "removebg", "crop"]

        for n, btn in self.tab_buttons.items():
            btn.setChecked(n == name)

        self.stack.setCurrentIndex(order.index(name))

        # show overlay only in crop mode
        self.stage_label.set_overlay_enabled(name == "crop" and self.params.crop_ratio != "Original")

        # leaving crop hides overlay (your old requirement still holds)
        if name != "crop":
            self.stage_label.set_overlay_enabled(False)

        self._refresh_apply_buttons()
        self.update_preview()

    # =========================
    # Filter presets
    # =========================
    def _set_active_preset(self, preset: str):
        self.params.preset = preset
        for n, b in self.preset_btns.items():
            b.blockSignals(True)
            b.setChecked(n == preset)
            b.blockSignals(False)

    def _on_preset_clicked(self, preset: str):
        self._set_active_preset(preset)
        self.params.mono = (preset == "mono")
        self.sig_params_changed.emit(self.params)
        self._push_history()

    # =========================
    # Crop ratio + crop updates
    # =========================
    def _set_active_crop(self, label: str):
        self.params.crop_ratio = label
        for n, b in self.crop_btns.items():
            b.blockSignals(True)
            b.setChecked(n == label)
            b.blockSignals(False)

    def _on_crop_ratio(self, label: str):
        self._set_active_crop(label)

        aspect_map = {
            "Original": None,
            "Custom": None,
            "Square": 1.0,
            "9:16": 9 / 16,
            "4:5": 4 / 5,
            "5:7": 5 / 7,
            "3:4": 3 / 4,
            "3:5": 3 / 5,
            "2:3": 2 / 3,
        }

        if label == "Original":
            self.stage_label.clear_crop()
            self.stage_label.set_overlay_enabled(False)
            self._crop_dirty = False
            self._refresh_apply_buttons()
            return

        self.stage_label.set_overlay_enabled(True)
        aspect = aspect_map.get(label, None)
        self.stage_label.set_aspect_ratio(aspect)

        if label == "Custom":
            self.stage_label.ensure_crop_box("custom")
        else:
            self.stage_label.ensure_crop_box("aspect")

        # picking a ratio counts as "modification"
        self._crop_dirty = True
        self._refresh_apply_buttons()

    def _on_crop_changed(self, crop_dict: dict):
        self.params.crop = crop_dict
        # user moved/resized box => modification
        if crop_dict.get("enabled", False):
            self._crop_dirty = True
        self._refresh_apply_buttons()

    def _force_crop_custom(self):
        if self.params.crop_ratio != "Custom":
            self._set_active_crop("Custom")
            self.stage_label.set_aspect_ratio(None)
            self._crop_dirty = True
            self._refresh_apply_buttons()

    # =========================
    # Remove BG
    # =========================
    def _set_ui_locked(self, locked: bool):
        for btn in self.tab_buttons.values():
            btn.setEnabled(not locked)

        for b in getattr(self, "preset_btns", {}).values():
            b.setEnabled(not locked)

        for i in range(self.stack.count()):
            self.stack.widget(i).setEnabled(not locked)

        # keep remove page enabled so cancel works
        self.stack.widget(2).setEnabled(True)

        self.remove_cancel_btn.setEnabled(locked)
        self.remove_start_btn.setEnabled(not locked)

        self._refresh_apply_buttons()
        self._update_undo_redo_buttons()

    def _on_removebg_start(self):
        if self._remove_running or self._base_idx < 0:
            return

        preview = self._render_current_preview()
        bgr, _ = split_bgra(preview)

        self._remove_running = True
        self._remove_cancelled = False
        self._remove_preview_bgra = None
        self._remove_dirty = False
        self._refresh_apply_buttons()

        self.remove_status.setText("Removing...")
        self._set_ui_locked(True)

        self._remove_thread = QThread()
        self._remove_worker = RemoveBgWorker(bgr.copy())
        self._remove_worker.moveToThread(self._remove_thread)

        self._remove_thread.started.connect(self._remove_worker.run)
        self._remove_worker.finished.connect(self._on_removebg_finished)
        self._remove_worker.error.connect(self._on_removebg_error)

        self._remove_worker.finished.connect(self._remove_thread.quit)
        self._remove_worker.error.connect(self._remove_thread.quit)
        self._remove_thread.finished.connect(self._remove_thread.deleteLater)

        self._remove_thread.start()

    def _on_removebg_cancel(self):
        if not self._remove_running:
            return
        self._remove_cancelled = True
        self.remove_status.setText("Cancelling...")

    def _on_removebg_finished(self, bgra_out):
        self._remove_running = False
        self._set_ui_locked(False)

        if self._remove_cancelled:
            self._remove_cancelled = False
            self.remove_status.setText("Cancelled")
            return

        if bgra_out is None:
            self.remove_status.setText("Failed")
            return

        self._remove_preview_bgra = bgra_out.copy()
        self._remove_dirty = True  # modification exists
        self.remove_status.setText("Done")

        self._refresh_apply_buttons()
        self.set_stage_pixmap(bgra_to_qpixmap(bgra_out))

    def _on_removebg_error(self, msg: str):
        self._remove_running = False
        self._set_ui_locked(False)

        if self._remove_cancelled:
            self._remove_cancelled = False
            self.remove_status.setText("Cancelled")
        else:
            self.remove_status.setText("Failed")

    def _apply_removebg(self):
        if self._remove_preview_bgra is None or self._remove_running:
            return

        self._base_images.append(self._remove_preview_bgra.copy())
        self._base_idx = len(self._base_images) - 1
        self._remove_preview_bgra = None
        self._remove_dirty = False

        self._reset_params_neutral()
        self._push_history()

        self.remove_status.setText("Applied")
        self._refresh_apply_buttons()
        self.update_preview()

    # =========================
    # Crop apply
    # =========================
    def _can_apply_crop(self) -> bool:
        return bool(self.params.crop and self.params.crop.get("enabled", False))

    def _apply_crop(self):
        if not self._can_apply_crop() or self._remove_running or self._base_idx < 0:
            return

        preview = self._render_current_preview()
        bgr, alpha = split_bgra(preview)

        cropped_bgr = apply_crop_local(bgr, self.params.crop)
        if alpha is not None:
            cropped_alpha = apply_crop_local(alpha, self.params.crop)
            cropped = merge_bgra(cropped_bgr, cropped_alpha)
        else:
            cropped = cropped_bgr

        self._base_images.append(cropped.copy())
        self._base_idx = len(self._base_images) - 1

        self.stage_label.clear_crop()
        self._set_active_crop("Original")
        self.stage_label.set_overlay_enabled(False)

        self._crop_dirty = False
        self._reset_params_neutral()

        self._push_history()
        self._refresh_apply_buttons()
        self.update_preview()

    # =========================
    # Apply button enable rules
    #   FIX #2: only blue (enabled) when modification exists
    # =========================
    def _refresh_apply_buttons(self):
        # removebg apply enabled only if: in removebg tab, not running, preview exists, and dirty
        remove_enable = (
            (self._current_tab == "removebg")
            and (not self._remove_running)
            and (self._remove_preview_bgra is not None)
            and self._remove_dirty
        )
        self.remove_apply_btn.setEnabled(remove_enable)

        # crop apply enabled only if: in crop tab, crop enabled, and crop dirty
        crop_enable = (
            (self._current_tab == "crop")
            and (not self._remove_running)
            and self._can_apply_crop()
            and self._crop_dirty
        )
        self.crop_apply_btn.setEnabled(crop_enable)

    # =========================
    # Reset params (after commits)
    # =========================
    def _reset_params_neutral(self):
        self._suspend_history = True
        try:
            self.params.brightness = 0
            self.params.sharpness = 0
            self.params.denoise = 0
            self.params.red = 0
            self.params.green = 0
            self.params.blue = 0
            self.params.mono = False
            self.params.preset = "none"
            self.params.crop_ratio = "Original"
            self.params.crop = {"enabled": False, "x": 0, "y": 0, "w": 1, "h": 1}

            for key, (slider, spin) in self._controls.items():
                slider.blockSignals(True)
                spin.blockSignals(True)
                slider.setValue(0)
                spin.setValue(0)
                slider.blockSignals(False)
                spin.blockSignals(False)

            self._set_active_preset("none")
            self._set_active_crop("Original")
        finally:
            self._suspend_history = False

    # =========================
    # Top bar
    # =========================
    def _on_cancel_clicked(self):
        self.sig_cancel.emit()
        self.close()

    def _on_done_clicked(self):
        out = self._render_current_preview()

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "download/image_01.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;WEBP (*.webp)"
        )
        if not path:
            return

        ext = path.lower().split(".")[-1]
        if out.ndim == 3 and out.shape[2] == 4:
            if ext in ("jpg", "jpeg", "bmp"):
                cv2.imwrite(path, out[:, :, :3])
            else:
                cv2.imwrite(path, out)
        else:
            cv2.imwrite(path, out)

        self.sig_done.emit(path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OperationWindow()
    win.show()
    sys.exit(app.exec())
