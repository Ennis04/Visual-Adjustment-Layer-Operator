import sys
import traceback
from pathlib import Path
import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget
)

from operation import OperationWindow


def cv_to_pixmap(bgr_img) -> QPixmap:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def make_upload_icon(size: int = 88) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.white)

    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)

    pen = QPen(Qt.black, 3)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)

    cx = size // 2
    top = int(size * 0.18)
    mid = int(size * 0.48)

    p.drawLine(cx, mid, cx, top)
    p.drawLine(cx, top, cx - 14, top + 14)
    p.drawLine(cx, top, cx + 14, top + 14)

    tray_y = int(size * 0.62)
    tray_w = int(size * 0.56)
    tray_h = int(size * 0.20)

    left = cx - tray_w // 2
    right = cx + tray_w // 2

    p.drawLine(left, tray_y, left, tray_y + tray_h)
    p.drawLine(right, tray_y, right, tray_y + tray_h)
    p.drawLine(left, tray_y + tray_h, right, tray_y + tray_h)

    p.end()
    return pm


class ThumbButton(QPushButton):
    def __init__(self, image_path: Path, on_click):
        super().__init__()
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(120, 80)
        self.setStyleSheet("QPushButton{border:none; padding:0; background:transparent;}")

        if image_path.exists():
            img = cv2.imread(str(image_path))
            if img is not None:
                pix = cv_to_pixmap(img).scaled(
                    self.size(),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation,
                )
                self.setIcon(pix)
                self.setIconSize(self.size())

        self.clicked.connect(on_click)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("V.A.L.O.")

        self._selected_pixmap: QPixmap | None = None
        self.op_window: OperationWindow | None = None

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Top bar
        topbar = QFrame()
        topbar.setFixedHeight(58)
        topbar.setObjectName("TopBar")
        top_layout = QHBoxLayout(topbar)
        top_layout.setContentsMargins(0, 0, 0, 0)

        title_badge = QLabel("V.A.L.O.")
        title_badge.setObjectName("TopTitleBadge")
        title_badge.setAlignment(Qt.AlignCenter)
        title_badge.setFixedHeight(46)
        title_badge.setFixedWidth(260)

        top_layout.addStretch(1)
        top_layout.addWidget(title_badge)
        top_layout.addStretch(1)
        root_layout.addWidget(topbar)

        # Main area
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(70, 50, 70, 50)
        content_layout.setSpacing(90)
        root_layout.addWidget(content, 1)

        # LEFT PANEL
        left_panel = QWidget()
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(10)

        welcome = QLabel("Welcome to V.A.L.O.")
        welcome.setObjectName("WelcomeTitle")

        subtitle = QLabel("Let's start by uploading an image.")
        subtitle.setObjectName("WelcomeSub")

        underline = QFrame()
        underline.setFixedHeight(2)
        underline.setFixedWidth(550)
        underline.setObjectName("Underline")

        hint = QLabel("No image?\nTry one of these samples:")
        hint.setObjectName("Hint")

        thumbs_row = QHBoxLayout()
        thumbs_row.setSpacing(14)

        for i in range(1, 5):
            img_path = Path("img") / f"sample{i}.jpg"
            btn = ThumbButton(img_path, on_click=lambda _, x=i: self.use_sample(x))
            thumbs_row.addWidget(btn)
        thumbs_row.addStretch(1)

        left.addStretch(1)
        left.addWidget(welcome)
        left.addWidget(subtitle)
        left.addWidget(underline)
        left.addSpacing(14)
        left.addWidget(hint)
        left.addLayout(thumbs_row)
        left.addStretch(1)

        content_layout.addWidget(left_panel, 1)

        # RIGHT PANEL (UPLOAD CARD)
        self.card = QFrame()
        self.card.setObjectName("UploadCard")
        self.card.setMinimumWidth(620)
        self.card.setMaximumWidth(720)
        self.card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Force Qt to actually paint background for frames/widgets
        self.card.setAttribute(Qt.WA_StyledBackground, True)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(44, 44, 44, 44)
        card_layout.setSpacing(18)
        card_layout.setAlignment(Qt.AlignCenter)

        self.upload_icon_box = QFrame()
        self.upload_icon_box.setObjectName("UploadIconBox")
        self.upload_icon_box.setFixedSize(380, 120)
        self.upload_icon_box.setAttribute(Qt.WA_StyledBackground, True)

        box_layout = QVBoxLayout(self.upload_icon_box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setAlignment(Qt.AlignCenter)

        self.upload_icon = QLabel()
        self.upload_icon.setPixmap(make_upload_icon(92))
        self.upload_icon.setAlignment(Qt.AlignCenter)
        box_layout.addWidget(self.upload_icon)

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.setObjectName("UploadButton")
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn.setFixedHeight(56)
        self.upload_btn.setFixedWidth(320)
        self.upload_btn.clicked.connect(self.pick_image)

        self.preview_container = QFrame()
        self.preview_container.setObjectName("PreviewPanel")
        self.preview_container.setAttribute(Qt.WA_StyledBackground, True)

        pv = QVBoxLayout(self.preview_container)
        pv.setContentsMargins(14, 14, 14, 14)
        pv.setSpacing(10)

        self.preview_img = QLabel()
        self.preview_img.setAlignment(Qt.AlignCenter)
        self.preview_img.setMinimumHeight(300)
        self.preview_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pv.addWidget(self.preview_img)

        self.preview_container.setVisible(False)

        self.action_row = QFrame()
        self.action_row.setObjectName("ActionRow")
        self.action_row.setAttribute(Qt.WA_StyledBackground, True)

        ar = QHBoxLayout(self.action_row)
        ar.setContentsMargins(0, 0, 0, 0)
        ar.setSpacing(12)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("CancelButton")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.clicked.connect(self.exit_preview)

        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("NextButton")
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.clicked.connect(self.next_step)

        ar.addWidget(self.cancel_btn)
        ar.addWidget(self.next_btn)

        self.action_row.setVisible(False)

        self.upload_mode = QWidget()
        self.upload_mode.setObjectName("UploadMode")
        self.upload_mode.setAttribute(Qt.WA_StyledBackground, True)

        upload_mode_layout = QVBoxLayout(self.upload_mode)
        upload_mode_layout.setContentsMargins(0, 0, 0, 0)
        upload_mode_layout.setSpacing(18)
        upload_mode_layout.setAlignment(Qt.AlignCenter)

        upload_mode_layout.addWidget(self.upload_icon_box, alignment=Qt.AlignHCenter)
        upload_mode_layout.addWidget(self.upload_btn, alignment=Qt.AlignHCenter)

        card_layout.addStretch(1)
        card_layout.addWidget(self.upload_mode, alignment=Qt.AlignCenter)
        card_layout.addWidget(self.preview_container)
        card_layout.addWidget(self.action_row)
        card_layout.addStretch(1)

        content_layout.addWidget(self.card, 0)

        self.setStyleSheet("""
            QWidget { background:#e9e6e3; color:#111; font-family: system-ui, Segoe UI, Arial; }
            #TopBar { background:#d9d9d9; }

            #TopTitleBadge {
                background:#d9d9d9; border-radius:6px; font-weight:900;
                letter-spacing:0.22em; font-size:34px; padding:6px 18px;
            }

            #WelcomeTitle { font-size:54px; font-weight:900; font-style:italic; }
            #WelcomeSub { font-size:20px; font-style:italic; color:#565656; }
            #Underline { background:#1f1f1f; border:none; }
            #Hint { font-size:14px; font-style:italic; color:#565656; }

            /* FORCE WHITE AREAS */
            #UploadCard { background:#ffffff; border:2px solid #cfcfcf; border-radius:14px; }
            #UploadMode { background:#ffffff; }
            #UploadIconBox { background:#ffffff; border-radius:8px; }
            #PreviewPanel { background:#ffffff; border:1px solid #e8e8e8; border-radius:10px; }
            #ActionRow { background:#ffffff; }

            #UploadButton, #NextButton {
                background:#1f74ff; color:#fff; border:none; border-radius:12px;
                font-size:22px; font-weight:800; padding:12px 18px;
            }
            #UploadButton:hover, #NextButton:hover { background:#1a66e6; }

            #CancelButton {
                background:#ffffff; border:2px solid #cfcfcf; border-radius:12px;
                font-size:18px; font-weight:700; padding:10px 18px;
            }
            #CancelButton:hover { background:#f5f5f5; }
        """)

        self.showMaximized()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview_pixmap()

    def _refresh_preview_pixmap(self):
        if self._selected_pixmap is None:
            return
        target = self.preview_img.size()
        if target.width() < 10 or target.height() < 10:
            return
        self.preview_img.setPixmap(
            self._selected_pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def pick_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an image", str(Path.cwd()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Could not open", "That file couldn't be read as an image.")
            return

        self.enter_preview(cv_to_pixmap(img))

    def use_sample(self, idx: int):
        sample_path = Path("img") / f"sample{idx}.jpg"
        if not sample_path.exists():
            QMessageBox.information(self, "Sample not found", f"Missing file:\n{sample_path}")
            return

        img = cv2.imread(str(sample_path))
        if img is None:
            QMessageBox.warning(self, "Could not open", "That sample image couldn't be read.")
            return

        self.enter_preview(cv_to_pixmap(img))

    def enter_preview(self, pixmap: QPixmap):
        self._selected_pixmap = pixmap

        self.upload_mode.setVisible(False)
        self.preview_container.setVisible(True)
        self.action_row.setVisible(True)

        # Ensure buttons are not covered by anything
        self.action_row.raise_()
        self.next_btn.raise_()

        self._refresh_preview_pixmap()

    def exit_preview(self):
        self._selected_pixmap = None
        self.preview_img.clear()

        self.preview_container.setVisible(False)
        self.action_row.setVisible(False)
        self.upload_mode.setVisible(True)

    def next_step(self):
        # If this shows, the button is being clicked.
        try:
            if self._selected_pixmap is None:
                QMessageBox.warning(self, "No image", "Please upload/select an image first.")
                return

            self.op_window = OperationWindow(self._selected_pixmap)
            self.op_window.sig_cancel.connect(self._on_operation_cancelled)

            self.op_window.show()
            self.hide()

        except Exception:
            QMessageBox.critical(self, "Next Error", traceback.format_exc())

    def _on_operation_cancelled(self):
        self.show()
        if self.op_window is not None:
            self.op_window.deleteLater()
            self.op_window = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
