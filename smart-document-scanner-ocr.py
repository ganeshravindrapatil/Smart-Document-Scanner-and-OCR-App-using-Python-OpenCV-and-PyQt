# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 10:46:09 2025

@author: Admin
"""

from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import cv2
import numpy as np

# ---------- Image processing helpers ----------

def order_points(pts):
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute width of new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    # compute height of new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # destination points for birds-eye view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def auto_detect_document(image):
    # Resize for processing
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    img = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        # fallback: return original if no document found
        return orig

    # Apply perspective transform
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    return warped

def enhance_document(image, block_size=25, C=10):
    # Ensure block_size is odd and >=3
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold for clean B/W
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    # Slight morphology to clean noise
    kernel = np.ones((2, 2), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)

    # Convert back to BGR for display
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

# ---------- PyQt5 GUI ----------

class DocScannerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.original = None
        self.doc_aligned = None
        self.processed = None

        self.block_size = 25
        self.C = 10

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Smart Document Scanner - Advanced")
        self.resize(1200, 700)

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central)

        # Left side: images (original + processed)
        view_layout = QtWidgets.QVBoxLayout()

        self.lbl_orig = QtWidgets.QLabel("Original")
        self.lbl_proc = QtWidgets.QLabel("Processed")
        self.lbl_orig.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_proc.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_orig.setStyleSheet("background-color: #202020; color: white;")
        self.lbl_proc.setStyleSheet("background-color: #202020; color: white;")

        view_layout.addWidget(QtWidgets.QLabel("Original / Aligned"))
        view_layout.addWidget(self.lbl_orig, 1)
        view_layout.addWidget(QtWidgets.QLabel("Enhanced Output"))
        view_layout.addWidget(self.lbl_proc, 1)

        # Right side: controls
        ctrl_layout = QtWidgets.QVBoxLayout()

        btn_open = QtWidgets.QPushButton("Open Image")
        btn_detect = QtWidgets.QPushButton("Auto Detect & Align")
        btn_enhance = QtWidgets.QPushButton("Enhance Document")
        btn_reset = QtWidgets.QPushButton("Reset")

        btn_open.clicked.connect(self.open_image)
        btn_detect.clicked.connect(self.auto_detect_page)
        btn_enhance.clicked.connect(self.apply_enhancement)
        btn_reset.clicked.connect(self.reset_all)

        ctrl_layout.addWidget(btn_open)
        ctrl_layout.addWidget(btn_detect)
        ctrl_layout.addWidget(btn_enhance)
        ctrl_layout.addWidget(btn_reset)

        ctrl_layout.addSpacing(15)
        ctrl_layout.addWidget(QtWidgets.QLabel("Adaptive Threshold Block Size"))

        self.slider_block = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_block.setRange(3, 51)
        self.slider_block.setValue(self.block_size)
        self.slider_block.setSingleStep(2)
        self.slider_block.valueChanged.connect(self.update_block_size)

        ctrl_layout.addWidget(self.slider_block)

        ctrl_layout.addWidget(QtWidgets.QLabel("Adaptive Threshold C (offset)"))

        self.slider_C = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_C.setRange(0, 20)
        self.slider_C.setValue(self.C)
        self.slider_C.valueChanged.connect(self.update_C)
        ctrl_layout.addWidget(self.slider_C)

        ctrl_layout.addStretch()

        main_layout.addLayout(view_layout, 3)
        main_layout.addLayout(ctrl_layout, 1)

        self.setCentralWidget(central)

        # Menu (optional)
        file_menu = self.menuBar().addMenu("&File")
        act_open = QtWidgets.QAction("Open", self)
        act_save = QtWidgets.QAction("Save Processed As...", self)
        act_exit = QtWidgets.QAction("Exit", self)

        act_open.triggered.connect(self.open_image)
        act_save.triggered.connect(self.save_processed)
        act_exit.triggered.connect(self.close)

        file_menu.addAction(act_open)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        file_menu.addAction(act_exit)

    # ---- Slots ----

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.original = cv2.imread(path)
            self.doc_aligned = None
            self.processed = None
            self.show_image(self.lbl_orig, self.original)
            self.lbl_proc.clear()
            self.lbl_proc.setText("Processed")

    def show_image(self, label, img):
        if img is None:
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            label.width() if label.width() > 0 else 600,
            label.height() if label.height() > 0 else 300,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        label.setPixmap(pix)

    def auto_detect_page(self):
        if self.original is None:
            QtWidgets.QMessageBox.information(self, "Info", "Please open an image first.")
            return
        self.doc_aligned = auto_detect_document(self.original)
        self.show_image(self.lbl_orig, self.doc_aligned)

    def apply_enhancement(self):
        base_img = None
        if self.doc_aligned is not None:
            base_img = self.doc_aligned
        elif self.original is not None:
            base_img = self.original
        else:
            QtWidgets.QMessageBox.information(self, "Info", "Please open an image first.")
            return

        self.processed = enhance_document(base_img, self.block_size, self.C)
        self.show_image(self.lbl_proc, self.processed)

    def reset_all(self):
        if self.original is None:
            return
        self.doc_aligned = None
        self.processed = None
        self.show_image(self.lbl_orig, self.original)
        self.lbl_proc.clear()
        self.lbl_proc.setText("Processed")

    def update_block_size(self, value):
        # keep it odd
        if value % 2 == 0:
            value += 1
        self.block_size = value
        if self.processed is not None:
            self.apply_enhancement()

    def update_C(self, value):
        self.C = int(value)
        if self.processed is not None:
            self.apply_enhancement()

    def save_processed(self):
        if self.processed is None:
            QtWidgets.QMessageBox.information(self, "Info", "No processed image to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Processed Image", "", "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        if path:
            cv2.imwrite(path, self.processed)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = DocScannerApp()
    win.show()
    sys.exit(app.exec_())
