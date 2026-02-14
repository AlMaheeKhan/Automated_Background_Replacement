# app/gui/app.py

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from app.services.segmentation_service import SegmentationService


class BackgroundApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Automated Background Replacement Tool")
        self.resize(600, 600)

        self.service = SegmentationService()

        self.image_path = None
        self.background_path = None

        self.label = QLabel("Select images to begin")
        self.label.setAlignment(Qt.AlignCenter)

        self.select_image_btn = QPushButton("Select Foreground Image")
        self.select_image_btn.clicked.connect(self.select_image)

        self.select_background_btn = QPushButton("Select Background Image")
        self.select_background_btn.clicked.connect(self.select_background)

        self.process_btn = QPushButton("Replace Background")
        self.process_btn.clicked.connect(self.process_image)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.select_image_btn)
        layout.addWidget(self.select_background_btn)
        layout.addWidget(self.process_btn)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Foreground Image")
        if file_path:
            self.image_path = file_path

    def select_background(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Background Image")
        if file_path:
            self.background_path = file_path

    def process_image(self):
        if not self.image_path:
            return

        with open(self.image_path, "rb") as f:
            image_bytes = f.read()

        background_bytes = None
        if self.background_path:
            with open(self.background_path, "rb") as f:
                background_bytes = f.read()

        result_bytes = self.service.process(image_bytes, background_bytes)

        # Decode result
        np_arr = np.frombuffer(result_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(
            image.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()

        self.result_label.setPixmap(QPixmap.fromImage(q_img))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BackgroundApp()
    window.show()
    sys.exit(app.exec_())
