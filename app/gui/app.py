# app/gui/app.py
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from app.inference import run_inference
from app.segmentation.model_loader import load_model
from app.config import MODEL_TYPE, MODEL_PATH

class BackgroundApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = load_model(MODEL_TYPE, MODEL_PATH)

        self.label = QLabel("Automated Background Replacement")
        self.button = QPushButton("Select Image")
        self.button.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image")
        if file_path:
            run_inference(self.model, file_path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BackgroundApp()
    window.show()
    sys.exit(app.exec_())
