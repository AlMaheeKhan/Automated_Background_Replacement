from app.gui.app import BackgroundApp
import sys
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BackgroundApp()
    window.show()
    sys.exit(app.exec_())

