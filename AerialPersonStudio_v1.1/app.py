from __future__ import annotations
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
SRC = APP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from aps.ui.main_window import MainWindow


def main():
    os.chdir(APP_ROOT)
    app = QApplication(sys.argv)
    app.setApplicationName("Aerial Person Studio")
    app.setOrganizationName("Aerial Person Project")
    window = MainWindow(APP_ROOT)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
