DARK_QSS = r'''
QWidget { background: #0f172a; color: #e5e7eb; font-family: "Segoe UI"; font-size: 10pt; }
QMainWindow { background: #0b1220; }
QFrame#Card { background: #111827; border: 1px solid #243044; border-radius: 12px; }
QLabel#Title { font-size: 20pt; font-weight: 700; color: #f8fafc; }
QLabel#Subtitle { color: #94a3b8; font-size: 10pt; }
QLabel#Metric { font-size: 22pt; font-weight: 700; color: #60a5fa; }
QPushButton { background: #1d4ed8; border: none; border-radius: 8px; padding: 8px 14px; font-weight: 600; }
QPushButton:hover { background: #2563eb; }
QPushButton:pressed { background: #1e40af; }
QPushButton[secondary="true"] { background: #1f2937; border: 1px solid #334155; }
QPushButton[secondary="true"]:hover { background: #273449; }
QPushButton[danger="true"] { background: #991b1b; }
QPushButton[danger="true"]:hover { background: #b91c1c; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QTextEdit {
    background: #0b1324; border: 1px solid #334155; border-radius: 7px; padding: 6px; selection-background-color: #2563eb;
}
QComboBox QAbstractItemView { background: #111827; selection-background-color: #1d4ed8; }
QTabWidget::pane { border: 1px solid #243044; border-radius: 10px; top: -1px; }
QTabBar::tab { background: #111827; color: #94a3b8; padding: 10px 16px; margin-right: 3px; border-radius: 7px; }
QTabBar::tab:selected { background: #1d4ed8; color: white; }
QGroupBox { border: 1px solid #334155; border-radius: 9px; margin-top: 12px; padding-top: 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #cbd5e1; }
QProgressBar { border: 1px solid #334155; border-radius: 7px; text-align: center; background: #0b1324; }
QProgressBar::chunk { background: #2563eb; border-radius: 6px; }
QSlider::groove:horizontal { background: #334155; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background: #60a5fa; width: 14px; margin: -4px 0; border-radius: 7px; }
QListWidget::item { padding: 6px; border-radius: 5px; }
QListWidget::item:selected { background: #1d4ed8; }
QScrollBar:vertical { background: #0b1324; width: 12px; }
QScrollBar::handle:vertical { background: #334155; border-radius: 6px; min-height: 30px; }
QToolTip { background: #111827; color: #f8fafc; border: 1px solid #334155; }
'''
