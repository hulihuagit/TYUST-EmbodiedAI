"""
无人机检测与追踪系统 —— 桌面应用入口

运行方式（开发）:
    cd D:/PythonProject3
    python app.py

打包为 exe:
    pyinstaller --onefile --windowed --name "无人机追踪系统" app.py
"""
import sys
import os


def get_base_path():
    """获取应用根目录，兼容开发环境和 PyInstaller 打包"""
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后，资源在 sys._MEIPASS
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))


# Ensure project root is on path
BASE_PATH = get_base_path()
sys.path.insert(0, BASE_PATH)

# Also add the parent of BASE_PATH for finding modules in development
if not getattr(sys, 'frozen', False):
    parent = os.path.dirname(BASE_PATH)
    if parent not in sys.path:
        sys.path.insert(0, parent)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from ui.main_window import MainWindow


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName('Drone Tracker')
    app.setOrganizationName('UAV-Tracking')
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
