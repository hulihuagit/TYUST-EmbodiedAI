"""
主窗口 —— 整合导入面板、配置面板、结果面板，管理运行状态。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QPushButton, QStatusBar, QMessageBox, QMenuBar, QMenu, QTabWidget,
    QLabel
)
from PySide6.QtCore import Qt, QTimer

from ui.import_panel import ImportPanel
from ui.config_panel import ConfigPanel
from ui.result_panel import ResultPanel
from ui.worker_thread import ProcessWorker


class MainWindow(QMainWindow):
    """桌面应用主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('无人机检测与追踪系统 —— 基于时间表面与事件相机')
        self.resize(1280, 780)
        self.setMinimumSize(1024, 640)

        self._worker = None
        self._setup_menu()
        self._setup_ui()
        self._connect_signals()

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu('文件(&F)')
        file_menu.addAction('退出(&Q)', self.close)

        help_menu = menubar.addMenu('帮助(&H)')
        help_menu.addAction('使用说明', self._show_help)
        help_menu.addAction('关于', self._show_about)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # ---- 左侧：导入 + 配置（垂直分割） ----
        left_splitter = QSplitter(Qt.Vertical)

        self.import_panel = ImportPanel()
        left_splitter.addWidget(self.import_panel)

        self.config_panel = ConfigPanel()
        left_splitter.addWidget(self.config_panel)

        left_splitter.setStretchFactor(0, 2)  # import panel gets more space
        left_splitter.setStretchFactor(1, 3)  # config panel gets more space

        # ---- 右侧：结果面板 ----
        self.result_panel = ResultPanel()

        # ---- 水平分割器：左 | 右 ----
        h_splitter = QSplitter(Qt.Horizontal)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Run button at bottom of left
        self.run_btn = QPushButton('▶  开始处理')
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet(
            'QPushButton {'
            '  background: #27ae60; color: white; font-size: 14px; font-weight: bold;'
            '  border-radius: 4px; padding: 8px;'
            '}'
            'QPushButton:hover { background: #2ecc71; }'
            'QPushButton:disabled { background: #555; color: #999; }'
        )
        self.run_btn.clicked.connect(self._toggle_run)

        left_layout.addWidget(left_splitter)
        left_layout.addWidget(self.run_btn)

        left_container.setMinimumWidth(420)
        left_container.setMaximumWidth(520)

        h_splitter.addWidget(left_container)
        h_splitter.addWidget(self.result_panel)
        h_splitter.setStretchFactor(0, 0)  # left fixed
        h_splitter.setStretchFactor(1, 1)  # right stretches

        main_layout.addWidget(h_splitter)

        # ---- 状态栏 ----
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('就绪 — 检查数据目录后点击"开始处理"')

    def _connect_signals(self):
        # Stop button
        self.result_panel.stop_btn.clicked.connect(self._stop_processing)

    def _toggle_run(self):
        """开始或停止处理"""
        if self._worker is not None and self._worker.isRunning():
            self._stop_processing()
            return

        # Gather all params
        import_params = self.import_panel.get_params()
        config_params = self.config_panel.get_params()
        params = {**import_params, **config_params}

        # Validate
        data_dir = params.get('data_dir', '')
        if not data_dir or not os.path.isdir(data_dir):
            QMessageBox.warning(self, '数据目录无效',
                                f'请检查数据目录路径:\n{data_dir}')
            return

        event_dir = os.path.join(data_dir, 'Event', 'Frames')
        if not os.path.isdir(event_dir):
            QMessageBox.warning(self, '数据不完整',
                                f'Event/Frames/ 目录不存在:\n{event_dir}')
            return

        # Start
        self._start_processing(params)

    def _start_processing(self, params):
        """启动后台处理线程"""
        self.run_btn.setText('⏹  停止处理')
        self.run_btn.setStyleSheet(
            'QPushButton {'
            '  background: #c0392b; color: white; font-size: 14px; font-weight: bold;'
            '  border-radius: 4px; padding: 8px;'
            '}'
            'QPushButton:hover { background: #e74c3c; }'
        )

        self.result_panel.on_started()
        self.status_bar.showMessage('处理中...')

        self.import_panel.setEnabled(False)
        self.config_panel.setEnabled(False)

        self._worker = ProcessWorker(params)
        self._worker.progress.connect(self.result_panel.update_progress)
        self._worker.preview_ready.connect(self.result_panel.update_preview)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop_processing(self):
        """停止后台处理"""
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait(3000)
        self._reset_ui_state()

    def _on_finished(self, output_path, metrics):
        """处理完成"""
        self.result_panel.on_finished(output_path, metrics)
        self.status_bar.showMessage(f'处理完成 — 输出: {output_path}')
        self._reset_ui_state()

    def _on_error(self, error_msg):
        """处理出错"""
        self.result_panel.on_error(error_msg)
        self.status_bar.showMessage('处理出错')
        QMessageBox.critical(self, '处理错误', error_msg[:1000])
        self._reset_ui_state()

    def _reset_ui_state(self):
        """恢复 UI 状态"""
        self.run_btn.setText('▶  开始处理')
        self.run_btn.setStyleSheet(
            'QPushButton {'
            '  background: #27ae60; color: white; font-size: 14px; font-weight: bold;'
            '  border-radius: 4px; padding: 8px;'
            '}'
            'QPushButton:hover { background: #2ecc71; }'
        )
        self.import_panel.setEnabled(True)
        self.config_panel.setEnabled(True)
        self.result_panel.stop_btn.setEnabled(False)

    def _show_help(self):
        QMessageBox.information(self, '使用说明',
                                '1. 在"数据目录"中选择数据集根目录（包含 Event/Frames/ 和 RGB/）\n'
                                '2. 在"算法配置"中调节参数\n'
                                '3. 点击"开始处理"运行管道\n'
                                '4. 在右侧查看实时预览和最终结果\n\n'
                                '预设方案保存在 presets/ 目录中，可快速切换不同参数组合。')

    def _show_about(self):
        QMessageBox.about(self, '关于',
                          '基于时间表面的高速运动目标（无人机）检测与追踪系统\n\n'
                          '核心技术：双极性指数衰减时间表面、多尺度检测、\n'
                          'KCF 核化相关滤波、卡尔曼滤波、匈牙利匹配\n\n'
                          '数据来源：FRED (Florence RGB-Event Drone Dataset)\n'
                          '版本：1.0  |  2026-07')

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self, '确认退出', '处理正在进行中，确定退出吗？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._worker.abort()
                self._worker.wait(3000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
