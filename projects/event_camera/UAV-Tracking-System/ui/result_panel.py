"""
结果查看面板 —— 实时处理预览、进度条、指标统计、导出。
"""
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QProgressBar, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QSplitter
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap


def surface_to_qimage(surface, target_w=320, target_h=240):
    """将时间表面 float32 [0,1] 转为 QImage，叠加伪彩色"""
    if surface is None:
        return None
    # Normalize and apply colormap
    vis = (surface * 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    color = cv2.resize(color, (target_w, target_h))
    h, w, ch = color.shape
    return QImage(color.data, w, h, ch * w, QImage.Format_BGR888)


def raw_to_qimage(raw_frame, target_w=320, target_h=240):
    """将灰度帧转为 QImage"""
    if raw_frame is None:
        return None
    if len(raw_frame.shape) == 2:
        vis = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
    else:
        vis = raw_frame.copy()
    if len(vis.shape) == 3 and vis.shape[2] == 3:
        vis = cv2.resize(vis, (target_w, target_h))
        h, w, ch = vis.shape
        return QImage(vis.data, w, h, ch * w, QImage.Format_BGR888)
    return None


class ResultPanel(QWidget):
    """结果面板：预览 + 指标 + 进度 + 导出"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_path = ''
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # ---- 实时预览 ----
        preview_group = QGroupBox('处理预览（每 100 帧更新）')
        preview_layout = QHBoxLayout(preview_group)

        self.surface_label = QLabel()
        self.surface_label.setFixedSize(320, 240)
        self.surface_label.setStyleSheet('background: #1a1a2e; border: 1px solid #333;')
        self.surface_label.setAlignment(Qt.AlignCenter)
        self.surface_label.setText('等待处理...')
        preview_layout.addWidget(self.surface_label)

        self.result_label = QLabel()
        self.result_label.setFixedSize(320, 240)
        self.result_label.setStyleSheet('background: #1a1a2e; border: 1px solid #333;')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setText('跟踪结果预览')
        preview_layout.addWidget(self.result_label)

        layout.addWidget(preview_group)

        # ---- 进度条 + 状态 ----
        prog_group = QGroupBox('处理进度')
        prog_layout = QVBoxLayout(prog_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        prog_layout.addWidget(self.progress_bar)

        self.status_label = QLabel('就绪')
        self.status_label.setStyleSheet('color: #888;')
        prog_layout.addWidget(self.status_label)

        self.stats_label = QLabel('')
        self.stats_label.setStyleSheet('font-size: 11px; color: #aaa;')
        prog_layout.addWidget(self.stats_label)

        layout.addWidget(prog_group)

        # ---- 最终指标表格 ----
        metrics_group = QGroupBox('处理结果')
        metrics_layout = QVBoxLayout(metrics_group)

        self.metrics_table = QTableWidget(3, 2)
        self.metrics_table.setHorizontalHeaderLabels(['指标', '数值'])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setMaximumHeight(120)
        self._reset_metrics_table()
        metrics_layout.addWidget(self.metrics_table)

        layout.addWidget(metrics_group)

        # ---- 操作按钮 ----
        btn_row = QHBoxLayout()

        self.stop_btn = QPushButton('⏹ 停止')
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            'QPushButton { background: #c0392b; color: white; padding: 6px 16px; }'
            'QPushButton:hover { background: #e74c3c; }'
            'QPushButton:disabled { background: #555; }')
        btn_row.addWidget(self.stop_btn)

        btn_row.addStretch()

        self.open_video_btn = QPushButton('▶ 打开视频')
        self.open_video_btn.setEnabled(False)
        self.open_video_btn.clicked.connect(self._open_video)
        btn_row.addWidget(self.open_video_btn)

        self.open_folder_btn = QPushButton('📂 打开输出目录')
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._open_folder)
        btn_row.addWidget(self.open_folder_btn)

        self.export_report_btn = QPushButton('📊 导出报告')
        self.export_report_btn.setEnabled(False)
        self.export_report_btn.clicked.connect(self._export_report)
        btn_row.addWidget(self.export_report_btn)

        layout.addLayout(btn_row)
        layout.addStretch()

    def _reset_metrics_table(self):
        for row, (metric, value) in enumerate([
            ('跟踪覆盖率', '—'),
            ('平均检测数/帧', '—'),
            ('有跟踪的帧数', '—'),
        ]):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(value))

    def update_preview(self, surface, raw_frame, det_boxes, track_boxes, track_ids, trails):
        """更新实时预览图像"""
        # Time surface preview
        qimg_surface = surface_to_qimage(surface)
        if qimg_surface:
            self.surface_label.setPixmap(QPixmap.fromImage(qimg_surface).scaled(
                320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Tracking result preview
        raw_bgr = raw_to_qimage(raw_frame.pixel(0, 0) if False else None, 0, 0)
        if raw_frame is not None:
            vis = raw_frame.copy()
            if len(vis.shape) == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            H, W = vis.shape[:2]

            # Draw detection boxes (orange)
            for box in det_boxes:
                x, y, w, h = box
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 140, 255), 1)

            # Draw tracking boxes (green) with IDs
            for tid in track_ids:
                if tid in track_boxes:
                    x, y, w, h = track_boxes[tid]
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(vis, f'#{tid}', (x, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Draw trails
            for tid in track_ids:
                if tid in trails:
                    pts = trails[tid]
                    for i in range(1, len(pts)):
                        alpha = i / max(len(pts), 1)
                        color = (int(100 * (1 - alpha)), int(240 * alpha), int(240 * (1 - alpha) + 60 * alpha))
                        cv2.line(vis, pts[i - 1], pts[i], color, 1)

            vis = cv2.resize(vis, (320, 240))
            h, w, ch = vis.shape
            qimg = QImage(vis.data, w, h, ch * w, QImage.Format_BGR888)
            self.result_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_progress(self, frame_idx, total, stats):
        """更新进度条和实时统计"""
        pct = int((frame_idx / max(total, 1)) * 100)
        self.progress_bar.setValue(pct)
        self.status_label.setText(
            f'处理中... {frame_idx}/{total} ({pct}%)  '
            f'| 事件: {stats.get("events", 0)}  '
            f'| 检测: {stats.get("detections", 0)}  '
            f'| 跟踪: {stats.get("tracks", 0)}'
        )

    def on_finished(self, output_path, metrics):
        """处理完成时更新界面"""
        self._output_path = output_path
        self.progress_bar.setValue(100)
        self.status_label.setText(f'✅ 处理完成！输出: {output_path}')
        self.status_label.setStyleSheet('color: #27ae60; font-weight: bold;')

        self.metrics_table.setItem(0, 1, QTableWidgetItem(f'{metrics.get("coverage_pct", 0)}%'))
        self.metrics_table.setItem(1, 1, QTableWidgetItem(str(metrics.get('avg_detections', 0))))
        self.metrics_table.setItem(2, 1, QTableWidgetItem(
            f'{metrics.get("frames_with_track", 0)} / {metrics.get("total_frames", 0)}'))

        self.open_video_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        self.export_report_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def on_error(self, error_msg):
        """处理出错"""
        self.progress_bar.setValue(0)
        self.status_label.setText(f'❌ 处理出错')
        self.status_label.setStyleSheet('color: #e74c3c; font-weight: bold;')
        self.stats_label.setText(error_msg[:500])

    def on_started(self):
        """处理开始"""
        self.progress_bar.setValue(0)
        self.status_label.setText('初始化...')
        self.status_label.setStyleSheet('color: #f39c12;')
        self._reset_metrics_table()
        self.open_video_btn.setEnabled(False)
        self.open_folder_btn.setEnabled(False)
        self.export_report_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.surface_label.setText('加载中...')
        self.result_label.setText('加载中...')

    def _open_video(self):
        """用默认播放器打开视频"""
        if self._output_path and os.path.exists(self._output_path):
            os.startfile(self._output_path)
        else:
            # Try tracking only
            track_path = self._output_path.replace('.mp4', '_tracking_only.mp4')
            if os.path.exists(track_path):
                os.startfile(track_path)

    def _open_folder(self):
        """打开输出目录"""
        folder = os.path.dirname(self._output_path) if self._output_path else ''
        if folder and os.path.exists(folder):
            os.startfile(folder)

    def _export_report(self):
        """导出文本报告"""
        if not self._output_path:
            return
        folder = os.path.dirname(self._output_path)
        report_path = os.path.join(folder, 'processing_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('无人机追踪处理报告\n')
            f.write('=' * 40 + '\n')
            f.write(f'输出视频: {self._output_path}\n')
            for row in range(self.metrics_table.rowCount()):
                metric = self.metrics_table.item(row, 0).text()
                value = self.metrics_table.item(row, 1).text()
                f.write(f'{metric}: {value}\n')
        os.startfile(report_path)
