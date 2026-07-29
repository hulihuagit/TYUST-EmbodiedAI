"""
数据导入面板 —— 选择数据集目录、校验文件、预览事件帧。
"""
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QFileDialog, QTextEdit, QComboBox,
    QSpinBox, QFormLayout, QMessageBox
)
from PySide6.QtCore import Qt, Signal


class ImportPanel(QWidget):
    """数据导入与校验面板"""

    dir_changed = Signal(str)       # 数据目录变更
    resolution_changed = Signal(int, int)  # width, height

    PRESET_RESOLUTIONS = [
        ('640 × 360（推荐：半分辨率）', 640, 360),
        ('1280 × 720（原始分辨率）', 1280, 720),
        ('426 × 240（快速预览）', 426, 240),
        ('320 × 180（极速预览）', 320, 180),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_dir = ''
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.setStyleSheet("""
            QGroupBox {
                font-size: 13px; font-weight: bold; color: #2c3e50;
                border: 1px solid #dcdde1; border-radius: 6px;
                margin-top: 14px; padding-top: 18px; padding-bottom: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #1a56db;
            }
            QLabel { font-size: 12px; color: #2d3436; }
            QLineEdit { font-size: 12px; padding: 4px 8px; min-height: 26px; }
            QComboBox { font-size: 12px; padding: 3px 6px; min-height: 26px; }
            QPushButton { font-size: 12px; padding: 4px 12px; min-height: 26px; }
            QSpinBox { font-size: 12px; padding: 3px 6px; min-height: 26px; }
            QTextEdit { font-size: 11px; }
        """)

        # ---- 数据目录 ----
        dir_group = QGroupBox('📁 数据目录')
        dir_layout = QHBoxLayout(dir_group)
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText('选择数据集根目录...')
        self.dir_edit.setText('D:\\PythonProject3\\data')
        self.dir_btn = QPushButton('📁')
        self.dir_btn.setFixedWidth(36)
        self.dir_btn.clicked.connect(self._select_dir)
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(self.dir_btn)
        layout.addWidget(dir_group)

        # ---- 校验结果 ----
        validate_group = QGroupBox('✅ 文件校验')
        self.validate_text = QTextEdit()
        self.validate_text.setReadOnly(True)
        self.validate_text.setMaximumHeight(120)
        self.validate_text.setStyleSheet('QTextEdit { font-size: 11px; }')
        val_layout = QVBoxLayout(validate_group)
        val_layout.addWidget(self.validate_text)
        layout.addWidget(validate_group)

        # ---- 分辨率 + 帧范围 ----
        proc_group = QGroupBox('⚙️ 处理选项')
        form = QFormLayout(proc_group)

        self.res_combo = QComboBox()
        for label, w, h in self.PRESET_RESOLUTIONS:
            self.res_combo.addItem(label, (w, h))
        self.res_combo.currentIndexChanged.connect(self._on_resolution_changed)
        form.addRow('处理分辨率:', self.res_combo)

        self.frame_limit = QSpinBox()
        self.frame_limit.setRange(0, 100000)
        self.frame_limit.setValue(0)
        self.frame_limit.setSpecialValueText('全部帧')
        self.frame_limit.setSuffix(' 帧')
        form.addRow('处理帧数:', self.frame_limit)

        layout.addWidget(proc_group)

        # ---- 预览 ----
        btn_row = QHBoxLayout()
        self.preview_event_btn = QPushButton('预览事件帧')
        self.preview_event_btn.clicked.connect(self._preview_event)
        self.preview_gt_btn = QPushButton('预览 GT 标注')
        self.preview_gt_btn.clicked.connect(self._preview_gt)
        btn_row.addWidget(self.preview_event_btn)
        btn_row.addWidget(self.preview_gt_btn)
        layout.addLayout(btn_row)

        layout.addStretch()

        # 初始化校验
        self._validate_directory()

    def _select_dir(self):
        path = QFileDialog.getExistingDirectory(self, '选择数据集目录', self.dir_edit.text())
        if path:
            self.dir_edit.setText(path)
            self._validate_directory()

    def _validate_directory(self):
        path = self.dir_edit.text().strip()
        if not path:
            self.validate_text.setPlainText('未选择目录')
            return

        data = Path(path)
        lines = []
        lines.append(f'目录: {path}')

        # Check event frames
        event_dir = data / 'Event' / 'Frames'
        if event_dir.exists():
            pngs = list(event_dir.glob('*.png'))
            lines.append(f'✅ 事件帧 PNG: {len(pngs)} 个')
        else:
            lines.append('❌ Event/Frames/ 目录不存在')

        # Check RGB
        rgb_dir = data / 'RGB'
        if rgb_dir.exists():
            jpgs = list(rgb_dir.glob('*.jpg'))
            lines.append(f'✅ RGB 帧 JPG: {len(jpgs)} 个')
        else:
            lines.append('⚠️ RGB/ 目录不存在（将使用时间表面代替）')

        # Check GT
        gt_file = data / 'coordinates.txt'
        if gt_file.exists():
            lines.append(f'✅ GT 坐标: coordinates.txt')
        else:
            lines.append('⚠️ 无 GT 坐标文件')

        # Check HDF5
        hdf5_file = data / 'Event' / 'events.hdf5'
        if hdf5_file.exists():
            size_mb = hdf5_file.stat().st_size / (1024 * 1024)
            lines.append(f'⚠️ HDF5: {size_mb:.0f} MB（滤镜未注册，将使用 PNG 提取）')
        else:
            lines.append('❌ 无 HDF5 事件文件')

        self.validate_text.setPlainText('\n'.join(lines))
        self.data_dir = path
        self.dir_changed.emit(path)

    def _on_resolution_changed(self):
        w, h = self.res_combo.currentData()
        self.resolution_changed.emit(w, h)

    def _preview_event(self):
        """打开事件帧预览窗口"""
        path = self.dir_edit.text().strip()
        event_dir = os.path.join(path, 'Event', 'Frames')
        if not os.path.exists(event_dir):
            QMessageBox.warning(self, '提示', '事件帧目录不存在')
            return

        import re
        import cv2
        files = sorted(os.listdir(event_dir),
                       key=lambda f: int(re.search(r'frame_(\d+)', f).group(1))
                       if re.search(r'frame_(\d+)', f) else 0)
        if not files:
            QMessageBox.warning(self, '提示', '目录下无 PNG 文件')
            return

        # Show first, middle, last frames
        import numpy as np
        indices = [0, len(files) // 2, len(files) - 1]
        imgs = []
        for idx in indices:
            if idx < len(files):
                fp = os.path.join(event_dir, files[idx])
                img = cv2.imread(fp)
                if img is not None:
                    img = cv2.resize(img, (426, 240))
                    cv2.putText(img, f'Frame {idx+1}/{len(files)}', (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    imgs.append(img)

        if imgs:
            combined = np.hstack(imgs) if len(imgs) > 1 else imgs[0]
            cv2.imshow('事件帧预览 (按任意键关闭)', combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _preview_gt(self):
        """预览 GT 标注叠加在 RGB 帧上"""
        path = self.dir_edit.text().strip()
        gt_file = os.path.join(path, 'coordinates.txt')
        if not os.path.exists(gt_file):
            QMessageBox.warning(self, '提示', 'GT 坐标文件不存在')
            return

        # Read GT
        import cv2
        import numpy as np
        import re

        gts = {}  # timestamp_us -> [(x1,y1,x2,y2)]
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ': ' not in line:
                    continue
                try:
                    t_str, coords_str = line.split(': ', 1)
                    t_us = int(float(t_str) * 1_000_000)
                    raw = coords_str.split(',')
                    vals = []
                    for v in raw[:4]:
                        try:
                            vals.append(float(v.strip()))
                        except ValueError:
                            break
                    if len(vals) == 4:
                        gts[t_us] = [int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])]
                except (ValueError, IndexError):
                    continue

        if not gts:
            QMessageBox.warning(self, '提示', '无法解析 GT 数据')
            return

        # Pick a middle GT entry
        keys = sorted(gts.keys())
        mid_ts = keys[len(keys) // 2]
        x1, y1, x2, y2 = gts[mid_ts]

        # Try to find corresponding RGB frame
        rgb_dir = os.path.join(path, 'RGB')
        if os.path.exists(rgb_dir):
            rgb_files = sorted(os.listdir(rgb_dir))
            if rgb_files:
                mid_rgb = rgb_files[len(rgb_files) // 2]
                img = cv2.imread(os.path.join(rgb_dir, mid_rgb))
                if img is not None:
                    img = cv2.resize(img, (640, 360))
                    sx, sy = 640 / 1280, 360 / 720
                    cv2.rectangle(img,
                                  (int(x1 * sx), int(y1 * sy)),
                                  (int(x2 * sx), int(y2 * sy)),
                                  (0, 255, 0), 2)
                    cv2.putText(img, f'GT: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})',
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow('GT 标注预览 (按任意键关闭)', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    return

        QMessageBox.information(self, 'GT 信息',
                                f'GT 条目数: {len(gts)}\n'
                                f'示例: t={mid_ts/1e6:.3f}s\n'
                                f'框: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] '
                                f'= {x2-x1:.0f}×{y2-y1:.0f} px')

    def get_params(self):
        """返回面板中的导入参数"""
        w, h = self.res_combo.currentData()
        return {
            'data_dir': self.dir_edit.text().strip(),
            'res_w': w,
            'res_h': h,
            'total_frames': self.frame_limit.value(),
        }
