"""
算法配置面板 —— 清晰的参数分组与调节。
"""
import json
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QCheckBox,
    QFormLayout, QMessageBox, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal


class ConfigPanel(QScrollArea):
    """算法参数配置面板（可滚动）"""

    params_changed = Signal(dict)

    PRESET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'presets')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet('QScrollArea { border: none; }')

        self._container = QWidget()
        self.setWidget(self._container)
        self._build_ui()
        os.makedirs(self.PRESET_DIR, exist_ok=True)

    def _build_ui(self):
        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ---------- 全局样式 ----------
        self.setStyleSheet("""
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #2c3e50;
                border: 1px solid #dcdde1;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 18px;
                padding-bottom: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #1a56db;
            }
            QLabel {
                font-size: 12px;
                color: #2d3436;
            }
            QSpinBox, QDoubleSpinBox {
                font-size: 12px;
                padding: 3px 6px;
                min-height: 26px;
            }
            QComboBox {
                font-size: 12px;
                padding: 3px 6px;
                min-height: 26px;
            }
            QCheckBox {
                font-size: 12px;
                color: #2d3436;
            }
            QPushButton {
                font-size: 12px;
                padding: 4px 12px;
                min-height: 26px;
            }
        """)

        # ====== 时间表面 ======
        ts_group = QGroupBox('⏱ 时间表面')
        ts_form = QFormLayout(ts_group)
        ts_form.setLabelAlignment(Qt.AlignRight)
        ts_form.setHorizontalSpacing(20)
        ts_form.setVerticalSpacing(8)

        self.tau_spin = QDoubleSpinBox()
        self.tau_spin.setRange(5, 100)
        self.tau_spin.setValue(20)
        self.tau_spin.setSuffix(' ms')
        self.tau_spin.setSingleStep(5)
        self.tau_spin.setToolTip('时间衰减常数。越小对快速运动越敏感，越大轨迹保留越久。'
                                 '高速无人机建议 10-20ms，慢速/悬停建议 30-50ms')
        self.tau_spin.valueChanged.connect(self._emit_params)
        tau_label = QLabel('衰减常数 τ')
        tau_label.setToolTip(self.tau_spin.toolTip())
        ts_form.addRow(tau_label, self.tau_spin)

        self.surface_type = QComboBox()
        self.surface_type.addItem('双极性指数衰减', 'bipolar')
        self.surface_type.setToolTip('双极性：分别处理 ON/OFF 事件，保留运动方向信息')
        self.surface_type.currentIndexChanged.connect(self._emit_params)
        ts_form.addRow('表面类型', self.surface_type)

        layout.addWidget(ts_group)

        # ====== 检测器 ======
        det_group = QGroupBox('🔍 检测器')
        det_form = QFormLayout(det_group)
        det_form.setLabelAlignment(Qt.AlignRight)
        det_form.setHorizontalSpacing(20)
        det_form.setVerticalSpacing(8)

        self.min_area = QSpinBox()
        self.min_area.setRange(1, 500)
        self.min_area.setValue(10)
        self.min_area.setSuffix(' px²')
        self.min_area.setToolTip('候选目标的最小面积（像素²）。值越小能检测越远的目标，'
                                 '但也容易把噪点当目标。远距离小无人机建议 5-15')
        min_label = QLabel('最小面积')
        min_label.setToolTip(self.min_area.toolTip())
        det_form.addRow(min_label, self.min_area)

        self.max_area = QSpinBox()
        self.max_area.setRange(10, 2000)
        self.max_area.setValue(300)
        self.max_area.setSuffix(' px²')
        self.max_area.setToolTip('候选目标的最大面积（像素²）。'
                                 '防止大块背景连通域被当成目标。近距离无人机建议 200-500')
        max_label = QLabel('最大面积')
        max_label.setToolTip(self.max_area.toolTip())
        det_form.addRow(max_label, self.max_area)

        self.flow_threshold = QDoubleSpinBox()
        self.flow_threshold.setRange(0.0, 2.0)
        self.flow_threshold.setValue(0.5)
        self.flow_threshold.setSingleStep(0.1)
        self.flow_threshold.setToolTip('光流运动验证阈值。候选框内平均光流低于此值的视为静止虚检。'
                                       '值越高过滤越严格。建议 0.3-0.8')
        flow_label = QLabel('光流阈值')
        flow_label.setToolTip(self.flow_threshold.toolTip())
        det_form.addRow(flow_label, self.flow_threshold)

        self.top_k = QSpinBox()
        self.top_k.setRange(1, 20)
        self.top_k.setValue(1)
        self.top_k.setToolTip('每帧最多保留的候选检测数。设为 1 即单目标模式。')
        det_form.addRow('保留检测数', self.top_k)

        self.enable_multiscale = QCheckBox('启用多尺度检测通道')
        self.enable_multiscale.setChecked(True)
        self.enable_multiscale.setToolTip('同时运行原始表面 + TopHat + DoG 三路检测。'
                                          '关掉只保留原始表面检测，速度快但可能漏检')
        self.enable_multiscale.stateChanged.connect(self._emit_params)
        det_form.addRow(self.enable_multiscale)

        layout.addWidget(det_group)

        # ====== 跟踪器 ======
        trk_group = QGroupBox('🎯 跟踪器')
        trk_form = QFormLayout(trk_group)
        trk_form.setLabelAlignment(Qt.AlignRight)
        trk_form.setHorizontalSpacing(20)
        trk_form.setVerticalSpacing(8)

        self.iou_threshold = QDoubleSpinBox()
        self.iou_threshold.setRange(0.01, 0.9)
        self.iou_threshold.setValue(0.1)
        self.iou_threshold.setSingleStep(0.05)
        self.iou_threshold.setToolTip('检测框与跟踪框匹配的 IoU 最低阈值。'
                                      '小目标框容易漂移，建议设低一些（0.05-0.15）')
        iou_label = QLabel('IoU 匹配阈值')
        iou_label.setToolTip(self.iou_threshold.toolTip())
        trk_form.addRow(iou_label, self.iou_threshold)

        self.max_lost = QSpinBox()
        self.max_lost.setRange(3, 50)
        self.max_lost.setValue(8)
        self.max_lost.setSuffix(' 帧')
        self.max_lost.setToolTip('目标连续丢失多少帧后判定为永久丢失并删除。'
                                 '值越大能容忍更长遮挡，但虚检也存活更久。建议 5-15')
        lost_label = QLabel('最大丢失帧数')
        lost_label.setToolTip(self.max_lost.toolTip())
        trk_form.addRow(lost_label, self.max_lost)

        self.min_hits = QSpinBox()
        self.min_hits.setRange(1, 30)
        self.min_hits.setValue(5)
        self.min_hits.setSuffix(' 帧')
        self.min_hits.setToolTip('新目标需连续命中多少帧后才确认输出。'
                                 '值越高虚检越少，但新目标响应变慢。建议 3-8')
        hits_label = QLabel('确认命中数')
        hits_label.setToolTip(self.min_hits.toolTip())
        trk_form.addRow(hits_label, self.min_hits)

        layout.addWidget(trk_group)

        # ====== 输出 ======
        out_group = QGroupBox('📂 输出')
        out_form = QFormLayout(out_group)
        out_form.setVerticalSpacing(8)
        self.output_dir_combo = QComboBox()
        self.output_dir_combo.setEditable(True)
        self.output_dir_combo.addItem('桌面 drone_tracking_output')
        self.output_dir_combo.addItem('D:\\PythonProject3\\output')
        self.output_dir_combo.currentTextChanged.connect(self._emit_params)
        out_form.addRow('输出目录', self.output_dir_combo)
        layout.addWidget(out_group)

        # ====== 预设方案 ======
        preset_group = QGroupBox('💾 预设方案')
        preset_layout = QHBoxLayout(preset_group)
        preset_layout.setSpacing(8)
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(False)
        self.preset_combo.setMinimumWidth(140)
        self._load_preset_list()
        preset_layout.addWidget(self.preset_combo, 1)

        load_btn = QPushButton('加载')
        load_btn.setToolTip('加载选中的预设参数')
        load_btn.clicked.connect(self._load_preset)
        preset_layout.addWidget(load_btn)

        save_btn = QPushButton('保存')
        save_btn.setToolTip('将当前参数保存为新预设')
        save_btn.clicked.connect(self._save_preset)
        preset_layout.addWidget(save_btn)

        layout.addWidget(preset_group)
        layout.addStretch()

    def _load_preset_list(self):
        self.preset_combo.clear()
        self.preset_combo.addItem('-- 选择预设 --')
        if os.path.exists(self.PRESET_DIR):
            for f in sorted(os.listdir(self.PRESET_DIR)):
                if f.endswith('.json'):
                    self.preset_combo.addItem(f.replace('.json', ''))

    def _save_preset(self):
        name = self.preset_combo.currentText().strip()
        if not name or name == '-- 选择预设 --':
            name = 'my_preset'
        path = os.path.join(self.PRESET_DIR, name.replace('.json', '') + '.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.get_params(), f, ensure_ascii=False, indent=2)
        self._load_preset_list()
        self.preset_combo.setCurrentText(os.path.basename(path).replace('.json', ''))
        QMessageBox.information(self, '已保存', f'预设已保存到:\n{path}')

    def _load_preset(self):
        name = self.preset_combo.currentText().strip()
        if not name or name == '-- 选择预设 --':
            return
        path = os.path.join(self.PRESET_DIR, name + '.json')
        if not os.path.exists(path):
            QMessageBox.warning(self, '错误', f'预设文件不存在:\n{path}')
            return
        with open(path, 'r', encoding='utf-8') as f:
            self.set_params(json.load(f))
        self._emit_params()

    def get_params(self):
        output_text = self.output_dir_combo.currentText()
        if '桌面' in output_text:
            output_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'drone_tracking_output')
        else:
            output_dir = output_text

        return {
            'tau_ms': int(self.tau_spin.value()),
            'surface_type': self.surface_type.currentData(),
            'min_area': self.min_area.value(),
            'max_area': self.max_area.value(),
            'flow_threshold': self.flow_threshold.value(),
            'top_k': self.top_k.value(),
            'enable_multiscale': self.enable_multiscale.isChecked(),
            'iou_threshold': self.iou_threshold.value(),
            'max_lost': self.max_lost.value(),
            'min_hits': self.min_hits.value(),
            'output_dir': output_dir,
        }

    def set_params(self, params):
        if 'tau_ms' in params:
            self.tau_spin.setValue(int(params['tau_ms']))
        if 'min_area' in params:
            self.min_area.setValue(int(params['min_area']))
        if 'max_area' in params:
            self.max_area.setValue(int(params['max_area']))
        if 'flow_threshold' in params:
            self.flow_threshold.setValue(float(params['flow_threshold']))
        if 'top_k' in params:
            self.top_k.setValue(int(params['top_k']))
        if 'enable_multiscale' in params:
            self.enable_multiscale.setChecked(bool(params['enable_multiscale']))
        if 'iou_threshold' in params:
            self.iou_threshold.setValue(float(params['iou_threshold']))
        if 'max_lost' in params:
            self.max_lost.setValue(int(params['max_lost']))
        if 'min_hits' in params:
            self.min_hits.setValue(int(params['min_hits']))

    def _emit_params(self):
        self.params_changed.emit(self.get_params())
