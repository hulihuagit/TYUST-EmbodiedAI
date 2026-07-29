"""
真实数据集加载器 —— 从 Event Frame PNGs 中提取事件数据，
替代 EventSimulator 用于真实事件相机数据的处理。

数据来源：data/ 目录
  - Event/Frames/*.png : 事件帧可视化（ON=白色, OFF=浅棕色, 背景=深棕色）
  - RGB/*.jpg : 同步RGB帧
  - coordinates.txt : 地面真值标注

原理：
  Event Frame PNGs 是事件相机的"积分图"——每帧累积显示一段时间内触发过
  事件的像素。通过比较连续帧之间的差异，可以提取出当前时间窗口内新产生的
  事件（位置和极性），然后送入时间表面管道处理。
"""

import cv2
import numpy as np
import os
import re
from pathlib import Path


class RealDataLoader:
    """
    从 Event Frame PNGs 中按时间顺序读取数据，
    通过帧间差分提取 (x, y, polarity, timestamp) 事件流。
    """

    # 事件帧中的颜色定义（BGR）
    BG_COLOR = np.array([52, 37, 30])       # 背景（无事件）
    OFF_COLOR = np.array([200, 126, 64])    # OFF 事件
    ON_COLOR = np.array([255, 255, 255])    # ON 事件

    def __init__(self, data_dir, target_size=None, color_tolerance=30):
        """
        参数:
          data_dir: 数据集根目录（包含 Event/Frames/, RGB/, coordinates.txt）
          target_size: (width, height) 目标处理分辨率，None 则保持原始 1280x720
          color_tolerance: 颜色匹配容差
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size  # (W, H) 或 None
        self.color_tolerance = color_tolerance

        # ---- 加载并排序事件帧 ----
        event_frame_dir = self.data_dir / "Event" / "Frames"
        self.event_files = []
        for f in event_frame_dir.glob("*.png"):
            m = re.search(r'frame_(\d+)', f.name)
            if m:
                ts_us = int(m.group(1))  # 微秒时间戳
                self.event_files.append((ts_us, str(f)))
        # 按时间戳排序
        self.event_files.sort(key=lambda x: x[0])

        # 过滤掉明显异常的时间戳（保留 > 0.03 秒的帧，对应 GT 数据从 ~13.5s 开始）
        self.event_files = [(ts, p) for ts, p in self.event_files if ts > 30000]
        print(f"[DataLoader] 加载了 {len(self.event_files)} 个事件帧")

        # ---- 加载并排序 RGB 帧 ----
        rgb_dir = self.data_dir / "RGB"
        self.rgb_files = []
        for f in rgb_dir.glob("*.jpg"):
            m = re.search(r'Video_0_(\d+)_(\d+)_(\d+)\.(\d+)', f.name)
            if m:
                h, mi, s, us = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                total_us = (h * 3600 + mi * 60 + s) * 1_000_000 + us
                self.rgb_files.append((total_us, str(f)))
        self.rgb_files.sort(key=lambda x: x[0])
        print(f"[DataLoader] 加载了 {len(self.rgb_files)} 个 RGB 帧")

        # ---- 加载地面真值坐标 ----
        self.gt_boxes = {}  # timestamp_us -> list of [x, y, w, h]
        coord_file = self.data_dir / "coordinates.txt"
        if coord_file.exists():
            with open(coord_file, 'r') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parts = line.split(': ', 1)
                        if len(parts) != 2:
                            continue
                        t_sec = float(parts[0])
                        t_us = int(t_sec * 1_000_000)
                        coords_str = parts[1]
                        # 格式: x1, y1, x2, y2, class_id, class_name
                        raw_vals = [x.strip() for x in coords_str.split(',')]
                        vals = []
                        for v in raw_vals[:4]:  # 仅取前4个数值
                            try:
                                vals.append(float(v))
                            except ValueError:
                                break
                        if len(vals) >= 4:
                            x1, y1, x2, y2 = vals[0:4]
                            w = x2 - x1
                            h = y2 - y1
                            self.gt_boxes.setdefault(t_us, []).append(
                                [int(x1), int(y1), int(w), int(h)])
                    except (ValueError, IndexError):
                        continue
            print(f"[DataLoader] 加载了 {len(self.gt_boxes)} 个地面真值时间点")

        # ---- 状态变量 ----
        self.frame_idx = 0
        self.prev_event_mask = None   # 上一帧的事件分类图 (H,W) int8: 0=bg, 1=ON, -1=OFF
        self.time = 0.0
        self.dt = 1.0 / 30.0  # ~30fps

        # 原始分辨率
        self.original_size = (1280, 720)  # (W, H)
        if self.target_size is None:
            self.target_size = self.original_size
        self.scale_x = self.target_size[0] / self.original_size[0]
        self.scale_y = self.target_size[1] / self.original_size[1]

        # 颜色匹配容差（欧氏距离）
        self.color_tol_sq = self.color_tolerance ** 2

    def _classify_pixels(self, bgr_img):
        """
        基于颜色对事件帧进行像素分类。
        返回: (H, W) int8 数组，0=背景, 1=ON事件, -1=OFF事件
        """
        H, W = bgr_img.shape[:2]
        img_flat = bgr_img.reshape(-1, 3).astype(np.int32)

        # 计算每个像素到各类颜色的欧氏距离
        bg_dist = np.sum((img_flat - self.BG_COLOR.astype(np.int32)) ** 2, axis=1)
        off_dist = np.sum((img_flat - self.OFF_COLOR.astype(np.int32)) ** 2, axis=1)
        on_dist = np.sum((img_flat - self.ON_COLOR.astype(np.int32)) ** 2, axis=1)

        # 初始化全为背景
        classification = np.zeros(H * W, dtype=np.int8)

        # 在容差范围内匹配 ON 事件
        on_match = on_dist <= self.color_tol_sq
        classification[on_match] = 1

        # 在容差范围内匹配 OFF 事件（OFF 优先级低于 ON，因为 ON 是纯白更容易匹配）
        off_match = off_dist <= self.color_tol_sq
        # 对于同时匹配 ON 和 OFF 的像素，选择距离更近的
        conflict = on_match & off_match
        on_closer = on_dist[conflict] <= off_dist[conflict]
        off_only = off_match & ~on_match
        classification[off_only] = -1
        # 冲突情况：ON 更近的保持 1，OFF 更近的设为 -1
        conflict_indices = np.where(conflict)[0]
        for ci in conflict_indices:
            if on_dist[ci] <= off_dist[ci]:
                classification[ci] = 1
            else:
                classification[ci] = -1

        return classification.reshape(H, W)

    def reset(self):
        """重置到起始位置"""
        self.frame_idx = 0
        self.prev_event_mask = None
        self.time = 0.0

    def step(self):
        """
        读取下一帧，返回 (events, rgb_frame)。

        使用颜色分类+帧间比较提取事件，避免 resize 插值伪影。

        返回:
          events: list of (x, y, polarity, timestamp_seconds)
          rgb_frame: (H, W) uint8 灰度图像，或 None
        """
        if self.frame_idx >= len(self.event_files):
            return [], None

        # ---- 读取事件帧（原始分辨率，用于颜色分类） ----
        ts_us, event_path = self.event_files[self.frame_idx]
        event_img_original = cv2.imread(event_path)
        if event_img_original is None:
            self.frame_idx += 1
            return [], None

        # ---- 颜色分类（在原始分辨率下进行，避免插值伪影） ----
        curr_mask = self._classify_pixels(event_img_original)

        # ---- 通过与上一帧比较提取新事件 ----
        events = []
        if self.prev_event_mask is not None:
            # 找出状态变化的像素
            # 新 ON 事件：上一帧不是 ON，当前帧是 ON
            new_on = (self.prev_event_mask != 1) & (curr_mask == 1)
            # 新 OFF 事件：上一帧不是 OFF，当前帧是 OFF
            new_off = (self.prev_event_mask != -1) & (curr_mask == -1)

            orig_h, orig_w = curr_mask.shape

            # 提取 ON 事件坐标，缩放到目标分辨率
            if np.any(new_on):
                ys, xs = np.where(new_on)
                t = self.time
                for ox, oy in zip(xs, ys):
                    tx = int(ox * self.scale_x)
                    ty = int(oy * self.scale_y)
                    events.append((tx, ty, 1, t))

            # 提取 OFF 事件坐标，缩放到目标分辨率
            if np.any(new_off):
                ys, xs = np.where(new_off)
                t = self.time
                for ox, oy in zip(xs, ys):
                    tx = int(ox * self.scale_x)
                    ty = int(oy * self.scale_y)
                    events.append((tx, ty, -1, t))

        # 保存当前分类结果
        self.prev_event_mask = curr_mask

        # ---- 准备可视化用的帧（缩放到目标分辨率） ----
        if self.target_size != self.original_size:
            event_img_viz = cv2.resize(event_img_original, self.target_size)
        else:
            event_img_viz = event_img_original
        event_gray = cv2.cvtColor(event_img_viz, cv2.COLOR_BGR2GRAY)

        # ---- 读取最接近的 RGB 帧 ----
        rgb_frame = self._find_closest_rgb(ts_us)

        # ---- 更新时间 ----
        self.time += self.dt
        self.frame_idx += 1

        return events, rgb_frame

    def _find_closest_rgb(self, event_ts_us):
        """
        找到最接近的 RGB 帧。
        事件帧和 RGB 帧是同步录制的，数量相同（各3233帧），
        但由于使用不同的时钟命名，直接按索引匹配。
        """
        if not self.rgb_files:
            return None

        # 按索引比例匹配（两个序列同步录制，帧率相同）
        event_count = len(self.event_files)
        rgb_count = len(self.rgb_files)
        ratio = self.frame_idx / max(event_count - 1, 1)
        closest_idx = min(int(ratio * (rgb_count - 1)), rgb_count - 1)

        rgb_path = self.rgb_files[closest_idx][1]
        rgb = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        if rgb is None:
            return None

        if self.target_size != self.original_size:
            rgb = cv2.resize(rgb, self.target_size)

        return rgb

    def get_ground_truth(self, frame_idx=None):
        """
        返回当前帧的地面真值边界框。
        如果找不到精确匹配，返回最接近的。
        """
        if frame_idx is None:
            frame_idx = self.frame_idx

        if frame_idx >= len(self.event_files):
            return []

        ts_us = self.event_files[frame_idx][0]

        # 找容差范围内的 GT
        tolerance_us = 50000  # 50ms
        for gt_ts in sorted(self.gt_boxes.keys()):
            if abs(gt_ts - ts_us) < tolerance_us:
                boxes = self.gt_boxes[gt_ts]
                # 缩放到目标分辨率
                scaled_boxes = []
                for box in boxes:
                    x, y, w, h = box
                    scaled_boxes.append([
                        int(x * self.scale_x),
                        int(y * self.scale_y),
                        int(w * self.scale_x),
                        int(h * self.scale_y)
                    ])
                return scaled_boxes

        return []

    def get_ground_truth_all_frames(self):
        """获取当前帧的真值"""
        return self.get_ground_truth()

    @property
    def height(self):
        return self.target_size[1]

    @property
    def width(self):
        return self.target_size[0]

    def __len__(self):
        return len(self.event_files)
