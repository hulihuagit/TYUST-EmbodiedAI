import numpy as np
import cv2

class TimeSurface:
    """
    将稀疏事件流转换为密集的时间表面表示。
    支持双极性分离、指数衰减和多尺度时间窗口（参考FRED方法）。
    """
    def __init__(self, height=260, width=346, tau=20e-3, decay_factor=0.95):
        self.height = height
        self.width = width
        self.tau = tau          # 主时间常数（秒）— 高速小目标用较短tau
        self.decay_factor = decay_factor

        # 存储每个像素最后触发事件的时间戳（双极性）
        self.last_timestamp_pos = np.full((height, width), -np.inf, dtype=np.float32)
        self.last_timestamp_neg = np.full((height, width), -np.inf, dtype=np.float32)

        # 事件计数器（用于自适应阈值）
        self.event_count = np.zeros((height, width), dtype=np.int32)
        self.max_events_per_pixel = 0

    def update(self, events, current_time):
        """
        输入: events - list of (x, y, polarity, timestamp)
              current_time - 当前时间戳 (秒)
        返回: surface (H,W) float32 归一化到 [0,1]
        """
        if not events:
            return self._compute_surface(current_time)

        # 向量化更新
        x = np.array([e[0] for e in events], dtype=np.int32)
        y = np.array([e[1] for e in events], dtype=np.int32)
        pol = np.array([e[2] for e in events], dtype=np.int8)
        ts = np.array([e[3] for e in events], dtype=np.float32)

        # 限制坐标范围
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)

        # 更新事件计数（用于活动度热图）
        np.add.at(self.event_count, (y, x), 1)
        self.max_events_per_pixel = max(self.max_events_per_pixel,
                                         self.event_count.max())

        # 分离极性
        pos_mask = pol > 0
        neg_mask = pol < 0

        if np.any(pos_mask):
            self.last_timestamp_pos[y[pos_mask], x[pos_mask]] = ts[pos_mask]
        if np.any(neg_mask):
            self.last_timestamp_neg[y[neg_mask], x[neg_mask]] = ts[neg_mask]

        return self._compute_surface(current_time)

    def get_multi_scale(self, current_time, tau_scales=None):
        """
        返回多尺度时间表面（类似FRED多时间窗口）。
        tau_scales: list of tau值，默认 [10ms, 20ms, 40ms, 80ms]
        返回: surfaces dict {tau_value: surface_array}
        """
        if tau_scales is None:
            tau_scales = [10e-3, 20e-3, 40e-3, 80e-3]

        surfaces = {}
        original_tau = self.tau

        for tau in tau_scales:
            self.tau = tau
            surfaces[tau] = self._compute_surface(current_time)

        self.tau = original_tau
        return surfaces

    def _compute_surface(self, current_time):
        """根据当前时间计算衰减后的时间表面"""
        dt_pos = current_time - self.last_timestamp_pos
        dt_neg = current_time - self.last_timestamp_neg

        # 避免负值
        dt_pos = np.maximum(dt_pos, 0)
        dt_neg = np.maximum(dt_neg, 0)

        # 指数衰减
        surf_pos = np.exp(-dt_pos / max(self.tau, 1e-9))
        surf_neg = np.exp(-dt_neg / max(self.tau, 1e-9))

        # 合并：正极性为正贡献，负极性为负贡献
        surface = surf_pos - surf_neg

        # 归一化到 [0,1]
        surface = (surface + 1.0) / 2.0
        surface = np.clip(surface, 0, 1)

        return surface.astype(np.float32)

    def get_activity_map(self):
        """
        返回归一化的事件活动热图。
        高频事件区域通常是运动目标所在。
        """
        if self.max_events_per_pixel == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)
        return (self.event_count.astype(np.float32) /
                max(self.max_events_per_pixel, 1))

    def reset(self):
        """重置所有状态"""
        self.last_timestamp_pos.fill(-np.inf)
        self.last_timestamp_neg.fill(-np.inf)
        self.event_count.fill(0)
        self.max_events_per_pixel = 0
