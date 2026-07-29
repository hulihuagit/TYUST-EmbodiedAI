import cv2
import numpy as np
import os
import time


class VideoWriter:
    """
    生成2×2仪表盘视图 + 纯跟踪视图的视频。
    仪表盘包含：
      左上：时间表面伪彩色
      右上：原始灰度帧 + 速度矢量
      左下：检测结果（橙色框）
      右下：跟踪结果（绿色框 + ID + 轨迹 + 十字准星）
    """
    def __init__(self, output_path, fps=30, panel_size=(320, 420)):
        self.output_path = output_path
        self.fps = fps
        self.panel_h, self.panel_w = panel_size
        self.total_h = self.panel_h * 2
        self.total_w = self.panel_w * 2

        # 仪表盘视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps,
                                      (self.total_w, self.total_h))

        # 纯跟踪视频（单独输出，更清晰）
        track_path = output_path.replace('.mp4', '_tracking_only.mp4')
        self.track_writer = cv2.VideoWriter(track_path, fourcc, fps,
                                            (self.panel_w, self.panel_h))

        # FPS 计时
        self._frame_count = 0
        self._start_time = time.time()
        self._fps_display = 0.0

    def write_frame(self, surface, raw_frame, det_boxes, track_boxes,
                    track_ids, trails, velocities=None):
        """
        组装并写入一帧。
        surface: (H,W) float32 [0,1]
        raw_frame: (H,W) uint8
        det_boxes: list of [x,y,w,h]
        track_boxes: dict id->[x,y,w,h]
        track_ids: list of ids
        trails: dict id->list of (cx,cy)
        velocities: dict id->(vx,vy) optional
        """
        H, W = surface.shape

        # FPS计算
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            elapsed = time.time() - self._start_time
            self._fps_display = 30.0 / max(elapsed, 0.001)
            self._start_time = time.time()

        # ---- 面板1: 时间表面伪彩色 ----
        surf_color = cv2.applyColorMap((surface * 255).astype(np.uint8),
                                        cv2.COLORMAP_JET)
        panel1 = cv2.resize(surf_color, (self.panel_w, self.panel_h))
        # 添加颜色条说明
        cv2.putText(panel1, "Recent", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel1, "Old", (5, self.panel_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ---- 面板2: 原始帧（增强对比度） ----
        raw_enhanced = cv2.equalizeHist(raw_frame)  # 直方图均衡增强可见性
        raw_bgr = cv2.cvtColor(raw_enhanced, cv2.COLOR_GRAY2BGR)
        panel2 = cv2.resize(raw_bgr, (self.panel_w, self.panel_h))

        # 在面板2上绘制速度矢量
        if velocities:
            for tid in track_ids:
                if tid in velocities and tid in track_boxes:
                    vx, vy = velocities[tid]
                    box = track_boxes[tid]
                    cx = int((box[0] + box[2] / 2) * self.panel_w / W)
                    cy = int((box[1] + box[3] / 2) * self.panel_h / H)
                    speed = np.sqrt(vx**2 + vy**2) * self.panel_w / W
                    if speed > 0.5:
                        angle = np.arctan2(vy, vx)
                        ex = int(cx + np.cos(angle) * speed * 0.5)
                        ey = int(cy + np.sin(angle) * speed * 0.5)
                        ex = np.clip(ex, 0, self.panel_w - 1)
                        ey = np.clip(ey, 0, self.panel_h - 1)
                        cv2.arrowedLine(panel2, (cx, cy), (ex, ey),
                                        (0, 255, 255), 1, tipLength=0.3)

        # ---- 面板3: 检测结果 ----
        panel3 = cv2.resize(raw_bgr, (self.panel_w, self.panel_h))
        for box in det_boxes:
            x, y, w, h = box
            sx = int(x * self.panel_w / W)
            sy = int(y * self.panel_h / H)
            sw = int(w * self.panel_w / W)
            sh = int(h * self.panel_h / H)
            # 橙色检测框
            cv2.rectangle(panel3, (sx, sy), (sx + sw, sy + sh),
                          (255, 140, 0), 1)
            # 十字准星
            cx, cy = sx + sw // 2, sy + sh // 2
            cross_len = max(3, min(sw, sh) // 2)
            cv2.line(panel3, (cx - cross_len, cy), (cx + cross_len, cy),
                     (255, 140, 0), 1)
            cv2.line(panel3, (cx, cy - cross_len), (cx, cy + cross_len),
                     (255, 140, 0), 1)

        # ---- 面板4: 跟踪结果（主要输出） ----
        panel4 = cv2.resize(raw_bgr, (self.panel_w, self.panel_h))

        # 绘制轨迹线
        for tid in track_ids:
            if tid in trails:
                trail = trails[tid]
                pts = []
                for pt in trail:
                    px = int(pt[0] * self.panel_w / W)
                    py = int(pt[1] * self.panel_h / H)
                    pts.append((px, py))
                # 轨迹线（渐变色表示时间方向）
                for i in range(1, len(pts)):
                    alpha = i / max(len(pts), 1)
                    color = (
                        int(0 * (1 - alpha) + 0 * alpha),
                        int(100 * (1 - alpha) + 240 * alpha),
                        int(240 * (1 - alpha) + 60 * alpha),
                    )
                    cv2.line(panel4, pts[i-1], pts[i], color, 1)

        # 绘制跟踪框、ID和十字准星
        for tid in track_ids:
            if tid not in track_boxes:
                continue
            box = track_boxes[tid]
            x, y, w, h = box
            sx = int(x * self.panel_w / W)
            sy = int(y * self.panel_h / H)
            sw = max(1, int(w * self.panel_w / W))
            sh = max(1, int(h * self.panel_h / H))

            # 为不同ID设置不同颜色
            hue = (tid * 137) % 180  # 黄金比例分散颜色
            color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]),
                                 cv2.COLOR_HSV2BGR)[0, 0]
            color = (int(color[0]), int(color[1]), int(color[2]))

            # 矩形框
            cv2.rectangle(panel4, (sx, sy), (sx + sw, sy + sh), color, 2)

            # 十字准星在目标中心
            cx, cy = sx + sw // 2, sy + sh // 2
            cross_len = max(2, min(sw, sh) // 2)
            cv2.line(panel4, (cx - cross_len, cy), (cx + cross_len, cy),
                     color, 1)
            cv2.line(panel4, (cx, cy - cross_len), (cx, cy + cross_len),
                     color, 1)

            # ID标签（带背景）
            label = f"Drone#{tid}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(panel4, (sx, sy - lh - 4), (sx + lw + 2, sy),
                          (50, 50, 50), -1)
            cv2.putText(panel4, label, (sx, sy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ---- 纯跟踪视图（单独视频） ----
        track_view = panel4.copy()

        # ---- 合成仪表盘 ----
        top = np.hstack([panel1, panel2])
        bottom = np.hstack([panel3, panel4])
        full = np.vstack([top, bottom])

        # 面板标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 半透明标签背景
        for i, (label, lx, ly) in enumerate([
            ("Time Surface", 10, 25),
            ("Raw Frame + Velocity", self.panel_w + 10, 25),
            ("Detection (Orange)", 10, self.panel_h + 25),
            ("Tracking (Colored + Trails)", self.panel_w + 10, self.panel_h + 25),
        ]):
            cv2.putText(full, label, (lx, ly), font, 0.5, (0, 0, 0), 3)
            cv2.putText(full, label, (lx, ly), font, 0.5, (255, 255, 255), 1)

        # FPS和帧计数
        info_text = f"FPS: {self._fps_display:.1f} | Frame: {self._frame_count}"
        cv2.putText(full, info_text, (10, self.total_h - 12),
                    font, 0.45, (0, 255, 0), 1)

        # 写入
        self.writer.write(full)
        self.track_writer.write(track_view)

    def close(self):
        self.writer.release()
        self.track_writer.release()
