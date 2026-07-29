import numpy as np
import cv2

class Drone:
    """单架无人机的运动模型"""
    def __init__(self, drone_id, height, width, rng):
        self.id = drone_id
        self.height = height
        self.width = width
        self.rng = rng  # 保存随机数生成器

        # 无人机尺寸（像素）—— 模拟不同距离/大小（3~10像素）
        self.size = int(rng.integers(3, 10))
        margin = self.size + 5

        # 初始位置
        self.pos = np.array([
            float(rng.integers(margin, width - margin)),
            float(rng.integers(margin, height - margin))
        ], dtype=float)

        # 速度（像素/秒）—— 无人机典型高速运动
        speed = float(rng.uniform(30, 120))
        angle = float(rng.uniform(0, 2 * np.pi))
        self.vel = np.array([speed * np.cos(angle), speed * np.sin(angle)], dtype=float)

        # 加速度（像素/秒²）
        self.accel = np.zeros(2, dtype=float)

        # 行为模式: 'cruise' | 'hover' | 'turn' | 'dash'
        self.mode = 'cruise'
        self.mode_timer = 0
        self.mode_duration = int(rng.integers(30, 80))  # 帧数
        self.intensity = float(rng.uniform(0.6, 1.0))

        # 轨迹历史（用于可视化）
        self.history = [self.pos.copy()]

    def update(self, dt):
        """更新无人机状态"""
        self.mode_timer -= 1
        if self.mode_timer <= 0:
            self._switch_mode()

        # 根据模式调整速度和加速度
        if self.mode == 'cruise':
            # 巡航：匀速，偶尔微调方向
            if self.rng.random() < 0.02:
                angle_change = self.rng.uniform(-0.3, 0.3)
                speed = np.linalg.norm(self.vel)
                angle = np.arctan2(self.vel[1], self.vel[0]) + angle_change
                self.vel = np.array([speed * np.cos(angle), speed * np.sin(angle)])

        elif self.mode == 'hover':
            # 悬停：缓慢减速至接近静止
            self.vel *= 0.9
            if np.linalg.norm(self.vel) < 5:
                # 微小漂移
                self.vel += self.rng.uniform(-2, 2, 2)

        elif self.mode == 'turn':
            # 急转弯：快速改变方向
            turn_rate = self.rng.uniform(0.05, 0.2)
            angle = np.arctan2(self.vel[1], self.vel[0]) + turn_rate
            speed = np.linalg.norm(self.vel)
            self.vel = np.array([speed * np.cos(angle), speed * np.sin(angle)])

        elif self.mode == 'dash':
            # 冲刺：加速直线运动
            direction = self.vel / (np.linalg.norm(self.vel) + 1e-6)
            self.vel += direction * self.rng.uniform(10, 30) * dt
            # 限速
            max_speed = 200
            speed = np.linalg.norm(self.vel)
            if speed > max_speed:
                self.vel = self.vel / speed * max_speed

        # 更新位置
        self.pos = self.pos + self.vel * dt

        # 边界反弹
        margin = self.size + 2
        if self.pos[0] < margin:
            self.pos[0] = margin
            self.vel[0] = abs(self.vel[0])
        elif self.pos[0] > self.width - margin:
            self.pos[0] = self.width - margin
            self.vel[0] = -abs(self.vel[0])

        if self.pos[1] < margin:
            self.pos[1] = margin
            self.vel[1] = abs(self.vel[1])
        elif self.pos[1] > self.height - margin:
            self.pos[1] = self.height - margin
            self.vel[1] = -abs(self.vel[1])

        # 记录历史
        self.history.append(self.pos.copy())
        if len(self.history) > 200:
            self.history = self.history[-200:]

    def _switch_mode(self):
        """切换飞行模式"""
        modes = ['cruise', 'cruise', 'hover', 'turn', 'dash']
        weights = [0.4, 0.2, 0.15, 0.15, 0.1]
        self.mode = self.rng.choice(modes, p=weights)
        self.mode_duration = int(self.rng.integers(20, 100))


class EventSimulator:
    """
    模拟DAVIS事件相机数据，生成稀疏事件流。
    模拟多架无人机在场景中飞行。
    """
    def __init__(self, height=260, width=346, num_targets=5,
                 noise_density=0.005, event_rate=5000, seed=42):
        self.height = height
        self.width = width
        self.num_targets = num_targets
        self.noise_density = noise_density
        self.event_rate = event_rate  # 每个目标每秒最大事件数
        self.time = 0.0
        self.dt = 1/1000  # 1ms 时间步长（模拟事件相机高速采样）

        self.rng = np.random.default_rng(seed)
        self.drones = []
        for i in range(num_targets):
            drone = Drone(i, height, width, self.rng)
            self.drones.append(drone)

        # 背景纹理（静态场景，用于生成灰度帧）
        self._bg_texture = self._generate_background()

    def _generate_background(self):
        """生成天空背景纹理"""
        bg = np.zeros((self.height, self.width), dtype=np.uint8)
        # 天空渐变
        for y in range(self.height):
            val = int(180 - (y / self.height) * 80 + self.rng.integers(-5, 5))
            bg[y, :] = np.clip(val, 30, 220)
        # 添加一些云状纹理
        for _ in range(5):
            cx = int(self.rng.integers(0, self.width))
            cy = int(self.rng.integers(0, self.height // 2))
            r = int(self.rng.integers(20, 60))
            cv2.circle(bg, (cx, cy), r,
                      int(200 + self.rng.integers(-10, 10)), -1)
        # 高斯模糊使纹理自然
        bg = cv2.GaussianBlur(bg, (21, 21), 10)
        return bg

    def step(self):
        """产生一帧事件流，返回 (events, frame)"""
        self.time += self.dt
        events = []

        # 更新每架无人机并生成事件
        for drone in self.drones:
            drone.update(self.dt)

            # 根据无人机尺寸和速度计算事件数量
            # 高速运动产生更多事件（边缘变化剧烈）
            speed = np.linalg.norm(drone.vel)
            base_rate = self.event_rate * (drone.size / 8.0)  # 尺寸影响
            speed_factor = 1.0 + speed / 100.0  # 速度影响
            rate = base_rate * speed_factor
            num_events = self.rng.poisson(rate * self.dt)

            if num_events > 0:
                cx, cy = int(drone.pos[0]), int(drone.pos[1])
                r = max(1, drone.size // 2)

                # 无人机形状：十字形 + 中心圆点
                # 生成主要分布在十字形轮廓上的事件
                for _ in range(num_events):
                    # 70%事件在圆形主体，30%在十字臂
                    if self.rng.random() < 0.7:
                        angle = self.rng.uniform(0, 2 * np.pi)
                        radius = self.rng.uniform(0, r)
                        ex = cx + int(radius * np.cos(angle))
                        ey = cy + int(radius * np.sin(angle))
                    else:
                        # 十字臂
                        arm = self.rng.choice(['h', 'v'])
                        offset = self.rng.uniform(-r*1.8, r*1.8)
                        if arm == 'h':
                            ex = cx + int(offset)
                            ey = cy + int(self.rng.integers(-1, 2))
                        else:
                            ex = cx + int(self.rng.integers(-1, 2))
                            ey = cy + int(offset)

                    ex = np.clip(ex, 0, self.width - 1)
                    ey = np.clip(ey, 0, self.height - 1)

                    # 极性：运动方向决定
                    # 前方边缘为正（变亮），后方边缘为负（变暗）
                    if speed > 1:
                        motion_angle = np.arctan2(drone.vel[1], drone.vel[0])
                        event_angle = np.arctan2(ey - cy, ex - cx)
                        angle_diff = (event_angle - motion_angle + np.pi) % (2 * np.pi) - np.pi
                        pol = 1 if abs(angle_diff) < np.pi / 2 else -1
                    else:
                        pol = int(self.rng.choice([-1, 1]))

                    events.append((ex, ey, pol, self.time))

        # 背景噪声事件（模拟传感器热噪声）
        noise_count = int(self.noise_density * self.height * self.width)
        if noise_count > 0:
            xs = self.rng.integers(0, self.width, noise_count)
            ys = self.rng.integers(0, self.height, noise_count)
            pol = self.rng.choice([-1, 1], noise_count)
            ts = np.full(noise_count, self.time)
            for i in range(noise_count):
                events.append((xs[i], ys[i], int(pol[i]), ts[i]))

        # 渲染灰度帧（用于可视化和光流辅助）
        frame = self._render_frame()
        return events, frame

    def _render_frame(self):
        """渲染当前场景为灰度图（模拟传统相机同步帧）"""
        frame = self._bg_texture.copy().astype(np.int16)

        # 添加高斯噪声模拟传感器噪声
        noise = self.rng.normal(0, 8, frame.shape).astype(np.int16)
        frame = frame + noise

        # 绘制每架无人机（暗色斑点——小型无人机在天空背景下为暗点）
        for drone in self.drones:
            cx, cy = int(drone.pos[0]), int(drone.pos[1])
            r = max(1, drone.size // 2)
            intensity = int(drone.intensity * 100)
            # 小型暗色目标
            cv2.circle(frame, (cx, cy), r, -intensity, -1)
            # 十字臂
            cv2.line(frame, (cx - r, cy), (cx + r, cy), -intensity, 1)
            cv2.line(frame, (cx, cy - r), (cx, cy + r), -intensity, 1)

        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def get_ground_truth(self):
        """返回当前所有无人机的位置和边界框（用于评估）"""
        boxes = []
        for drone in self.drones:
            r = max(1, drone.size // 2)
            x = int(drone.pos[0] - r * 1.8)
            y = int(drone.pos[1] - r * 1.8)
            w = int(r * 3.6)
            h = int(r * 3.6)
            boxes.append([x, y, w, h])
        return boxes
