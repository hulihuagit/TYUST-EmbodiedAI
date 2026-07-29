import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    """
    基于卡尔曼滤波的边界框状态估计器。
    状态: [x, y, w, h, vx, vy] (中心位置+尺寸+速度)
    """
    def __init__(self, bbox):
        self.kf = cv2.KalmanFilter(6, 4)  # 6状态, 4测量
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        x, y, w, h = bbox
        self.kf.statePost = np.array([
            [x + w/2], [y + h/2], [w], [h], [0], [0]
        ], dtype=np.float32)

        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0

    def predict(self):
        """预测下一帧状态，返回预测边界框"""
        pred = self.kf.predict()
        cx, cy, w, h = pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0]
        self.age += 1
        self.time_since_update += 1
        return [int(cx - w/2), int(cy - h/2), max(1, int(w)), max(1, int(h))]

    def update(self, bbox):
        """用检测结果更新卡尔曼滤波器"""
        x, y, w, h = bbox
        measurement = np.array([
            [x + w/2], [y + h/2], [w], [h]
        ], dtype=np.float32)
        self.kf.correct(measurement)
        self.time_since_update = 0
        self.hit_streak = min(self.hit_streak + 1, 100)

    def get_state(self):
        """获取当前边界框 [x, y, w, h]"""
        s = self.kf.statePost
        cx, cy, w, h = s[0, 0], s[1, 0], max(1, s[2, 0]), max(1, s[3, 0])
        return [int(cx - w/2), int(cy - h/2), int(w), int(h)]

    def get_velocity(self):
        """获取当前估计速度"""
        return (self.kf.statePost[4, 0], self.kf.statePost[5, 0])


class KCFTracker:
    """
    核化相关滤波器跟踪器（KCF），使用NumPy FFT实现。
    针对小目标优化：更小的padding、自适应sigma。
    """
    def __init__(self, padding=1.5, sigma=0.5, lambda_=1e-4, interp_factor=0.1):
        self.padding = padding      # 小目标用小padding减少背景干扰
        self.sigma = sigma          # 高斯核带宽——小目标用较小sigma
        self.lambda_ = lambda_
        self.interp_factor = interp_factor

        self._alphaf = None   # 频域滤波器
        self._xf = None       # 频域模板
        self._cx = 0
        self._cy = 0
        self._w = 0
        self._h = 0
        self._window_sz = None
        self._target_sz = None
        self._cos_window = None

    # ---------- 公共接口 ----------
    def init(self, image, bbox):
        """初始化跟踪器"""
        gray = self._to_gray(image)
        self._init_features(gray, bbox)
        self._train()

    def update(self, image):
        """更新跟踪位置，返回 (bbox, confidence)"""
        gray = self._to_gray(image)

        # 1. 提取当前帧特征
        z = self._get_features(gray)

        # 2. 频域相关响应
        Z = np.fft.fft2(z, axes=(0, 1))
        X = self._xf
        kzf = self._gaussian_kernel(Z, X)

        # 3. 计算响应图
        response_f = self._alphaf * kzf
        response = np.real(np.fft.ifft2(response_f, axes=(0, 1)))

        # 4. 找峰值
        cy_idx, cx_idx = np.unravel_index(np.argmax(response), response.shape[:2])
        peak_value = float(response[cy_idx, cx_idx])

        # 5. 更新中心位置
        h, w = response.shape[:2]
        dy = cy_idx - h // 2
        dx = cx_idx - w // 2
        self._cx = int(np.clip(self._cx + dx, 0, gray.shape[1] - 1))
        self._cy = int(np.clip(self._cy + dy, 0, gray.shape[0] - 1))

        # 6. 在线更新模板
        new_x = self._get_features(gray)
        self._xf = (1 - self.interp_factor) * self._xf + self.interp_factor * new_x
        self._train()

        bbox = [int(self._cx - self._w // 2),
                int(self._cy - self._h // 2),
                int(self._w), int(self._h)]
        return bbox, peak_value

    # ---------- 内部方法 ----------
    def _to_gray(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    def _init_features(self, gray, bbox):
        x, y, w, h = bbox
        self._cx = x + w // 2
        self._cy = y + h // 2
        self._w = max(w, 3)   # 小目标保证至少3像素
        self._h = max(h, 3)
        self._target_sz = np.array([self._h, self._w])
        self._window_sz = np.floor(self._target_sz * (1 + self.padding)).astype(int)
        self._window_sz = np.maximum(self._window_sz, [5, 5])  # 最小窗口
        self._cos_window = np.outer(
            np.hanning(self._window_sz[0]),
            np.hanning(self._window_sz[1])
        )
        self._xf = self._get_features(gray)

    def _get_features(self, gray):
        """提取多通道特征: 灰度 + 梯度幅值 + LoG"""
        patch = cv2.getRectSubPix(gray, tuple(self._window_sz[::-1]),
                                   (float(self._cx), float(self._cy))).astype(np.float32)

        # 梯度幅值
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)

        # LoG (高斯拉普拉斯) - 增强小斑点
        blur = cv2.GaussianBlur(patch, (0, 0), 1.0)
        log = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        log = np.abs(log)

        feat = np.stack([patch / 255.0, mag / 255.0, log / 255.0], axis=-1)

        for c in range(feat.shape[-1]):
            feat[:, :, c] *= self._cos_window
        return feat.astype(np.float32)

    def _train(self):
        X = np.fft.fft2(self._xf, axes=(0, 1))
        kf = self._gaussian_kernel(X, X)
        target = self._create_gaussian_target()
        Y = np.fft.fft2(target)
        self._alphaf = Y / (kf + self.lambda_)

    def _gaussian_kernel(self, X, Y):
        """高斯核相关"""
        n = max(X.shape[0] * X.shape[1], 1)
        xx = np.sum(np.conj(X) * X, axis=-1)
        yy = np.sum(np.conj(Y) * Y, axis=-1)
        xy = np.sum(np.conj(X) * Y, axis=-1)
        xy_real = np.real(np.fft.ifft2(xy, axes=(0, 1)))
        norm = (xx.real + yy.real - 2 * xy_real) / n
        return np.exp(-norm / max(self.sigma ** 2, 1e-6))

    def _create_gaussian_target(self):
        sz = self._window_sz
        ys, xs = np.meshgrid(np.arange(sz[0]), np.arange(sz[1]), indexing='ij')
        cy = sz[0] // 2
        cx = sz[1] // 2
        sigma = np.sqrt(float(self._target_sz[0]) * float(self._target_sz[1])) * 0.125
        return np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / max(2 * sigma ** 2, 1e-6)).astype(np.float32)


class MultiObjectTracker:
    """
    多目标跟踪管理器，维护多个KCF跟踪器 + 卡尔曼滤波器。
    使用匈牙利算法进行检测-跟踪关联。
    """
    def __init__(self, iou_threshold=0.15, max_lost=10, min_hits=2):
        self.kcf_trackers = {}      # tid -> KCFTracker
        self.kalman_trackers = {}   # tid -> KalmanBoxTracker
        self.boxes = {}             # tid -> [x,y,w,h]
        self.trails = {}            # tid -> list of (cx,cy)
        self.velocities = {}        # tid -> (vx,vy)
        self.lost_count = {}        # tid -> int
        self.hit_streak = {}        # tid -> int
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits    # 最少命中次数才确认输出

    def update(self, image, detections):
        """
        更新所有跟踪器。
        返回: (active_ids, track_boxes, trails)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
               if len(image.shape) == 3 else image

        # === 步骤1: 卡尔曼预测 ===
        predicted_boxes = {}
        for tid in list(self.kcf_trackers.keys()):
            if tid in self.kalman_trackers:
                predicted_boxes[tid] = self.kalman_trackers[tid].predict()
            else:
                predicted_boxes[tid] = self.boxes.get(tid, [0, 0, 1, 1])

        # === 步骤2: KCF更新 ===
        kcf_boxes = {}
        for tid in list(self.kcf_trackers.keys()):
            bbox, conf = self.kcf_trackers[tid].update(gray)
            kcf_boxes[tid] = (bbox, conf)
            self.lost_count[tid] = self.lost_count.get(tid, 0) + 1

        # === 步骤3: 匈牙利算法数据关联 ===
        matches, unmatched_dets, unmatched_tracks = self._associate(
            detections, predicted_boxes)

        # === 步骤4: 更新匹配的跟踪 ===
        for det_idx, tid in matches:
            det_box = detections[det_idx]
            # 卡尔曼校正
            if tid in self.kalman_trackers:
                self.kalman_trackers[tid].update(det_box)
            # KCF重新初始化
            self.kcf_trackers[tid].init(gray, det_box)
            self.boxes[tid] = det_box
            self.lost_count[tid] = 0
            self.hit_streak[tid] = self.hit_streak.get(tid, 0) + 1
            # 更新轨迹
            cx = det_box[0] + det_box[2] // 2
            cy = det_box[1] + det_box[3] // 2
            self.trails.setdefault(tid, []).append((cx, cy))
            if len(self.trails[tid]) > 150:
                self.trails[tid] = self.trails[tid][-150:]
            # 更新速度
            if tid in self.kalman_trackers:
                self.velocities[tid] = self.kalman_trackers[tid].get_velocity()

        # === 步骤5: 处理未匹配的跟踪（使用卡尔曼预测） ===
        for tid in unmatched_tracks:
            self.boxes[tid] = predicted_boxes[tid]
            # 更新轨迹
            bbox = predicted_boxes[tid]
            cx = bbox[0] + bbox[2] // 2
            cy = bbox[1] + bbox[3] // 2
            self.trails.setdefault(tid, []).append((cx, cy))
            if len(self.trails[tid]) > 150:
                self.trails[tid] = self.trails[tid][-150:]

        # === 步骤6: 新建跟踪器（未匹配的检测） ===
        for det_idx in unmatched_dets:
            det_box = detections[det_idx]
            tid = self.next_id
            self.next_id += 1

            kcf = KCFTracker(padding=1.5, sigma=0.5)
            kcf.init(gray, det_box)
            self.kcf_trackers[tid] = kcf

            kf = KalmanBoxTracker(det_box)
            self.kalman_trackers[tid] = kf

            self.boxes[tid] = det_box
            self.trails[tid] = [(det_box[0] + det_box[2] // 2,
                                 det_box[1] + det_box[3] // 2)]
            self.lost_count[tid] = 0
            self.hit_streak[tid] = 1

        # === 步骤7: 清理丢失目标 ===
        lost_ids = [tid for tid, cnt in self.lost_count.items()
                    if cnt > self.max_lost]
        for tid in lost_ids:
            for d in [self.kcf_trackers, self.kalman_trackers, self.boxes,
                      self.trails, self.lost_count, self.hit_streak, self.velocities]:
                d.pop(tid, None)

        # === 步骤8: 返回已确认的活跃跟踪 ===
        active_ids = [tid for tid in self.boxes.keys()
                      if self.hit_streak.get(tid, 0) >= self.min_hits]
        return active_ids, self.boxes, self.trails, self.velocities

    def _associate(self, detections, predicted_boxes):
        """
        使用匈牙利算法关联检测和跟踪。
        返回: (matches, unmatched_detections, unmatched_tracks)
        """
        if not detections:
            return [], list(range(len(detections))), list(predicted_boxes.keys())

        if not predicted_boxes:
            return [], list(range(len(detections))), []

        track_ids = list(predicted_boxes.keys())
        num_dets = len(detections)
        num_tracks = len(track_ids)

        # 构建成本矩阵（1 - IoU）
        cost_matrix = np.ones((num_dets, num_tracks), dtype=np.float32)

        for i, det in enumerate(detections):
            for j, tid in enumerate(track_ids):
                pred_box = predicted_boxes[tid]
                iou = self._iou(det, pred_box)
                cost_matrix[i, j] = 1.0 - iou

        # 匈牙利算法求解
        det_indices, track_indices = linear_sum_assignment(cost_matrix)

        matches = []
        used_dets = set()
        used_tracks = set()

        for d, t in zip(det_indices, track_indices):
            if cost_matrix[d, t] < (1.0 - self.iou_threshold):
                matches.append((d, track_ids[t]))
                used_dets.add(d)
                used_tracks.add(track_ids[t])

        unmatched_dets = [d for d in range(num_dets) if d not in used_dets]
        unmatched_tracks = [tid for tid in track_ids if tid not in used_tracks]

        return matches, unmatched_dets, unmatched_tracks

    @staticmethod
    def _iou(a, b):
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = a
        x2, y2, w2, h2 = b
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        return inter / max(union, 1)
