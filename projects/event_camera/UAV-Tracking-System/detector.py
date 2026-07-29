import cv2
import numpy as np

class MotionDetector:
    """
    基于时间表面的运动小目标检测器。
    针对无人机等高速小目标优化：多尺度检测、Top-Hat增强、自适应阈值。
    """
    def __init__(self, min_area=3, max_area=200, flow_threshold=0.2, top_k=None):
        self.min_area = min_area    # 降低以适配2-5像素目标
        self.max_area = max_area
        self.flow_threshold = flow_threshold
        self.top_k = top_k          # 只保留最佳的 K 个检测（None=全部保留）
        self.prev_gray = None
        self.flow_cache = None      # 缓存整帧光流
        self.flow_frame_skip = 3    # 每N帧计算一次光流
        self.frame_count = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=36, detectShadows=False
        )

    def detect(self, surface, gray_frame=None):
        """
        输入: surface - (H,W) float32 [0,1] 时间表面
              gray_frame - (H,W) uint8 灰度图像 (可选)
        返回: boxes - list of [x,y,w,h]
        """
        self.frame_count += 1
        H, W = surface.shape
        surf_uint8 = (surface * 255).astype(np.uint8)

        # === 阶段1: 多尺度时间表面检测 ===
        all_boxes = []

        # 尺度1: 原始时间表面
        boxes1 = self._detect_from_surface(surf_uint8, "original")
        all_boxes.extend(boxes1)

        # 尺度2: Top-Hat增强（突出小亮点）
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(surf_uint8, cv2.MORPH_TOPHAT, kernel_small)
        boxes2 = self._detect_from_surface(tophat, "tophat")
        all_boxes.extend(boxes2)

        # 尺度3: 高斯差分（DoG，检测blob状小目标）
        g1 = cv2.GaussianBlur(surf_uint8, (0, 0), 1.0)
        g2 = cv2.GaussianBlur(surf_uint8, (0, 0), 3.0)
        dog = cv2.subtract(g1, g2)
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        boxes3 = self._detect_from_surface(dog, "dog")
        all_boxes.extend(boxes3)

        # === 阶段2: 去重与合并 ===
        if not all_boxes:
            return []

        boxes = self._merge_boxes(all_boxes)

        # === 阶段3: 运动一致性验证（仅每N帧计算一次光流） ===
        if gray_frame is not None and len(boxes) > 0:
            if self.frame_count % self.flow_frame_skip == 0:
                boxes = self._verify_by_motion_optimized(gray_frame, boxes)

        # === 阶段4: 只保留最佳 K 个检测（按面积+光流幅值打分） ===
        if self.top_k is not None and len(boxes) > self.top_k:
            boxes = self._select_top_k(boxes, gray_frame)

        return boxes

    def _detect_from_surface(self, image, source_name):
        """对单张表面图做检测"""
        boxes = []

        # 自适应阈值（Otsu）
        try:
            _, thresh = cv2.threshold(image, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except:
            return boxes

        # 如果Otsu阈值过低，使用固定阈值
        if np.count_nonzero(thresh) > image.size * 0.8:
            thresh_val = np.percentile(image, 90)
            _, thresh = cv2.threshold(image, thresh_val, 255, cv2.THRESH_BINARY)

        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 轮廓提取
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect < 4.0:
                    H, W = image.shape
                    if w < W * 0.15 and h < H * 0.15:
                        boxes.append([x, y, w, h])

        return boxes

    def _merge_boxes(self, boxes, iou_threshold=0.3):
        """合并重复检测框（非极大抑制）"""
        if len(boxes) <= 1:
            return boxes

        boxes_array = np.array(boxes)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 0] + boxes_array[:, 2]
        y2 = boxes_array[:, 1] + boxes_array[:, 3]

        areas = boxes_array[:, 2] * boxes_array[:, 3]
        order = areas.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou < iou_threshold]

        return [boxes[i] for i in keep]

    def _select_top_k(self, boxes, gray_frame=None):
        """按质量打分，只保留最佳的 K 个检测框。
        打分依据：面积合理性 + 局部对比度 + 光流幅值
        """
        if len(boxes) <= self.top_k:
            return boxes

        scores = []
        for box in boxes:
            x, y, w, h = box
            # 面积分：越接近 (min_area+max_area)/2 越好
            area = w * h
            ideal_area = (self.min_area + self.max_area) / 2
            area_score = 1.0 - abs(area - ideal_area) / max(ideal_area, 1)

            # 对比度分：框内像素标准差越大越好（目标 vs 背景）
            contrast_score = 0.0
            if gray_frame is not None and self.flow_cache is not None:
                H, W = gray_frame.shape[:2]
                x2 = min(x + w, W)
                y2 = min(y + h, H)
                if x2 > x and y2 > y:
                    roi = gray_frame[y:y2, x:x2]
                    if roi.size > 0:
                        std = float(np.std(roi.astype(np.float32)))
                        contrast_score = min(std / 30.0, 1.0)

            # 光流分：框内平均运动幅值越大越好
            flow_score = 0.0
            if self.flow_cache is not None:
                H, W = self.flow_cache.shape[:2]
                x2 = min(x + w, W)
                y2 = min(y + h, H)
                if x2 > x and y2 > y:
                    roi_flow = self.flow_cache[y:y2, x:x2]
                    mag = np.sqrt(roi_flow[..., 0]**2 + roi_flow[..., 1]**2)
                    mean_mag = float(np.mean(mag)) if mag.size > 0 else 0.0
                    flow_score = min(mean_mag / 3.0, 1.0)

            total = area_score * 0.3 + contrast_score * 0.3 + flow_score * 0.4
            scores.append(total)

        # 按分数排序，取 top_k
        ranked = sorted(zip(scores, boxes), key=lambda x: x[0], reverse=True)
        return [b for _, b in ranked[:self.top_k]]

    def _verify_by_motion_optimized(self, gray_frame, boxes):
        """使用光流验证运动一致性（优化版：光流仅计算一次）"""
        gray = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY) \
               if len(gray_frame.shape) == 3 else gray_frame

        # 仅当有前一帧时计算光流
        if self.prev_gray is None:
            self.prev_gray = gray
            return boxes

        # 只计算一次整帧光流
        self.flow_cache = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.prev_gray = gray

        verified = []
        for box in boxes:
            x, y, w, h = box
            if x >= self.flow_cache.shape[1] or y >= self.flow_cache.shape[0]:
                verified.append(box)
                continue
            x2 = min(x + w, self.flow_cache.shape[1])
            y2 = min(y + h, self.flow_cache.shape[0])
            if x2 <= x or y2 <= y:
                verified.append(box)
                continue

            roi_flow = self.flow_cache[y:y2, x:x2]
            mag = np.sqrt(roi_flow[..., 0]**2 + roi_flow[..., 1]**2)
            mean_mag = float(np.mean(mag)) if mag.size > 0 else 0.0

            if mean_mag > self.flow_threshold:
                verified.append(box)
            else:
                # 也检查局部对比度（可能是悬停目标）
                roi_gray = gray[y:y2, x:x2]
                std_val = float(np.std(roi_gray.astype(np.float32))) if roi_gray.size > 0 else 0.0
                if std_val > 10:
                    verified.append(box)

        return verified if verified else boxes
