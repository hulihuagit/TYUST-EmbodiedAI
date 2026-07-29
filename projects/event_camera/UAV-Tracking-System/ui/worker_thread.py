"""
后台处理线程 —— 在 QThread 中运行事件相机追踪管道，
通过信号向前端报告进度和结果。
"""
import sys
import os
import traceback

# Ensure parent directory is importable (dev + frozen)
if getattr(sys, 'frozen', False):
    BASE = sys._MEIPASS
else:
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from PySide6.QtCore import QThread, Signal
import cv2
import numpy as np

# Top-level imports for PyInstaller static analysis
from data_loader import RealDataLoader
from time_surface import TimeSurface
from detector import MotionDetector
from tracker import MultiObjectTracker
from video_writer import VideoWriter


class ProcessWorker(QThread):
    """后台管道处理线程"""
    progress = Signal(int, int, dict)   # frame_idx, total, {events, detections, tracks}
    finished = Signal(str, dict)        # output_path, metrics
    error = Signal(str)                 # error message
    preview_ready = Signal(object, object, list, dict, list, dict)
    # surface, raw_frame, det_boxes, track_boxes_dict, track_ids, trails

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            p = self.params

            # --- Init loader ---
            data_dir = p.get('data_dir', 'data')
            res_w = p.get('res_w', 640)
            res_h = p.get('res_h', 360)
            loader = RealDataLoader(data_dir, target_size=(res_w, res_h))

            total_frames = p.get('total_frames', 0)
            if total_frames == 0 or total_frames > len(loader):
                total_frames = len(loader)

            # --- Init modules ---
            tau = p.get('tau_ms', 20) / 1000.0
            ts = TimeSurface(height=res_h, width=res_w, tau=tau)

            detector = MotionDetector(
                min_area=p.get('min_area', 10),
                max_area=p.get('max_area', 300),
                flow_threshold=p.get('flow_threshold', 0.5),
                top_k=p.get('top_k', 1),
            )
            detector.flow_frame_skip = 1

            tracker = MultiObjectTracker(
                iou_threshold=p.get('iou_threshold', 0.1),
                max_lost=p.get('max_lost', 8),
                min_hits=p.get('min_hits', 5),
            )

            # --- Init video writer ---
            output_dir = p.get('output_dir', '')
            if not output_dir:
                output_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'drone_tracking_output')
            os.makedirs(output_dir, exist_ok=True)
            dashboard_path = os.path.join(output_dir, 'tracking_dashboard.mp4')
            panel_h = 420
            panel_w = int(panel_h * res_w / max(res_h, 1))
            writer = VideoWriter(dashboard_path, fps=30, panel_size=(panel_w, panel_h))

            # --- Track cumulative metrics ---
            total_detections = 0
            total_tracks = 0
            frames_with_track = 0
            frames_with_detection = 0

            # --- Main loop ---
            for frame_idx in range(total_frames):
                if self._abort:
                    break

                events, rgb_frame = loader.step()
                current_time = loader.time

                if frame_idx == 0 and not events:
                    continue

                surface = ts.update(events, current_time)

                if rgb_frame is not None:
                    raw_frame = rgb_frame
                else:
                    raw_frame = (surface * 255).astype(np.uint8)

                det_boxes = detector.detect(surface, raw_frame)
                frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
                active_ids, track_boxes, trails, velocities = tracker.update(
                    frame_bgr, det_boxes)

                # Single-target enforcement
                if p.get('top_k', 1) == 1 and len(active_ids) > 1:
                    best_id = max(active_ids,
                                  key=lambda tid: tracker.hit_streak.get(tid, 0))
                    active_ids = [best_id]

                writer.write_frame(
                    surface, raw_frame, det_boxes, track_boxes,
                    active_ids, trails, velocities)

                # Accumulate stats
                total_detections += len(det_boxes)
                total_tracks += len(active_ids)
                if len(det_boxes) > 0:
                    frames_with_detection += 1
                if len(active_ids) > 0:
                    frames_with_track += 1

                # Emit progress every 30 frames
                if frame_idx % 30 == 0:
                    stats = {
                        'events': len(events),
                        'detections': len(det_boxes),
                        'tracks': len(active_ids),
                    }
                    self.progress.emit(frame_idx, total_frames, stats)

                # Emit preview every 100 frames
                if frame_idx % 100 == 0:
                    self.preview_ready.emit(
                        surface, raw_frame, det_boxes,
                        track_boxes, active_ids, trails
                    )

            writer.close()

            # --- Compute final metrics ---
            coverage = (frames_with_track / max(total_frames, 1)) * 100
            avg_detections = total_detections / max(total_frames, 1)
            metrics = {
                'total_frames': total_frames,
                'frames_with_detection': frames_with_detection,
                'frames_with_track': frames_with_track,
                'coverage_pct': round(coverage, 1),
                'avg_detections': round(avg_detections, 2),
                'total_events_processed': 'see log',
            }
            self.finished.emit(dashboard_path, metrics)

        except Exception as e:
            self.error.emit(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
