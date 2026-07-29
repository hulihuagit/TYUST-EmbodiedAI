"""
基于时间表面的高速运动小目标（无人机）检测与追踪系统

参考：FRED (Florence RGB-Event Drone Dataset)
原理：利用时间表面（Time Surface）将事件相机的稀疏事件流转换为密集图像表示，
      结合多尺度检测、光流验证、KCF跟踪器和卡尔曼滤波，实现多无人机实时跟踪。

支持两种数据模式：
  1. 模拟模式（默认）：使用 EventSimulator 生成合成无人机事件数据
  2. 真实数据模式：从 data/ 目录读取真实事件相机数据

输出：桌面 drone_tracking_output/ 目录
  - tracking_dashboard.mp4  : 2×2 仪表盘视频
  - tracking_dashboard_tracking_only.mp4 : 纯跟踪视角视频
  - 使用说明.txt             : 运行说明
  - 算法原理.txt             : 算法原理文档
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from time_surface import TimeSurface
from detector import MotionDetector
from tracker import MultiObjectTracker
from event_simulator import EventSimulator
from video_writer import VideoWriter


def run_simulation(args):
    """使用模拟数据运行（原有模式）"""
    HEIGHT, WIDTH = 260, 346
    NUM_DRONES = args.num_drones
    TOTAL_FRAMES = args.frames
    NOISE_DENSITY = 0.001
    EVENT_RATE = 5000
    TAU = 20e-3
    MIN_AREA = 5
    MAX_AREA = 300
    FLOW_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.15
    MAX_LOST = 8
    MIN_HITS = 3

    DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
    OUTPUT_DIR = os.path.join(DESKTOP, "drone_tracking_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  基于时间表面的高速运动小目标（无人机）检测与追踪系统")
    print("  Time-Surface-Based High-Speed Drone Detection & Tracking")
    print("=" * 60)
    print()
    print(f"  [数据模式] 模拟数据")
    print(f"  无人机数量: {NUM_DRONES}")
    print(f"  分辨率: {WIDTH}x{HEIGHT}")
    print(f"  总帧数: {TOTAL_FRAMES}")
    print(f"  时间表面 tau: {TAU*1000:.0f}ms")
    print(f"  输出目录: {OUTPUT_DIR}")
    print()

    sim = EventSimulator(
        height=HEIGHT, width=WIDTH,
        num_targets=NUM_DRONES,
        noise_density=NOISE_DENSITY,
        event_rate=EVENT_RATE,
        seed=42
    )

    ts = TimeSurface(height=HEIGHT, width=WIDTH, tau=TAU)
    detector = MotionDetector(
        min_area=MIN_AREA, max_area=MAX_AREA, flow_threshold=FLOW_THRESHOLD)
    detector.flow_frame_skip = 1
    multi_tracker = MultiObjectTracker(
        iou_threshold=IOU_THRESHOLD, max_lost=MAX_LOST, min_hits=MIN_HITS)

    dashboard_path = os.path.join(OUTPUT_DIR, "tracking_dashboard.mp4")
    writer = VideoWriter(dashboard_path, fps=30, panel_size=(320, 420))

    print("开始模拟与跟踪...")
    print()

    for frame_idx in range(TOTAL_FRAMES):
        events, raw_frame = sim.step()
        current_time = sim.time
        surface = ts.update(events, current_time)
        det_boxes = detector.detect(surface, raw_frame)
        frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
        active_ids, track_boxes, trails, velocities = multi_tracker.update(
            frame_bgr, det_boxes)
        writer.write_frame(
            surface, raw_frame, det_boxes, track_boxes,
            active_ids, trails, velocities)

        if (frame_idx + 1) % 200 == 0:
            pct = (frame_idx + 1) / TOTAL_FRAMES * 100
            n_tracks = len(active_ids)
            print(f"  进度: {frame_idx+1}/{TOTAL_FRAMES} ({pct:.0f}%) | "
                  f"活跃跟踪: {n_tracks} | 模拟时间: {sim.time:.2f}s")

    writer.close()

    print()
    print("=" * 60)
    print("  处理完成！")
    print("=" * 60)
    print()
    print(f"  [仪表盘视频] {dashboard_path}")
    track_only = dashboard_path.replace('.mp4', '_tracking_only.mp4')
    print(f"  [跟踪视频]   {track_only}")
    print(f"  所有文件已保存到桌面: {OUTPUT_DIR}")


def run_real_data(args):
    """使用真实数据集运行"""
    from data_loader import RealDataLoader

    # ---- 参数配置 ----
    DATA_DIR = args.data_dir
    if args.resolution:
        parts = args.resolution.split('x')
        TARGET_W, TARGET_H = int(parts[0]), int(parts[1])
    else:
        TARGET_W, TARGET_H = 640, 360  # 半分辨率（原始 1280x720）

    TOTAL_FRAMES = args.frames if args.frames > 0 else 0  # 0 = 全部
    TAU = args.tau_ms / 1000.0  # 默认 20ms
    MIN_AREA = args.min_area
    MAX_AREA = args.max_area
    FLOW_THRESHOLD = args.flow_threshold
    IOU_THRESHOLD = args.iou_threshold
    MAX_LOST = args.max_lost
    MIN_HITS = args.min_hits
    TOP_K = args.top_k  # 每帧最多保留的检测数（单无人机=1）

    DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
    OUTPUT_DIR = os.path.join(DESKTOP, "drone_tracking_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  基于时间表面的高速运动小目标（无人机）检测与追踪系统")
    print("  Time-Surface-Based High-Speed Drone Detection & Tracking")
    print("=" * 60)
    print()
    print(f"  [数据模式] 真实事件相机数据")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  目标分辨率: {TARGET_W}x{TARGET_H}")
    print(f"  时间表面 tau: {TAU*1000:.0f}ms")
    print(f"  检测面积范围: {MIN_AREA}-{MAX_AREA} px²")
    print(f"  每帧保留检测数: {TOP_K}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print()

    # ---- 初始化数据加载器 ----
    loader = RealDataLoader(DATA_DIR, target_size=(TARGET_W, TARGET_H))
    HEIGHT, WIDTH = TARGET_H, TARGET_W

    if TOTAL_FRAMES == 0 or TOTAL_FRAMES > len(loader):
        TOTAL_FRAMES = len(loader)

    print(f"  实际处理帧数: {TOTAL_FRAMES}")
    print()

    # ---- 初始化处理模块 ----
    ts = TimeSurface(height=HEIGHT, width=WIDTH, tau=TAU)
    detector = MotionDetector(
        min_area=MIN_AREA, max_area=MAX_AREA, flow_threshold=FLOW_THRESHOLD,
        top_k=TOP_K)
    detector.flow_frame_skip = 1
    multi_tracker = MultiObjectTracker(
        iou_threshold=IOU_THRESHOLD, max_lost=MAX_LOST, min_hits=MIN_HITS)

    # 计算面板大小（保持宽高比）
    panel_h = 420
    panel_w = int(panel_h * WIDTH / HEIGHT)
    dashboard_path = os.path.join(OUTPUT_DIR, "tracking_dashboard.mp4")
    writer = VideoWriter(dashboard_path, fps=30, panel_size=(panel_w, panel_h))

    # ---- 主循环 ----
    print("开始处理真实数据...")
    print()

    for frame_idx in range(TOTAL_FRAMES):
        # 1. 读取事件数据和 RGB 帧
        events, rgb_frame = loader.step()
        current_time = loader.time

        if frame_idx == 0 and not events:
            print("  [警告] 第一帧无事件数据（差分需要至少2帧），跳过...")
            continue

        # 2. 更新时间表面
        surface = ts.update(events, current_time)

        # 3. 使用 RGB 帧或时间表面作为原始帧
        if rgb_frame is not None:
            raw_frame = rgb_frame
        else:
            raw_frame = (surface * 255).astype(np.uint8)

        # 4. 运动目标检测
        det_boxes = detector.detect(surface, raw_frame)

        # 5. 多目标跟踪
        frame_bgr = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
        active_ids, track_boxes, trails, velocities = multi_tracker.update(
            frame_bgr, det_boxes)

        # 5.5 单目标模式：只保留最持久（hit_streak 最高）的那条跟踪
        if TOP_K == 1 and len(active_ids) > 1:
            best_id = max(active_ids,
                          key=lambda tid: multi_tracker.hit_streak.get(tid, 0))
            active_ids = [best_id]

        # 6. 获取地面真值（用于可视化验证）
        gt_boxes = loader.get_ground_truth(frame_idx)

        # 7. 写入视频帧
        writer.write_frame(
            surface, raw_frame, det_boxes, track_boxes,
            active_ids, trails, velocities)

        # 进度显示
        if (frame_idx + 1) % 100 == 0:
            pct = (frame_idx + 1) / TOTAL_FRAMES * 100
            n_tracks = len(active_ids)
            n_events = len(events)
            n_dets = len(det_boxes)
            print(f"  进度: {frame_idx+1}/{TOTAL_FRAMES} ({pct:.0f}%) | "
                  f"事件: {n_events} | 检测: {n_dets} | 跟踪: {n_tracks}")

    writer.close()

    # ---- 输出结果 ----
    print()
    print("=" * 60)
    print("  处理完成！")
    print("=" * 60)
    print()
    print(f"  [仪表盘视频] {dashboard_path}")
    track_only = dashboard_path.replace('.mp4', '_tracking_only.mp4')
    print(f"  [跟踪视频]   {track_only}")
    print(f"  所有文件已保存到桌面: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description='基于时间表面的高速运动小目标检测与追踪系统')

    # 数据模式
    parser.add_argument('--mode', type=str, default='real',
                        choices=['sim', 'real'],
                        help='数据模式: sim=模拟数据, real=真实数据 (默认: real)')
    parser.add_argument('--data-dir', type=str,
                        default='D:\\PythonProject3\\data',
                        help='真实数据集目录 (默认: data/)')

    # 模拟参数
    parser.add_argument('--num-drones', type=int, default=2,
                        help='模拟模式下的无人机数量 (默认: 2)')

    # 通用参数
    parser.add_argument('--frames', type=int, default=0,
                        help='处理帧数 (0=全部, 默认: 0)')
    parser.add_argument('--resolution', type=str, default=None,
                        help='目标分辨率 WxH (默认: 640x360)')
    parser.add_argument('--tau-ms', type=float, default=20.0,
                        help='时间表面衰减常数/毫秒 (默认: 20)')
    parser.add_argument('--min-area', type=int, default=10,
                        help='最小检测面积/px² (默认: 10)')
    parser.add_argument('--max-area', type=int, default=300,
                        help='最大检测面积/px² (默认: 300)')
    parser.add_argument('--flow-threshold', type=float, default=0.5,
                        help='光流验证阈值 (默认: 0.5)')
    parser.add_argument('--iou-threshold', type=float, default=0.1,
                        help='跟踪IoU阈值 (默认: 0.1)')
    parser.add_argument('--max-lost', type=int, default=8,
                        help='目标最大丢失帧数 (默认: 8)')
    parser.add_argument('--min-hits', type=int, default=5,
                        help='最少命中次数确认跟踪 (默认: 5)')
    parser.add_argument('--top-k', type=int, default=1,
                        help='每帧最多保留的检测数，单目标=1 (默认: 1)')

    args = parser.parse_args()

    if args.mode == 'real':
        run_real_data(args)
    else:
        run_simulation(args)


if __name__ == "__main__":
    main()
