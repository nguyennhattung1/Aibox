"""
rtsp_pipeline_demo.py
======================
Task 1.6 – Demo với 1 camera thực tế & Đánh giá Pipeline Phase 1
- Input: RTSP stream từ camera thật
- Detection & Tracking: YOLOv8n/s + ByteTrack
- Real-time Heatmap: Density matrix accumulation + Gaussian blur + JET/INFERNO colormap
- Benchmark & Metrics: Real-time FPS, Inference time breakdown, Active & Total Tracked IDs
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG DEFAULT
# ─────────────────────────────────────────────
DEFAULT_RTSP  = "rtsp://admin:L2E7132F@192.168.1.21:554/cam/realmonitor?channel=1&subtype=1"
MODEL_NAME    = "yolov8n.pt"
PERSON_CLASS  = 0
CONF_THRESH   = 0.35
IOU_THRESH    = 0.45
HEATMAP_ALPHA = 0.55
SIGMA         = 10
COLORMAP      = cv2.COLORMAP_JET
SPLASH_RADIUS = 3
MIN_THRESHOLD = 15


def run_rtsp_demo(
    rtsp_url: str,
    output_dir: str,
    duration: float = 30.0,
    model_path: str = MODEL_NAME,
    conf_thresh: float = CONF_THRESH,
    display: bool = False,
    save_video: bool = True
):
    # Enable TCP transport for OpenCV RTSP stream stability
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    out_video_path = output_dir / f"rtsp_phase1_demo_{timestamp_str}.mp4"
    out_heatmap_path = output_dir / f"rtsp_phase1_heatmap_{timestamp_str}.png"
    out_summary_path = output_dir / f"rtsp_phase1_summary_{timestamp_str}.png"

    print("=" * 65)
    print("  PHASE 1 POC – RTSP CAMERA DEMO & PIPELINE BENCHMARK")
    print("=" * 65)
    print(f"  RTSP Stream : {rtsp_url.split('@')[-1] if '@' in rtsp_url else rtsp_url}")
    print(f"  Model       : {model_path} (Confidence threshold: {conf_thresh})")
    print(f"  Tracker     : ByteTrack")
    print(f"  Target Run  : {duration} seconds")
    print(f"  Output Dir  : {output_dir}")
    print("=" * 65)

    # 1. Initialize YOLO Model
    print("\n[1/3] Loading YOLO model...")
    model = YOLO(model_path)

    # 2. Connect RTSP Stream
    print("[2/3] Connecting to RTSP stream...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("[ERROR] Could not open RTSP stream. Check camera IP/Credentials/Network.")
        return None

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"  RTSP Stream Connected: {W}x{H} @ {reported_fps:.1f} FPS (reported)")

    # Video Writer
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, reported_fps, (W, H))

    # Density Matrix
    density = np.zeros((H, W), dtype=np.float32)

    # Tracker colors
    rng = np.random.default_rng(42)
    id_colors = {}

    def get_color(tid: int):
        if tid not in id_colors:
            id_colors[tid] = tuple(rng.integers(80, 255, size=3).tolist())
        return id_colors[tid]

    # Metrics
    frame_idx = 0
    all_tracked_ids = set()
    total_detections = 0
    
    t_read_list = []
    t_infer_list = []
    t_heatmap_list = []
    t_total_list = []

    print("\n[3/3] Running Pipeline Phase 1...")
    start_time = time.time()
    last_frame = None

    try:
        while True:
            t0 = time.time()

            # Read frame
            ret, frame = cap.read()
            t1 = time.time()
            if not ret:
                print("\n[WARN] Failed to read frame from RTSP stream.")
                break

            last_frame = frame.copy()
            frame_idx += 1
            t_read = (t1 - t0) * 1000  # ms

            # Model tracking
            results = model.track(
                frame,
                persist=True,
                classes=[PERSON_CLASS],
                conf=conf_thresh,
                iou=IOU_THRESH,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            t2 = time.time()
            t_infer = (t2 - t1) * 1000  # ms

            annotated = frame.copy()
            boxes = results[0].boxes

            current_people = 0
            if boxes is not None and boxes.id is not None:
                box_coords = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().numpy()
                confidences = boxes.conf.cpu().numpy()

                current_people = len(track_ids)
                total_detections += current_people

                for box, tid, conf in zip(box_coords, track_ids, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    tid = int(tid)
                    all_tracked_ids.add(tid)

                    # Foot-point extraction (X_center, Y_max)
                    fx = (x1 + x2) // 2
                    fy = min(y2, H - 1)

                    # Accumulate 2D density splash
                    yr0, yr1 = max(fy - SPLASH_RADIUS, 0), min(fy + SPLASH_RADIUS + 1, H)
                    xr0, xr1 = max(fx - SPLASH_RADIUS, 0), min(fx + SPLASH_RADIUS + 1, W)
                    density[yr0:yr1, xr0:xr1] += 1.0

                    # Draw BBox + ID + Confidence
                    color = get_color(tid)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{tid} ({conf:.2f})"
                    cv2.putText(annotated, label, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Draw Foot-point
                    cv2.circle(annotated, (fx, fy), 4, (0, 0, 255), -1)

            # Build Live Heatmap Overlay
            smooth = gaussian_filter(density, sigma=SIGMA)
            if smooth.max() > 0:
                norm = (smooth / smooth.max() * 255).astype(np.uint8)
                heat_color = cv2.applyColorMap(norm, COLORMAP)
                mask = (norm > MIN_THRESHOLD).astype(np.float32)[..., np.newaxis]
                annotated = (annotated * (1 - mask * HEATMAP_ALPHA)
                             + heat_color * mask * HEATMAP_ALPHA).astype(np.uint8)

            t3 = time.time()
            t_heatmap = (t3 - t2) * 1000  # ms
            t_total = (t3 - t0) * 1000    # ms

            t_read_list.append(t_read)
            t_infer_list.append(t_infer)
            t_heatmap_list.append(t_heatmap)
            t_total_list.append(t_total)

            current_fps = 1000.0 / t_total if t_total > 0 else 0
            avg_fps = frame_idx / (time.time() - start_time)

            # Draw Realtime Metrics Dashboard on Frame
            cv2.rectangle(annotated, (10, 10), (340, 125), (0, 0, 0), -1)
            cv2.rectangle(annotated, (10, 10), (340, 125), (0, 255, 255), 1)

            cv2.putText(annotated, f"RTSP LIVE PIPELINE PHASE 1", (18, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(annotated, f"Realtime FPS : {current_fps:4.1f} (Avg: {avg_fps:4.1f})", (18, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated, f"Active People: {current_people} (Total IDs: {len(all_tracked_ids)})", (18, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, f"Infer Time   : {t_infer:4.1f} ms | Heatmap: {t_heatmap:4.1f} ms", (18, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(annotated, f"Frame Count  : {frame_idx}", (18, 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow("RTSP Phase 1 Pipeline Demo", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] Stopped by user.")
                    break

            elapsed = time.time() - start_time
            if duration > 0 and elapsed >= duration:
                print(f"\n[INFO] Target duration of {duration}s reached.")
                break

            if frame_idx % 25 == 0:
                print(f"\r  Running... Frame {frame_idx:4d} | Elapsed: {elapsed:4.1f}s | FPS: {avg_fps:4.1f} | Active People: {current_people} | Unique IDs: {len(all_tracked_ids)}", end="", flush=True)

    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    final_avg_fps = frame_idx / total_time if total_time > 0 else 0

    avg_read_ms = np.mean(t_read_list) if t_read_list else 0
    avg_infer_ms = np.mean(t_infer_list) if t_infer_list else 0
    avg_heat_ms = np.mean(t_heatmap_list) if t_heatmap_list else 0
    avg_total_ms = np.mean(t_total_list) if t_total_list else 0

    # Save Final Heatmap PNG
    smooth_final = gaussian_filter(density, sigma=SIGMA)
    if smooth_final.max() > 0:
        norm_final = (smooth_final / smooth_final.max() * 255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(norm_final, COLORMAP)
        cv2.imwrite(str(out_heatmap_path), heatmap_img)
        print(f"\n  Final Heatmap PNG saved → {out_heatmap_path}")

        # Blend on last frame for final summary preview image
        if last_frame is not None:
            mask_final = (norm_final > MIN_THRESHOLD).astype(np.float32)[..., np.newaxis]
            summary_img = (last_frame * (1 - mask_final * HEATMAP_ALPHA) + heatmap_img * mask_final * HEATMAP_ALPHA).astype(np.uint8)

            # Draw summary card
            cv2.rectangle(summary_img, (20, 20), (550, 180), (0, 0, 0), -1)
            cv2.rectangle(summary_img, (20, 20), (550, 180), (0, 255, 255), 2)
            cv2.putText(summary_img, "PHASE 1 POC - EVALUATION SUMMARY REPORT", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(summary_img, f"Resolution: {W}x{H} | Total Frames: {frame_idx} ({total_time:.1f}s)", (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(summary_img, f"Pipeline Avg FPS: {final_avg_fps:.2f} FPS (Latency: {avg_total_ms:.1f} ms)", (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(summary_img, f"Inference Breakdown: YOLO+ByteTrack: {avg_infer_ms:.1f}ms | Heatmap: {avg_heat_ms:.1f}ms", (35, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(summary_img, f"Unique Person IDs Tracked: {len(all_tracked_ids)}", (35, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

            cv2.imwrite(str(out_summary_path), summary_img)
            print(f"  Summary Report Image saved → {out_summary_path}")

    # Print Final Evaluation Report
    print("\n" + "=" * 65)
    print("  PHASE 1 PIPELINE EVALUATION METRICS REPORT")
    print("=" * 65)
    print(f"  Total Duration Processed  : {total_time:.2f} seconds")
    print(f"  Total Frames Processed    : {frame_idx} frames")
    print(f"  Average Pipeline Speed    : {final_avg_fps:.2f} FPS")
    print(f"  Average Latency per Frame : {avg_total_ms:.2f} ms")
    print(f"  Latency Breakdown:")
    print(f"    - RTSP Capture/Decode   : {avg_read_ms:.2f} ms")
    print(f"    - YOLOv8 + ByteTrack    : {avg_infer_ms:.2f} ms")
    print(f"    - Heatmap Render (KDE)  : {avg_heat_ms:.2f} ms")
    print(f"  Unique Tracked Object IDs : {len(all_tracked_ids)}")
    print(f"  Total Person Detections   : {total_detections}")
    print("=" * 65 + "\n")

    return {
        "fps": final_avg_fps,
        "frames": frame_idx,
        "duration": total_time,
        "unique_ids": len(all_tracked_ids),
        "out_video": str(out_video_path) if save_video else None,
        "out_heatmap": str(out_heatmap_path),
        "out_summary": str(out_summary_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1.6 - RTSP Camera Phase 1 Pipeline Demo")
    parser.add_argument("--rtsp", default=DEFAULT_RTSP, help="RTSP Camera URL")
    parser.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds (default: 30.0)")
    parser.add_argument("--model", default=MODEL_NAME, help="YOLO model path (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=CONF_THRESH, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--output", default="temp", help="Output directory for temp artifacts (default: temp)")
    parser.add_argument("--display", action="store_true", help="Display live window")
    parser.add_argument("--no-video", action="store_true", help="Do not save output video")

    args = parser.parse_args()

    run_rtsp_demo(
        rtsp_url=args.rtsp,
        output_dir=args.output,
        duration=args.duration,
        model_path=args.model,
        conf_thresh=args.conf,
        display=args.display,
        save_video=not args.no_video
    )
