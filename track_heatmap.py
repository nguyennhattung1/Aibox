"""
track_heatmap.py
================
Task 1.2 – Detection & Tracking + Heatmap Visualization
- Model    : YOLOv8n (person class only)
- Tracker  : ByteTrack (built-in ultralytics)
- Heatmap  : Gaussian KDE kernel → INFERNO colormap → alpha-blend overlay
- Output   : annotated video (bbox + ID + heatmap) + final heatmap PNG
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME   = "yolov8n.pt"      # change to yolov8s.pt for higher accuracy
PERSON_CLASS = 0                  # COCO class index for 'person'
CONF_THRESH  = 0.35               # minimum detection confidence
IOU_THRESH   = 0.45               # NMS IOU threshold
HEATMAP_ALPHA = 0.55              # overlay blend ratio  (0=only raw, 1=only heatmap)
SIGMA        = 20                 # Gaussian blur sigma for heatmap smoothing
COLORMAP     = cv2.COLORMAP_INFERNO


def run_pipeline(video_path: str, output_dir: str):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    out_video_path  = output_dir / f"{stem}_tracked.mp4"
    out_heatmap_path = output_dir / f"{stem}_heatmap.png"

    print(f"\n{'='*60}")
    print(f"  Input   : {video_path.name}")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Tracker : ByteTrack")
    print(f"{'='*60}")

    # ── Load model ───────────────────────────
    model = YOLO(MODEL_NAME)

    # ── Open video ───────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution : {W}x{H}  |  FPS: {FPS:.1f}  |  Frames: {TOTAL}")

    # ── Video writer ─────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, FPS, (W, H))

    # ── Density matrix ───────────────────────
    density = np.zeros((H, W), dtype=np.float32)

    # ── Track ID colour palette ──────────────
    rng = np.random.default_rng(42)
    id_colors: dict[int, tuple] = {}

    def get_color(tid: int) -> tuple:
        if tid not in id_colors:
            c = rng.integers(80, 255, size=3).tolist()
            id_colors[tid] = tuple(c)
        return id_colors[tid]

    frame_idx = 0
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == 1:
            pct = frame_idx / max(TOTAL, 1) * 100
            bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
            print(f"\r  [{bar}] {pct:5.1f}%  frame {frame_idx}/{TOTAL}", end="", flush=True)

        # ── YOLOv8 + ByteTrack ───────────────
        results = model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS],
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        boxes = results[0].boxes
        annotated = frame.copy()

        # ── Accumulate density + draw ─────────
        if boxes is not None and boxes.id is not None:
            for box, tid in zip(boxes.xyxy.cpu().numpy(), boxes.id.int().cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                tid = int(tid)

                # Foot-point (centre bottom)
                fx = (x1 + x2) // 2
                fy = min(y2, H - 1)

                # Accumulate a small Gaussian splash at foot point
                yr0, yr1 = max(fy - 8, 0), min(fy + 9, H)
                xr0, xr1 = max(fx - 8, 0), min(fx + 9, W)
                density[yr0:yr1, xr0:xr1] += 1.0

                # Draw bounding box + ID label
                color = get_color(tid)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{tid}"
                cv2.putText(annotated, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # Draw foot-point dot
                cv2.circle(annotated, (fx, fy), 4, color, -1)

        # ── Build live heatmap ────────────────
        smooth = gaussian_filter(density, sigma=SIGMA)
        if smooth.max() > 0:
            norm = (smooth / smooth.max() * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(norm, COLORMAP)
            # Mask: only show where density > tiny threshold to keep background clean
            mask = (norm > 8).astype(np.float32)[..., np.newaxis]
            annotated = (annotated * (1 - mask * HEATMAP_ALPHA)
                         + heat_color * mask * HEATMAP_ALPHA).astype(np.uint8)

        # ── Frame info overlay ────────────────
        cv2.putText(annotated, f"Frame {frame_idx}/{TOTAL}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        n_people = len(boxes.id) if (boxes is not None and boxes.id is not None) else 0
        cv2.putText(annotated, f"People: {n_people}", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        writer.write(annotated)

    print(f"\n\n  Processing complete.")

    # ── Save final heatmap PNG ────────────────
    smooth_final = gaussian_filter(density, sigma=SIGMA)
    if smooth_final.max() > 0:
        norm_final = (smooth_final / smooth_final.max() * 255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(norm_final, COLORMAP)
        cv2.imwrite(str(out_heatmap_path), heatmap_img)
        print(f"  Heatmap PNG saved  → {out_heatmap_path}")
    else:
        print("  [WARN] No density accumulated – heatmap skipped.")

    cap.release()
    writer.release()
    print(f"  Output video saved → {out_video_path}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack + INFERNO Heatmap")
    parser.add_argument("--videos", nargs="+",
                        default=[
                            "Videos/C01 South First Indoor 02.mp4",
                            "Videos/CCTV Indoor Lobby 01.mp4",
                        ],
                        help="Path(s) to input video file(s)")
    parser.add_argument("--output", default="outputs",
                        help="Output directory (default: outputs/)")
    args = parser.parse_args()

    for vp in args.videos:
        if not os.path.isfile(vp):
            print(f"[SKIP] File not found: {vp}")
            continue
        run_pipeline(vp, args.output)

    print("All done!")
