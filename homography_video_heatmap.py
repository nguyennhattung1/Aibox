"""
homography_video_heatmap.py
============================
Task 2.1b – Homography trên Full Video
Kết hợp:
  - YOLOv8n + ByteTrack (từ track_heatmap.py)
  - Interactive 4-point ROI + Perspective Transformation (từ interactive_homography.py)

Pipeline:
  1. Mở frame đầu tiên → User click 4 điểm ROI → tính H matrix
  2. Duyệt toàn bộ video → detect + track → ánh xạ foot-point sang Floorplan
  3. Tích lũy density trên canvas Floorplan (Gaussian + INFERNO)
  4. Xuất video side-by-side: [Camera View | Floorplan Heatmap]
  5. Xuất PNG heatmap cuối cùng trên Floorplan

Usage:
  python homography_video_heatmap.py --video "Videos/C01 South First Indoor 02.mp4"
  python homography_video_heatmap.py --list-videos
  python homography_video_heatmap.py --all-videos
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
MODEL_NAME    = "yolov8n.pt"
PERSON_CLASS  = 0
CONF_THRESH   = 0.35
IOU_THRESH    = 0.45
HEATMAP_ALPHA = 0.55          # Tỷ lệ hòa trộn vừa phải giúp nền floorplan rõ nét
SIGMA         = 10            # Giảm sigma từ 18 -> 10 giúp mây nhiệt sắc nét, bớt thô
COLORMAP      = cv2.COLORMAP_JET # Chuyển sang JET cho dải màu tương phản và sắc nét hơn
SPLASH_RADIUS = 3             # Bán kính tích lũy điểm chân gọn gàng (3px thay vì 8px)
MIN_THRESHOLD = 15            # Ngưỡng lọc nhiễu nền (chỉ hiển thị vùng có mật độ đáng kể)

VIDEO_DIR     = Path("Videos")
W_FLOOR       = 600
H_FLOOR       = 700

# ─────────────────────────────────────────────
# GLOBAL STATE – Interactive ROI selection
# ─────────────────────────────────────────────
_selected_points = []
_window_name = "Select 4 Floor Corners  [TL -> TR -> BR -> BL]"


def _mouse_callback(event, x, y, flags, param):
    global _selected_points
    if event == cv2.EVENT_LBUTTONDOWN and len(_selected_points) < 4:
        _selected_points.append((x, y))
        print(f"  [Point {len(_selected_points)}/4] ({x}, {y})")


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def list_videos(video_dir: Path):
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(p for p in video_dir.iterdir() if p.suffix.lower() in exts)


def compute_homography(src_pts, dst_pts):
    return cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))


def transform_points(pts_cam, H):
    if len(pts_cam) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts_h = np.column_stack([pts_cam, np.ones(len(pts_cam))])
    t = np.dot(pts_h, H.T)
    w = t[:, 2:3]
    w[np.abs(w) < 1e-6] = 1e-6
    return (t[:, :2] / w).astype(np.float32)


def get_id_color(tid, palette, rng):
    if tid not in palette:
        palette[tid] = tuple(rng.integers(80, 255, size=3).tolist())
    return palette[tid]


# ─────────────────────────────────────────────
# STEP 1 – Interactive ROI selection
# ─────────────────────────────────────────────

def select_roi_interactive(first_frame):
    global _selected_points
    _selected_points = []

    H_cam, W_cam = first_frame.shape[:2]
    labels = ["1.TL", "2.TR", "3.BR", "4.BL"]
    colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80)]

    print("\n" + "=" * 65)
    print("  INTERACTIVE ROI SELECTION")
    print("=" * 65)
    print("  Click 4 goc vung san (floor area) theo thu tu:")
    print("    1. Top-Left (TL)     -> 2. Top-Right (TR)")
    print("    3. Bottom-Right (BR) -> 4. Bottom-Left (BL)")
    print("  Phim: [c] Confirm  |  [r] Reset  |  [q/ESC] Quit")
    print("=" * 65 + "\n")

    gui_ok = True
    try:
        cv2.namedWindow(_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_window_name, 1280, 720)
        cv2.setMouseCallback(_window_name, _mouse_callback)
    except cv2.error as e:
        print(f"  [WARN] GUI khong kha dung: {e}")
        gui_ok = False

    if not gui_ok:
        _selected_points = [
            (int(W_cam * 0.20), int(H_cam * 0.30)),
            (int(W_cam * 0.80), int(H_cam * 0.30)),
            (int(W_cam * 0.95), int(H_cam * 0.95)),
            (int(W_cam * 0.05), int(H_cam * 0.95)),
        ]
        print("  [FALLBACK] Dung diem mac dinh (khong co GUI).")
    else:
        while True:
            display = first_frame.copy()
            cv2.rectangle(display, (0, 0), (W_cam, 50), (20, 20, 20), -1)
            if len(_selected_points) < 4:
                info = f"Da chon {len(_selected_points)}/4 diem  ->  Click {labels[len(_selected_points)]}"
            else:
                info = "Du 4 diem  ->  [c] Confirm  |  [r] Reset"
            cv2.putText(display, info, (15, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 230), 2)

            for i, pt in enumerate(_selected_points):
                cv2.circle(display, pt, 9, colors[i], -1)
                cv2.circle(display, pt, 9, (255, 255, 255), 2)
                cv2.putText(display, labels[i], (pt[0] + 12, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
                if i > 0:
                    cv2.line(display, _selected_points[i - 1], pt, (0, 220, 220), 2)
            if len(_selected_points) == 4:
                cv2.line(display, _selected_points[3], _selected_points[0], (0, 220, 220), 2)

            cv2.imshow(_window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('r'):
                _selected_points = []
                print("  [RESET] Da xoa diem.")
            elif key == ord('c'):
                if len(_selected_points) == 4:
                    print("  [CONFIRM] 4 diem da chon xong.")
                    break
                else:
                    print(f"  [WARN] Can du 4 diem (hien {len(_selected_points)}/4).")
            elif key in (ord('q'), 27):
                print("  [QUIT] Thoat.")
                cv2.destroyAllWindows()
                sys.exit(0)

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    if len(_selected_points) < 4:
        _selected_points = [
            (int(W_cam * 0.20), int(H_cam * 0.30)),
            (int(W_cam * 0.80), int(H_cam * 0.30)),
            (int(W_cam * 0.95), int(H_cam * 0.95)),
            (int(W_cam * 0.05), int(H_cam * 0.95)),
        ]

    return np.float32(_selected_points)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(video_path: str, output_dir: str):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    out_video_path   = output_dir / f"{stem}_floorplan_heatmap.mp4"
    out_heatmap_path = output_dir / f"{stem}_floorplan_heatmap_final.png"

    print(f"\n{'=' * 65}")
    print(f"  Task 2.1b - Homography Full Video Heatmap")
    print(f"  Input  : {video_path.name}")
    print(f"  Model  : {MODEL_NAME}  |  Tracker: ByteTrack")
    print(f"  Floor  : {W_FLOOR}x{H_FLOOR} px")
    print(f"{'=' * 65}")

    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Khong the mo video: {video_path}")
        return

    W_cam  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_cam  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    TOTAL  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Camera : {W_cam}x{H_cam}  |  FPS: {FPS:.1f}  |  Frames: {TOTAL}\n")

    ret, first_frame = cap.read()
    if not ret:
        print("  [ERROR] Khong doc duoc frame dau tien.")
        cap.release()
        return

    src_pts = select_roi_interactive(first_frame)

    dst_pts = np.float32([
        [0,           0],
        [W_FLOOR - 1, 0],
        [W_FLOOR - 1, H_FLOOR - 1],
        [0,           H_FLOOR - 1],
    ])

    H_matrix = compute_homography(src_pts, dst_pts)

    print("\n  Homography Matrix H:")
    print(H_matrix)
    print()

    # ROI contour dùng cho pointPolygonTest (lọc foot-point)
    roi_contour = src_pts.astype(np.int32).reshape((-1, 1, 2))

    floorplan_bg      = cv2.warpPerspective(first_frame, H_matrix, (W_FLOOR, H_FLOOR))
    floorplan_bg_dark = (floorplan_bg * 0.35).astype(np.uint8)

    density_floor = np.zeros((H_FLOOR, W_FLOOR), dtype=np.float32)

    scale      = H_FLOOR / H_cam
    W_cam_disp = int(W_cam * scale)
    OUT_W      = W_cam_disp + W_FLOOR
    OUT_H      = H_FLOOR

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, FPS, (OUT_W, OUT_H))

    rng = np.random.default_rng(42)
    id_palette = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
        cam_annotated = frame.copy()
        foot_pts_cam  = []
        track_ids     = []

        if boxes is not None and boxes.id is not None:
            for box, tid in zip(boxes.xyxy.cpu().numpy(), boxes.id.int().cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                tid = int(tid)
                fx  = (x1 + x2) // 2
                fy  = min(y2, H_cam - 1)

                # ── Lọc: chỉ xử lý nếu foot-point nằm trong ROI polygon ────
                inside = cv2.pointPolygonTest(roi_contour, (float(fx), float(fy)), measureDist=False)
                if inside < 0:
                    continue  # foot-point ngoài ROI → ẩn hoàn toàn

                foot_pts_cam.append([fx, fy])
                track_ids.append(tid)

                color = get_id_color(tid, id_palette, rng)
                cv2.rectangle(cam_annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(cam_annotated, f"ID:{tid}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.circle(cam_annotated, (fx, fy), 5, color, -1)

        # ── Vẽ khung vàng ROI cố định lên mỗi frame camera ────────────────
        cv2.polylines(cam_annotated, [roi_contour], isClosed=True,
                      color=(0, 220, 220), thickness=2)
        # Vẽ 4 điểm góc
        corner_labels = ["TL", "TR", "BR", "BL"]
        corner_colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 255, 80)]
        for idx, pt in enumerate(src_pts.astype(np.int32)):
            cv2.circle(cam_annotated, tuple(pt), 6, corner_colors[idx], -1)

        n_people = len(track_ids)
        floorplan_frame = floorplan_bg_dark.copy()

        if foot_pts_cam:
            pts_cam_arr   = np.array(foot_pts_cam, dtype=np.float32)
            pts_floor_arr = transform_points(pts_cam_arr, H_matrix)

            for (xf, yf), tid in zip(pts_floor_arr, track_ids):
                xi, yi = int(xf), int(yf)
                if 0 <= xi < W_FLOOR and 0 <= yi < H_FLOOR:
                    yr0, yr1 = max(yi - SPLASH_RADIUS, 0), min(yi + SPLASH_RADIUS + 1, H_FLOOR)
                    xr0, xr1 = max(xi - SPLASH_RADIUS, 0), min(xi + SPLASH_RADIUS + 1, W_FLOOR)
                    density_floor[yr0:yr1, xr0:xr1] += 1.0

                    color = get_id_color(tid, id_palette, rng)
                    cv2.circle(floorplan_frame, (xi, yi), 6, color, -1)
                    cv2.circle(floorplan_frame, (xi, yi), 6, (255, 255, 255), 1)

        smooth = gaussian_filter(density_floor, sigma=SIGMA)
        if smooth.max() > 0:
            norm       = (smooth / smooth.max() * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(norm, COLORMAP)
            mask       = (norm > MIN_THRESHOLD).astype(np.float32)[..., np.newaxis]
            floorplan_frame = (
                floorplan_frame * (1 - mask * HEATMAP_ALPHA)
                + heat_color * mask * HEATMAP_ALPHA
            ).astype(np.uint8)

        # Overlays
        cv2.rectangle(cam_annotated, (0, 0), (W_cam, 42), (10, 10, 10), -1)
        cv2.putText(cam_annotated, f"Frame {frame_idx}/{TOTAL}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(cam_annotated, f"People: {n_people}",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 100), 2)
        cv2.putText(cam_annotated, "Camera View",
                    (W_cam - 160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.rectangle(floorplan_frame, (0, 0), (W_FLOOR, 42), (10, 10, 10), -1)
        cv2.putText(floorplan_frame, "2D Floorplan Heatmap",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
        peak = int(density_floor.max())
        cv2.putText(floorplan_frame, f"Peak density: {peak}",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 2)

        cam_resized  = cv2.resize(cam_annotated, (W_cam_disp, H_FLOOR))
        side_by_side = np.hstack([cam_resized, floorplan_frame])
        cv2.line(side_by_side, (W_cam_disp, 0), (W_cam_disp, H_FLOOR), (60, 60, 60), 2)

        writer.write(side_by_side)

    print(f"\n\n  Done! {frame_idx} frames processed.")

    smooth_final = gaussian_filter(density_floor, sigma=SIGMA)
    if smooth_final.max() > 0:
        norm_final = (smooth_final / smooth_final.max() * 255).astype(np.uint8)
        heat_final = cv2.applyColorMap(norm_final, COLORMAP)
        mask_f     = (norm_final > MIN_THRESHOLD).astype(np.float32)[..., np.newaxis]
        final_img  = (
            floorplan_bg_dark * (1 - mask_f * HEATMAP_ALPHA)
            + heat_final * mask_f * HEATMAP_ALPHA
        ).astype(np.uint8)
        cv2.putText(final_img, "2D Floorplan Heatmap (Final)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)
        cv2.imwrite(str(out_heatmap_path), final_img)
        print(f"  PNG  -> {out_heatmap_path}")
    else:
        print("  [WARN] No density accumulated.")

    cap.release()
    writer.release()
    print(f"  Video -> {out_video_path}")
    print(f"{'=' * 65}\n")


# ─────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Task 2.1b: Homography Full Video Heatmap on 2D Floorplan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du su dung:
  python homography_video_heatmap.py --list-videos
  python homography_video_heatmap.py --video "Videos/C01 South First Indoor 02.mp4"
  python homography_video_heatmap.py --all-videos
        """
    )
    p.add_argument("--video", "-v", type=str, default=None,
                   help="Duong dan den file video (vd: 'Videos/cam1.mp4')")
    p.add_argument("--list-videos", "-l", action="store_true",
                   help="Liet ke tat ca video co san trong thu muc Videos/")
    p.add_argument("--all-videos", "-a", action="store_true",
                   help="Xu ly tat ca video trong thu muc Videos/ lan luot")
    p.add_argument("--output", "-o", type=str, default="outputs",
                   help="Thu muc xuat ket qua (mac dinh: outputs/)")
    p.add_argument("--video-dir", type=str, default=str(VIDEO_DIR),
                   help=f"Thu muc chua video (mac dinh: {VIDEO_DIR})")
    return p


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    video_dir = Path(args.video_dir)

    if args.list_videos:
        videos = list_videos(video_dir)
        if not videos:
            print(f"[INFO] Khong tim thay video trong: {video_dir}")
        else:
            print(f"\nDanh sach video trong '{video_dir}':")
            print("-" * 55)
            for i, v in enumerate(videos, 1):
                size_mb = v.stat().st_size / 1024 / 1024
                print(f"  [{i}] {v.name:<45} ({size_mb:.1f} MB)")
            print("-" * 55)
            print(f"  Tong: {len(videos)} video\n")
        sys.exit(0)

    if args.all_videos:
        videos = list_videos(video_dir)
        if not videos:
            print(f"[ERROR] Khong tim thay video trong: {video_dir}")
            sys.exit(1)
        for v in videos:
            run_pipeline(str(v), args.output)
        print("\nAll done!")
        sys.exit(0)

    if args.video is None:
        videos = list_videos(video_dir)
        if not videos:
            print(f"[ERROR] Khong tim thay video nao trong '{video_dir}'.")
            sys.exit(1)
        print(f"\n[INFO] Khong truyen --video. Tu dong chon video dau tien: {videos[0].name}\n")
        target = videos[0]
    else:
        target = Path(args.video)

    if not target.exists():
        print(f"[ERROR] File khong ton tai: {target}")
        sys.exit(1)

    run_pipeline(str(target), args.output)
    print("Done!")
