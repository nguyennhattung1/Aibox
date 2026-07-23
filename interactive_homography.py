"""
interactive_homography.py
==========================
Interactive UI Tool for selecting 4 floor corners on a video frame
and performing Perspective Transformation (Homography Matrix - H).

Controls:
- Left Mouse Click : Select 4 floor corner points in order:
                     1. Top-Left (TL)
                     2. Top-Right (TR)
                     3. Bottom-Right (BR)
                     4. Bottom-Left (BL)
- Press 'c'        : Confirm selection & compute Homography transformation
- Press 'r'        : Reset / Clear selected points
- Press 'q' / ESC  : Quit application
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np


# ─────────────────────────────────────────────
# GLOBAL STATE FOR MOUSE CALLBACK
# ─────────────────────────────────────────────
selected_points = []
window_name = "Select 4 Floor Corners (TL -> TR -> BR -> BL)"


def mouse_callback(event, x, y, flags, param):
    """Mouse click handler to record up to 4 ROI corner points."""
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 4:
            selected_points.append((x, y))
            print(f"  [Point {len(selected_points)}/4] Clicked at: ({x}, {y})")


def compute_homography_matrix(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Computes 3x3 Homography Matrix H."""
    return cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))


def transform_points_batch(pts_cam: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Transforms N points from Camera View to Floorplan View."""
    if len(pts_cam) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts_homogeneous = np.column_stack([pts_cam, np.ones(len(pts_cam))])
    transformed_homo = np.dot(pts_homogeneous, H.T)

    w = transformed_homo[:, 2:3]
    w[np.abs(w) < 1e-6] = 1e-6

    pts_floor = transformed_homo[:, :2] / w
    return pts_floor.astype(np.float32)


def run_interactive_ui(video_path: str, output_dir: str = "outputs"):
    global selected_points

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    ret, original_frame = cap.read()
    cap.release()

    if not ret or original_frame is None:
        print("[ERROR] Failed to read frame from video.")
        return

    H_cam, W_cam = original_frame.shape[:2]

    print("\n" + "=" * 65)
    print("  INTERACTIVE HOMOGRAPHY UI")
    print("=" * 65)
    print("  INSTRUCTIONS:")
    print("  1. Click 4 floor corners on the window in order:")
    print("     - Point 1: Top-Left (TL)")
    print("     - Point 2: Top-Right (TR)")
    print("     - Point 3: Bottom-Right (BR)")
    print("     - Point 4: Bottom-Left (BL)")
    print("  2. Press 'c' to CONFIRM & compute Homography transform")
    print("  3. Press 'r' to RESET selected points")
    print("  4. Press 'q' or ESC to QUIT")
    print("=" * 65 + "\n")

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, mouse_callback)
    except cv2.error as e:
        print(f"[WARN] GUI Display not available: {e}")
        print("Fallback to automatic default points for test execution.")
        selected_points = [
            (int(W_cam * 0.20), int(H_cam * 0.30)),
            (int(W_cam * 0.80), int(H_cam * 0.30)),
            (int(W_cam * 0.95), int(H_cam * 0.95)),
            (int(W_cam * 0.05), int(H_cam * 0.95))
        ]

    labels = ["1. TL", "2. TR", "3. BR", "4. BL"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    gui_available = True
    while gui_available:
        display_frame = original_frame.copy()

        # Draw instructions on top of window
        cv2.rectangle(display_frame, (0, 0), (W_cam, 45), (0, 0, 0), -1)
        info_str = f"Selected {len(selected_points)}/4 points. "
        if len(selected_points) < 4:
            info_str += f"Click point {labels[len(selected_points)]}"
        else:
            info_str += "Press 'c' to Confirm | 'r' to Reset"

        cv2.putText(display_frame, info_str, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw selected points & connecting lines
        for idx, pt in enumerate(selected_points):
            col = colors[idx]
            cv2.circle(display_frame, pt, 8, col, -1)
            cv2.putText(display_frame, labels[idx], (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

            if idx > 0:
                cv2.line(display_frame, selected_points[idx - 1], pt, (0, 255, 255), 2)

        if len(selected_points) == 4:
            cv2.line(display_frame, selected_points[3], selected_points[0], (0, 255, 255), 2)

        try:
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(30) & 0xFF
        except cv2.error:
            gui_available = False
            break

        if key == ord('r'):
            selected_points = []
            print("  [RESET] Cleared selected points.")
        elif key == ord('c'):
            if len(selected_points) == 4:
                print("  [CONFIRM] 4 points selected. Processing...")
                break
            else:
                print(f"  [WARN] Please select exactly 4 points first (currently {len(selected_points)}/4).")
        elif key == ord('q') or key == 27:
            print("  [QUIT] Exiting...")
            cv2.destroyAllWindows()
            return

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    if len(selected_points) < 4:
        print("Using default 4-point ROI for fallback.")
        src_pts = np.float32([
            [int(W_cam * 0.20), int(H_cam * 0.30)],
            [int(W_cam * 0.80), int(H_cam * 0.30)],
            [int(W_cam * 0.95), int(H_cam * 0.95)],
            [int(W_cam * 0.05), int(H_cam * 0.95)]
        ])
    else:
        src_pts = np.float32(selected_points)

    # ── Target 2D Floorplan Canvas Dimensions ─────────────────────────────
    W_floor, H_floor = 600, 700
    dst_pts = np.float32([
        [0, 0],
        [W_floor - 1, 0],
        [W_floor - 1, H_floor - 1],
        [0, H_floor - 1]
    ])

    # ── Compute Homography Matrix H ────────────────────────────────────────
    H_matrix = compute_homography_matrix(src_pts, dst_pts)

    print("\n" + "=" * 65)
    print("  COMPUTED HOMOGRAPHY MATRIX (H):")
    print(H_matrix)
    print("=" * 65 + "\n")

    # ── Warp Perspective to Bird's-Eye Floorplan ──────────────────────────
    floorplan_warped = cv2.warpPerspective(original_frame, H_matrix, (W_floor, H_floor))

    # ── Annotate Camera View & Floorplan ──────────────────────────────────
    cam_vis = original_frame.copy()
    pts_int = src_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(cam_vis, [pts_int], isClosed=True, color=(0, 255, 255), thickness=3)

    corner_names = ["TL", "TR", "BR", "BL"]
    for (x, y), name, col in zip(src_pts, corner_names, colors):
        cv2.circle(cam_vis, (int(x), int(y)), 8, col, -1)
        cv2.putText(cam_vis, name, (int(x) + 12, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    # Sample Test Foot Points
    test_foot_points_cam = np.array([
        [W_cam * 0.50, H_cam * 0.65],
        [W_cam * 0.35, H_cam * 0.80],
        [W_cam * 0.65, H_cam * 0.50],
        [W_cam * 0.45, H_cam * 0.40],
    ], dtype=np.float32)

    test_foot_points_floor = transform_points_batch(test_foot_points_cam, H_matrix)

    for idx, (pt_c, pt_f) in enumerate(zip(test_foot_points_cam, test_foot_points_floor)):
        xc, yc = int(pt_c[0]), int(pt_c[1])
        xf, yf = int(pt_f[0]), int(pt_f[1])

        cv2.circle(cam_vis, (xc, yc), 8, (0, 0, 255), -1)
        cv2.putText(cam_vis, f"P{idx+1}", (xc + 10, yc + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if 0 <= xf < W_floor and 0 <= yf < H_floor:
            cv2.circle(floorplan_warped, (xf, yf), 8, (0, 0, 255), -1)
            cv2.putText(floorplan_warped, f"P{idx+1} ({xf},{yf})", (xf + 10, yf + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ── Combine Side-by-Side Result ────────────────────────────────────────
    cam_resized = cv2.resize(cam_vis, (int(W_cam * (H_floor / H_cam)), H_floor))

    cv2.putText(cam_resized, "Camera Perspective View (Selected ROI)", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(floorplan_warped, "2D Bird's-Eye Floorplan", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    side_by_side = np.hstack([cam_resized, floorplan_warped])

    out_file = output_dir / f"{video_path.stem}_interactive_homography.png"
    cv2.imwrite(str(out_file), side_by_side)
    print(f"  [SUCCESS] Interactive result saved → {out_file}\n")


if __name__ == "__main__":
    video_sample = "Videos/C01 South First Indoor 02.mp4"
    output_sample = "outputs"

    if len(sys.argv) > 1:
        video_sample = sys.argv[1]
    if len(sys.argv) > 2:
        output_sample = sys.argv[2]

    if os.path.exists(video_sample):
        run_interactive_ui(video_sample, output_sample)
    else:
        print(f"[SKIP] Video file not found: {video_sample}")
