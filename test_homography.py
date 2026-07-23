"""
test_homography.py
==================
Standalone test module for Perspective Transformation (Homography Matrix - H).
Maps foot-point coordinates (X_camera, Y_camera) -> (X_floor, Y_floor).

Features:
1. Calculates Homography Matrix H using cv2.getPerspectiveTransform().
2. Transforms single or batch 2D points using matrix H.
3. Renders side-by-side visualization: Camera View vs. Bird's-Eye 2D Floorplan.
"""

import os
from pathlib import Path
import cv2
import numpy as np


def compute_homography_matrix(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Computes 3x3 Homography Matrix H from 4 source points and 4 destination points.

    :param src_pts: np.ndarray of shape (4, 2), float32 - 4 corners in Camera View
    :param dst_pts: np.ndarray of shape (4, 2), float32 - 4 corners in Floorplan View
    :return: 3x3 Homography Matrix H
    """
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return H


def transform_foot_point(x_cam: float, y_cam: float, H: np.ndarray) -> tuple[float, float]:
    """
    Transforms a single (X_camera, Y_camera) point to (X_floor, Y_floor) using H matrix.

    [x', y', w]^T = H * [x_cam, y_cam, 1]^T
    X_floor = x' / w,  Y_floor = y' / w

    :param x_cam: X coordinate in camera view
    :param y_cam: Y coordinate in camera view
    :param H: 3x3 Homography Matrix
    :return: Tuple (x_floor, y_floor)
    """
    pt_cam = np.array([x_cam, y_cam, 1.0], dtype=np.float64)
    pt_floor_homo = np.dot(H, pt_cam)
    
    w = pt_floor_homo[2]
    if abs(w) < 1e-6:
        w = 1e-6

    x_floor = pt_floor_homo[0] / w
    y_floor = pt_floor_homo[1] / w
    return float(x_floor), float(y_floor)


def transform_points_batch(pts_cam: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Transforms N points of shape (N, 2) from Camera View to Floorplan View.

    :param pts_cam: np.ndarray of shape (N, 2)
    :param H: 3x3 Homography Matrix
    :return: np.ndarray of shape (N, 2) representing transformed points
    """
    if len(pts_cam) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts_homogeneous = np.column_stack([pts_cam, np.ones(len(pts_cam))])
    transformed_homo = np.dot(pts_homogeneous, H.T)

    w = transformed_homo[:, 2:3]
    w[np.abs(w) < 1e-6] = 1e-6

    pts_floor = transformed_homo[:, :2] / w
    return pts_floor.astype(np.float32)


def run_homography_demo(video_path: str, output_dir: str = "outputs"):
    """
    Test demo: Reads sample video frame, defines 4 ROI points, warps frame to
    Bird's-Eye Floorplan view, and projects sample foot points.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[ERROR] Failed to read frame from video.")
        return

    H_cam, W_cam = frame.shape[:2]

    # ── 1. Define 4 Source Points (Camera View - Trapezoid ROI) ───────────────
    # Top-Left, Top-Right, Bottom-Right, Bottom-Left
    src_pts = np.float32([
        [int(W_cam * 0.30), int(H_cam * 0.40)],  # Top-Left
        [int(W_cam * 0.70), int(H_cam * 0.40)],  # Top-Right
        [int(W_cam * 0.90), int(H_cam * 0.90)],  # Bottom-Right
        [int(W_cam * 0.10), int(H_cam * 0.90)]   # Bottom-Left
    ])

    # ── 2. Target 2D Floorplan Canvas Dimensions ─────────────────────────────
    W_floor, H_floor = 500, 600  # 500px width x 600px height top-down view

    # ── 3. Define 4 Destination Points (Floorplan View - Rectangle) ──────────
    dst_pts = np.float32([
        [0, 0],              # Top-Left
        [W_floor - 1, 0],    # Top-Right
        [W_floor - 1, H_floor - 1],  # Bottom-Right
        [0, H_floor - 1]     # Bottom-Left
    ])

    # ── 4. Compute Homography Matrix H ────────────────────────────────────────
    H_matrix = compute_homography_matrix(src_pts, dst_pts)

    print("\n" + "=" * 60)
    print("  HOMOGRAPHY MATRIX (H):")
    print(H_matrix)
    print("=" * 60 + "\n")

    # ── 5. Warp Camera Frame to Top-Down Bird's-Eye View ──────────────────────
    floorplan_warped = cv2.warpPerspective(frame, H_matrix, (W_floor, H_floor))

    # ── 6. Create Visual Annotations ──────────────────────────────────────────
    cam_vis = frame.copy()

    # Draw Source Quadrilateral ROI on Camera View
    pts_int = src_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(cam_vis, [pts_int], isClosed=True, color=(0, 255, 255), thickness=3)

    # Label ROI Corners
    corner_names = ["TL", "TR", "BR", "BL"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for (x, y), name, col in zip(src_pts, corner_names, colors):
        cv2.circle(cam_vis, (int(x), int(y)), 7, col, -1)
        cv2.putText(cam_vis, name, (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    # Sample Test Foot Points in Camera View
    test_foot_points_cam = np.array([
        [W_cam * 0.50, H_cam * 0.65],  # Center middle
        [W_cam * 0.35, H_cam * 0.80],  # Left bottom
        [W_cam * 0.65, H_cam * 0.50],  # Right top
    ], dtype=np.float32)

    # Transform Foot Points to Floorplan View
    test_foot_points_floor = transform_points_batch(test_foot_points_cam, H_matrix)

    # Draw Foot Points on Camera View & Floorplan View
    for idx, (pt_c, pt_f) in enumerate(zip(test_foot_points_cam, test_foot_points_floor)):
        xc, yc = int(pt_c[0]), int(pt_c[1])
        xf, yf = int(pt_f[0]), int(pt_f[1])

        # Draw on Camera View
        cv2.circle(cam_vis, (xc, yc), 8, (0, 0, 255), -1)
        cv2.putText(cam_vis, f"P{idx+1}", (xc + 10, yc + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw on Floorplan View
        if 0 <= xf < W_floor and 0 <= yf < H_floor:
            cv2.circle(floorplan_warped, (xf, yf), 8, (0, 0, 255), -1)
            cv2.putText(floorplan_warped, f"P{idx+1} ({xf},{yf})", (xf + 10, yf + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ── 7. Combine & Save Output Image ────────────────────────────────────────
    # Resize camera view to match floorplan height for side-by-side display
    cam_resized = cv2.resize(cam_vis, (int(W_cam * (H_floor / H_cam)), H_floor))

    # Add Titles
    cv2.putText(cam_resized, "Camera Perspective View", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(floorplan_warped, "2D Bird's-Eye Floorplan", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    side_by_side = np.hstack([cam_resized, floorplan_warped])

    out_file = output_dir / "test_homography_result.png"
    cv2.imwrite(str(out_file), side_by_side)
    print(f"  [SUCCESS] Demo result saved → {out_file}\n")


if __name__ == "__main__":
    video_sample = "Videos/C01 South First Indoor 02.mp4"
    if os.path.exists(video_sample):
        run_homography_demo(video_sample)
    else:
        print(f"[SKIP] Video not found: {video_sample}")
