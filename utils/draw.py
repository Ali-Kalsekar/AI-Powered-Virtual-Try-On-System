from __future__ import annotations

from typing import Optional

import cv2
import matplotlib.cm as cm
import numpy as np

from pose_estimation.pose_detector import PoseData


_CMAP = cm.get_cmap("viridis")


def _color_from_key(name: str) -> tuple[int, int, int]:
    idx = (sum(ord(c) for c in name) % 256) / 255.0
    rgba = _CMAP(idx)
    return int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255)


def draw_pose_visuals(frame_bgr: np.ndarray, pose_data: Optional[PoseData]) -> np.ndarray:
    if pose_data is None:
        return frame_bgr

    for name, point in pose_data.points.items():
        cv2.circle(frame_bgr, point, 5, _color_from_key(name), -1, lineType=cv2.LINE_AA)

    edges = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_eye", "right_eye"),
    ]
    for a, b in edges:
        if a in pose_data.points and b in pose_data.points:
            cv2.line(
                frame_bgr,
                pose_data.points[a],
                pose_data.points[b],
                (60, 220, 220),
                2,
                lineType=cv2.LINE_AA,
            )

    if pose_data.bbox is not None:
        x1, y1, x2, y2 = pose_data.bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2, lineType=cv2.LINE_AA)

    return frame_bgr


def draw_status_panel(
    frame_bgr: np.ndarray,
    fps: float,
    selected_items: set[str],
    scale_factor: float,
    mirror_mode: bool,
    rotation_correction: bool,
) -> np.ndarray:
    cv2.rectangle(frame_bgr, (10, 10), (470, 165), (15, 15, 15), -1)
    cv2.rectangle(frame_bgr, (10, 10), (470, 165), (90, 190, 190), 1)

    item_text = "None" if not selected_items else ", ".join(sorted(selected_items))

    lines = [
        f"FPS: {fps:.1f}",
        f"Items: {item_text}",
        f"Scale: {scale_factor:.2f}",
        f"Mirror: {'ON' if mirror_mode else 'OFF'}",
        f"Rotation: {'ON' if rotation_correction else 'OFF'}",
    ]

    y = 35
    for line in lines:
        cv2.putText(frame_bgr, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 245, 245), 2, cv2.LINE_AA)
        y += 28

    help_text = "1 Shirt | 2 Jacket | 3 Glasses | +/- Size | M Mirror | R Rotation | C Capture | 0 Clear | Q Quit"
    cv2.putText(frame_bgr, help_text, (10, frame_bgr.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235, 235, 235), 1, cv2.LINE_AA)

    return frame_bgr
