from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseData:
    """Container for detected pose points in pixel coordinates."""

    points: Dict[str, Tuple[int, int]]
    bbox: Optional[Tuple[int, int, int, int]]


class PoseDetector:
    """MediaPipe Pose wrapper for stable landmark detection and tracking."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        tracking_confidence: float = 0.6,
        smooth_factor: float = 0.7,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.smooth_factor = np.clip(smooth_factor, 0.0, 1.0)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=tracking_confidence,
        )

        self._smoothed_points: Dict[str, Tuple[float, float]] = {}

    def _extract_point(
        self,
        landmarks,
        landmark_idx,
        frame_width: int,
        frame_height: int,
    ) -> Optional[Tuple[int, int]]:
        lm = landmarks[landmark_idx]
        if lm.visibility < self.confidence_threshold:
            return None

        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        return x, y

    def _smooth(self, key: str, point: Tuple[int, int]) -> Tuple[int, int]:
        if key not in self._smoothed_points:
            self._smoothed_points[key] = (float(point[0]), float(point[1]))
            return point

        prev_x, prev_y = self._smoothed_points[key]
        new_x = self.smooth_factor * prev_x + (1.0 - self.smooth_factor) * point[0]
        new_y = self.smooth_factor * prev_y + (1.0 - self.smooth_factor) * point[1]

        self._smoothed_points[key] = (new_x, new_y)
        return int(new_x), int(new_y)

    def detect(self, frame_bgr: np.ndarray) -> Optional[PoseData]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(frame_rgb)

        if not result.pose_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        lms = result.pose_landmarks.landmark

        mapping = {
            "left_shoulder": self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            "right_shoulder": self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            "left_hip": self.mp_pose.PoseLandmark.LEFT_HIP,
            "right_hip": self.mp_pose.PoseLandmark.RIGHT_HIP,
            "left_eye": self.mp_pose.PoseLandmark.LEFT_EYE,
            "right_eye": self.mp_pose.PoseLandmark.RIGHT_EYE,
            "nose": self.mp_pose.PoseLandmark.NOSE,
        }

        points: Dict[str, Tuple[int, int]] = {}
        for name, idx in mapping.items():
            p = self._extract_point(lms, idx, w, h)
            if p is not None:
                points[name] = self._smooth(name, p)

        if "left_shoulder" in points and "right_shoulder" in points:
            neck_x = (points["left_shoulder"][0] + points["right_shoulder"][0]) // 2
            neck_y = (points["left_shoulder"][1] + points["right_shoulder"][1]) // 2
            points["neck"] = self._smooth("neck", (neck_x, neck_y))

        required_for_bbox = [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        ]
        bbox = None
        if all(k in points for k in required_for_bbox):
            xs = [points[k][0] for k in required_for_bbox]
            ys = [points[k][1] for k in required_for_bbox]
            pad_x = max(20, int((max(xs) - min(xs)) * 0.25))
            pad_y = max(30, int((max(ys) - min(ys)) * 0.35))

            x1 = max(0, min(xs) - pad_x)
            y1 = max(0, min(ys) - pad_y)
            x2 = min(w - 1, max(xs) + pad_x)
            y2 = min(h - 1, max(ys) + pad_y)
            bbox = (x1, y1, x2, y2)

        return PoseData(points=points, bbox=bbox)

    def close(self) -> None:
        self.pose.close()
