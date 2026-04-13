from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

from pose_estimation.pose_detector import PoseData


class ClothingOverlay:
    """Loads clothing assets and overlays them using pose landmarks."""

    def __init__(self, assets_root: str) -> None:
        self.assets_root = Path(assets_root)
        self.catalog = {
            "shirt": self._discover_assets("shirts"),
            "jacket": self._discover_assets("jackets"),
            "glasses": self._discover_assets("glasses"),
        }
        self.cache: Dict[str, np.ndarray] = {}

    def _discover_assets(self, folder: str) -> list[Path]:
        base = self.assets_root / folder
        if not base.exists():
            return []

        assets = []
        for ext in ("*.png", "*.webp"):
            assets.extend(base.glob(ext))
        return sorted(assets)

    def _load_rgba(self, path: Path) -> Optional[np.ndarray]:
        key = str(path.resolve())
        if key in self.cache:
            return self.cache[key].copy()

        if not path.exists():
            return None

        pil = Image.open(path).convert("RGBA")
        rgba = np.array(pil)
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

        self.cache[key] = bgra
        return bgra.copy()

    def _rotate(self, img_bgra: np.ndarray, angle_deg: float) -> np.ndarray:
        h, w = img_bgra.shape[:2]
        center = (w / 2.0, h / 2.0)

        mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cos = abs(mat[0, 0])
        sin = abs(mat[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        mat[0, 2] += (new_w / 2) - center[0]
        mat[1, 2] += (new_h / 2) - center[1]

        return cv2.warpAffine(
            img_bgra,
            mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    def _resize_rgba(self, img_bgra: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        target_w = max(1, target_w)
        target_h = max(1, target_h)

        pil = Image.fromarray(cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA))
        pil = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        resized = np.array(pil)
        return cv2.cvtColor(resized, cv2.COLOR_RGBA2BGRA)

    def _alpha_blend(
        self,
        frame_bgr: np.ndarray,
        overlay_bgra: np.ndarray,
        top_left: Tuple[int, int],
    ) -> None:
        x, y = top_left
        h, w = overlay_bgra.shape[:2]
        fh, fw = frame_bgr.shape[:2]

        if x >= fw or y >= fh or x + w <= 0 or y + h <= 0:
            return

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)

        ov_x1 = x1 - x
        ov_y1 = y1 - y
        ov_x2 = ov_x1 + (x2 - x1)
        ov_y2 = ov_y1 + (y2 - y1)

        roi = frame_bgr[y1:y2, x1:x2]
        ov = overlay_bgra[ov_y1:ov_y2, ov_x1:ov_x2]

        alpha = ov[:, :, 3:4].astype(np.float32) / 255.0
        inv_alpha = 1.0 - alpha

        roi[:] = (alpha * ov[:, :, :3] + inv_alpha * roi).astype(np.uint8)

    def _default_asset_for(self, item: str) -> Optional[np.ndarray]:
        if item not in self.catalog or not self.catalog[item]:
            return None
        return self._load_rgba(self.catalog[item][0])

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

    @staticmethod
    def _angle_deg(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0])))

    def _overlay_torso_item(
        self,
        frame_bgr: np.ndarray,
        pose_data: PoseData,
        item: str,
        scale_factor: float,
        rotation_correction: bool,
    ) -> None:
        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "neck"]
        if not all(k in pose_data.points for k in required):
            return

        cloth = self._default_asset_for(item)
        if cloth is None:
            return

        ls = pose_data.points["left_shoulder"]
        rs = pose_data.points["right_shoulder"]
        lh = pose_data.points["left_hip"]
        rh = pose_data.points["right_hip"]
        neck = pose_data.points["neck"]

        shoulder_w = self._distance(ls, rs)
        torso_h = max(1.0, (self._distance(ls, lh) + self._distance(rs, rh)) * 0.5)

        target_w = int(shoulder_w * 1.8 * scale_factor)
        target_h = int(torso_h * 1.75 * scale_factor)

        cloth = self._resize_rgba(cloth, target_w, target_h)
        if rotation_correction:
            cloth = self._rotate(cloth, self._angle_deg(ls, rs))

        top_left_x = int(neck[0] - cloth.shape[1] * 0.5)
        top_left_y = int(neck[1] - cloth.shape[0] * 0.2)

        self._alpha_blend(frame_bgr, cloth, (top_left_x, top_left_y))

    def _overlay_glasses(
        self,
        frame_bgr: np.ndarray,
        pose_data: PoseData,
        scale_factor: float,
        rotation_correction: bool,
    ) -> None:
        if "left_eye" not in pose_data.points or "right_eye" not in pose_data.points:
            return

        glasses = self._default_asset_for("glasses")
        if glasses is None:
            return

        le = pose_data.points["left_eye"]
        re = pose_data.points["right_eye"]
        eye_distance = self._distance(le, re)

        target_w = int(max(20.0, eye_distance * 2.2 * scale_factor))
        aspect = glasses.shape[0] / max(1, glasses.shape[1])
        target_h = int(target_w * aspect)

        glasses = self._resize_rgba(glasses, target_w, target_h)
        if rotation_correction:
            glasses = self._rotate(glasses, self._angle_deg(le, re))

        center_x = int((le[0] + re[0]) * 0.5)
        center_y = int((le[1] + re[1]) * 0.5)
        top_left_x = int(center_x - glasses.shape[1] * 0.5)
        top_left_y = int(center_y - glasses.shape[0] * 0.45)

        self._alpha_blend(frame_bgr, glasses, (top_left_x, top_left_y))

    def apply(
        self,
        frame_bgr: np.ndarray,
        pose_data: Optional[PoseData],
        selected_items: Set[str],
        scale_factor: float = 1.0,
        rotation_correction: bool = True,
    ) -> np.ndarray:
        if pose_data is None or not selected_items:
            return frame_bgr

        ordered_items = [item for item in ("shirt", "jacket", "glasses") if item in selected_items]

        for item in ordered_items:
            if item in ("shirt", "jacket"):
                self._overlay_torso_item(
                    frame_bgr,
                    pose_data,
                    item=item,
                    scale_factor=scale_factor,
                    rotation_correction=rotation_correction,
                )
            elif item == "glasses":
                self._overlay_glasses(
                    frame_bgr,
                    pose_data,
                    scale_factor=scale_factor,
                    rotation_correction=rotation_correction,
                )

        return frame_bgr
