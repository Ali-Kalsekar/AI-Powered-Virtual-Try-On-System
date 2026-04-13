from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt

from clothing.clothing_overlay import ClothingOverlay
from pose_estimation.pose_detector import PoseDetector
from utils.draw import draw_pose_visuals, draw_status_panel
from utils.fps import FPSCounter
from utils.logger import TryOnLogger


def load_simple_yaml(config_path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if not config_path.exists():
        return config

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value.lower() in {"true", "false"}:
            config[key] = value.lower() == "true"
        else:
            try:
                if "." in value:
                    config[key] = float(value)
                else:
                    config[key] = int(value)
            except ValueError:
                config[key] = value.strip("\"'")

    return config


def resize_for_speed(frame, target_width: int):
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame

    scale = target_width / float(w)
    new_h = int(h * scale)
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def save_screenshot(frame_bgr, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"tryon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    out_path = output_dir / filename
    cv2.imwrite(str(out_path), frame_bgr)
    return out_path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config = load_simple_yaml(project_root / "config" / "config.yaml")

    camera_index = int(config.get("camera_index", 0))
    confidence_threshold = float(config.get("confidence_threshold", 0.6))
    frame_width = int(config.get("frame_width", 960))
    mirror_mode = bool(config.get("mirror_mode", True))
    rotation_correction = bool(config.get("rotation_correction", True))

    default_item = str(config.get("default_item", "shirt")).lower().strip()

    pose_detector = PoseDetector(confidence_threshold=confidence_threshold, tracking_confidence=0.6)
    overlay_engine = ClothingOverlay(str(project_root / "assets"))
    fps_counter = FPSCounter(avg_window=25)
    logger = TryOnLogger(str(project_root / "output" / "tryon_log.csv"))

    plt.ioff()

    selected_items: set[str] = set()
    if default_item in {"shirt", "jacket", "glasses"}:
        selected_items.add(default_item)

    scale_factor = 1.0

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Unable to open camera/video source.")

    window_name = "AI Virtual Try-On System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    logger.log("startup", selected_items, scale_factor)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror_mode:
                frame = cv2.flip(frame, 1)

            frame = resize_for_speed(frame, frame_width)

            pose_data = pose_detector.detect(frame)
            output_frame = overlay_engine.apply(
                frame_bgr=frame,
                pose_data=pose_data,
                selected_items=selected_items,
                scale_factor=scale_factor,
                rotation_correction=rotation_correction,
            )

            output_frame = draw_pose_visuals(output_frame, pose_data)
            fps = fps_counter.update()
            output_frame = draw_status_panel(
                output_frame,
                fps=fps,
                selected_items=selected_items,
                scale_factor=scale_factor,
                mirror_mode=mirror_mode,
                rotation_correction=rotation_correction,
            )

            cv2.imshow(window_name, output_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.log("quit", selected_items, scale_factor)
                break
            if key == ord("1"):
                if "shirt" in selected_items:
                    selected_items.remove("shirt")
                    logger.log("remove_shirt", selected_items, scale_factor)
                else:
                    selected_items.add("shirt")
                    logger.log("add_shirt", selected_items, scale_factor)
            if key == ord("2"):
                if "jacket" in selected_items:
                    selected_items.remove("jacket")
                    logger.log("remove_jacket", selected_items, scale_factor)
                else:
                    selected_items.add("jacket")
                    logger.log("add_jacket", selected_items, scale_factor)
            if key == ord("3"):
                if "glasses" in selected_items:
                    selected_items.remove("glasses")
                    logger.log("remove_glasses", selected_items, scale_factor)
                else:
                    selected_items.add("glasses")
                    logger.log("add_glasses", selected_items, scale_factor)
            if key == ord("0"):
                selected_items.clear()
                logger.log("clear_items", selected_items, scale_factor)
            if key in (ord("+"), ord("=")):
                scale_factor = min(2.0, scale_factor + 0.05)
                logger.log("scale_up", selected_items, scale_factor)
            if key in (ord("-"), ord("_")):
                scale_factor = max(0.5, scale_factor - 0.05)
                logger.log("scale_down", selected_items, scale_factor)
            if key == ord("m"):
                mirror_mode = not mirror_mode
                logger.log("toggle_mirror", selected_items, scale_factor, extra=str(mirror_mode))
            if key == ord("r"):
                rotation_correction = not rotation_correction
                logger.log("toggle_rotation", selected_items, scale_factor, extra=str(rotation_correction))
            if key == ord("c"):
                saved = save_screenshot(output_frame, project_root / "output")
                logger.log("screenshot", selected_items, scale_factor, extra=str(saved.name))

    finally:
        pose_detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
