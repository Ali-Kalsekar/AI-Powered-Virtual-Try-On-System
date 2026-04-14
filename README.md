# AI-Powered Virtual Try-On System
> Last automated login update: 2026-04-14 12:41:38

Production-ready real-time virtual try-on system built with Python, OpenCV, and MediaPipe Pose.

Users can try shirt, jacket, and glasses overlays live from webcam input with landmark-aware alignment, dynamic scaling, rotation correction, and FPS monitoring.

## Features

- Real-time body pose detection with MediaPipe Pose
- Landmark tracking for shoulders, hips, neck, and eyes
- Dynamic clothing and accessory alignment
- Multiple item support:
  - Shirt
  - Jacket
  - Glasses
- Live FPS counter and visual HUD
- Mirror mode toggle
- Rotation correction toggle
- Clothing size adjustment
- Screenshot capture and save
- Event logging to CSV
- Modular project architecture for scalability

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Pillow
- Matplotlib
- datetime

## Project Structure

```text
virtual_tryon_system/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ main.py
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ requirements.txt
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ README.md
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ pose_estimation/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ pose_detector.py
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ clothing/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ clothing_overlay.py
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ utils/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ __init__.py
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ draw.py
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ fps.py
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ logger.py
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ config/
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ config.yaml
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ assets/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ shirts/
Ã¢â€â€š   Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ jackets/
Ã¢â€â€š   Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ glasses/
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ output/
    Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ tryon_log.csv
```

## Installation

1. Clone the repository
2. Navigate to project folder
3. Install dependencies

```bash
pip install -r requirements.txt
```

## Asset Setup

Place your assets in these folders:

- assets/shirts/
- assets/jackets/
- assets/glasses/

Recommended asset format:

- Transparent PNG (preferred)
- Front-facing clothing/accessory images
- Centered object with minimal background margins

## Run

```bash
python main.py
```

## Controls

- 1: Toggle Shirt
- 2: Toggle Jacket
- 3: Toggle Glasses
- 0: Remove All Items
- + / =: Increase item size
- - / _: Decrease item size
- M: Toggle mirror mode
- R: Toggle rotation correction
- C: Capture screenshot
- Q: Quit

## Configuration

Edit config/config.yaml:

```yaml
camera_index: 0
default_item: shirt
confidence_threshold: 0.6
frame_width: 960
mirror_mode: true
rotation_correction: true
```

## Output

- Screenshots are saved in output/
- Session events are logged in output/tryon_log.csv

## Performance Notes

- Frame resizing is used for real-time speed
- Lightweight pose tracking is enabled
- OpenCV buffer settings reduce latency
- Optional GPU acceleration can be enabled if your OpenCV build supports CUDA

## Troubleshooting

- Camera not opening:
  - Check camera_index in config/config.yaml
  - Ensure no other app is using the webcam
- Overlay not visible:
  - Use transparent PNG assets
  - Ensure assets are placed in correct folders
- Low FPS:
  - Lower frame_width in config/config.yaml
  - Close background applications

## Future Enhancements

- Asset cycling (next/previous per category)
- Multi-person support
- Segmentation-based depth layering
- Web deployment with WebRTC

## License

MIT
