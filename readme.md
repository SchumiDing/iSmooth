<p align="center">
  <svg width="820" height="160" viewBox="0 0 820 160" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="iSmooth">
    <defs>
      <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop stop-color="#12c2e9" offset="0%"/>
        <stop stop-color="#c471ed" offset="50%"/>
        <stop stop-color="#f64f59" offset="100%"/>
      </linearGradient>
      <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
        <feMerge>
          <feMergeNode in="coloredBlur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
    </defs>
    <rect fill="none" width="100%" height="100%"/>
    <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
          font-family="system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, sans-serif"
          font-size="64" font-weight="800" fill="url(#g)" filter="url(#glow)">iSmooth</text>
  </svg>
</p>
<h5 align="center">Lightweight Video Object Detection and Motion Coherence / Stutter Analysis</h5>

## Features

- Object detection with Ultralytics YOLOv8/YOLO11 (via the `ultralytics` package)
- Simple object association using template similarity (OpenCV `cv2.matchTemplate`)
- Per-frame displacement vectors and segment-level metrics:
  - Start/end positions (cumulative displacement)
  - Cosine similarity between consecutive displacement segments
  - Segment length (magnitude)
  - Timestamp range for each segment
- Output `visData.json` can be used to evaluate motion coherence/stutter in videos
- Default weights: `Model/weights/yolo11n.pt`

## Use case: Detecting stutter in frame-interpolated videos

Frame interpolation inserts in-between frames to increase perceived smoothness. When interpolation fails, you may observe:

- Sudden direction reversals or jitter (cosine similarity drops near 0 or negative)
- Unstable speed (spiky/impulsive changes in segment length)
- Track breaks or ID swaps due to association errors

The exported `visData.json` contains direction consistency (cosine similarity) and motion magnitude (length) for each segment, serving as the basis for stutter detection.

## Project structure

```
.
├── run.py                 # One-click run script
├── test.mp4               # Sample video (bring your own / replace)
├── visData.json           # Visualization/analysis data (generated)
├── requirements.txt       # Project dependencies
├── Model/
│   ├── config/globe.py    # Model weight path configuration
│   ├── Modules/locater.py # Core logic (detect, associate, movement, export)
│   └── weights/yolo11n.pt # Default YOLO weights
└── LICENSE                # License
```

## Requirements

- Python 3.9+ (3.10/3.11 recommended)
- macOS/Linux/Windows
- Optional: NVIDIA GPU + CUDA (significant speedup, not required)

## Installation (macOS/Linux)

```bash
# Optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install -U pip
pip install -r requirements.txt
```

## Installation (Windows, PowerShell)

```powershell
# Optional: create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install -U pip
pip install -r requirements.txt
```

Notes for Windows:

- If activation is blocked, run PowerShell as Administrator and execute:
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- For GPU acceleration, install the matching CUDA-enabled PyTorch build per your CUDA version (see pytorch.org). Then install `ultralytics`.

## Quick start

1) Prepare a video: put your file as `test.mp4` in the project root, or edit `run.py` to change `video_path`.
2) Weights: default `Model/weights/yolo11n.pt`. You can swap the model in `Model/config/globe.py` (see below).
3) Run:

```bash
python run.py
```

This will generate `visData.json` in the project root.

## Configure and swap YOLO weights (globe.py)

Edit `Model/config/globe.py` to point to the weights you want to use:

```python
model = "yolo11n.pt"                # replace with your downloaded model, e.g. "yolo11m.pt", "yolo11x.pt", or a custom .pt
model_path = f"Model/weights/{model}"
```

How to use higher-accuracy models:

- Download a different YOLO model from Ultralytics (e.g., `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`, or YOLOv8 variants) and place it under `Model/weights/`.
- Or set an absolute path to your weights file:
  ```python
  model_path = "/absolute/path/to/your_model.pt"
  ```
- Heavier models may require more VRAM and will be slower but can improve detection stability in complex scenes.

Thresholds (in `run.py`):

```python
# Template similarity threshold (0~1): higher is stricter
detector.setThreshold(0.7)
# YOLO detection confidence threshold (0~1)
detector.setYoloThreshold(0.6)
```

## visData.json schema

Data is grouped by object key (e.g., `person_0`) and stores per-segment info between consecutive frames:

```json
{
  "person_0": [
    {
      "start": [x_prev, y_prev],
      "end": [x_curr, y_curr],
      "cosine_similarity": 0.95,
      "length": 12.34,
      "timestamp": [prev_frame_idx, curr_frame_idx]
    }
  ],
  "car_1": [ ... ]
}
```

- Units: pixels. `start`/`end` are cumulative displacements relative to the previous/current object center.
- `cosine_similarity`: direction similarity between the previous and current displacement vectors (range: -1 to 1).
- `length`: magnitude of the current displacement vector.
- `timestamp`: frame index range covered by the segment (inclusive start and end).

## Example: computing a simple coherence index

This is a minimal, heuristic example; feel free to refine to your needs.

```python
import json
import numpy as np

data = json.load(open('visData.json'))

obj_scores = {}

for obj, segs in data.items():
    if not segs:
        continue
    cos = np.array([s['cosine_similarity'] for s in segs])
    leng = np.array([s['length'] for s in segs])

    # Directional consistency: map cos from [-1,1] to [0,1] and weight by length
    cos01 = (cos + 1) / 2
    w = leng / (leng.sum() + 1e-8)
    dir_consistency = float((cos01 * w).sum())

    # Speed stability: penalize large ratio changes between consecutive segment lengths
    if len(leng) > 1:
        r = leng[1:] / (leng[:-1] + 1e-6)
        speed_stability = float(np.exp(-np.abs(np.log(r))).mean())
    else:
        speed_stability = 1.0

    score = 0.6 * dir_consistency + 0.4 * speed_stability
    obj_scores[obj] = score

overall_score = float(np.mean(list(obj_scores.values()))) if obj_scores else 0.0

print('Per-object coherence:', obj_scores)
print('Overall coherence score:', overall_score)
```

Suggested (empirical) ranges — tune for your data:

- overall_score ≥ 0.85: smooth
- 0.70 ~ 0.85: mildly unstable, occasional jitter
- < 0.70: likely stutter/jitter

Tips:

- Many near-zero/negative `cosine_similarity` values or spiky/zig-zag `length` patterns usually indicate poor interpolation quality.
- For complex scenes, consider a stronger multi-object tracker (e.g., DeepSORT/ByteTrack) instead of template matching to reduce ID switches.

## Programmatic usage

```python
from Model.Modules.locater import objectRecorder
from Model.config.globe import model_path

rec = objectRecorder(video_file="your_video.mp4", Model=model_path)
rec.setThreshold(0.7)
rec.setYoloThreshold(0.6)
rec.detect()
vis_data = rec.prepareVisData()
```

## FAQ

- How do I tell if a frame-interpolated video is stuttering?
  - Look for low/negative `cosine_similarity` distribution and abrupt spikes in `length`; combine with the coherence index above.
- ModuleNotFoundError: No module named 'ultralytics'
  - Install dependencies via `pip install -r requirements.txt`.
- Cannot open video / failed to read
  - Ensure `test.mp4` path is correct or edit `run.py` to set `video_path`.
  - Some codecs are not well supported by OpenCV; try re-encoding to H.264/MP4.
- Detection is slow
  - Use a smaller model (e.g., `yolo11n.pt`), downscale input, or enable GPU.
- ID switches / unstable association
  - Tune `setThreshold` for template similarity, or switch to a stronger tracker.

## License

See `LICENSE`.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV, NumPy, Matplotlib
