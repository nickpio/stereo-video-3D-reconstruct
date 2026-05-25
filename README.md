# Stereo Video 3D Reconstruction Pipeline

Classical (OpenCV SGBM) + Neural (Depth-Anything-V2) stereo/monodepth reconstruction from nuScenes multi-camera "stereo video" sequences, with known ego poses for global fusion.

## Features

- Classical (SGBM) and neural (Depth-Anything-V2) depth from stereo pairs (CAM_FRONT_LEFT / RIGHT)
- LiDAR-grounded evaluation (MAE, RMSE, % valid pixels)
- Proper devkit-based stereo rectification
- Multi-frame fusion (point cloud or TSDF via Open3D)
- Side-by-side 3D visualization of classical vs neural reconstructions

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Basic run with LiDAR metrics
python scripts/run_demo.py --scene scene-0061 --num-frames 3 --eval

# With TSDF fusion (needs open3d)
python scripts/run_demo.py --scene scene-0061 --num-frames 2 --fusion tsdf --eval
```

See `scripts/run_demo.py --help` for all options.

## Dataset

Uses the provided `v1.0-mini/` (nuScenes mini).
