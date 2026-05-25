# Stereo Video 3D Reconstruction Pipeline

Classical (OpenCV SGBM) + Neural (Depth-Anything-V2) stereo/monodepth reconstruction from nuScenes multi-camera "stereo video" sequences, with known ego poses for global fusion.

## Features

- Classical (SGBM) and neural (Depth-Anything-V2) depth from stereo pairs (CAM_FRONT_LEFT / RIGHT)
- LiDAR-grounded evaluation (MAE, RMSE, % valid pixels). Aggregate statistics are computed **only over frames with sufficient LiDAR coverage** (`num_lidar_points_projected > 1000` by default). Each per-frame evaluation includes a `used_for_summary` flag.
- Proper devkit-based stereo rectification
- Multi-frame fusion (point cloud or TSDF via Open3D)
- Side-by-side 3D visualization of classical vs neural reconstructions

## Evaluation Notes

When `--eval` is used:
- Per-frame metrics (MAE, RMSE, % valid pixels, etc.) are always computed against projected LiDAR.
- **Aggregate statistics** (mean MAE, etc.) in `reconstruction_stats.json` are calculated **only over frames with ≥ 1000 projected LiDAR points** for reliability.
- Each frame's `eval` entry includes `used_for_summary: true/false` so you can identify which frames contributed to the summary.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# One-command complete sample (recommended — runs with eval + TSDF fusion + 3D previews)
python scripts/run_demo.py --complete-sample --scene scene-0061

# Or run specific configurations manually
python scripts/run_demo.py --scene scene-0061 --num-frames 3 --eval
python scripts/run_demo.py --scene scene-0061 --num-frames 2 --fusion tsdf --eval --save-previews
```

See `scripts/run_demo.py --help` for all options (including `--complete-sample` for a full end-to-end run).

## Dataset

Uses the provided `v1.0-mini/` (nuScenes mini).
