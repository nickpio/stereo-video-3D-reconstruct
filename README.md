# Stereo Video 3D Reconstruction Pipeline

Classical (OpenCV SGBM) + Neural (Depth-Anything-V2) stereo/monodepth reconstruction from nuScenes multi-camera "stereo video" sequences, with known ego poses for global fusion.

## Features

- Classical (SGBM) and neural (Depth-Anything-V2) depth from stereo pairs (default: fixed CAM_FRONT_LEFT + CAM_FRONT_RIGHT; use `--dynamic-pairs` for per-frame best-overlap selection)
- LiDAR-grounded evaluation (MAE, RMSE, % valid pixels). Aggregate statistics are computed **only over frames with sufficient LiDAR coverage** (`num_lidar_points_projected > 1000` by default). Each per-frame evaluation includes a `used_for_summary` flag.
- Proper devkit-based stereo rectification
- Multi-frame fusion (point cloud or TSDF via Open3D)
- Side-by-side 3D visualization of classical vs neural reconstructions

### Dynamic Camera Pair Selection (`--dynamic-pairs`)

The `--dynamic-pairs` flag enables per-frame best-overlap selection using geometric metrics instead of the fixed FL+FR pair (CAM_FRONT_LEFT + CAM_FRONT_RIGHT).

Example command:
```bash
python scripts/run_demo.py --scene scene-0061 --dynamic-pairs --num-frames 5 --eval
```

Selection info (chosen cameras, overlap/quality) is printed during the run.

This is powered by the underlying `compute_pair_overlap_metrics` function. The pipeline remains fully compatible with both modes (fixed pair is the default when the flag is omitted).

## Pipeline Overview

The pipeline processes nuScenes stereo video sequences through these main stages:

1. **Stereo Pair Selection** — Loads synchronized camera pairs. Defaults to fixed `CAM_FRONT_LEFT` + `CAM_FRONT_RIGHT`. With `--dynamic-pairs`, it selects the best-overlapping pair per frame using geometric overlap + baseline metrics.

2. **Depth Estimation** — Computes depth using classical stereo (OpenCV SGBM) and neural depth (Depth-Anything-V2, mono or stereo).

3. **LiDAR Evaluation** — Projects LiDAR points into images to compute quantitative metrics (MAE, RMSE, % valid) for both depth sources.

4. **Multi-frame Fusion** — Fuses per-frame depth into a global 3D model using either point clouds or dense TSDF (via Open3D).

5. **Visualization & Export** — Generates comparison images/video, point clouds, surface meshes (with TSDF), and 3D previews.

The most complete and accurate results are produced with:
```bash
python scripts/run_demo.py --complete-sample --dynamic-pairs --scene scene-0061
```
This combination enables dynamic pair selection, TSDF fusion, stereo neural depth, full evaluation, and rich visualizations.

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
