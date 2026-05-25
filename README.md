# Stereo Video 3D Reconstruction Pipeline

Classical (OpenCV SGBM) + Neural (Depth-Anything-V2) stereo/monodepth reconstruction from nuScenes multi-camera "stereo video" sequences, with known ego poses for global fusion.

## Features

- **Stereo pair selection**: CAM_FRONT_LEFT + CAM_FRONT_RIGHT (~1m baseline)
- **Classical**: Full stereo rectification + Semi-Global Block Matching (SGBM) + reproject to 3D
- **Neural**: Monocular depth (Depth-Anything-V2-Small via Hugging Face) with scale alignment to classical or LiDAR
- **Multi-frame fusion**: Unproject + transform depths to world coordinates using nuScenes ego + sensor poses. Two modes:
  - Simple point cloud accumulation (default, fast)
  - TSDF volumetric fusion (dense surface via Open3D — ` --fusion tsdf`)
- **Evaluation**: Quantitative metrics vs LiDAR GT (MAE, RMSE, AbsRel, δ<1.25, etc.) using projected LiDAR depths + error heatmaps (`--eval`)
- **Visualization & exports**:
  - Per-frame side-by-side depth map comparisons (color-mapped, now with optional LiDAR error)
  - Depth video (MP4 via OpenCV)
  - Per-frame + global PLY point clouds (Open3D) + optional surface mesh from TSDF
  - Full quantitative stats in reconstruction_stats.json
- **Demo script**: End-to-end run on any scene with frame limit for quick iteration

## Dataset

Uses the provided `v1.0-mini/` (nuScenes mini, 10 scenes, ~40 keyframes each).

## Quick Start

```bash
# 1. (Recommended) Create venv
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# 2. Install (core classical + loader; neural is optional extra)
pip install -r requirements.txt

# Note: nuscenes-devkit + torch + open3d are heavy. For neural only:
#   pip install torch torchvision transformers  (CPU: add --index-url https://download.pytorch.org/whl/cpu)

# 3. Run the demo (processes first N frames of a scene, writes to outputs/)
python scripts/run_demo.py --scene scene-0061 --num-frames 3 --eval

# With dense TSDF fusion (requires open3d) + full eval
python scripts/run_demo.py --scene scene-0061 --num-frames 2 --fusion tsdf --eval

# Fast classical-only (no torch needed)
python scripts/run_demo.py --scene scene-0061 --num-frames 2 --no-neural

# Or list scenes
python scripts/run_demo.py --list-scenes
```

## Verified Example Output (classical path, 1 frame)

After running the 1-frame classical-only demo on scene-0061:

- `depth_comparison_000.png` — 2x2 panel (left image | SGBM depth | neural slot | diff)
- `pointcloud_classical.ply` (3.9 MB, 60k points, world-frame, colored)
- `pointcloud_lidar.ply` (reference)
- `reconstruction_stats.json` with median depth, counts

The pipeline successfully:
- Loaded nuScenes calibration + poses via devkit
- Computed relative pose + rectified the ~1m baseline stereo pair
- Ran SGBM → dense depth → back-projected + pose-transformed to world
- Fused (trivial 1-frame) + exported PLY (ascii fallback when open3d absent)
- Generated comparison viz

Full neural path (Depth-Anything-V2 + median alignment to classical) follows identical flow once `torch` + `transformers` are present.

Outputs will include:
- `depth_comparison_XXXX.png` — 4-panel viz (left image, classical depth, neural depth, diff)
- `depth_comparison.mp4` — video of the sequence
- `pointcloud_classical.ply` / `pointcloud_neural.ply` / `pointcloud_lidar.ply`
- `stats.json` — timing, depth stats, rough accuracy vs LiDAR

## Methods

### Classical Stereo
1. Load intrinsics + sensor2ego for both cameras at the sample.
2. Compute relative R, T between left and right.
3. `cv2.stereoRectify` + initUndistortRectifyMap → rectified pair.
4. `cv2.StereoSGBM_create` (tuned params for driving scenes) → disparity.
5. `cv2.reprojectImageTo3D` or custom → camera-frame point cloud + colors.
6. Back-project to ego/world using calibration + ego_pose.

### Neural Monodepth
1. Load `depth-anything/Depth-Anything-V2-Small-hf` (or chosen) via `transformers`.
2. Run inference → relative inverse depth (high quality edges, no calibration needed).
3. **Scale recovery**: robust median alignment of (valid) depths to the classical stereo point cloud (or to LiDAR where projected).
4. Same unprojection + world transform as classical.

### Fusion
- Depths masked (min/max range 2-80m typical for driving).
- Colors from left image.
- Transformed via `ego_from_world @ sensor_from_ego @ cam_points`.
- Accumulate with simple voxel downsample (via open3d) for global map.
- Optional: fuse both methods or keep separate for comparison.

## Limitations & Notes

- Monocular depth is **scale-ambiguous**; alignment is heuristic (works well for relative structure but absolute metric error ~10-30% without LiDAR).
- No temporal consistency / filtering across frames (each frame independent).
- nuScenes cameras are **not hardware-synchronized stereo**; small time deltas + rolling shutter possible. Good enough for demo.
- SGBM params tuned heuristically for nuScenes resolution/appearance.
- Neural inference on CPU can be slow (5-30s/frame for 1600x900 input; use `--resize 512` or CUDA).
- Global fusion assumes perfect poses (GT from nuScenes); no drift correction.
- LiDAR comparison is approximate (occlusions, density, beam divergence).

## Development

```bash
# After changes
python -m pytest tests/  # (if added)

# Profile one frame (classical)
python scripts/run_demo.py --scene scene-0061 --num-frames 1 --no-neural --no-video --save-previews
```

## Roadmap / Future

- Add RAFT-Stereo or CREStereo for learned stereo matching (neural + stereo constraints)
- Bundle adjustment or pose graph optimization (instead of / in addition to GT poses)
- Simple TSDF fusion or Gaussian Splatting export (advanced)
- Real-time mode with streaming

## Acknowledgments

- nuScenes by Motional
- Depth-Anything-V2 by Lihe Yang et al.
- OpenCV, Open3D, Hugging Face teams

See [GROK.md](GROK.md) for project coding guidelines.
