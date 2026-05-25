#!/usr/bin/env python3
"""End-to-end demo runner for stereo video 3D reconstruction on nuScenes mini.

Classical (SGBM) + Neural (Depth-Anything-V2) with global fusion (points or TSDF)
and quantitative LiDAR evaluation.

Usage examples:
  python scripts/run_demo.py --list-scenes
  python scripts/run_demo.py --scene scene-0061 --num-frames 3 --eval
  python scripts/run_demo.py --scene scene-0061 --num-frames 2 --fusion tsdf --eval
  python scripts/run_demo.py --scene scene-0061 --num-frames 1 --no-neural
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Add parent to path for direct script run
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.nuscenes_loader import NuScenesStereoLoader, StereoPair
from pipeline.reconstruction import (
    accumulate_reconstruction,
    export_reconstruction,
)
from pipeline import evaluation as eval_mod
from pipeline import fusion as fusion_mod
from pipeline.viz import (
    export_depth_video,
    make_comparison_image,
    save_comparison_png,
    simple_pointcloud_preview,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stereo Video 3D Reconstruction Demo (nuScenes mini)")
    p.add_argument("--scene", default="scene-0061", help="Scene name from mini (default: scene-0061)")
    p.add_argument("--num-frames", type=int, default=4, help="Max frames to process (default 4 for speed)")
    p.add_argument("--output-dir", default=None, help="Output directory (default: outputs/<scene>_<timestamp>)")
    p.add_argument("--list-scenes", action="store_true", help="List available scenes and exit")
    p.add_argument("--no-neural", action="store_true", help="Skip neural depth (much faster, classical only)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for neural model")
    p.add_argument("--no-video", action="store_true", help="Skip MP4 export")
    p.add_argument("--save-previews", action="store_true", help="Save simple matplotlib 3D pointcloud previews")
    p.add_argument("--voxel", type=float, default=0.15, help="Voxel size for global PLY downsampling")
    p.add_argument("--max-points", type=int, default=60000, help="Max points kept per frame (memory)")
    # New in Phase 3 (plan)
    p.add_argument("--fusion", choices=["points", "tsdf"], default="points",
                   help="Multi-frame fusion mode: 'points' (default, fast) or 'tsdf' (dense surface, requires open3d)")
    p.add_argument("--eval", action="store_true", help="Compute quantitative metrics vs LiDAR GT (uses project_lidar)")
    p.add_argument("--voxel-size", type=float, default=0.08, help="TSDF voxel size (meters) when --fusion tsdf")
    # Item 3
    p.add_argument("--neural-model", choices=["mono", "stereo"], default="mono",
                   help="Neural depth model type (mono = default Depth-Anything-V2, stereo = stereo-consistent variant)")
    return p.parse_args()


def get_loader() -> NuScenesStereoLoader:
    return NuScenesStereoLoader(dataroot="v1.0-mini", version="v1.0-mini", verbose=False)


def main() -> int:
    args = parse_args()

    loader = get_loader()

    if args.list_scenes:
        print("Available scenes in v1.0-mini:")
        for s in loader.list_scenes():
            print(f"  {s['name']:12s}  samples={s['n_samples']:3d}  {s['description'][:70]}...")
        return 0

    scene_name = args.scene
    print(f"=== Stereo 3D Reconstruction Demo ===")
    print(f"Scene: {scene_name}")
    print(f"Max frames: {args.num_frames}")
    print(f"Neural enabled: {not args.no_neural}")

    # 1. Load sequence
    print("\n[1/5] Loading stereo sequence (CAM_FRONT_LEFT + CAM_FRONT_RIGHT)...")
    t0 = time.time()
    pairs: List[StereoPair] = loader.get_stereo_sequence(
        scene_name, max_frames=args.num_frames
    )
    if not pairs:
        print(f"ERROR: No stereo pairs found for scene {scene_name}")
        return 2
    print(f"  Loaded {len(pairs)} stereo pairs (dt={time.time()-t0:.1f}s)")

    # 2. Neural model (optional)
    neural_model = None
    if not args.no_neural:
        model_type = args.neural_model
        print(f"\n[2/5] Loading neural depth model ({model_type}) (Depth-Anything-V2-Small, may download ~100MB)...")
        try:
            if model_type == "stereo":
                from pipeline.stereo_depth import load_stereo_depth_model
                neural_model = load_stereo_depth_model(device=args.device, force_cpu=(args.device == "cpu"))
            else:
                from pipeline.neural_depth import load_depth_model
                neural_model = load_depth_model(device=args.device, force_cpu=(args.device == "cpu"))
            print(f"  Model loaded on {neural_model['device']} (type={model_type})")
        except Exception as e:
            print(f"  WARNING: Could not load neural model ({e}). Continuing with classical only.")
            neural_model = None
    else:
        print("\n[2/5] Neural disabled by flag.")

    # 3. Run reconstruction (classical always + neural if available)
    print("\n[3/5] Running reconstruction (classical SGBM + neural monodepth + world fusion)...")
    t1 = time.time()
    rec = accumulate_reconstruction(
        loader,
        pairs,
        neural_model=neural_model,
        scene_name=scene_name,
        max_points_per_frame=args.max_points,
        # New Phase 3 options (forwarded; reconstruction will use what it can)
        fusion_mode=args.fusion,
        do_eval=args.eval,
        tsdf_voxel_size=args.voxel_size,
        neural_model_type=args.neural_model,  # Item 3
    )
    print(f"  Done in {time.time()-t1:.1f}s. "
          f"Points: classical={rec.stats.get('total_points_classical',0):,}, "
          f"neural={rec.stats.get('total_points_neural',0):,}")
    if rec.eval_summary.get("has_lidar_eval"):
        print(f"  Eval (LiDAR): classical_mae={rec.eval_summary.get('classical_mean_mae'):.2f}, "
              f"neural_mae={rec.eval_summary.get('neural_mean_mae'):.2f}")
        # Item 1: Show per-frame % valid (completeness) for hard evidence
        if rec.frames:
            first_eval = rec.frames[0].eval_results
            c_pct = first_eval.get("classical", {}).get("percent_valid", 0)
            n_pct = first_eval.get("neural", {}).get("percent_valid", 0)
            print(f"         % valid pixels: classical={c_pct:.1f}%, neural={n_pct:.1f}% "
                  f"(out of {first_eval.get('num_lidar_points_projected', 0)} LiDAR projections)")

    # 4. Export point clouds + stats
    print("\n[4/5] Exporting point clouds and stats...")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"{scene_name}_{ts}"
    written = export_reconstruction(rec, out_dir, voxel_down=args.voxel, save_per_frame=False)
    print(f"  Wrote:")
    for k, p in written.items():
        print(f"    {k:12s} -> {p}")

    # 5. Visualizations (comparison panels + optional video + 3D previews)
    print("\n[5/5] Generating visualizations...")
    panels: List[np.ndarray] = []
    for i, fr in enumerate(rec.frames):
        # Reload left for clean viz (cheap)
        left_img = cv2.imread(fr.left_path)
        if left_img is None:
            continue

        # Use the depths stored (note: classical may be rect-res but viz resizes)
        panel = make_comparison_image(
            left_img,
            fr.classical_depth,
            fr.neural_depth if fr.neural_depth is not None else fr.classical_depth,
            title=f"{scene_name} frame {i+1}/{len(rec.frames)}",
        )
        panels.append(panel)

        # Save individual
        save_comparison_png(panel, out_dir / f"depth_comparison_{i:03d}.png")

    video_path = ""
    if panels and not args.no_video:
        video_path = export_depth_video(panels, out_dir / "depth_comparison.mp4", fps=4)
        if video_path:
            print(f"  Video: {video_path}")

    # Optional 3D preview images (one per method)
    if args.save_previews:
        if rec.global_points_classical is not None and len(rec.global_points_classical) > 0:
            simple_pointcloud_preview(
                rec.global_points_classical,
                rec.global_colors_classical,
                title=f"Classical - {scene_name}",
                out_png=out_dir / "preview_classical_3d.png",
            )
        if rec.global_points_neural is not None and len(rec.global_points_neural) > 0:
            simple_pointcloud_preview(
                rec.global_points_neural,
                rec.global_colors_neural,
                title=f"Neural - {scene_name}",
                out_png=out_dir / "preview_neural_3d.png",
            )
        if rec.lidar_points is not None and len(rec.lidar_points) > 0:
            simple_pointcloud_preview(
                rec.lidar_points,
                title=f"LiDAR ref - {scene_name}",
                out_png=out_dir / "preview_lidar_3d.png",
            )

    # Final summary
    print("\n" + "=" * 60)
    print("DONE. Reconstruction complete.")
    print(f"Output directory: {out_dir.resolve()}")
    print("\nKey artifacts:")
    print("  - depth_comparison_*.png   : per-frame classical vs neural vs diff")
    if video_path:
        print("  - depth_comparison.mp4     : video of the sequence")
    print("  - pointcloud_*.ply         : global fused maps (open in Open3D / CloudCompare / MeshLab)")
    if args.fusion == "tsdf":
        print("  - surface_*.ply            : TSDF extracted surface points (when successful)")
    print("  - reconstruction_stats.json: summary numbers + eval (if --eval)")
    if args.save_previews:
        print("  - preview_*_3d.png         : quick 3D scatter previews")
    if neural_model is None:
        print("  (Re-ran with neural model for full comparison: pip install torch transformers)")

    print("\nTips:")
    print("  - For interactive 3D: python -c 'import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud(\"pointcloud_classical.ply\")])'")
    print("  - Compare classical (geometric) vs neural (learned) structure.")
    print("  - LiDAR PLY + new --eval gives quantitative MAE/RMSE vs GT (see reconstruction_stats.json).")
    print("  - Try --fusion tsdf (requires open3d) for dense surface reconstruction.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
