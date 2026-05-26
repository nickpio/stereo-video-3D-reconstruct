"""Multi-frame 3D reconstruction and fusion.

- Computes per-frame classical + neural depth
- Unprojects + transforms to world frame using nuScenes poses
- Accumulates colored point clouds (Open3D)
- Exports PLY (per-frame optional + global)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.classical_stereo import (
    compute_classical_stereo,
    colorize_depth,
)
from pipeline.neural_depth import (
    align_depth_to_reference,
    compute_neural_depth,
    load_depth_model,
    predict_relative_depth,
)
from pipeline.nuscenes_loader import (
    CameraData,
    NuScenesStereoLoader,
    StereoPair,
    get_stereo_calibration,
)
from pipeline import evaluation as eval_mod
from pipeline import fusion as fusion_mod


try:
    import open3d as o3d
    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False
    o3d = None  # type: ignore


@dataclass
class FrameResult:
    sample_token: str
    timestamp: int
    left_path: str
    # Depths (original left camera resolution for viz)
    classical_depth: np.ndarray          # meters, 0=invalid
    neural_depth: np.ndarray             # meters after alignment
    # Point clouds in world (for accumulation)
    world_points_classical: np.ndarray   # N x 3
    world_colors_classical: np.ndarray   # N x 3 uint8
    world_points_neural: np.ndarray
    world_colors_neural: np.ndarray
    # Info
    n_points_classical: int
    n_points_neural: int
    classical_stats: Dict[str, float]
    neural_info: Dict[str, Any]
    # Evaluation vs LiDAR (populated when available)
    eval_results: Dict[str, Any] = field(default_factory=dict)

    # Phase 4 dynamic pair selection metadata (sourced from StereoPair in process_frame).
    # Added with defaults for non-breaking compatibility with any prior FrameResult constructions.
    selected_pair_channels: Optional[Tuple[str, str]] = None
    overlap_score: float = 0.0
    quality_score: float = 0.0
    selection_strategy: str = "fixed"


@dataclass
class Reconstruction:
    scene_name: str
    frames: List[FrameResult] = field(default_factory=list)
    global_points_classical: Optional[np.ndarray] = None
    global_colors_classical: Optional[np.ndarray] = None
    global_points_neural: Optional[np.ndarray] = None
    global_colors_neural: Optional[np.ndarray] = None
    lidar_points: Optional[np.ndarray] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    # Aggregate evaluation results (populated when --eval / LiDAR available)
    eval_summary: Dict[str, Any] = field(default_factory=dict)
    # TSDF fusion results (when fusion_mode="tsdf" and open3d available)
    tsdf_classical: Optional["fusion_mod.TSDFResult"] = None
    tsdf_neural: Optional["fusion_mod.TSDFResult"] = None


def backproject_depth_to_cam(
    depth: np.ndarray,
    K: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backproject depth image to 3D points in camera frame. Returns pts (N,3), valid_mask (HxW)."""
    h, w = depth.shape[:2]
    if mask is None:
        mask = (depth > 0.5) & np.isfinite(depth)

    j, i = np.where(mask)  # row, col
    if len(j) == 0:
        return np.empty((0, 3), np.float32), mask

    z = depth[j, i]
    x = (i - K[0, 2]) * z / K[0, 0]
    y = (j - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    if max_points is not None and len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        # Note: mask not updated (caller uses full for viz)

    return pts, mask


def transform_points_world(pts_cam: np.ndarray, world_from_sensor: np.ndarray) -> np.ndarray:
    """pts_cam (N,3) -> world (N,3)"""
    if len(pts_cam) == 0:
        return pts_cam
    pts_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1), dtype=np.float32)])
    pts_w = (world_from_sensor @ pts_h.T).T[:, :3]
    return pts_w.astype(np.float32)


def colors_from_image(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract RGB colors (N,3) uint8 for the True pixels in mask (HxW)."""
    # Open3D prefers float 0-1, but we keep uint8 for now
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return rgb[mask]


def process_frame(
    loader: NuScenesStereoLoader,
    pair: StereoPair,
    classical_model: Optional[Dict] = None,  # not used, params inside
    neural_model: Optional[Dict[str, Any]] = None,
    *,
    min_depth: float = 2.0,
    max_depth: float = 80.0,
    neural_align: bool = True,
    max_points_per_frame: int = 120_000,
    neural_model_type: str = "mono",  # "mono" or "stereo" (Item 3)
) -> FrameResult:
    """Run full classical + neural on one stereo pair. Return FrameResult with world points."""
    left_img = loader.load_image(pair.left)
    right_img = loader.load_image(pair.right)

    # --- Classical ---
    classical_out = compute_classical_stereo(
        pair, left_img, right_img,
        min_depth=min_depth, max_depth=max_depth,
        use_wls=True,   # WLS post-filtering for much better classical disparity density
    )
    classical_depth = classical_out["depth_rect"]  # in rectified, but close enough; for viz we can warp or use as proxy
    # For consistency with neural (original res), re-unproject using original K on the depth we have?
    # The depth_rect is valid on rectified image. For viz simplicity we will use it on left_rect later.
    # For points we already have good ones in classical_out["points_left_cam"]

    pts_classical_cam = classical_out["points_left_cam"]
    cols_classical = classical_out["colors"] if classical_out["colors"] is not None else np.zeros((0, 3), np.uint8)

    # Transform to world
    world_pts_c = transform_points_world(pts_classical_cam, pair.left.world_from_sensor)
    # subsample for memory
    if len(world_pts_c) > max_points_per_frame:
        idx = np.random.choice(len(world_pts_c), max_points_per_frame, replace=False)
        world_pts_c = world_pts_c[idx]
        cols_classical = cols_classical[idx] if len(cols_classical) > 0 else cols_classical

    # --- Neural (on original left image) ---
    neural_depth_rel = None
    neural_depth = None
    neural_info: Dict[str, Any] = {}
    world_pts_n = np.empty((0, 3), np.float32)
    cols_n = np.empty((0, 3), np.uint8)

    if neural_model is not None:
        if neural_model_type == "stereo":
            # Item 3: use stereo-aware path (currently consistency wrapper around mono)
            from pipeline.stereo_depth import compute_stereo_neural_depth
            try:
                stereo_calib = get_stereo_calibration(pair)  # from nuscenes_loader
            except Exception:
                stereo_calib = None

            neural_depth_rel, neural_info = compute_stereo_neural_depth(
                left_img, right_img, neural_model,
                stereo_calib=stereo_calib,
                ref_depth=classical_out["depth_rect"],
                align=neural_align
            )
        else:
            # Default mono path
            neural_depth_rel, neural_info = compute_neural_depth(
                left_img, neural_model, ref_depth=classical_out["depth_rect"], align=neural_align
            )

        neural_depth = neural_depth_rel  # already aligned inside if ref given

        # Prefer direct LiDAR alignment when available (much better scale for absolute metrics)
        if pair.lidar_filepath:
            try:
                from pipeline import evaluation as eval_mod_local
                lidar_world = loader.load_lidar_points(pair, in_world=True)
                uv, gt_depths = loader.project_lidar_to_image(
                    lidar_world, pair.left, img_shape=neural_depth.shape[:2]
                )
                if len(gt_depths) > 20:
                    neural_depth = eval_mod_local.align_depth_to_lidar(neural_depth, gt_depths, uv)
                    neural_info["aligned_to_lidar"] = True
            except Exception:
                pass

        # Backproject using original intrinsics + neural_depth (now metric)
        # Note: neural_depth may be at slightly different resolution if model resized, but HF post restores orig
        pts_n_cam, n_mask = backproject_depth_to_cam(
            neural_depth, pair.left.intrinsic, max_points=max_points_per_frame
        )
        if len(pts_n_cam) > 0:
            world_pts_n = transform_points_world(pts_n_cam, pair.left.world_from_sensor)
            cols_n = colors_from_image(left_img, n_mask)[:len(world_pts_n)]

    # Stats
    c_stats = {
        "median_depth": float(np.nanmedian(classical_out["depth_rect"][classical_out["depth_rect"] > 0])),
        "n_points_cam": int(len(pts_classical_cam)),
    }

    # --- Evaluation vs LiDAR (if available for this pair) ---
    eval_res: Dict[str, Any] = {}
    try:
        if pair.lidar_filepath:
            eval_res = eval_mod.evaluate_frame_depths(
                loader,
                pair,
                classical_out["depth_rect"],
                neural_depth if neural_depth is not None else np.zeros_like(classical_out["depth_rect"]),
                pair.left,
            )
    except Exception as e:
        eval_res = {"error": str(e)}

    return FrameResult(
        sample_token=pair.sample_token,
        timestamp=pair.timestamp,
        left_path=pair.left.filepath,
        classical_depth=classical_out["depth_rect"],  # may be rect res; viz will handle
        neural_depth=neural_depth if neural_depth is not None else np.zeros_like(classical_out["depth_rect"]),
        world_points_classical=world_pts_c,
        world_colors_classical=cols_classical,
        world_points_neural=world_pts_n,
        world_colors_neural=cols_n,
        n_points_classical=len(world_pts_c),
        n_points_neural=len(world_pts_n),
        classical_stats=c_stats,
        neural_info=neural_info,
        eval_results=eval_res,
        # Phase 4: propagate selection metadata from StereoPair (non-breaking for pair objects without attrs)
        selected_pair_channels=getattr(pair, "selected_pair_channels", None),
        overlap_score=getattr(pair, "overlap_score", 0.0),
        quality_score=getattr(pair, "quality_score", 0.0),
        selection_strategy=getattr(pair, "selection_strategy", "fixed"),
    )


def accumulate_reconstruction(
    loader: NuScenesStereoLoader,
    pairs: List[StereoPair],
    neural_model: Optional[Dict] = None,
    *,
    scene_name: str = "unknown",
    max_frames: Optional[int] = None,
    max_points_per_frame: int = 80_000,
    voxel_size: float = 0.15,  # for optional downsample on globals
    # Phase 3+ options (accepted for CLI compatibility; full use in later wiring)
    fusion_mode: str = "points",
    do_eval: bool = False,
    tsdf_voxel_size: float = 0.08,
    neural_model_type: str = "mono",  # "mono" or "stereo" (Item 3)
    **kwargs,
) -> Reconstruction:
    """Process a list of stereo pairs into a full reconstruction."""
    frames: List[FrameResult] = []
    all_c_pts: List[np.ndarray] = []
    all_c_cols: List[np.ndarray] = []
    all_n_pts: List[np.ndarray] = []
    all_n_cols: List[np.ndarray] = []

    # Data collected for optional TSDF fusion
    tsdf_classical_depths: List[np.ndarray] = []
    tsdf_neural_depths: List[np.ndarray] = []
    tsdf_colors: List[np.ndarray] = []
    tsdf_world_from_cams: List[np.ndarray] = []
    tsdf_intrinsics: List[np.ndarray] = []

    n = len(pairs) if max_frames is None else min(len(pairs), max_frames)

    do_tsdf = (fusion_mode == "tsdf" and fusion_mod.HAS_OPEN3D)

    for i, pair in enumerate(pairs[:n]):
        print(f"  Processing frame {i+1}/{n} (sample {pair.sample_token[:8]})...")
        fr = process_frame(
            loader, pair,
            neural_model=neural_model,
            max_points_per_frame=max_points_per_frame,
            neural_model_type=neural_model_type,
        )
        frames.append(fr)
        if len(fr.world_points_classical) > 0:
            all_c_pts.append(fr.world_points_classical)
            all_c_cols.append(fr.world_colors_classical)
        if len(fr.world_points_neural) > 0:
            all_n_pts.append(fr.world_points_neural)
            all_n_cols.append(fr.world_colors_neural)

        # Collect data for TSDF if requested
        if do_tsdf:
            try:
                left_img = loader.load_image(pair.left)
                tsdf_classical_depths.append(fr.classical_depth.copy())
                tsdf_neural_depths.append(fr.neural_depth.copy())
                tsdf_colors.append(left_img)
                tsdf_world_from_cams.append(pair.left.world_from_sensor.copy())
                tsdf_intrinsics.append(pair.left.intrinsic.copy())
            except Exception as e:
                print(f"    [warn] Could not collect TSDF data for frame {i}: {e}")

    # Concat globals
    g_pts_c = np.concatenate(all_c_pts, axis=0) if all_c_pts else np.empty((0, 3), np.float32)
    g_cols_c = np.concatenate(all_c_cols, axis=0) if all_c_cols else np.empty((0, 3), np.uint8)
    g_pts_n = np.concatenate(all_n_pts, axis=0) if all_n_pts else np.empty((0, 3), np.float32)
    g_cols_n = np.concatenate(all_n_cols, axis=0) if all_n_cols else np.empty((0, 3), np.uint8)

    # Optional lidar reference (first few frames for speed)
    lidar_pts = None
    try:
        lidar_list = []
        for p in pairs[: min(3, len(pairs))]:
            if p.lidar_filepath and os.path.exists(p.lidar_filepath):
                pts = loader.load_lidar_points(p, in_world=True)
                if len(pts) > 50000:
                    pts = pts[np.random.choice(len(pts), 50000, replace=False)]
                lidar_list.append(pts)
        if lidar_list:
            lidar_pts = np.concatenate(lidar_list, axis=0)
    except Exception as e:
        print(f"  [warn] Could not load LiDAR reference: {e}")

    # Simple aggregate eval summary from per-frame results.
    # Only include frames with sufficient LiDAR projections (>=1000) for reliable metrics.
    # For the per-method means, further require num_valid > 0 (i.e. at least some
    # predicted depths matched valid GT after range clipping) to avoid NaN pollution.
    MIN_LIDAR_POINTS_FOR_EVAL = 1000

    eval_summary: Dict[str, Any] = {
        "has_lidar_eval": False,
        "frames_with_eval": 0,
        "frames_with_sufficient_lidar": 0,
        "min_lidar_points_threshold": MIN_LIDAR_POINTS_FOR_EVAL,
    }
    classical_maes = []
    neural_maes = []
    classical_maes_aligned = []
    neural_maes_aligned = []

    for fr in frames:
        er = getattr(fr, "eval_results", {})
        num_lidar = er.get("num_lidar_points_projected", 0)
        has_lidar = bool(er.get("has_lidar"))
        c = er.get("classical", {}) or {}
        n = er.get("neural", {}) or {}
        c_nv = int(c.get("num_valid", 0))
        n_nv = int(n.get("num_valid", 0))
        has_valid = (c_nv > 0) or (n_nv > 0)

        if has_lidar and num_lidar >= MIN_LIDAR_POINTS_FOR_EVAL:
            eval_summary["has_lidar_eval"] = True
            eval_summary["frames_with_sufficient_lidar"] += 1
            if has_valid:
                eval_summary["frames_with_eval"] += 1
            # Mark as used for summary only if it actually contributes usable (non-nan) metrics
            er["used_for_summary"] = bool(has_valid)

            if c_nv > 0 and "mae" in c and np.isfinite(c["mae"]):
                classical_maes.append(float(c["mae"]))
            if n_nv > 0 and "mae" in n and np.isfinite(n["mae"]):
                neural_maes.append(float(n["mae"]))

            # Also collect scale-aligned MAEs when available (and valid)
            ca = c.get("aligned", {}) or {}
            na = n.get("aligned", {}) or {}
            ca_nv = int(ca.get("num_valid", 0))
            na_nv = int(na.get("num_valid", 0))
            if ca_nv > 0 and "mae" in ca and np.isfinite(ca["mae"]):
                classical_maes_aligned.append(float(ca["mae"]))
            if na_nv > 0 and "mae" in na and np.isfinite(na["mae"]):
                neural_maes_aligned.append(float(na["mae"]))

        elif has_lidar:
            # Frame had some LiDAR but below threshold
            er["used_for_summary"] = False
            eval_summary["has_lidar_eval"] = True
        else:
            er["used_for_summary"] = False

    if classical_maes:
        eval_summary["classical_mean_mae"] = float(np.mean(classical_maes))
    if neural_maes:
        eval_summary["neural_mean_mae"] = float(np.mean(neural_maes))

    if classical_maes_aligned:
        eval_summary["classical_mean_mae_aligned"] = float(np.mean(classical_maes_aligned))
    if neural_maes_aligned:
        eval_summary["neural_mean_mae_aligned"] = float(np.mean(neural_maes_aligned))

    # --- Optional TSDF fusion (only if requested and open3d available) ---
    tsdf_classical = None
    tsdf_neural = None
    if do_tsdf and tsdf_classical_depths:
        print(f"  Running TSDF fusion (voxel_size={tsdf_voxel_size}) ...")
        try:
            tsdf_classical = fusion_mod.fuse_sequence_with_tsdf(
                tsdf_classical_depths,
                tsdf_colors,
                tsdf_world_from_cams,
                tsdf_intrinsics,
                voxel_size=tsdf_voxel_size,
            )
            print(f"    Classical TSDF: {len(tsdf_classical.points):,} surface points")
        except Exception as e:
            print(f"    Classical TSDF failed: {e}")

        try:
            tsdf_neural = fusion_mod.fuse_sequence_with_tsdf(
                tsdf_neural_depths,
                tsdf_colors,
                tsdf_world_from_cams,
                tsdf_intrinsics,
                voxel_size=tsdf_voxel_size,
            )
            print(f"    Neural TSDF: {len(tsdf_neural.points):,} surface points")
        except Exception as e:
            print(f"    Neural TSDF failed: {e}")

    rec = Reconstruction(
        scene_name=scene_name,
        frames=frames,
        global_points_classical=g_pts_c,
        global_colors_classical=g_cols_c,
        global_points_neural=g_pts_n,
        global_colors_neural=g_cols_n,
        lidar_points=lidar_pts,
        stats={
            "n_frames": len(frames),
            "total_points_classical": len(g_pts_c),
            "total_points_neural": len(g_pts_n),
            "has_lidar": lidar_pts is not None,
            "fusion_mode": fusion_mode,
        },
        eval_summary=eval_summary,
        tsdf_classical=tsdf_classical,
        tsdf_neural=tsdf_neural,
    )
    return rec


def save_pointcloud_ply(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    path: str | Path,
    *,
    voxel_down: Optional[float] = None,
) -> str:
    """Save colored point cloud as PLY. Returns saved path."""
    if not HAS_OPEN3D:
        # Fallback: write simple ASCII PLY manually
        return _save_ply_ascii(points, colors, path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    if voxel_down and voxel_down > 0 and len(points) > 1000:
        pcd = pcd.voxel_down_sample(voxel_down)

    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)
    return str(path)


def _save_ply_ascii(points: np.ndarray, colors: Optional[np.ndarray], path: str | Path) -> str:
    """Very small dependency-free PLY writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    has_col = colors is not None and len(colors) == n

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_col:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            if has_col:
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x} {y} {z}\n")
    return str(path)


def export_reconstruction(
    rec: Reconstruction,
    out_dir: str | Path,
    *,
    save_per_frame: bool = False,
    voxel_down: float = 0.12,
) -> Dict[str, str]:
    """Export global (and optionally per-frame) PLYs + stats.json. Returns paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    # Globals
    if len(rec.global_points_classical) > 0:
        p = out_dir / "pointcloud_classical.ply"
        save_pointcloud_ply(rec.global_points_classical, rec.global_colors_classical, p, voxel_down=voxel_down)
        written["classical"] = str(p)

    if len(rec.global_points_neural) > 0:
        p = out_dir / "pointcloud_neural.ply"
        save_pointcloud_ply(rec.global_points_neural, rec.global_colors_neural, p, voxel_down=voxel_down)
        written["neural"] = str(p)

    if rec.lidar_points is not None and len(rec.lidar_points) > 0:
        p = out_dir / "pointcloud_lidar.ply"
        save_pointcloud_ply(rec.lidar_points, None, p, voxel_down=voxel_down * 1.5)
        written["lidar"] = str(p)

    # TSDF surface results (if fusion was tsdf and succeeded)
    if rec.tsdf_classical is not None and rec.tsdf_classical.has_surface:
        p = out_dir / "surface_classical.ply"
        save_pointcloud_ply(rec.tsdf_classical.points, rec.tsdf_classical.colors, p)
        written["surface_classical"] = str(p)

        # Save mesh if available (better for full 3D reconstruction)
        if HAS_OPEN3D and rec.tsdf_classical.mesh_vertices is not None and rec.tsdf_classical.mesh_triangles is not None:
            try:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(rec.tsdf_classical.mesh_vertices)
                mesh.triangles = o3d.utility.Vector3iVector(rec.tsdf_classical.mesh_triangles)
                if rec.tsdf_classical.colors is not None and len(rec.tsdf_classical.colors) == len(rec.tsdf_classical.mesh_vertices):
                    mesh.vertex_colors = o3d.utility.Vector3dVector(rec.tsdf_classical.colors.astype(np.float64) / 255.0)
                o3d.io.write_triangle_mesh(str(out_dir / "surface_classical_mesh.ply"), mesh)
                written["surface_classical_mesh"] = str(out_dir / "surface_classical_mesh.ply")
            except Exception as e:
                print(f"  [warn] Failed to save classical TSDF mesh: {e}")

    if rec.tsdf_neural is not None and rec.tsdf_neural.has_surface:
        p = out_dir / "surface_neural.ply"
        save_pointcloud_ply(rec.tsdf_neural.points, rec.tsdf_neural.colors, p)
        written["surface_neural"] = str(p)

        if HAS_OPEN3D and rec.tsdf_neural.mesh_vertices is not None and rec.tsdf_neural.mesh_triangles is not None:
            try:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(rec.tsdf_neural.mesh_vertices)
                mesh.triangles = o3d.utility.Vector3iVector(rec.tsdf_neural.mesh_triangles)
                if rec.tsdf_neural.colors is not None and len(rec.tsdf_neural.colors) == len(rec.tsdf_neural.mesh_vertices):
                    mesh.vertex_colors = o3d.utility.Vector3dVector(rec.tsdf_neural.colors.astype(np.float64) / 255.0)
                o3d.io.write_triangle_mesh(str(out_dir / "surface_neural_mesh.ply"), mesh)
                written["surface_neural_mesh"] = str(out_dir / "surface_neural_mesh.ply")
            except Exception as e:
                print(f"  [warn] Failed to save neural TSDF mesh: {e}")

    # Stats
    stats_path = out_dir / "reconstruction_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "scene": rec.scene_name,
            "n_frames": rec.stats.get("n_frames", len(rec.frames)),
            "points": {
                "classical": int(rec.stats.get("total_points_classical", 0)),
                "neural": int(rec.stats.get("total_points_neural", 0)),
                "lidar": int(len(rec.lidar_points)) if rec.lidar_points is not None else 0,
            },
            "eval_summary": rec.eval_summary,
            "fusion": {
                "mode": getattr(rec, "stats", {}).get("fusion_mode", "points"),
                "tsdf_classical_points": int(len(rec.tsdf_classical.points)) if rec.tsdf_classical and rec.tsdf_classical.points is not None else 0,
                "tsdf_neural_points": int(len(rec.tsdf_neural.points)) if rec.tsdf_neural and rec.tsdf_neural.points is not None else 0,
                "tsdf_classical_integrated_frames": getattr(rec.tsdf_classical, 'num_integrated_frames', 0) if rec.tsdf_classical else 0,
                "tsdf_neural_integrated_frames": getattr(rec.tsdf_neural, 'num_integrated_frames', 0) if rec.tsdf_neural else 0,
            },
            # Phase 4: top-level pair_selection summary (derived from enriched per-frame data; empty for old runs)
            "pair_selection": {
                "strategy": (getattr(rec.frames[0], "selection_strategy", None) if rec.frames else None),
                "num_frames_with_selection": len(rec.frames),
                "dynamic_pairs_count": sum(1 for fr in rec.frames if getattr(fr, "selection_strategy", None) == "best_overlap"),
            } if rec.frames and any(getattr(fr, "selected_pair_channels", None) for fr in rec.frames) else {},
            "per_frame": [
                {
                    "sample": fr.sample_token[:10],
                    "n_classical": fr.n_points_classical,
                    "n_neural": fr.n_points_neural,
                    "classical_median_depth": fr.classical_stats.get("median_depth"),
                    "eval": getattr(fr, "eval_results", {}),
                    # Phase 4: include selection metadata from StereoPair (via FrameResult); safe for old runs
                    "selected_pair_channels": getattr(fr, "selected_pair_channels", None),
                    "overlap_score": getattr(fr, "overlap_score", None),
                    "quality_score": getattr(fr, "quality_score", None),
                    "selection_strategy": getattr(fr, "selection_strategy", None),
                }
                for fr in rec.frames
            ],
        }, f, indent=2)
    written["stats"] = str(stats_path)

    return written
