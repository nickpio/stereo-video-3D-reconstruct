"""Quantitative evaluation of predicted depth maps against LiDAR ground truth.

Uses the existing project_lidar_to_image capability in the loader for
sparse but accurate per-pixel GT depths in the left camera image plane.

**Important**: Both classical SGBM and Depth-Anything-V2 can have large
scale bias. This module therefore reports two sets of metrics per method:
- Raw (as produced by the reconstruction)
- "aligned": after robust median scaling to the LiDAR GT on that frame.
  The aligned numbers are the ones you should trust for absolute error.

Provides standard depth estimation metrics (MAE, RMSE, AbsRel, δ<1.25, etc.)
used in monocular/stereo depth papers on nuScenes, KITTI, etc.

Also generates error heatmaps for visualization.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pipeline.nuscenes_loader import CameraData, NuScenesStereoLoader, StereoPair


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    max_depth: float = 80.0,
    min_depth: float = 2.0,
) -> Dict[str, float]:
    """Compute standard depth estimation metrics on valid pixels.

    Args:
        pred_depth: (H, W) predicted depth in meters.
        gt_depth:   (H, W) or (N,) ground truth depths (same units).
        valid_mask: Optional boolean mask of same shape as pred_depth (or broadcastable).
                    If None, computed from finite + in [min_depth, max_depth].
        max_depth / min_depth: Clipping range for valid GT.

    Returns dict with:
        mae, rmse, abs_rel, sq_rel, delta1, delta2, delta3,
        num_valid, pred_median, gt_median, etc.
    """
    if valid_mask is None:
        valid_mask = (
            np.isfinite(gt_depth)
            & np.isfinite(pred_depth)
            & (gt_depth >= min_depth)
            & (gt_depth <= max_depth)
            & (pred_depth >= min_depth)
            & (pred_depth <= max_depth)
        )

    if gt_depth.ndim == 1:
        # Sparse case: gt_depth and pred_depth are already sampled arrays
        p = pred_depth[valid_mask] if valid_mask.ndim > 0 else pred_depth
        g = gt_depth[valid_mask] if valid_mask.ndim > 0 else gt_depth
    else:
        p = pred_depth[valid_mask]
        g = gt_depth[valid_mask]

    if len(p) < 5:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "abs_rel": float("nan"),
            "sq_rel": float("nan"),
            "delta1": float("nan"),
            "delta2": float("nan"),
            "delta3": float("nan"),
            "num_valid": int(len(p)),
            "pred_median": float(np.nanmedian(p)) if len(p) > 0 else float("nan"),
            "gt_median": float(np.nanmedian(g)) if len(g) > 0 else float("nan"),
        }

    p = p.astype(np.float64)
    g = g.astype(np.float64)

    err = np.abs(p - g)
    rel = err / (g + 1e-6)

    mae = np.mean(err)
    rmse = np.sqrt(np.mean(err**2))
    abs_rel = np.mean(rel)
    sq_rel = np.mean(err**2 / (g + 1e-6))

    # Threshold accuracy (standard in depth papers)
    max_ratio = np.maximum(p / g, g / p)
    delta1 = np.mean(max_ratio < 1.25)
    delta2 = np.mean(max_ratio < 1.25**2)
    delta3 = np.mean(max_ratio < 1.25**3)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3),
        "num_valid": int(len(p)),
        "pred_median": float(np.median(p)),
        "gt_median": float(np.median(g)),
    }


def evaluate_frame_depths(
    loader: NuScenesStereoLoader,
    pair: StereoPair,
    classical_depth: np.ndarray,
    neural_depth: np.ndarray,
    left_cam: CameraData,
    *,
    max_depth: float = 80.0,
    min_depth: float = 2.0,
) -> Dict[str, Any]:
    """Evaluate both classical and neural depth for one stereo frame vs LiDAR.

    Returns dict with 'classical' and 'neural' metric dicts + 'num_lidar_points_projected'.
    """
    if pair.lidar_filepath is None:
        return {
            "classical": {"num_valid": 0, "mae": float("nan"), "percent_valid": 0.0},
            "neural": {"num_valid": 0, "mae": float("nan"), "percent_valid": 0.0},
            "num_lidar_points_projected": 0,
            "has_lidar": False,
        }

    try:
        lidar_world = loader.load_lidar_points(pair, in_world=True)
    except Exception:
        return {
            "classical": {"num_valid": 0, "mae": float("nan")},
            "neural": {"num_valid": 0, "mae": float("nan")},
            "num_lidar_points_projected": 0,
            "has_lidar": False,
        }

    uv, gt_depths = loader.project_lidar_to_image(
        lidar_world, left_cam, img_shape=classical_depth.shape[:2]
    )

    if len(gt_depths) < 5:
        return {
            "classical": {"num_valid": 0, "mae": float("nan"), "percent_valid": 0.0},
            "neural": {"num_valid": 0, "mae": float("nan"), "percent_valid": 0.0},
            "num_lidar_points_projected": len(gt_depths),
            "has_lidar": True,
        }

    # Sample predictions at projected locations (nearest neighbor)
    h, w = classical_depth.shape[:2]
    us = np.clip(uv[:, 0].astype(int), 0, w - 1)
    vs = np.clip(uv[:, 1].astype(int), 0, h - 1)

    c_sampled = classical_depth[vs, us]
    n_sampled = neural_depth[vs, us]

    classical_metrics = compute_depth_metrics(
        c_sampled, gt_depths, valid_mask=None, min_depth=min_depth, max_depth=max_depth
    )
    neural_metrics = compute_depth_metrics(
        n_sampled, gt_depths, valid_mask=None, min_depth=min_depth, max_depth=max_depth
    )

    # Compute % valid (completeness)
    num_proj = len(gt_depths)
    pct_valid_classical = (classical_metrics.get("num_valid", 0) / num_proj * 100.0) if num_proj > 0 else 0.0
    pct_valid_neural = (neural_metrics.get("num_valid", 0) / num_proj * 100.0) if num_proj > 0 else 0.0

    classical_metrics["percent_valid"] = float(pct_valid_classical)
    neural_metrics["percent_valid"] = float(pct_valid_neural)

    # === Critical: also compute scale-aligned metrics to LiDAR ===
    # This makes absolute numbers (MAE, RMSE, delta thresholds) meaningful
    # even when the raw predictions have large scale bias.
    classical_aligned = align_depth_to_lidar(classical_depth, gt_depths, uv)
    neural_aligned = align_depth_to_lidar(neural_depth, gt_depths, uv)

    c_aligned_sampled = classical_aligned[vs, us]
    n_aligned_sampled = neural_aligned[vs, us]

    classical_aligned_m = compute_depth_metrics(
        c_aligned_sampled, gt_depths, valid_mask=None, min_depth=min_depth, max_depth=max_depth
    )
    neural_aligned_m = compute_depth_metrics(
        n_aligned_sampled, gt_depths, valid_mask=None, min_depth=min_depth, max_depth=max_depth
    )

    classical_metrics["aligned"] = classical_aligned_m
    neural_metrics["aligned"] = neural_aligned_m

    return {
        "classical": classical_metrics,
        "neural": neural_metrics,
        "num_lidar_points_projected": int(num_proj),
        "has_lidar": True,
    }


def make_depth_error_heatmap(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    *,
    vmin: float = 0.0,
    vmax: float = 10.0,
    colormap: int = 2,  # cv2.COLORMAP_JET
) -> np.ndarray:
    """Create a BGR error heatmap image for |pred - gt| (only where both valid).

    Returns uint8 BGR image same size as inputs.
    """
    import cv2

    diff = np.abs(pred_depth.astype(np.float32) - gt_depth.astype(np.float32))
    diff = np.clip(diff, vmin, vmax)

    # Normalize to 0-255
    if vmax > vmin:
        norm = ((diff - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(diff, dtype=np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(norm, colormap)

    # Mask invalid regions (where either pred or gt is invalid or zero)
    invalid = (pred_depth <= 0.5) | (gt_depth <= 0.5) | ~np.isfinite(pred_depth) | ~np.isfinite(gt_depth)
    heatmap[invalid] = (0, 0, 0)

    return heatmap


def add_error_to_comparison_panel(
    base_panel: np.ndarray,
    error_heatmap: np.ndarray,
    label: str = "Error vs LiDAR",
) -> np.ndarray:
    """Append or overlay an error heatmap row to an existing comparison panel.

    Simple implementation: stacks below the existing 2x2 panel.
    """
    import cv2

    h, w = base_panel.shape[:2]
    err_resized = cv2.resize(error_heatmap, (w, h // 2), interpolation=cv2.INTER_AREA)

    # Add label
    cv2.putText(
        err_resized,
        label,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    combined = np.vstack([base_panel, err_resized])
    return combined


def align_depth_to_lidar(
    pred_depth: np.ndarray,
    lidar_depths: np.ndarray,
    uv: np.ndarray,
) -> np.ndarray:
    """Robustly scale a (possibly relative) depth map to metric LiDAR.

    Tries both direct and inverse scaling hypotheses using median and picks
    the one with lower relative error on the sparse LiDAR points.
    """
    if len(lidar_depths) < 10:
        return pred_depth

    h, w = pred_depth.shape
    us = np.clip(uv[:, 0].astype(int), 0, w - 1)
    vs = np.clip(uv[:, 1].astype(int), 0, h - 1)
    pred_sparse = pred_depth[vs, us]

    mask = (lidar_depths > 0.5) & (pred_sparse > 0.5)
    if mask.sum() < 5:
        return pred_depth

    ref = lidar_depths[mask]
    pred = pred_sparse[mask]

    # Direct scaling
    s1 = np.median(ref / np.maximum(pred, 1e-6))
    aligned1 = s1 * pred_depth

    # Inverse scaling (very common for monocular networks)
    s2 = np.median(ref * np.maximum(pred, 1e-6))
    aligned2 = s2 / np.maximum(pred_depth, 1e-6)

    def median_rel_err(aligned):
        a = aligned[vs[mask], us[mask]]
        return np.median(np.abs(ref - a) / (ref + 1e-6))

    return aligned1 if median_rel_err(aligned1) < median_rel_err(aligned2) else aligned2
