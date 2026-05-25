"""Visualization helpers: comparison panels, depth videos, simple 3D previews."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pipeline.classical_stereo import colorize_depth
from pipeline import evaluation as eval_mod


def colorize_depth_jet(depth: np.ndarray, vmin: float = 2.0, vmax: float = 60.0) -> np.ndarray:
    """BGR jet colorized depth (invalid black)."""
    return colorize_depth(depth, vmin=vmin, vmax=vmax)


def make_comparison_image(
    left_bgr: np.ndarray,
    classical_depth: np.ndarray,
    neural_depth: np.ndarray,
    *,
    title: str = "",
    vmin: float = 2.0,
    vmax: float = 60.0,
    target_width: int = 1600,
) -> np.ndarray:
    """Create 2x2 comparison panel (BGR). Resizes to fit target_width."""
    h, w = left_bgr.shape[:2]

    # Make sure depth maps match left resolution (nearest resize if rect vs orig differ)
    cdepth = cv2.resize(classical_depth, (w, h), interpolation=cv2.INTER_NEAREST)
    ndepth = cv2.resize(neural_depth, (w, h), interpolation=cv2.INTER_NEAREST)

    cvis = colorize_depth_jet(cdepth, vmin, vmax)
    nvis = colorize_depth_jet(ndepth, vmin, vmax)

    # Diff (absolute)
    diff = np.abs(cdepth - ndepth)
    diff_vis = colorize_depth_jet(diff, vmin=0.0, vmax=20.0)

    # Resize factor for panel
    scale = target_width / (2 * w + 20)
    small_w = int(w * scale)
    small_h = int(h * scale)

    def prep(img):
        r = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        return r

    left_s = prep(left_bgr)
    c_s = prep(cvis)
    n_s = prep(nvis)
    d_s = prep(diff_vis)

    # 2x2 grid with small gap
    gap = 8
    panel_h = 2 * small_h + gap
    panel_w = 2 * small_w + gap
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    panel[:small_h, :small_w] = left_s
    panel[:small_h, small_w + gap:] = c_s
    panel[small_h + gap:, :small_w] = n_s
    panel[small_h + gap:, small_w + gap:] = d_s

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.6
    cv2.putText(panel, "Left Image", (10, 25), font, fs, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "Classical (SGBM)", (small_w + gap + 10, 25), font, fs, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "Neural (Depth-Anything-V2)", (10, small_h + gap + 25), font, fs, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "Abs Diff (c-n)", (small_w + gap + 10, small_h + gap + 25), font, fs, (255, 255, 255), 2, cv2.LINE_AA)

    if title:
        cv2.putText(panel, title, (10, panel_h - 10), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return panel


def export_depth_video(
    comparison_images: List[np.ndarray],
    out_path: str | Path,
    fps: int = 6,
) -> str:
    """Write list of comparison panels to MP4 using OpenCV."""
    if not comparison_images:
        return ""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = comparison_images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for img in comparison_images:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(img)
    writer.release()
    return str(out_path)


def save_comparison_png(
    panel: np.ndarray,
    path: str | Path,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)
    return str(path)


def simple_pointcloud_preview(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = "Point Cloud",
    max_points: int = 8000,
    out_png: Optional[str | Path] = None,
) -> Optional[np.ndarray]:
    """Very lightweight 3D scatter preview using matplotlib. Returns RGB array or saves PNG."""
    if len(points) == 0:
        return None

    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        pts = points[idx]
        cols = colors[idx] / 255.0 if colors is not None and len(colors) > 0 else None
    else:
        pts = points
        cols = colors / 255.0 if colors is not None and len(colors) > 0 else None

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    if cols is not None:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=1, alpha=0.6)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.5, c=pts[:, 2])

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()

    if out_png:
        out_path = Path(out_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None

    # Return as array for further use
    fig.canvas.draw()
    arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return arr


def make_lidar_error_heatmap(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Thin wrapper around evaluation.make_depth_error_heatmap for convenience in viz flows."""
    return eval_mod.make_depth_error_heatmap(pred_depth, gt_depth, **kwargs)
