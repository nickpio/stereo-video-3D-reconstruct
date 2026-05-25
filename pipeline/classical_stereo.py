"""Classical stereo reconstruction using OpenCV SGBM + rectification.

Designed for nuScenes camera pairs (pinhole intrinsics, no distortion params).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from pipeline.nuscenes_loader import CameraData, StereoPair, get_stereo_calibration


@dataclass
class RectificationData:
    """Precomputed rectification maps and matrices."""
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray
    map2y: np.ndarray
    Q: np.ndarray          # 4x4 reproject matrix
    P1: np.ndarray         # 3x4 projection for rectified left
    P2: np.ndarray
    R1: np.ndarray         # Rectification rotation for left (orig -> rect)
    R2: np.ndarray
    R: np.ndarray          # right_from_left rotation
    t: np.ndarray          # right_from_left translation (baseline mostly in x after rect)
    roi1: Tuple[int, int, int, int]
    roi2: Tuple[int, int, int, int]
    baseline: float        # meters, |t| after rectification (approx)


def get_relative_pose(left: CameraData, right: CameraData) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (R, t) such that X_right_sensor ≈ R @ X_left_sensor + t."""
    # right_from_left = right.sensor_from_world @ world_from_left
    right_from_left = (
        right.sensor_from_ego
        @ right.ego_from_world
        @ left.world_from_sensor
    )
    R = right_from_left[:3, :3]
    t = right_from_left[:3, 3:4].flatten()
    return R, t


def compute_rectification(
    left_intr: np.ndarray,
    right_intr: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    img_size: Tuple[int, int],  # (width, height)
    alpha: float = 0.0,
) -> RectificationData:
    """Compute stereo rectification for a pair. Assumes zero distortion."""
    w, h = img_size
    D = np.zeros(5, dtype=np.float64)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_intr, D,
        right_intr, D,
        (w, h),
        R, t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        left_intr, D, R1, P1, (w, h), cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        right_intr, D, R2, P2, (w, h), cv2.CV_32FC1
    )

    baseline = float(np.linalg.norm(t))

    return RectificationData(
        map1x=map1x, map1y=map1y,
        map2x=map2x, map2y=map2y,
        Q=Q,
        P1=P1, P2=P2,
        R1=R1, R2=R2,
        R=R, t=t,
        roi1=roi1, roi2=roi2,
        baseline=baseline,
    )


def rectify_images(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    rect: RectificationData,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remap images to rectified views (BGR uint8)."""
    left_rect = cv2.remap(
        left_bgr, rect.map1x, rect.map1y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    right_rect = cv2.remap(
        right_bgr, rect.map2x, rect.map2y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return left_rect, right_rect


def compute_disparity(
    left_rect: np.ndarray,
    right_rect: np.ndarray,
    *,
    min_disp: int = 0,
    num_disp: int = 160,   # multiple of 16, larger for ~1m baseline @ nuScenes res
    block_size: int = 7,
    uniqueness: int = 12,
    speckle_window: int = 150,
    speckle_range: int = 2,
) -> np.ndarray:
    """Semi-global block matching. Returns disparity in pixels (float32, NaN for invalid)."""
    if num_disp % 16 != 0:
        num_disp = ((num_disp // 16) + 1) * 16

    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        disp12MaxDiff=2,
        uniquenessRatio=uniqueness,
        speckleWindowSize=speckle_window,
        speckleRange=speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(left_rect, right_rect).astype(np.float32) / 16.0
    # Mark clearly invalid
    disp[disp <= min_disp] = np.nan
    return disp


def disparity_to_depth_rect(
    disp: np.ndarray,
    rect: RectificationData,
    *,
    min_depth: float = 2.0,
    max_depth: float = 80.0,
) -> np.ndarray:
    """Convert disparity (rectified) to depth in the rectified left camera frame (meters)."""
    f = rect.P1[0, 0]  # rectified focal (x)
    B = rect.baseline
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (f * B) / disp
    depth[~np.isfinite(depth)] = 0.0
    depth[(depth < min_depth) | (depth > max_depth)] = 0.0
    return depth


def depth_to_pointcloud(
    depth_rect: np.ndarray,
    rect: RectificationData,
    left_color_rect: Optional[np.ndarray] = None,
    *,
    min_depth: float = 2.0,
    max_depth: float = 80.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """3D points (N,3) in the *original left camera* frame + optional RGB colors.

    Uses reproject + transform by R1 to go from rectified -> original sensor coords.
    """
    # Use OpenCV reproject for speed/accuracy (gives points in rectified left cam)
    points_rect = cv2.reprojectImageTo3D(depth_rect.astype(np.float32), rect.Q)  # HxWx3

    valid = (depth_rect > min_depth) & (depth_rect < max_depth) & (depth_rect > 0)
    pts_rect = points_rect[valid]  # N x 3

    if len(pts_rect) == 0:
        return np.empty((0, 3), dtype=np.float32), None

    # Transform from rectified left camera frame back to original left camera frame
    # The rectification rotates the camera: R1 is such that new basis = R1 @ old?
    # Reprojected points live in the coordinate system after applying R1 to the camera.
    # To go back: X_orig = R1.T @ X_rect   (R1 is orthogonal)
    R1 = rect.R1
    pts_orig = (R1.T @ pts_rect.T).T.astype(np.float32)

    colors = None
    if left_color_rect is not None:
        colors = left_color_rect[valid].astype(np.uint8)

    return pts_orig, colors


def compute_classical_stereo(
    pair: StereoPair,
    left_img: np.ndarray,
    right_img: np.ndarray,
    *,
    min_depth: float = 2.0,
    max_depth: float = 80.0,
    num_disp: int = 160,
    block_size: int = 7,
    use_devkit_calibration: bool = True,
) -> Dict[str, Any]:
    """End-to-end classical stereo for one pair.

    When use_devkit_calibration=True (default), the relative pose is obtained
    using the recommended devkit-derived calibration helper (proper use of
    calibrated_sensor records). This is the "proper" path requested for
    rigorous use of nuScenes calibrations.

    The math is equivalent to the previous manual chaining, but now explicitly
    sourced from the devkit for traceability and maintainability.

    Returns:
      - rect: RectificationData
      - left_rect, right_rect
      - disp (rect)
      - depth_rect (meters in rect cam)
      - points_left_cam (N,3) in original left sensor frame
      - colors (N,3) uint8 or None
      - K_left_orig, etc for downstream use
      - calibration_source: "devkit_calibrated_sensor" or "legacy"
    """
    h, w = left_img.shape[:2]
    img_size = (w, h)

    if use_devkit_calibration:
        calib = get_stereo_calibration(pair)
        K1, K2 = calib["K1"], calib["K2"]
        R, t = calib["R"], calib["t"]
        calibration_source = calib["source"]
    else:
        R, t = get_relative_pose(pair.left, pair.right)
        K1, K2 = pair.left.intrinsic, pair.right.intrinsic
        calibration_source = "legacy"

    rect = compute_rectification(K1, K2, R, t, img_size)

    left_r, right_r = rectify_images(left_img, right_img, rect)
    disp = compute_disparity(left_r, right_r, num_disp=num_disp, block_size=block_size)
    depth_r = disparity_to_depth_rect(disp, rect, min_depth=min_depth, max_depth=max_depth)

    pts, cols = depth_to_pointcloud(
        depth_r, rect, left_r if left_img is not None else None,
        min_depth=min_depth, max_depth=max_depth
    )

    result = {
        "rect": rect,
        "left_rect": left_r,
        "right_rect": right_r,
        "disp": disp,
        "depth_rect": depth_r,
        "points_left_cam": pts,
        "colors": cols,
        "left_cam": pair.left,
        "K_orig": pair.left.intrinsic,
        "calibration_source": calibration_source,
    }
    return result


def colorize_depth(depth: np.ndarray, vmin: float = 2.0, vmax: float = 60.0) -> np.ndarray:
    """Return BGR uint8 colorized depth using JET (invalid=0 -> black)."""
    d = depth.copy()
    d = np.clip(d, vmin, vmax)
    d = (d - vmin) / (vmax - vmin + 1e-6)
    d_uint = (d * 255).astype(np.uint8)
    color = cv2.applyColorMap(d_uint, cv2.COLORMAP_JET)
    color[depth <= 0] = (0, 0, 0)
    return color
