"""NuScenes data loader for stereo video sequences.

Provides easy access to synchronized stereo camera pairs (LEFT/RIGHT),
their calibrations, ego poses, and optional LiDAR reference for a scene sequence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from pyquaternion import Quaternion


# Phase 0 foundations: constants for dynamic camera pair selection + overlap metrics.
# (module-level for reference/override; used by NuScenesStereoLoader methods and CANDIDATE_PAIRS)
NUSCENES_CAM_IMG_SIZE: Tuple[int, int] = (1600, 900)  # (width, height) fallback for grid/bounds
DEFAULT_OVERLAP_DEPTHS: List[float] = [5.0, 15.0, 30.0, 50.0]
DEFAULT_GRID_SIZE: int = 32  # pixel step for uniform grid sampling (~1400 pts @ 1600x900; vectorized & fast)
OVERLAP_THRESH: float = 0.20
MIN_BASELINE_M: float = 0.30

# Phase 4: temporal smoothing / hysteresis constant for dynamic pair selection.
# Small bonus added to quality score of the pair selected on the previous sample
# (in the sequence walker). Reduces jittery switching between similar-scoring pairs
# across consecutive frames. Only affects _select_best_pair_for_sample (dynamic/"best_overlap" path).
# Controllable at construction time via the temporal_smoothing_bonus= kwarg (default from this const).
# Value in [0.05, 0.10] recommended; 0.0 disables effect. Does not affect fixed pair path at all.
TEMPORAL_SMOOTHING_BONUS: float = 0.08


@dataclass
class CameraData:
    """Per-camera data for one sample."""
    channel: str
    token: str
    filepath: str
    timestamp: int
    intrinsic: np.ndarray  # 3x3
    # sensor_from_ego (i.e. T_sensor_ego): 4x4, points in ego -> sensor
    sensor_from_ego: np.ndarray
    # ego_from_world: 4x4 (pose of ego in world)
    ego_from_world: np.ndarray
    calibrated_sensor_token: Optional[str] = None  # For proper devkit traceability
    # Convenience: world_from_ego
    @property
    def world_from_ego(self) -> np.ndarray:
        return np.linalg.inv(self.ego_from_world)

    @property
    def ego_from_sensor(self) -> np.ndarray:
        return np.linalg.inv(self.sensor_from_ego)

    @property
    def world_from_sensor(self) -> np.ndarray:
        return self.world_from_ego @ self.ego_from_sensor


@dataclass
class StereoPair:
    """One synchronized stereo frame."""
    sample_token: str
    timestamp: int
    left: CameraData
    right: CameraData
    lidar_token: Optional[str] = None
    lidar_filepath: Optional[str] = None

    # Phase 1 dynamic pair selection metadata (plan: docs/dynamic_pair_selection_plan.md).
    # Added at end with defaults for full backward compatibility with all existing
    # StereoPair(...) constructions (fixed-path and internal dummy). Lightweight
    # per-frame info for debugging/traceability; downstream code ignores them.
    selected_pair_channels: Optional[Tuple[str, str]] = None
    overlap_score: float = 0.0
    quality_score: float = 0.0
    selection_strategy: str = "fixed"


class NuScenesStereoLoader:
    """High-level loader for stereo sequences from nuScenes."""

    # Recommended stereo pair with good overlap + baseline
    DEFAULT_LEFT = "CAM_FRONT_LEFT"
    DEFAULT_RIGHT = "CAM_FRONT_RIGHT"

    # Phase 0: prioritized candidate pairs for dynamic selection (front-heavy as per plan)
    # Order reflects preference: strong forward first, then near-front, then side/rear fallbacks.
    # Only pairs expected to have >~20% overlap and >0.3m baseline in typical rigs.
    CANDIDATE_PAIRS: List[Tuple[str, str]] = [
        ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"),   # 1. original DEFAULT; wide forward baseline (~2.16m)
        ("CAM_FRONT_LEFT", "CAM_FRONT"),         # 2. ~1.02m baseline, excellent central overlap
        ("CAM_FRONT_RIGHT", "CAM_FRONT"),        # 3. symmetric ~1.29m
        ("CAM_FRONT_LEFT", "CAM_BACK_LEFT"),     # 4. left shoulder coverage (turns, close obstacles)
        ("CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"),   # 5.
        ("CAM_FRONT", "CAM_FRONT_LEFT"),         # 6. reverse ordering (asymmetric FOVs)
        ("CAM_FRONT", "CAM_FRONT_RIGHT"),        # 7.
        ("CAM_BACK_LEFT", "CAM_BACK_RIGHT"),     # 8. rear fallback
        ("CAM_BACK_LEFT", "CAM_BACK"),
        ("CAM_BACK_RIGHT", "CAM_BACK"),
    ]

    def __init__(self, dataroot: str = "v1.0-mini", version: str = "v1.0-mini", verbose: bool = False, temporal_smoothing_bonus: float = TEMPORAL_SMOOTHING_BONUS):
        self.dataroot = Path(dataroot).resolve()
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=verbose)
        self._sensor_cache: Dict[str, Dict[str, Any]] = {}
        # Simple in-memory cache for pair geometry (relative poses etc). Keyed by (left_ch, right_ch).
        # Populated on-demand for use_sensor_only=True (rig-constant, pose-independent).
        self._pair_geom_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._last_selected_pair: Optional[Tuple[str, str]] = None  # Phase 4: for temporal smoothing hysteresis in dynamic mode
        self._temporal_smoothing_bonus: float = temporal_smoothing_bonus
        self._ensure_pair_cache()  # Phase 1: hook for pre-warming (currently lazy in metrics)

    def list_scenes(self) -> List[Dict[str, Any]]:
        """Return list of {name, description, token, n_samples}."""
        scenes = []
        for scene in self.nusc.scene:
            nsamples = scene.get("nbr_samples", 0)
            scenes.append({
                "name": scene["name"],
                "description": scene["description"],
                "token": scene["token"],
                "n_samples": nsamples,
            })
        return scenes

    def get_scene(self, scene_name: str) -> Dict[str, Any]:
        for scene in self.nusc.scene:
            if scene["name"] == scene_name:
                return scene
        raise ValueError(f"Scene not found: {scene_name}. Available: {[s['name'] for s in self.nusc.scene]}")

    def get_stereo_sequence(
        self,
        scene_name: str,
        left_channel: str = DEFAULT_LEFT,
        right_channel: str = DEFAULT_RIGHT,
        max_frames: Optional[int] = None,
        only_keyframes: bool = True,
        pair_strategy: str = "fixed",  # "fixed" (BC default) or "best_overlap"/"dynamic" -> selector
    ) -> List[StereoPair]:
        """Return ordered list of StereoPair for the scene.

        Walks sample -> next chain starting from scene first_sample.
        For each sample finds the matching camera sample_data for left/right.
        Phase 1 extension: pair_strategy dispatches (default "fixed" preserves BC exactly).
        When "best_overlap", left/right args are ignored; uses _select_best_pair_for_sample
        per sample (dynamic pairs allowed in output list). only_keyframes + LiDAR still apply.
        References: docs/dynamic_pair_selection_plan.md Phase 1.
        """
        scene = self.get_scene(scene_name)
        sample_token = scene["first_sample_token"]

        pairs: List[StereoPair] = []
        count = 0
        while sample_token and (max_frames is None or count < max_frames):
            sample = self.nusc.get("sample", sample_token)
            if pair_strategy == "fixed":
                pair = self._build_stereo_pair(sample, left_channel, right_channel, only_keyframes)
            else:
                # Dynamic: selector builds (or None), then apply only_keyframes filter using
                # the chosen pair's sd tokens (kf check lives here to preserve exact behavior).
                dyn_pair = self._select_best_pair_for_sample(sample)
                if dyn_pair is None:
                    pair = None
                elif only_keyframes:
                    try:
                        lsd = self.nusc.get("sample_data", dyn_pair.left.token)
                        rsd = self.nusc.get("sample_data", dyn_pair.right.token)
                        if not lsd.get("is_key_frame", False) or not rsd.get("is_key_frame", False):
                            pair = None
                        else:
                            pair = dyn_pair
                    except Exception:
                        pair = None
                else:
                    pair = dyn_pair
            if pair is not None:
                pairs.append(pair)
                count += 1
            sample_token = sample.get("next")

        return pairs

    def get_dynamic_stereo_sequence(
        self,
        scene_name: str,
        max_frames: Optional[int] = None,
        only_keyframes: bool = True,
    ) -> List[StereoPair]:
        """Phase 1 public entry point for dynamic best-overlapping camera pair selection.

        Walks the sample chain exactly like get_stereo_sequence but per-sample invokes
        the selector (_select_best_pair_for_sample) instead of fixed left/right channels.
        Output List[StereoPair] may contain mixed pairs across frames (each with its
        selected_pair_channels, overlap_score, quality_score, selection_strategy="best_overlap").

        All other behaviors preserved: only_keyframes filtering, LiDAR attachment,
        caching, error modes, etc. Thin wrapper that reuses the extended get_stereo_sequence
        with pair_strategy dispatch (ensures single implementation of walker logic).

        Example:
            loader = get_default_loader()
            dyn_pairs = loader.get_dynamic_stereo_sequence("scene-0061", max_frames=5)
            for p in dyn_pairs:
                print(p.selected_pair_channels, p.quality_score)

        References: docs/dynamic_pair_selection_plan.md (API Changes, Phase 1 deliverables,
        Success Criteria). Backward compatible; fixed path unchanged.
        """
        return self.get_stereo_sequence(
            scene_name,
            max_frames=max_frames,
            only_keyframes=only_keyframes,
            pair_strategy="best_overlap",
            # left/right deliberately omitted (ignored under this strategy)
        )

    def _build_stereo_pair(
        self, sample: Dict[str, Any], left_ch: str, right_ch: str, only_keyframes: bool
    ) -> Optional[StereoPair]:
        """Find left and right sample_data for this keyframe sample."""
        data_tokens = sample["data"]  # dict channel -> sample_data_token

        left_sd_token = data_tokens.get(left_ch)
        right_sd_token = data_tokens.get(right_ch)
        if not left_sd_token or not right_sd_token:
            return None

        left_sd = self.nusc.get("sample_data", left_sd_token)
        right_sd = self.nusc.get("sample_data", right_sd_token)

        if only_keyframes and (not left_sd["is_key_frame"] or not right_sd["is_key_frame"]):
            return None

        # LiDAR for reference (LIDAR_TOP)
        lidar_sd_token = data_tokens.get("LIDAR_TOP")
        lidar_fp = None
        if lidar_sd_token:
            lidar_sd = self.nusc.get("sample_data", lidar_sd_token)
            lidar_fp = str(self.dataroot / lidar_sd["filename"])

        left_cd = self._make_camera_data(left_sd, left_ch)
        right_cd = self._make_camera_data(right_sd, right_ch)

        # Phase 1: populate selection metadata for fixed path too (for uniformity with dynamic).
        # Compute metrics (caches geometry); scores reflect the (fixed) pair quality.
        metrics = self.compute_pair_overlap_metrics(left_cd, right_cd, use_sensor_only=True)
        return StereoPair(
            sample_token=sample["token"],
            timestamp=left_sd["timestamp"],  # or avg
            left=left_cd,
            right=right_cd,
            lidar_token=lidar_sd_token,
            lidar_filepath=lidar_fp,
            selected_pair_channels=(left_ch, right_ch),
            overlap_score=metrics["overlap"],
            quality_score=metrics["quality"],
            selection_strategy="fixed",
        )

    def _make_camera_data(self, sd: Dict[str, Any], channel: str) -> CameraData:
        """Build CameraData with full transforms from a sample_data record."""
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])

        intrinsic = np.array(cs["camera_intrinsic"], dtype=np.float64)  # 3x3
        assert intrinsic.shape == (3, 3)

        # sensor_from_ego (T_sensor<-ego): rotation + translation from ego vehicle frame to sensor
        sensor_from_ego = transform_matrix(
            cs["translation"],
            Quaternion(cs["rotation"]),
            inverse=False,  # makes sensor_from_ego
        )

        # ego_from_world (pose of the ego at this timestamp)
        ego_from_world = transform_matrix(
            ep["translation"],
            Quaternion(ep["rotation"]),
            inverse=False,
        )

        filepath = str(self.dataroot / sd["filename"])

        return CameraData(
            channel=channel,
            token=sd["token"],
            filepath=filepath,
            timestamp=sd["timestamp"],
            intrinsic=intrinsic,
            sensor_from_ego=sensor_from_ego,
            ego_from_world=ego_from_world,
            calibrated_sensor_token=sd["calibrated_sensor_token"],
        )

    def _select_best_pair_for_sample(self, sample: Dict[str, Any]) -> Optional[StereoPair]:
        """Private Phase 1 helper (core of dynamic selection).

        Given a nuScenes sample dict:
        - Collect available channels from sample["data"].
        - Iterate self.CANDIDATE_PAIRS (in priority order), filter to pairs where both
          channels exist for this sample.
        - For each valid candidate: build temporary CameraData via _make_camera_data
          (reused), call compute_pair_overlap_metrics (hits cache after first), use
          'quality' (hybrid overlap*baseline_score) as score.
        - Return StereoPair for highest-scoring usable pair (usable=True preferred;
          if none usable, highest quality as fallback). Includes full LiDAR attachment
          (same as fixed path) + per-frame selection metadata.
        - If no candidate channels available at all: return None (caller may fallback).

        Temporal smoothing / hysteresis (Phase 4): a small bonus (controlled by
        TEMPORAL_SMOOTHING_BONUS or the temporal_smoothing_bonus= constructor kwarg)
        is added to the *selection score* (not the stored quality_score) of the pair
        that matches self._last_selected_pair (from the immediately prior sample in
        a dynamic sequence walk). This provides simple hysteresis to reduce jitter
        without affecting raw metrics or fixed-path behavior.

        - Only active for dynamic paths (pair_strategy != "fixed", get_dynamic_stereo_sequence).
        - Fixed path (_build_stereo_pair) and non-dynamic calls are completely unaffected.
        - _last_selected_pair is updated on every call to this selector (per-sample),
          so it chains across the sequence even if only_keyframes filtering skips some.
        - Bonus value 0.0 disables; default 0.08 (tunable 0.05-0.10 range suggested).
        - Stored quality_score / overlap_score on returned StereoPair remain the raw
          (unboosted) values from compute_pair_overlap_metrics.

        Keyframe filtering is NOT applied here (selector is sample-agnostic); it is
        handled by the sequence methods to keep only_keyframes behavior identical.
        Always uses use_sensor_only=True for speed (rig geometry cached).

        References: dynamic_pair_selection_plan.md sections 2 (Core Abstraction,
        Overlap/Quality Metric, Fallbacks), 3 (Phase 1: select_best_pair_for_sample);
        Phase 4 implementation of the temporal smoothing hook noted in earlier phases.
        """
        self._ensure_pair_cache()

        data_tokens: Dict[str, str] = sample.get("data", {})
        available = set(data_tokens.keys())

        # Build CameraData (and keep sd refs for ts) only for camera channels present
        # (at most 6; cheap; avoids repeated nusc.get in scoring loop).
        cam_datas: Dict[str, CameraData] = {}
        sd_for_cam: Dict[str, Dict[str, Any]] = {}
        for ch in available:
            if ch.startswith("CAM_"):
                sd_token = data_tokens[ch]
                sd = self.nusc.get("sample_data", sd_token)
                sd_for_cam[ch] = sd
                cam_datas[ch] = self._make_camera_data(sd, ch)

        # Score candidates
        candidates: List[Tuple[float, float, bool, str, str, CameraData, CameraData, Dict[str, Any]]] = []
        for left_ch, right_ch in self.CANDIDATE_PAIRS:
            if left_ch not in cam_datas or right_ch not in cam_datas:
                continue
            lcd = cam_datas[left_ch]
            rcd = cam_datas[right_ch]
            metrics = self.compute_pair_overlap_metrics(lcd, rcd, use_sensor_only=True)
            q = float(metrics.get("quality", 0.0))
            o = float(metrics.get("overlap", 0.0))
            usable = bool(metrics.get("usable", False))
            lsd = sd_for_cam[left_ch]
            candidates.append((q, o, usable, left_ch, right_ch, lcd, rcd, lsd))

        if not candidates:
            return None

        # Phase 4 temporal smoothing: apply small hysteresis bonus to prev pair's *selection score*.
        # Uses raw quality from metrics for storage; bonus only tips the argmax here.
        # (Only reached in dynamic paths; fixed path never calls this.)
        bonus = self._temporal_smoothing_bonus
        last_pair = self._last_selected_pair
        def _selection_key(item: Tuple[float, float, bool, str, str, CameraData, CameraData, Dict[str, Any]]) -> Tuple[bool, float]:
            q, _, usable, lch, rch, *_ = item
            boost = bonus if (last_pair is not None and (lch, rch) == last_pair) else 0.0
            return (usable, q + boost)

        # Prefer usable (True first), then highest (boosted) quality desc.
        candidates.sort(key=_selection_key, reverse=True)
        top_q, top_o, _, lch, rch, lcd, rcd, lsd = candidates[0]

        # Record for next sample's temporal smoothing (hysteresis state on loader instance).
        self._last_selected_pair = (lch, rch)

        # LiDAR attachment (independent of chosen pair; same for sample)
        lidar_sd_token = data_tokens.get("LIDAR_TOP")
        lidar_fp = None
        if lidar_sd_token:
            lidar_sd = self.nusc.get("sample_data", lidar_sd_token)
            lidar_fp = str(self.dataroot / lidar_sd["filename"])

        return StereoPair(
            sample_token=sample["token"],
            timestamp=lsd["timestamp"],
            left=lcd,
            right=rcd,
            lidar_token=lidar_sd_token,
            lidar_filepath=lidar_fp,
            selected_pair_channels=(lch, rch),
            overlap_score=top_o,
            quality_score=top_q,
            selection_strategy="best_overlap",
        )

    def load_image(self, cam: CameraData) -> np.ndarray:
        """Load BGR image as uint8 HxWx3 (OpenCV convention)."""
        import cv2
        img = cv2.imread(cam.filepath, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {cam.filepath}")
        return img

    def load_lidar_points(self, pair: StereoPair, in_world: bool = True) -> np.ndarray:
        """Load LiDAR pointcloud (N, 3 or 4) optionally transformed to world frame."""
        if not pair.lidar_filepath or not os.path.exists(pair.lidar_filepath):
            raise FileNotFoundError(f"LiDAR file missing: {pair.lidar_filepath}")

        pc = LidarPointCloud.from_file(pair.lidar_filepath)
        points = pc.points[:3].T.copy()  # N x 3

        if in_world:
            # Need ego pose for this lidar sample_data
            # We already have it from the pair? But lidar may have slightly different timestamp.
            # For simplicity, use the sample's ego (close enough for demo)
            # Better: load the actual lidar ego_pose
            lidar_sd = self.nusc.get("sample_data", pair.lidar_token)
            ep = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            ego_from_world = transform_matrix(
                ep["translation"], Quaternion(ep["rotation"]), inverse=False
            )
            # Lidar sensor extrinsics
            lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
            lidar_from_ego = transform_matrix(
                lidar_cs["translation"], Quaternion(lidar_cs["rotation"]), inverse=False
            )
            world_from_lidar = np.linalg.inv(ego_from_world) @ np.linalg.inv(lidar_from_ego)
            # Transform points (homogeneous)
            pts_h = np.hstack([points, np.ones((len(points), 1))])
            points = (world_from_lidar @ pts_h.T).T[:, :3]
        return points

    def project_lidar_to_image(
        self, points_world: np.ndarray, cam: CameraData, img_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project world points into camera image plane. Returns (uv, depths) valid only.

        uv: (M, 2) float, depths: (M,) in meters.
        """
        # world -> sensor
        pts_sensor = (cam.sensor_from_ego @ cam.ego_from_world @ 
                      np.hstack([points_world, np.ones((len(points_world), 1))]).T)[:3]

        # filter behind camera
        depths = pts_sensor[2]
        mask = depths > 0.1
        pts_sensor = pts_sensor[:, mask]
        depths = depths[mask]

        # project
        uv = view_points(pts_sensor, cam.intrinsic, normalize=True)[:2].T  # M x 2

        if img_shape is not None:
            h, w = img_shape
            valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            uv = uv[valid]
            depths = depths[valid]

        return uv, depths

    def _generate_image_grid_rays(
        self, intrinsic: np.ndarray, img_size: Tuple[int, int], grid_step: int = DEFAULT_GRID_SIZE
    ) -> np.ndarray:
        """Small internal helper: vectorized generation of backprojection rays for uniform pixel grid.

        Returns (3, N) dirs such that 3D point in cam coords at Z-depth d is: pts = d * dirs.
        Uses K^{-1} @ [u, v, 1]^T formulation (standard for metric depth along Z).
        Efficient numpy meshgrid + matmul; no loops. Grid step controls density (default 32px).
        """
        width, height = img_size
        us = np.arange(0, width, grid_step, dtype=np.float64)
        vs = np.arange(0, height, grid_step, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)  # shape (nrows, ncols); ravel order is fine (row-major)
        n = uu.size
        uv1 = np.stack([uu.ravel(), vv.ravel(), np.ones(n, dtype=np.float64)], axis=0)  # 3 x N
        Kinv = np.linalg.inv(intrinsic.astype(np.float64))
        dirs = Kinv @ uv1
        return dirs

    def _ensure_pair_cache(self) -> None:
        """Phase 1 small hook (per task): ensure _pair_geom_cache ready for CANDIDATE_PAIRS.
        Currently a no-op (metrics lazily populate on first use via use_sensor_only=True).
        Called from __init__ and can be called by selector. Future pre-warm possible.
        References: dynamic_pair_selection_plan.md (cache at init or first use).
        """
        pass  # Lazy population is sufficient and keeps __init__ fast.

    def compute_pair_overlap_metrics(
        self,
        cam_l: "CameraData",
        cam_r: "CameraData",
        depths: Optional[List[float]] = None,
        grid_step: int = DEFAULT_GRID_SIZE,
        use_sensor_only: bool = True,
        overlap_thresh: float = OVERLAP_THRESH,
        min_baseline: float = MIN_BASELINE_M,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Compute hybrid multi-depth projected-grid overlap + baseline metrics for any camera pair.

        Follows the exact research-recommended approach:
        - Multi-depth (default [5,15,30,50]m) uniform image grid rays lifted from cam_l via intrinsics.
        - Transform via relative pose (full world+ego or sensor_only approx for rig-static speed).
        - Project to cam_r; fraction inside bounds + positive depth, averaged over depths -> 'overlap'.
        - 'baseline' = ||t|| of right_from_left.
        - 'quality' = overlap * f(baseline) hybrid (mild preference for ~0.8-2m baselines typical in rigs).
        - 'usable' flag based on thresholds.
        - Caches sensor-only geometry keyed by channel pair ( (l,r) ) at loader level.
        - Pure numpy + existing view_points; <1ms/pair typical. No images loaded.
        - Supports use_sensor_only=True (recommended default for same-sample calls).

        Returns at minimum keys: {'overlap': float, 'baseline': float, 'quality': float, 'usable': bool}.
        May also include 'rectified_overlap' (from thin helper).

        Backward compatible addition; does not affect any existing public method.
        """
        if depths is None:
            depths = DEFAULT_OVERLAP_DEPTHS[:]
        if img_size is None:
            img_size = NUSCENES_CAM_IMG_SIZE
        width, height = img_size

        # --- 1. Relative pose (right_from_left) + baseline + cache ---
        key = (cam_l.channel, cam_r.channel)
        if use_sensor_only and key in self._pair_geom_cache:
            geom = self._pair_geom_cache[key]
            R = geom["R"].copy()
            t = geom["t"].copy()
            baseline = float(geom["baseline"])
        else:
            if use_sensor_only:
                # Rig-constant approximation (ignores ~10-30ms timestamp diff in same sample's ego_poses).
                # Xr_sensor = right.sensor_from_ego @ left.ego_from_sensor @ Xl_sensor
                rf_l = cam_r.sensor_from_ego @ cam_l.ego_from_sensor
                R = rf_l[:3, :3].copy()
                t = rf_l[:3, 3].copy()
            else:
                # Full transform using actual per-camera ego poses at capture time
                rf_l = (
                    cam_r.sensor_from_ego
                    @ cam_r.ego_from_world
                    @ cam_l.world_from_sensor
                )
                R = rf_l[:3, :3].copy()
                t = rf_l[:3, 3].copy()
            baseline = float(np.linalg.norm(t))
            if use_sensor_only:
                self._pair_geom_cache[key] = {
                    "R": R.copy(),
                    "t": t.copy(),
                    "baseline": baseline,
                }

        # --- 2. Grid rays from left ---
        dirs = self._generate_image_grid_rays(cam_l.intrinsic, img_size, grid_step)
        N = dirs.shape[1]
        if N == 0:
            overlap = 0.0
        else:
            # --- 3. Multi-depth projection + in-bounds count (vectorized over grid) ---
            K_r = cam_r.intrinsic.astype(np.float64)
            fracs: List[float] = []
            for d in depths:
                pts_l = float(d) * dirs  # 3 x N  (Z = d convention)
                # Transform to right sensor frame
                pts_r = R @ pts_l + t.reshape(3, 1)
                z_r = pts_r[2]
                pos_mask = z_r > 0.1
                # Project (reuse nuscenes view_points for consistency with project_lidar_to_image)
                uv_r = view_points(pts_r, K_r, normalize=True)[:2]  # 2 x N
                u, v = uv_r[0], uv_r[1]
                in_bounds = (
                    (u >= 0)
                    & (u < width)
                    & (v >= 0)
                    & (v < height)
                    & pos_mask
                )
                frac = float(np.mean(in_bounds)) if N > 0 else 0.0
                fracs.append(frac)
            overlap = float(np.mean(fracs)) if fracs else 0.0

        # --- 4. Hybrid quality ---
        b = baseline
        if b < min_baseline:
            bscore = 0.0
        elif b > 3.5:
            bscore = 0.65
        else:
            # Mild preference: peak near 1.3m (common good stereo baseline); floor 0.6
            bscore = float(np.clip(1.0 - 0.2 * abs(b - 1.3), 0.6, 1.0))
        quality = float(overlap * bscore)

        usable = bool((overlap >= overlap_thresh) and (baseline >= min_baseline))

        result: Dict[str, Any] = {
            "overlap": float(overlap),
            "baseline": float(baseline),
            "quality": float(quality),
            "usable": usable,  # bool as specified
        }

        # --- 5. Optional rectified overlap (thin, safe) ---
        try:
            rect_pct = self._compute_rectified_overlap_pct(cam_l, cam_r, img_size=img_size)
            result["rectified_overlap"] = float(rect_pct)
        except Exception:
            result["rectified_overlap"] = -1.0

        return result

    def _compute_rectified_overlap_pct(
        self,
        cam_l: "CameraData",
        cam_r: "CameraData",
        img_size: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Thin private helper for rectified-ROI overlap % (companion to grid metric).

        Safely lazy-imports classical_stereo (avoids import-time circularity: classical imports
        CameraData/StereoPair/get_stereo_calibration from here; runtime import succeeds post-module-load).
        Uses stereoRectify ROIs intersection / image area as proxy for post-rect valid overlap.
        Returns [0,1] or 0.0 on failure (missing cv2, bad data, etc). Non-intrusive.
        """
        if img_size is None:
            img_size = NUSCENES_CAM_IMG_SIZE
        width, height = img_size
        total_area = float(width * height)
        if total_area <= 0.0:
            return 0.0
        try:
            import pipeline.classical_stereo as cs  # lazy, runtime only
            # Minimal valid StereoPair (optionals default to None)
            dummy_pair = StereoPair(
                sample_token="phase0_metric_dummy",
                timestamp=int(cam_l.timestamp),
                left=cam_l,
                right=cam_r,
            )
            calib = get_stereo_calibration(dummy_pair)
            rect = cs.compute_rectification(
                calib["K1"], calib["K2"], calib["R"], calib["t"], (width, height), alpha=0.0
            )
            r1 = rect.roi1  # (x, y, w, h)
            r2 = rect.roi2
            # Handle OpenCV quirk: (0,0,0,0) or zero-area ROI with alpha=0 conventionally means "full image valid"
            def _expand_roi(roi, fw, fh):
                if len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0:
                    return (0, 0, fw, fh)
                return roi
            r1 = _expand_roi(r1, width, height)
            r2 = _expand_roi(r2, width, height)
            ix1 = max(r1[0], r2[0])
            iy1 = max(r1[1], r2[1])
            ix2 = min(r1[0] + r1[2], r2[0] + r2[2])
            iy2 = min(r1[1] + r1[3], r2[1] + r2[3])
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter_area = float(iw * ih)
            return inter_area / total_area
        except Exception:
            return 0.0


def get_default_loader() -> NuScenesStereoLoader:
    """Convenience for the local mini dataset."""
    return NuScenesStereoLoader(dataroot="v1.0-mini", version="v1.0-mini", verbose=False)


def get_stereo_calibration(pair: StereoPair) -> Dict[str, Any]:
    """Return stereo calibration parameters derived properly from nuScenes devkit.

    This is the recommended "proper" way (using official calibrated_sensor records)
    to obtain inputs for rectification.

    Returns dict with: K1, K2, R, t, baseline, and source tokens.
    """
    left = pair.left
    right = pair.right

    # right_from_left: X_right = R @ X_left + t
    right_from_left = (
        right.sensor_from_ego
        @ right.ego_from_world
        @ left.world_from_sensor
    )
    R = right_from_left[:3, :3]
    t = right_from_left[:3, 3].copy()

    return {
        "K1": left.intrinsic.copy(),
        "K2": right.intrinsic.copy(),
        "R": R.copy(),
        "t": t.copy(),
        "baseline": float(np.linalg.norm(t)),
        "left_calibrated_sensor_token": left.calibrated_sensor_token,
        "right_calibrated_sensor_token": right.calibrated_sensor_token,
        "source": "devkit_calibrated_sensor",
    }
