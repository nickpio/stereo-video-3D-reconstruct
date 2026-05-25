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


class NuScenesStereoLoader:
    """High-level loader for stereo sequences from nuScenes."""

    # Recommended stereo pair with good overlap + baseline
    DEFAULT_LEFT = "CAM_FRONT_LEFT"
    DEFAULT_RIGHT = "CAM_FRONT_RIGHT"

    def __init__(self, dataroot: str = "v1.0-mini", version: str = "v1.0-mini", verbose: bool = False):
        self.dataroot = Path(dataroot).resolve()
        self.version = version
        self.nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=verbose)
        self._sensor_cache: Dict[str, Dict[str, Any]] = {}

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
    ) -> List[StereoPair]:
        """Return ordered list of StereoPair for the scene.

        Walks sample -> next chain starting from scene first_sample.
        For each sample finds the matching camera sample_data for left/right.
        """
        scene = self.get_scene(scene_name)
        sample_token = scene["first_sample_token"]

        pairs: List[StereoPair] = []
        count = 0
        while sample_token and (max_frames is None or count < max_frames):
            sample = self.nusc.get("sample", sample_token)
            pair = self._build_stereo_pair(sample, left_channel, right_channel, only_keyframes)
            if pair is not None:
                pairs.append(pair)
                count += 1
            sample_token = sample.get("next")

        return pairs

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

        return StereoPair(
            sample_token=sample["token"],
            timestamp=left_sd["timestamp"],  # or avg
            left=left_cd,
            right=right_cd,
            lidar_token=lidar_sd_token,
            lidar_filepath=lidar_fp,
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
