"""Multi-frame dense fusion using TSDF (when Open3D is available).

Provides a clean wrapper around Open3D's ScalableTSDFVolume for integrating
per-frame depth (classical or neural) + known poses from nuScenes.

Falls back gracefully when open3d is not installed (returns empty results).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False
    o3d = None  # type: ignore


@dataclass
class TSDFResult:
    """Result of TSDF fusion for one method (classical or neural)."""
    points: np.ndarray          # (N, 3) surface points (or empty)
    colors: np.ndarray          # (N, 3) uint8 or empty
    mesh_vertices: Optional[np.ndarray] = None
    mesh_triangles: Optional[np.ndarray] = None
    has_surface: bool = False
    voxel_size: float = 0.08
    num_integrated_frames: int = 0


class TSDFIntegrator:
    """Wrapper for Open3D TSDF integration of depth sequences with known poses.

    Usage:
        integrator = TSDFIntegrator(voxel_size=0.08, sdf_trunc=0.3)
        for depth, color, world_from_cam, K in sequence:
            integrator.integrate(depth, color, world_from_cam, K)
        result = integrator.extract()
    """

    def __init__(
        self,
        voxel_size: float = 0.08,
        sdf_trunc: float = 0.3,
        min_depth: float = 2.0,
        max_depth: float = 80.0,
    ):
        self.voxel_size = float(voxel_size)
        self.sdf_trunc = float(sdf_trunc)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self._volume = None
        self._integrated = 0
        self._color_type = None

    def _ensure_volume(self, img_shape: Tuple[int, int]):
        if not HAS_OPEN3D:
            return
        if self._volume is not None:
            return

        h, w = img_shape
        # Use a reasonable intrinsic (will be overridden per integrate call)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w, h, fx=1000, fy=1000, cx=w/2, cy=h/2
        )

        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self._intrinsic = intrinsic  # placeholder

    def integrate(
        self,
        depth: np.ndarray,           # (H, W) float32 meters
        color_bgr: np.ndarray,       # (H, W, 3) uint8
        world_from_cam: np.ndarray,  # 4x4
        intrinsic: np.ndarray,       # 3x3 K
    ) -> bool:
        """Integrate one depth+color frame. Returns True on success."""
        if not HAS_OPEN3D:
            return False

        if depth is None or depth.size == 0:
            return False

        h, w = depth.shape[:2]
        self._ensure_volume((h, w))

        # Prepare Open3D images
        color_rgb = color_bgr[..., ::-1].astype(np.uint8)  # BGR -> RGB
        color_o3d = o3d.geometry.Image(color_rgb)

        # Depth must be float32 in meters for create_from_color_and_depth
        depth_f32 = depth.astype(np.float32)
        depth_o3d = o3d.geometry.Image(depth_f32)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=1.0,          # already in meters
            depth_trunc=self.max_depth,
            convert_rgb_to_intensity=False,
        )

        # Build 3x3 intrinsic for this frame
        K = intrinsic
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        cam_intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        # Open3D expects extrinsic as camera-to-world? Actually for integrate it is
        # the transformation from world to camera (i.e. extrinsic matrix in many conventions).
        # We have world_from_cam → cam_from_world = inv(world_from_cam)
        cam_from_world = np.linalg.inv(world_from_cam)

        self._volume.integrate(rgbd, cam_intr, cam_from_world)
        self._integrated += 1
        return True

    def extract(self) -> TSDFResult:
        """Extract surface point cloud (and optionally mesh)."""
        if not HAS_OPEN3D or self._volume is None or self._integrated == 0:
            return TSDFResult(
                points=np.empty((0, 3), np.float32),
                colors=np.empty((0, 3), np.uint8),
                has_surface=False,
                voxel_size=self.voxel_size,
                num_integrated_frames=self._integrated,
            )

        # Extract colored point cloud
        pcd = self._volume.extract_point_cloud()
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = (np.asarray(pcd.colors, dtype=np.float32) * 255).astype(np.uint8)

        # Also try to extract a mesh (nice for users)
        mesh = None
        try:
            mesh = self._volume.extract_triangle_mesh()
        except Exception:
            pass

        mesh_verts = None
        mesh_tris = None
        if mesh is not None and len(mesh.vertices) > 0:
            mesh_verts = np.asarray(mesh.vertices, dtype=np.float32)
            mesh_tris = np.asarray(mesh.triangles, dtype=np.int32)

        return TSDFResult(
            points=points,
            colors=colors,
            mesh_vertices=mesh_verts,
            mesh_triangles=mesh_tris,
            has_surface=len(points) > 0,
            voxel_size=self.voxel_size,
            num_integrated_frames=self._integrated,
        )


def fuse_sequence_with_tsdf(
    depths: List[np.ndarray],
    colors: List[np.ndarray],
    world_from_cams: List[np.ndarray],
    intrinsics: List[np.ndarray],
    *,
    voxel_size: float = 0.08,
    sdf_trunc: float = 0.3,
) -> TSDFResult:
    """Convenience: fuse a list of frames in one call."""
    if not depths:
        return TSDFResult(
            points=np.empty((0, 3), np.float32),
            colors=np.empty((0, 3), np.uint8),
        )

    integrator = TSDFIntegrator(voxel_size=voxel_size, sdf_trunc=sdf_trunc)
    for d, c, T, K in zip(depths, colors, world_from_cams, intrinsics):
        integrator.integrate(d, c, T, K)
    return integrator.extract()
