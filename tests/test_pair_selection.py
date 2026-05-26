"""Tests for Phase 0/1 dynamic pair selection foundations.

Covers the new `compute_pair_overlap_metrics` (and supporting
CANDIDATE_PAIRS / cache / helpers) added to `pipeline/nuscenes_loader.py`.

These tests use real nuScenes mini data (scene-0061) for realism while
remaining fast and deterministic. They form the foundation for future
dynamic selector logic (select_best_pair_for_sample, get_dynamic_stereo_sequence, etc.).

Run:
  python -m pytest tests/test_pair_selection.py -q
  # or standalone:
  python tests/test_pair_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

# Support both `python -m pytest ...` (with PYTHONPATH or installed package)
# and direct `python tests/test_pair_selection.py` (standalone).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.nuscenes_loader import (
    CameraData,
    NuScenesStereoLoader,
    get_default_loader,
)


# --------------------------------------------------------------------------- #
# Helpers (test-only; keep minimal and local to this file)
# --------------------------------------------------------------------------- #

def _make_synthetic_camera_pair(
    yaw_deg: float = 90.0,
    baseline_m: float = 1.0,
    img_w: int = 1600,
    img_h: int = 900,
) -> Tuple[CameraData, CameraData]:
    """Create two minimal CameraData instances with controlled relative pose.

    Used for edge-case testing (e.g. near-zero overlap). Does not rely on
    real dataset. Intrinsics are plausible pinhole; poses use simple yaw
    rotation + lateral translation so that grid rays from left mostly miss
    the right camera frustum.
    """
    fx = fy = 1260.0  # approx nuScenes focal
    cx, cy = img_w / 2.0, img_h / 2.0
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    c, s = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    # right_from_left rotation (sensor frame) + translation
    # We encode it in the *sensor_from_ego* of the right camera while
    # keeping ego poses identity (simplest model for relative-only test).
    Rmat = np.array(
        [[c, 0.0, s, baseline_m], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )

    # Unique channel names per call (incorporating yaw) so that the loader's
    # _pair_geom_cache (keyed only by channel tuple) does not accidentally
    # return a previous synthetic's geometry when the test exercises multiple
    # different poses. Real camera channels are globally unique anyway.
    chan_l = f"SYNTH_LEFT_{yaw_deg:g}"
    chan_r = f"SYNTH_RIGHT_{yaw_deg:g}"

    cam_l = CameraData(
        channel=chan_l,
        token="synth_l",
        filepath="",
        timestamp=0,
        intrinsic=K.copy(),
        sensor_from_ego=np.eye(4, dtype=np.float64),
        ego_from_world=np.eye(4, dtype=np.float64),
    )
    cam_r = CameraData(
        channel=chan_r,
        token="synth_r",
        filepath="",
        timestamp=0,
        intrinsic=K.copy(),
        sensor_from_ego=Rmat.copy(),
        ego_from_world=np.eye(4, dtype=np.float64),
    )
    return cam_l, cam_r


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def loader_and_sample_cams() -> Tuple[NuScenesStereoLoader, Dict[str, CameraData]]:
    """Real NuScenesStereoLoader + one sample's 6 CameraData objects.

    Uses the first keyframe sample from scene-0061 (well-known test scene
    in the v1.0-mini dataset that ships with the repo).

    Loads the six cameras exactly once (module scope) for speed:
        CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT,
        CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT.

    All downstream tests receive the *same* calibrated rig + slightly
    time-skewed ego poses that the real metric must handle.

    Why module scope: The nuscenes JSON parsing + calibrated_sensor/ego_pose
    lookups are the dominant cost; per-function reload would be wasteful
    for 5+ fast geometric tests.
    """
    loader = get_default_loader()
    scene = loader.get_scene("scene-0061")
    sample = loader.nusc.get("sample", scene["first_sample_token"])

    cam_channels = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    cams: Dict[str, CameraData] = {}
    for ch in cam_channels:
        sd_token = sample["data"][ch]
        sd = loader.nusc.get("sample_data", sd_token)
        cams[ch] = loader._make_camera_data(sd, ch)
    return loader, cams


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_metrics_basic_real_data(loader_and_sample_cams: Tuple[NuScenesStereoLoader, Dict[str, CameraData]]) -> None:
    """Exercise compute_pair_overlap_metrics on the original DEFAULT pair plus two
    other sensible candidates using real calibrated data from scene-0061.

    Asserts:
      - Required keys present (overlap, baseline, quality, usable) + optional rectified.
      - overlap in [0, 1]
      - baseline > 0.3 m (MIN_BASELINE_M) for these good pairs
      - usable is Python bool
      - quality is reasonable (non-negative, <= 1.0)

    Why this matters for the dynamic selector:
    The future selector will call this exact method (often with use_sensor_only=True)
    on every CANDIDATE_PAIR for every sample. If the metric returns garbage,
    NaNs, or inverted usability on real rig data, pair selection will be random
    or always fall back to DEFAULT. This test is the primary guardrail.
    """
    loader, cams = loader_and_sample_cams

    default = (NuScenesStereoLoader.DEFAULT_LEFT, NuScenesStereoLoader.DEFAULT_RIGHT)
    other_pairs = [
        ("CAM_FRONT_LEFT", "CAM_FRONT"),
        ("CAM_FRONT_RIGHT", "CAM_FRONT"),
    ]
    pairs_to_test = [default] + other_pairs

    for left_ch, right_ch in pairs_to_test:
        cl = cams[left_ch]
        cr = cams[right_ch]
        metrics = loader.compute_pair_overlap_metrics(cl, cr, use_sensor_only=True)

        # Keys
        required = {"overlap", "baseline", "quality", "usable"}
        assert required.issubset(metrics.keys()), f"missing keys in {metrics.keys()}"
        assert "rectified_overlap" in metrics  # bonus from the helper

        # Ranges & types (realistic expectations on good forward-ish pairs)
        overlap = metrics["overlap"]
        baseline = metrics["baseline"]
        quality = metrics["quality"]
        usable = metrics["usable"]

        assert 0.0 <= overlap <= 1.0, f"overlap {overlap} out of [0,1]"
        assert baseline > 0.3, f"baseline {baseline} not > 0.3m for a candidate pair"
        assert isinstance(usable, bool), f"usable must be bool, got {type(usable)}"
        assert 0.0 <= quality <= 1.0, f"quality {quality} unreasonable"
        # rectified is either [0,1] or sentinel -1.0 on failure path
        rect = metrics["rectified_overlap"]
        assert -1.0 <= rect <= 1.0


def test_metrics_sensor_vs_full(
    loader_and_sample_cams: Tuple[NuScenesStereoLoader, Dict[str, CameraData]]
) -> None:
    """Compare use_sensor_only=True (rig-static cache path) vs False (full per-timestamp
    ego-pose path) on the exact same real CameraData pair.

    The two results must be close (overlap delta < 0.05, baseline delta < 0.3 m)
    but are allowed (and expected) to differ slightly because the six cameras in
    one nuScenes sample have 10-30 ms timestamp skew → different ego_poses.

    Why this matters for the dynamic selector:
    The production selector will prefer the fast sensor_only path for speed
    (and to populate the _pair_geom_cache). This test guarantees the
    approximation is faithful enough that selection decisions will not flip
    spuriously between the two modes. It also exercises the cache population
    side-effect.
    """
    loader, cams = loader_and_sample_cams
    cl = cams["CAM_FRONT_LEFT"]
    cr = cams["CAM_FRONT_RIGHT"]

    m_sensor = loader.compute_pair_overlap_metrics(cl, cr, use_sensor_only=True)
    m_full = loader.compute_pair_overlap_metrics(cl, cr, use_sensor_only=False)

    # Must be semantically close despite timestamp skew
    assert abs(m_sensor["overlap"] - m_full["overlap"]) < 0.05
    assert abs(m_sensor["baseline"] - m_full["baseline"]) < 0.30
    # Usability decision must be identical for a good pair
    assert m_sensor["usable"] == m_full["usable"]

    # Sanity: the cache should now contain the entry from the sensor_only call
    key = ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
    assert key in loader._pair_geom_cache


def test_candidate_pairs_defined_and_sensible() -> None:
    """Static sanity check on the CANDIDATE_PAIRS class attribute that the dynamic
    selector will iterate.

    Asserts:
      - At least 5 entries (current design has 10)
      - Contains the original DEFAULT (CAM_FRONT_LEFT, CAM_FRONT_RIGHT) for BC
      - Every entry is a 2-tuple of distinct, known valid camera channels

    Why this matters for the dynamic selector:
    The list is the search space. An empty / corrupted / duplicate-containing
    list would either make selection a no-op or cause KeyErrors when indexing
    into per-sample data dicts. This test is the contract test for the
    "prioritized candidate pairs" section of the plan.
    """
    pairs = NuScenesStereoLoader.CANDIDATE_PAIRS
    assert len(pairs) >= 5, "CANDIDATE_PAIRS must contain enough options for dynamic choice"

    default_pair = (
        NuScenesStereoLoader.DEFAULT_LEFT,
        NuScenesStereoLoader.DEFAULT_RIGHT,
    )
    assert default_pair in pairs, "original DEFAULT must remain in CANDIDATE_PAIRS for backward compat"

    valid_channels = {
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    }

    for p in pairs:
        assert isinstance(p, tuple) and len(p) == 2, f"bad pair entry: {p}"
        left, right = p
        assert left in valid_channels and right in valid_channels, f"unknown channel in {p}"
        assert left != right, f"self-pair not allowed: {p}"


def test_low_overlap_pair(loader_and_sample_cams: Tuple[NuScenesStereoLoader, Dict[str, CameraData]]) -> None:
    """Verify that compute_pair_overlap_metrics correctly flags geometrically poor
    pairs as low-overlap / not usable.

    Uses a synthetic pair (90° yaw, 1 m baseline) constructed to have essentially
    zero frustum overlap under the multi-depth grid projection. Also exercises
    the baseline < MIN_BASELINE_M rejection path with a second synthetic.

    Real data pairs from the 6 cams are all reasonably usable; a pure mock is
    required for the "bad" case.

    Why this matters for the dynamic selector:
    The selector must be able to *reject* candidates (e.g. during a hard turn
    a side-rear pair may momentarily have terrible overlap). If the metric
    never returns usable=False, the selector has no signal and will happily
    pick garbage pairs. This test protects the "only consider pairs with
    reasonable geometric overlap" requirement from the plan.
    """
    loader, _ = loader_and_sample_cams  # we only need a live loader instance

    # Case 1: 90° relative yaw → rays from left miss right camera almost completely
    cl, cr = _make_synthetic_camera_pair(yaw_deg=90.0, baseline_m=1.0)
    m = loader.compute_pair_overlap_metrics(cl, cr, use_sensor_only=True)
    assert m["overlap"] < 0.10, f"expected near-zero overlap for 90deg pair, got {m['overlap']}"
    assert m["usable"] is False

    # Case 2: tiny baseline (even with perfect overlap the pair is rejected)
    cl2, cr2 = _make_synthetic_camera_pair(yaw_deg=0.0, baseline_m=0.05)
    m2 = loader.compute_pair_overlap_metrics(cl2, cr2, use_sensor_only=True)
    assert m2["baseline"] < 0.30
    assert m2["usable"] is False
    assert m2["quality"] == 0.0  # explicit in implementation


def test_cache_population(loader_and_sample_cams: Tuple[NuScenesStereoLoader, Dict[str, CameraData]]) -> None:
    """Confirm that successful calls with use_sensor_only=True populate the
    internal _pair_geom_cache as documented.

    After two distinct channel-pair queries the cache must contain the
    expected (left, right) keys with the proper sub-dict (R, t, baseline).

    Why this matters for the dynamic selector:
    Phase 0 explicitly calls for "Precompute rig geometry cache". All future
    dynamic loading code will rely on this cache for speed (O(1) lookup after
    first touch). A broken cache would cause repeated expensive matrix math
    and/or wrong results on subsequent frames. This test is the regression
    guard for the caching contract.
    """
    loader, cams = loader_and_sample_cams

    # Use two different pairs; clear any prior state for isolation (harmless in practice)
    loader._pair_geom_cache.clear()

    p1 = ("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
    p2 = ("CAM_FRONT_LEFT", "CAM_FRONT")

    loader.compute_pair_overlap_metrics(cams[p1[0]], cams[p1[1]], use_sensor_only=True)
    loader.compute_pair_overlap_metrics(cams[p2[0]], cams[p2[1]], use_sensor_only=True)

    assert p1 in loader._pair_geom_cache
    assert p2 in loader._pair_geom_cache

    entry = loader._pair_geom_cache[p1]
    assert isinstance(entry, dict)
    assert {"R", "t", "baseline"} <= entry.keys()
    assert isinstance(entry["R"], np.ndarray) and entry["R"].shape == (3, 3)
    assert isinstance(entry["t"], np.ndarray) and entry["t"].shape == (3,)
    assert float(entry["baseline"]) > 0.3


def test_rectified_overlap_smoke(loader_and_sample_cams: Tuple[NuScenesStereoLoader, Dict[str, CameraData]]) -> None:
    """Smoke test for the rectified-overlap companion metric (the private
    _compute_rectified_overlap_pct helper exposed via the public metrics dict).

    We only assert that the key is present and in a sane numeric range; we do
    not assert exact values (those depend on OpenCV stereoRectify + alpha=0
    ROI semantics).

    Why this matters for the dynamic selector:
    The original research plan listed "post-rectification valid ROI area %"
    as a secondary / hybrid term. Even though the primary implementation uses
    the grid-projection overlap, the rectified value is still computed on
    every call. A crash here (import cycle, missing get_stereo_calibration,
    cv2 failure) would break the entire metrics path for future dynamic code.
    """
    loader, cams = loader_and_sample_cams
    cl = cams["CAM_FRONT_LEFT"]
    cr = cams["CAM_FRONT_RIGHT"]

    m = loader.compute_pair_overlap_metrics(cl, cr, use_sensor_only=True)

    assert "rectified_overlap" in m
    val = m["rectified_overlap"]
    assert isinstance(val, (int, float))
    # On success [0,1]; on any failure path the implementation returns -1.0 sentinel
    assert -1.0 <= val <= 1.0


# --------------------------------------------------------------------------- #
# Standalone entry point (supports `python tests/test_pair_selection.py`)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # When executed directly, delegate to pytest for collection + reporting.
    # Keeps behavior identical to `python -m pytest ... -q`.
    import pytest as _pytest

    # -q = quiet, --tb=line for compact failures, exit status propagated
    sys.exit(_pytest.main([str(Path(__file__).resolve()), "-q", "--tb=line"]))
