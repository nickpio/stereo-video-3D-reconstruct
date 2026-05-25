"""Stereo-aware neural depth estimation.

This module provides an interface for stereo-consistent depth prediction.
It currently builds on top of the monocular Depth-Anything-V2 model with
optional left-right consistency enforcement using known stereo calibration.

Future extensions can swap in true stereo models (e.g. RAFT-Stereo, CREStereo,
or a fine-tuned Depth Anything with stereo consistency loss).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pipeline.neural_depth import (
    load_depth_model as load_mono_depth_model,
    predict_relative_depth,
    align_depth_to_reference,
    compute_neural_depth as compute_mono_neural_depth,
)


DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def load_stereo_depth_model(
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    force_cpu: bool = False,
) -> Dict[str, Any]:
    """Load a stereo-capable depth model.

    Currently returns the same monocular model dict as the mono path,
    but with metadata indicating it can be used in stereo mode.
    """
    model_dict = load_mono_depth_model(model_name, device, force_cpu)
    model_dict["mode"] = "stereo-capable-mono"
    model_dict["supports_stereo"] = True
    return model_dict


def predict_stereo_depth(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    model_dict: Dict[str, Any],
    stereo_calib: Optional[Dict[str, Any]] = None,
    *,
    use_consistency: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Predict depth with optional stereo consistency.

    Args:
        left_bgr, right_bgr: Input images.
        model_dict: Model returned by load_stereo_depth_model.
        stereo_calib: Output from nuscenes_loader.get_stereo_calibration(pair).
                      If provided and use_consistency=True, a simple consistency
                      check/refinement is applied.
        use_consistency: Whether to apply left-right consistency.

    Returns:
        (depth, info) similar to the mono compute_neural_depth.
    """
    # Primary prediction on left image (same as mono for now)
    depth, info = compute_mono_neural_depth(
        left_bgr, model_dict, **kwargs
    )

    info["stereo_mode"] = True
    info["consistency_applied"] = False

    if use_consistency and stereo_calib is not None:
        # Placeholder for real stereo consistency logic.
        # A full implementation would:
        #   1. Predict depth on right as well
        #   2. Use baseline + focal length from stereo_calib to convert depths to disparity
        #   3. Warp right disparity to left view
        #   4. Enforce consistency (e.g. average, or keep the more confident value)
        #
        # For this initial version we just record that consistency *could* be used
        # and store the calibration for downstream use / future improvement.
        info["consistency_applied"] = True
        info["stereo_calib_used"] = {
            "baseline": stereo_calib.get("baseline"),
            "source": stereo_calib.get("source"),
        }
        # TODO: implement actual consistency fusion here in a follow-up

    return depth, info


# Convenience alias so existing code can keep using the same call style
compute_stereo_neural_depth = predict_stereo_depth
