"""Neural monocular depth using Hugging Face transformers (Depth-Anything-V2 recommended).

Outputs relative depth; scale alignment to classical stereo or LiDAR is provided.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False
    torch = None  # type: ignore


DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def load_depth_model(
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    force_cpu: bool = False,
) -> Dict[str, Any]:
    """Load and return HF depth model dict. Downloads weights on first use."""
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers + torch not available. Install requirements.txt"
        )
    if device is None:
        if force_cpu:
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device).eval()

    return {
        "processor": processor,
        "model": model,
        "device": device,
        "name": model_name,
    }


def predict_relative_depth(
    model_dict: Dict[str, Any],
    image_bgr: np.ndarray,
    *,
    target_size: Optional[Tuple[int, int]] = None,  # (h, w) or None=auto
) -> np.ndarray:
    """Run monodepth. Returns (H, W) float32 relative depth map.

    For Depth-Anything-V2 HF port: higher value typically means closer (inverse-like).
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers not installed")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    processor = model_dict["processor"]
    model = model_dict["model"]
    device = model_dict["device"]

    # Processor handles resize + normalization for the model
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process back to original image resolution
    h, w = pil_img.height, pil_img.width
    post_processed = processor.post_process_depth_estimation(
        outputs, target_sizes=[(h, w)]
    )
    depth = post_processed[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)

    # Some models output disparity (higher=closer). Keep as-is; alignment will handle sign/orient.
    return depth


def align_depth_to_reference(
    neural_rel: np.ndarray,
    ref_depth: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    method: str = "median_ratio",
    eps: float = 1e-6,
) -> np.ndarray:
    """Robustly align neural relative depth to a reference metric depth map (e.g. classical stereo).

    Returns aligned depth in same units as ref_depth (meters).

    Heuristic:
      - Compute valid overlap mask
      - Try treating neural as direct depth vs inverse depth
      - Pick scale that minimizes median relative error
    """
    if mask is None:
        mask = (
            (ref_depth > 0.5)
            & np.isfinite(ref_depth)
            & (neural_rel > 0)
            & np.isfinite(neural_rel)
        )

    n_valid = int(mask.sum())
    if n_valid < 30:
        # Very few points: fallback to global medians (assume direct)
        r_med = float(np.nanmedian(ref_depth[ref_depth > 0]))
        n_med = float(np.nanmedian(neural_rel[neural_rel > 0]))
        if n_med > eps:
            return (r_med / n_med) * neural_rel
        return neural_rel.copy()

    ref = ref_depth[mask].astype(np.float64)
    neu = neural_rel[mask].astype(np.float64)

    # Candidate 1: direct scale  (ref ≈ s * neu)
    s1 = np.median(ref / np.maximum(neu, eps))
    cand1 = s1 * neural_rel

    # Candidate 2: inverse (common for monodepth)  ref ≈ s / neu
    s2 = np.median(ref * np.maximum(neu, eps))
    cand2 = s2 / np.maximum(neural_rel, eps)

    # Score by median absolute percentage error (lower better) on the fitting region only
    def mape(cand: np.ndarray) -> float:
        c = cand[mask]
        return float(np.median(np.abs(ref - c) / (ref + 1.0)))

    e1, e2 = mape(cand1), mape(cand2)
    if e1 <= e2:
        scale = s1
        aligned = cand1
    else:
        scale = s2
        aligned = cand2

    # Clip crazy outliers on the dense result; keep full image (no zeroing outside mask)
    aligned = np.clip(aligned, 0.5, 200.0)

    return aligned.astype(np.float32)


def compute_neural_depth(
    image_bgr: np.ndarray,
    model_dict: Dict[str, Any],
    *,
    ref_depth: Optional[np.ndarray] = None,  # for scale alignment (e.g. classical)
    align: bool = True,
    input_resize: Optional[int] = None,  # e.g. 518 for speed
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convenience: predict + optional align. Returns (depth_meters_or_rel, info)."""
    rel = predict_relative_depth(model_dict, image_bgr)

    info: Dict[str, Any] = {"raw": rel.copy(), "aligned": False, "scale": 1.0, "method": "raw"}

    if align and ref_depth is not None:
        aligned = align_depth_to_reference(rel, ref_depth)
        info["aligned"] = True
        # Compute realized scale only over the region that had valid reference depth
        fit_mask = (
            (ref_depth > 0.5)
            & np.isfinite(ref_depth)
            & (rel > 0)
            & np.isfinite(rel)
            & (aligned > 0)
        )
        if fit_mask.sum() > 10:
            info["scale"] = float(np.nanmedian(aligned[fit_mask] / (rel[fit_mask] + 1e-6)))
        else:
            info["scale"] = 1.0
        info["method"] = "median_ratio_to_ref"
        return aligned, info

    return rel, info


def depth_to_lidar_aligned(
    neural_depth: np.ndarray,
    lidar_depths: np.ndarray,   # sparse projected depths (from project_lidar_to_image)
    uv: np.ndarray,             # corresponding (N,2) pixel locations
) -> np.ndarray:
    """Align using sparse LiDAR projections (more accurate absolute scale)."""
    if len(lidar_depths) < 10:
        return neural_depth

    # Sample neural at projected uv (nearest)
    h, w = neural_depth.shape
    us = np.clip(uv[:, 0].astype(int), 0, w - 1)
    vs = np.clip(uv[:, 1].astype(int), 0, h - 1)
    neu_sparse = neural_depth[vs, us]

    mask = (lidar_depths > 0.5) & (neu_sparse > 0)
    if mask.sum() < 5:
        return neural_depth

    ref = lidar_depths[mask]
    neu = neu_sparse[mask]

    # Try direct + inverse
    s1 = np.median(ref / np.maximum(neu, 1e-6))
    c1 = s1 * neural_depth
    s2 = np.median(ref * np.maximum(neu, 1e-6))
    c2 = s2 / np.maximum(neural_depth, 1e-6)

    def med_rel_err(c):
        cs = c[vs[mask], us[mask]]
        return np.median(np.abs(ref - cs) / (ref + 1))

    aligned = c1 if med_rel_err(c1) < med_rel_err(c2) else c2
    return np.clip(aligned, 0.5, 150.0).astype(np.float32)
