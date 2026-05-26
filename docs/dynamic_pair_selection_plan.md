# Plan: Dynamic Best-Overlapping Camera Pair Selection for Stereo Reconstruction

**Date**: 2026 (current session)  
**Goal**: Replace the current fixed stereo pair (`CAM_FRONT_LEFT` + `CAM_FRONT_RIGHT`) with per-frame (or per-sample) dynamic selection of the "best" overlapping camera pair from the 6 nuScenes cameras. This should improve reconstruction coverage, density, and robustness across varying scene geometries, motion, and lighting without breaking the existing `StereoPair` / reconstruction pipeline contract.

**Status**: Plan authored after deep exploration (sub-agent + manual) + metric research. Ready for phased execution via sub-agents.

## 1. Background & Problem Statement

### Current State (from exploration)
- **Fixed pair**: Hardcoded in `NuScenesStereoLoader` (`DEFAULT_LEFT = "CAM_FRONT_LEFT"`, `DEFAULT_RIGHT = "CAM_FRONT_RIGHT"`) — see `pipeline/nuscenes_loader.py:63-64`.
- `get_stereo_sequence(scene, left_channel=..., right_channel=..., ...)` walks samples and builds `StereoPair` using only those two channels (`_build_stereo_pair`, lines 119-153).
- `StereoPair` is a simple dataclass holding two `CameraData` (with full intrinsics + `sensor_from_ego` + `ego_from_world` transforms) + optional LiDAR.
- `get_stereo_calibration` and `classical_stereo.get_relative_pose` compute `right_from_left` on the fly (some duplication).
- **Pipeline is mostly pair-agnostic** once a valid `StereoPair` is provided:
  - `process_frame` (reconstruction.py:133+) uses `pair.left` as the reference for backprojection, world transforms, TSDF, eval, viz. `pair.right` only for input to classical stereo.
  - `compute_classical_stereo` recomputes rectification per pair (perfect for dynamic).
  - TSDF fusion (`fusion.py`) explicitly supports per-frame varying `world_from_cam` + `K`.
  - Global point accumulation and evaluation work regardless of which cam is "left".
- **No overlap/selection logic exists** today. The comment on line 62 calls FL+FR "Recommended ... with good overlap + baseline", but it is static for the entire sequence.
- Usage sites (demo, README) hardcode or default to the fixed pair.
- 6 cameras available in v1.0-mini (CAM_FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_LEFT, BACK_RIGHT).

### Why Dynamic Selection?
- Fixed pair can have suboptimal overlap in some frames (e.g., during sharp turns, close obstacles, or when side/rear coverage would be superior).
- Different pairs offer different baselines (depth precision vs. matching difficulty tradeoff) and FOV coverage.
- Enables higher-quality, more complete 3D reconstructions, especially for full-surround or adaptive "stereo video".
- Low risk: The math and data structures already support arbitrary per-frame pairs.

**Non-goals** (for v1): Image-based selection at runtime (keep pure geometry for speed); full 360° surround reconstruction; automatic fallback to mono depth.

## 2. High-Level Design

### Core Abstraction
- Introduce **pair selection strategy** (heuristic / callable) that, given a nuScenes `sample` dict + loader access to calibrated sensors, returns the best `(left_channel, right_channel)` for that frame.
- New or extended loader API that can produce a `List[StereoPair]` where each element may use a different pair (still valid synchronized data from the same sample).
- `StereoPair` and downstream contracts unchanged — only the *source* of the pairs changes.

### Candidate Pairs (Prioritized)
From rig analysis (extrinsics + observed overlaps):
1. `("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")` — current default; strong forward, ~1m baseline.
2. `("CAM_FRONT_LEFT", "CAM_FRONT")` and symmetric `("CAM_FRONT_RIGHT", "CAM_FRONT")`.
3. `("CAM_FRONT_LEFT", "CAM_BACK_LEFT")` / side-rear equivalents (for shoulders/turns).
4. Rear pairs (`BACK_LEFT + BACK_RIGHT`, etc.) as lower priority for reverse or full-surround.

Only consider pairs with reasonable geometric overlap (>~25-30%) and baseline (>0.3-0.5m).

### Overlap / Quality Metric (Primary Recommendation from Research)
**Hybrid score (primary implementation target)**:
- **Core**: Multi-depth projected grid overlap fraction (lift uniform image grid rays from cam L at depths [5,15,30,50]m using intrinsics, apply relative pose, count fraction landing inside cam R image bounds + positive depth). Average across depths. Vectorized numpy, <1ms/pair.
- **Hybrid**: `quality = overlap * f(baseline)` (normalize baseline around typical good values ~1-1.5m; or use rectified ROI area % as secondary term).
- Optional: Forward-motion bias using ego velocity (delta `ego_from_world` between consecutive samples) to prefer pairs aligned with travel direction.
- Fallback: If no pair meets thresholds, use DEFAULT (FL+FR) or highest-scoring available.

This reuses existing `get_relative_pose` / transform logic and is directly inspired by techniques in cuVSLAM / multi-view stereo view selection literature.

Alternative / companion: Post-rectification valid ROI area % (directly from `compute_rectification`).

**Why this metric?**
- Image-free (fast, works at load time).
- Directly predicts dense correspondence potential for SGBM + neural.
- Handles nuScenes' angled/verged cameras.
- Cheap to precompute/cache (rig geometry is nearly static).

See research sub-agent output for full pseudocode and variants.

### API Changes (Minimal & Backward Compatible)
- `NuScenesStereoLoader`:
  - New class attr or config: `CANDIDATE_PAIRS: List[Tuple[str,str]] = [...]`
  - New method: `get_dynamic_stereo_sequence(scene_name, max_frames=None, selector: Optional[Callable] = None, ...)` 
    - Or extend `get_stereo_sequence(..., pair_strategy="best_overlap" | "fixed" | callable)`.
  - Helper: `compute_pair_overlap_metrics(cam1, cam2) -> Dict` (returns overlap, baseline, quality, usable).
  - Internal: `select_best_pair_for_sample(sample) -> Optional[StereoPair]`.
  - Precompute/cache pair geometries/scores at `__init__` or first use (using representative intrinsics + sensor extrinsics).
- `StereoPair` (optional): Add `selected_by: str = "fixed"`, `overlap_score: float = 0.0` for debugging/traceability (non-breaking).
- Expose in `scripts/run_demo.py`: New CLI flag `--dynamic-pairs` / `--pair-strategy` (default: off for backward compat).
- Update `accumulate_reconstruction` / docs minimally (mostly logging "using dynamic pairs" + per-frame channel names in debug output).
- No changes needed in fusion, classical stereo, reconstruction core, evaluation, viz (already flexible).

### Fallbacks & Robustness
- If selected pair has missing data or low score → fallback to DEFAULT or skip frame (current behavior for missing channels).
- Temporal smoothing: small bonus for pair used in previous frame (reduce jitter).
- Logging: Per-frame "Selected CAM_XXX + CAM_YYY (quality=0.XX, baseline=1.2m)" at INFO level.
- Configurability: Allow user-provided candidate list or custom selector callable.

### Testing & Validation Strategy
- Unit tests for `compute_pair_overlap_metrics` (synthetic + real calibrated_sensor data).
- Integration: Run `get_dynamic...` on scene-0061 (and others), assert non-empty list, varying pairs in output, reasonable scores.
- End-to-end: Full demo run with `--dynamic-pairs --num-frames 10 --eval` → inspect `reconstruction_stats.json` (per-frame "selected_pair" or similar), compare point counts / eval metrics vs fixed-pair baseline.
- Visual: Generate depth_comparison videos/PNGs and manually or heuristically check for improved coverage (more points in side areas, fewer holes).
- Regression: Fixed-pair mode must produce identical results to today.
- Edge cases: Stationary ego, turns, night scenes (if data), partial camera dropout.

## 3. Implementation Phases (Prioritized, Incremental)

**Phase 0: Foundations (Low risk, enables everything)**
- Add candidate pair list + `compute_pair_overlap_metrics(...)` (and supporting `compute_rectified_overlap_pct` helper) to `nuscenes_loader.py`.
- Add unit tests (new `tests/test_pair_selection.py` or inline).
- Precompute rig geometry cache.
- Validate metrics against manual inspection of a few samples (use existing `project_lidar...` as oracle where available).

**Phase 1: Core Dynamic Loading**
- Implement `select_best_pair_for_sample` + `get_dynamic_stereo_sequence` (or strategy param on existing method).
- Support velocity bias (simple delta ego pose).
- Update `StereoPair` optionally with selection metadata.
- Ensure `only_keyframes` and lidar attachment still work.

**Phase 2: Integration & CLI**
- Wire into `scripts/run_demo.py` (new `--dynamic-pairs` flag, updated prints, per-frame logging of chosen channels).
- Update README, help text, example commands.
- Minor: Add selection info to `reconstruction_stats.json` "per_frame" entries (e.g. `"selected_pair": ["CAM_FRONT_LEFT", "CAM_FRONT"]`).

**Phase 3: Polish, Evaluation & Documentation**
- Temporal smoothing + configurable thresholds/weights.
- Enhanced logging / stats (e.g., histogram of selected pairs across sequence).
- Full end-to-end validation run + comparison report (fixed vs dynamic: point density, eval MAE coverage, visual quality).
- Update any affected docs (GROK.md if exists, README).
- Optional: Expose custom selector callable for advanced users.

**Phase 4 (Future / Stretch)**: Image-based refinement (post-load texture stats), learned selector, full surround fusion modes.

**Order**: Phase 0 → 1 (core functionality) → 2 (usable) → 3 (validated). Do not skip validation between phases.

## 4. Risks & Mitigations

- **Risk**: Performance (negligible — 6 cams × ~10 pairs × 1ms = tiny).
- **Risk**: Quality regression on non-front cams (classical SGBM params, neural appearance domain). *Mitigation*: Start with front-heavy candidates; make classical params tunable; document that neural is more robust.
- **Risk**: Jittery pair switching. *Mitigation*: Temporal smoothing + hysteresis.
- **Risk**: "left" reference changes break user expectations in viz/stats. *Mitigation*: Per-frame channel recording; "left" is always the reference for that frame's data (already the case).
- **Dupe code**: Relative pose calc. *Opportunity*: Centralize in loader during implementation.
- **Testing surface**: Small because core pipeline is flexible.

## 5. Success Criteria

- `loader.get_dynamic_stereo_sequence(...)` returns valid `List[StereoPair]` (possibly with mixed pairs) for real scenes.
- At least one sequence shows different pairs selected across frames with plausible scores.
- End-to-end run completes without regression in fixed mode.
- Dynamic mode produces reconstructions (measurable via point count, eval coverage, or visual inspection) that are equal or better in coverage/quality on test scenes.
- Code is tested, documented, and the plan's phases are all closed.

## 6. Files to Change (Approximate)

- `pipeline/nuscenes_loader.py` (main — ~150-250 new LOC for metrics + selector).
- `scripts/run_demo.py` (CLI + usage).
- `README.md` + any usage examples.
- `tests/` (new or extended).
- `docs/` (this plan + possibly usage notes).
- Minor touches: `reconstruction.py` (optional logging), `classical_stereo.py` (possible dedup of relative pose).

## 7. Sub-Agent Execution Strategy (for b/c/d)

- Use specialized sub-agents (explore, implement, review, test-runner).
- One sub-agent per major phase or component (e.g., "implement metrics + loader helpers", "update demo + CLI", "write tests", "run validation").
- After each sub-agent: human (or reviewer sub-agent) validates output before next.
- Iterate until all phases complete + final E2E demo run succeeds with measurable improvement or at least no regression + dynamic behavior demonstrated.

## 8. Open Questions / Decisions for Implementation

- Exact candidate list & priority order (front-heavy vs. include sides?).
- Default behavior in `get_stereo_sequence` (keep fixed as default for BC; new method for dynamic).
- Weighting of hybrid score (baseline vs overlap) — tune on real data during Phase 3.
- Whether to add `overlap_score` etc. to `StereoPair` or keep internal.
- Velocity source (simple pose delta is sufficient for v1; can_bus for later).
- Alpha for rectification in hybrid metric.

This plan is concrete, leverages the excellent existing geometry infrastructure, minimizes risk, and delivers real value for the reconstruction quality.

**Next Step**: Begin Phase 0 via sub-agent orchestration.
