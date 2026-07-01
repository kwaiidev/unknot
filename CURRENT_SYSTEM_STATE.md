# Current System State

This document captures the current 2D LiDAR SLAM baseline. It is descriptive only; it does not define new behavior.

## Project Overview

The project reconstructs a 2D trajectory from `challenge/lidar.json` using keyframe LiDAR odometry, correlative scan-to-submap matching, loop closure, and GTSAM pose graph optimization. The frontend remains a discrete distance-field matcher, not ICP. The backend optimizes accepted odometry and loop factors, then renders the final trajectory against `challenge/Output_Trajectory.png` for visual scoring.

The CLI remains:

```bash
.venv/bin/python main.py stage11 --out-dir outputs/benchmarks/gt_17 --debug-matching
```

`main.py` is a thin compatibility wrapper around the `slam` package.

## Frontend

### Scan Preprocessing

- Reads NDJSON LiDAR scans from `challenge/lidar.json`.
- Converts polar returns into local `(x, y)` points.
- Filters by minimum/maximum range.
- Limits point count per scan for bounded matching cost.
- Selects keyframes by scan stride.

### Saliency Weighting

- Classifies scan points as `linear`, `neutral`, or `salient`.
- Linear shelves/walls receive lower score weight.
- Corners, endpoints, gaps, high-curvature regions, and isolated features receive higher weight.
- Weights are normalized so scans remain comparable even when salient features are sparse.

### Hierarchical Scan Search

- Uses the previous motion delta to predict the next pose.
- Builds a local submap from recent keyframes.
- Searches candidate `(x, y, theta)` poses against a submap distance field.
- Uses coarse-to-fine search stages around the prediction.
- Recovery search can expand the search window when confidence gating rejects a match.

### Confidence Estimation

For each scan match, the frontend records:

- best score
- second-best score
- score gap
- normalized score gap
- score variance
- near-best candidate count
- candidate count

The local score surface is analyzed with finite-difference Hessian estimation. Hessian eigenvalues, weak directions, condition number, and pose covariance are derived from that surface.

### Degeneracy Detection

`slam/frontend/degeneracy.py` estimates:

- orientation entropy
- PCA linearity
- dominant direction
- aisle confidence
- mode: `AISLE`, `INTERSECTION`, or `NORMAL`

AISLE mode applies additional scoring safeguards, including aisle-aligned downweighting, cross-structure emphasis, yaw regularization, and yaw bias correction.

### Multi-Hypothesis Tracking

The frontend maintains a small set of pose hypotheses (`K <= 5`, default `3`) to reduce early commitment in repetitive aisles. Each hypothesis is propagated through the motion model, matched independently, weighted by scan-match likelihood, and softly pruned. The best hypothesis is inserted into the pose graph, but alternate hypotheses inform ambiguity gating and loop-closure safety.

### Corridor Frame Estimation

The aisle frame is stabilized with a persistent corridor state:

- raw PCA axis per scan
- corridor-level axis state
- exponential moving average axis
- update gating by confidence, MHT entropy, and intersection state
- reset logic for large axis jumps, position gaps, or prolonged low confidence

### Anisotropic Constraints

In aisle-like scenes, odometry covariance is aligned with the stabilized corridor frame:

- along-aisle motion is constrained by the motion prior
- cross-aisle motion is constrained more by scan matching
- longitudinal drift regularization reduces sliding in long corridors
- covariance is rotated back into the factor frame before GTSAM insertion

## Backend

### GTSAM Integration

Stage 10 and stage 11 construct a GTSAM `NonlinearFactorGraph` with:

- prior factor on the first pose
- between factors for adjacent odometry
- robust loop-closure factors for accepted scan-matched closures

Odometry factors use adaptive covariance from scan-match confidence and anisotropic corridor-frame modeling when applicable.

### Adaptive Covariance

Measurement covariance is derived from:

- Hessian curvature
- score ambiguity
- score gap deficit
- MHT ambiguity
- anisotropic aisle-frame covariance

Poorly constrained directions get larger covariance, so GTSAM trusts uncertain matches less.

### Loop Closure

Loop closure uses:

- range-signature candidate filtering
- scan-to-scan correlative matching
- score and score-gap thresholds
- MHT agreement checks
- ambiguity deferral

Accepted candidates become GTSAM loop factors. All evaluated candidates are recorded for diagnostics and offline re-ranking.

### Corridor Graph

After GTSAM optimization, an additive corridor-segment graph is built for topological diagnostics:

- segments represent continuous corridor-like traversals
- edges represent sequential transitions, loop transitions, or ambiguity transitions
- segment signatures include axis histogram, entropy, linearity, saliency ratio, turns, intersections, resets, and drift
- conflicts are reported when corridor-level transforms disagree with pose-level loop transforms

The corridor graph is diagnostic only and does not feed back into GTSAM.

### Optimization

The active optimizer remains GTSAM. Final rendering applies a similarity alignment against the provided reference image only for visualization/fit scoring; it is not part of SLAM state estimation.

## Diagnostics

Stage 11 can generate:

- `10_final_trajectory.png`: final rendered trajectory.
- `match_diagnostics_stage11.jsonl`: per-keyframe scan-match confidence, covariance, degeneracy, MHT, and aisle-frame data.
- `aisle_frame_diagnostics_stage11.png`: drift, covariance, and corridor-axis stability plots.
- `loop_closure_diagnostics_stage11.jsonl`: one record per evaluated loop candidate.
- `loop_closure_analysis_stage11.json`: post-run loop closure ranking and summary statistics.
- `loop_closure_report_stage11.md`: human-readable loop closure analysis.
- `loop_closure_candidates_stage11.png`: pose graph with accepted/rejected/ambiguous/conflicting loop candidates.
- `corridor_graph_stage11.json`: corridor segment graph data.
- `corridor_graph_stage11.png`: corridor graph visualization.
- `corridor_graph_consistency_stage11.png`: corridor consistency mismatch heatmap.
- `loop_replay_stage11_<id>.json`: isolated loop closure replay metadata.
- `loop_replay_stage11_<id>_score_surface.jsonl`: isolated loop closure score surface samples.

## Repository Structure

- `main.py`: backward-compatible CLI wrapper.
- `degeneracy.py`: backward-compatible re-export of packaged degeneracy metrics.
- `slam/pipeline.py`: behavior-preserving pipeline implementation and stage orchestration.
- `slam/cli.py`: CLI entry point exports.
- `slam/config.py`: configuration dataclasses and parser helpers.
- `slam/models.py`: shared dataclass/type exports.
- `slam/frontend/`: scan matching, saliency, hierarchy, MHT, descriptors, degeneracy, and corridor frame facades.
- `slam/backend/`: loop closure, covariance, pose graph, optimization, and corridor graph facades.
- `slam/mapping/`: submap, occupancy, and distance-field facades.
- `slam/io/`: dataset, serialization, and report facades.
- `slam/diagnostics/`: diagnostic logging, plots, metrics, and replay facades.
- `slam/visualization/`: rendering and overlay facades.
- `slam/utils/`: geometry, math, transform, and parsing helpers.
- `challenge/`: input LiDAR data and reference output image.
- `outputs/benchmarks/`: retained full benchmark runs.
- `outputs/generated/`: generated one-off artifacts such as `00_stats.txt`.
- `outputs/debug/`: short debug/check runs.
- `outputs/diagnostics/`: standalone diagnostic exports.
- `outputs/replay/`: loop closure replay artifacts.

## Configuration Summary

### Dataset and Keyframes

- `--lidar`
- `--limit`
- `--stride`
- `--min-valid-points`
- `--max-points`
- `--min-range-m`
- `--max-range-m`

### Submap and Search

- `--cell-size-m`
- `--inflation-cells`
- `--submap-keyframes`
- `--search-xy-m`
- `--search-step-m`
- `--search-theta-deg`
- `--search-theta-step-deg`
- `--max-step-m`

### Confidence and Covariance

- `--confidence-threshold`
- `--score-gap-threshold`
- `--near-best-score-percent`
- `--max-near-best-candidates`
- `--covariance-scaling`
- `--hessian-conditioning-threshold`
- `--hessian-min-curvature`
- `--expanded-search-xy-m`
- `--expanded-search-theta-deg`

### Saliency and Degeneracy

- `--feature-weighting` / `--no-feature-weighting`
- `--linear-feature-weight`
- `--salient-feature-weight`
- `--saliency-neighbor-count`
- `--saliency-linear-curvature-deg`
- `--saliency-corner-curvature-deg`
- `--saliency-gap-m`
- `--degeneracy-scoring` / `--no-degeneracy-scoring`
- `--aisle-aligned-feature-weight`
- `--aisle-cross-feature-weight`
- `--aisle-yaw-lambda`
- `--aisle-theta-window-scale`
- `--yaw-bias-correction-alpha`
- `--yaw-bias-top-score-percent`

### Multi-Hypothesis Tracking

- `--hypothesis-count`
- `--hypothesis-weight-threshold`
- `--hypothesis-ambiguous-weight-threshold`
- `--hypothesis-score-temperature`
- `--hypothesis-ambiguous-temperature-scale`
- `--hypothesis-branch-score-percent`
- `--hypothesis-close-score-gap`
- `--hypothesis-near-best-count-threshold`
- `--hypothesis-divergence-threshold-m`
- `--hypothesis-theta-distance-weight-m`
- `--hypothesis-ambiguity-covariance-scale`

### Anisotropic and Corridor Frame

- `--anisotropic-constraints` / `--no-anisotropic-constraints`
- `--anisotropic-min-confidence`
- `--anisotropic-aisle-linearity-threshold`
- `--anisotropic-along-sigma-m`
- `--anisotropic-cross-sigma-m`
- `--anisotropic-theta-sigma-deg`
- `--anisotropic-scan-covariance-floor-m`
- `--anisotropic-drift-gain`
- `--corridor-frame` / `--no-corridor-frame`
- `--corridor-axis-alpha`
- `--corridor-ema-beta`
- `--corridor-update-min-confidence`
- `--corridor-update-min-linearity`
- `--corridor-max-hypothesis-entropy`
- `--corridor-reset-axis-deg`
- `--corridor-reset-position-gap-m`
- `--corridor-reset-low-confidence-frames`

### Loop Closure and Backend

- `--signature-bins`
- `--signature-max-range-m`
- `--min-loop-separation`
- `--max-loop-distance-m`
- `--max-signature-distance`
- `--min-overlap-bins`
- `--max-closures`
- `--loop-weight`
- `--odometry-weight`
- `--anchor-weight`
- `--loop-match-search-xy-m`
- `--loop-match-search-step-m`
- `--loop-match-search-theta-deg`
- `--loop-match-search-theta-step-deg`
- `--min-loop-match-score`
- `--min-loop-match-score-gap`
- `--loop-hypothesis-agreement-m`
- `--loop-hypothesis-agreement-theta-deg`
- `--loop-strong-score-gap`

### Corridor Graph and Diagnostics

- `--corridor-graph` / `--no-corridor-graph`
- `--corridor-segment-axis-threshold-deg`
- `--corridor-segment-entropy-threshold`
- `--corridor-segment-confidence-jump`
- `--corridor-segment-drift-reset-m`
- `--corridor-segment-min-poses`
- `--corridor-edge-support-required`
- `--corridor-similarity-low-confidence`
- `--corridor-graph-conflict-distance-m`
- `--corridor-graph-conflict-theta-deg`
- `--corridor-graph-opt-iterations`
- `--corridor-graph-opt-step`
- `--debug-matching`
- `--debug-score-heatmaps`
- `--replay-loop-closure`

## Current Performance

Latest verified full benchmark:

- Run: `outputs/benchmarks/gt_17`
- Estimated poses: `2184`
- Accepted loop closures: `11 / 40`
- Alignment fit score: `54.50`
- Corridor graph: `244` nodes, `250` edges, `64` ambiguous edges, `3` conflicts
- Match diagnostics: `2184` accepted matches, `111` rejected matches, `123` expanded retries, `83` degenerate surfaces

### Strengths

- Stable frontend tracking on the current warehouse dataset.
- Robustness improvements for repetitive aisles through saliency weighting, MHT, and corridor-frame covariance.
- Diagnostic visibility into scan-match confidence, loop-closure decisions, and corridor topology.
- GTSAM backend remains stable and compatible with adaptive covariance.

### Known Weaknesses

- Loop closure discrimination remains conservative in highly repetitive regions.
- Corridor graph is diagnostic and does not yet correct topology.
- Place identity is still inferred from geometry rather than explicit semantic or structural descriptors.
- Long-range repeated-aisle ambiguity can still produce uncertain loop candidates.

## Remaining Technical Challenges

- Improve loop closure quality without accepting wrong-aisle matches.
- Distinguish same-corridor and parallel-corridor revisits more explicitly.
- Reduce dependence on local geometric similarity in repetitive shelves.
- Move more logic out of `slam/pipeline.py` into true subsystem implementations.

## Future Work

- Global structural descriptors for corridor-level place recognition.
- Scan Context or similar rotation-aware place descriptors.
- Learned place recognition for repetitive warehouse scenes.
- Semantic landmarks such as aisle ends, crossings, rack gaps, and open areas.
- Multi-resolution descriptors for loop closure prefiltering.
- Corridor graph feedback into loop-closure validation after sufficient diagnostics.
