# Unknot

2D LiDAR SLAM diagnostics and trajectory reconstruction workspace.

The executable CLI remains backward compatible:

```bash
.venv/bin/python main.py stage11 --out-dir outputs/benchmarks/gt_1 --debug-matching
```

`main.py` is now a compatibility wrapper around the `slam` package.

## Project Layout

```text
slam/
  cli.py                  CLI entry points
  config.py               configuration dataclasses and argparse helpers
  pipeline.py             behavior-preserving stage orchestration
  models.py               shared dataclass API
  frontend/               scan matching, scoring, saliency, MHT, corridor frame
  backend/                loop closure, covariance, GTSAM, corridor graph
  mapping/                submap, occupancy, distance field helpers
  io/                     dataset and serialization/report helpers
  diagnostics/            JSONL, plots, metrics, replay diagnostics
  visualization/          trajectory and scan rendering
  utils/                  geometry, math, transforms, parser helpers
```

The current refactor is intentionally conservative: subsystem modules expose
stable import boundaries while preserving the existing algorithms, defaults,
CLI flags, and output files.

## Data

- `challenge/lidar.json`: NDJSON LiDAR scans.
- `challenge/Output_Trajectory.png`: reference image used only by final render alignment.

Each line in `challenge/lidar.json` has this shape:

```json
[epochMs, [[distanceMm, intensity, angleQ6], ...]]
```

`angleQ6` is `degrees * 64`; `distanceMm` is `0` when there is no return.

## Setup

Use the checked-in virtual environment if present:

```bash
.venv/bin/python -m py_compile main.py degeneracy.py $(find slam -name '*.py' | sort)
```

If creating a fresh environment, install the runtime dependencies used by the
pipeline:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy pillow scipy gtsam
```

## Common Runs

Generate odometry previews:

```bash
.venv/bin/python main.py stage6 --limits 1000,3000 --out-dir outputs/debug/stage6_check
```

Run full GTSAM loop-closure output:

```bash
.venv/bin/python main.py stage11 --out-dir outputs/benchmarks/gt_1 --debug-matching
```

Replay one loop closure candidate:

```bash
.venv/bin/python main.py stage11 --out-dir outputs/replay/replay_0 --replay-loop-closure 0
```

Disable optional corridor graph diagnostics:

```bash
.venv/bin/python main.py stage11 --out-dir outputs/debug/no_corridor_graph --no-corridor-graph
```

## Outputs

Generated artifacts are organized under:

- `outputs/benchmarks/`: retained full stage11 benchmark runs.
- `outputs/generated/`: one-off generated stage outputs such as `00_stats.txt`.
- `outputs/debug/`: short debug/check runs.
- `outputs/diagnostics/`: standalone diagnostic exports.
- `outputs/replay/`: loop closure replay artifacts.

Stage 11 writes:

- `10_final_trajectory.png`
- `match_diagnostics_stage11.jsonl`
- `aisle_frame_diagnostics_stage11.png`
- `loop_closure_diagnostics_stage11.jsonl`
- `loop_closure_analysis_stage11.json`
- `loop_closure_report_stage11.md`
- `loop_closure_candidates_stage11.png`
- `corridor_graph_stage11.json`
- `corridor_graph_stage11.png`
- `corridor_graph_consistency_stage11.png`

## Verification

For behavior-preserving refactors, run:

```bash
.venv/bin/python -m py_compile main.py degeneracy.py $(find slam -name '*.py' | sort)
.venv/bin/python main.py stage6 --limits 1000,3000 --out-dir outputs/debug/refactor_stage6
.venv/bin/python main.py stage11 --out-dir outputs/benchmarks/gt_1 --debug-matching
```

Compare pose count, accepted loop closures, alignment score, and generated
diagnostic files against the previous run.
