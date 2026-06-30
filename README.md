# Unknot

LiDAR trajectory reconstruction challenge workspace.

## Contents

- `challenge/lidar.json`: raw NDJSON LiDAR scans, stored with Git LFS because it is larger than GitHub's regular file limit.
- `challenge/Output_Trajectory.png`: reference trajectory image.
- `plan.md`: reconstruction approach.
- `plan-reference.md`: visual checkpoint workflow for testing progress.
- `main.py`: starter entry point.

## Data Format

Each line in `challenge/lidar.json` is one 360-degree scan:

```json
[epochMs, [[distanceMm, intensity, angleQ6], ...]]
```

`angleQ6` is `degrees * 64`, and `distanceMm` is `0` when there is no return.
