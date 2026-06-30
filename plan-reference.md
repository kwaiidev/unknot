# Plan Reference: Visual LiDAR Pipeline

This is the execution plan for turning `challenge/lidar.json` into a 2D trajectory PNG in a way that can be tested visually at each stage.

The main rule is: do not jump straight from raw LiDAR to the final image. Each stage should save a diagnostic artifact that answers one specific question before moving to the next stage.

## Development Loop

Use this loop for every stage:

1. Implement one small stage.
2. Generate its diagnostic PNG or text output.
3. Visually inspect the result.
4. Fix obvious parsing, geometry, filtering, or alignment issues.
5. Move to the next stage only when the current output looks sane.

## Stage 1: Data Sanity

Generate:

```text
outputs/00_stats.txt
outputs/01_valid_points.png
```

Check:

- Scan count is about `22,950`.
- Timestamps increase.
- Most scans have hundreds of valid points.
- There are no huge unexpected time gaps.
- Empty or sparse scan regions are visible.

## Stage 2: Single-Scan Plots

Generate 10 to 20 local scan plots:

```text
outputs/scans/scan_00000.png
outputs/scans/scan_01000.png
outputs/scans/scan_02000.png
```

Check:

- Points form recognizable lines, walls, racks, or aisle boundaries.
- There is no circular explosion caused by angle conversion errors.
- There is no degree/radian bug.
- The scan orientation is consistent from sample to sample.

## Stage 3: Range Image

Generate:

```text
outputs/02_range_image.png
```

This image should use scan index on one axis, LiDAR angle on the other axis, and distance as color.

Check:

- Smooth bands appear over time.
- Turns show visible shifts in the bands.
- Bad, empty, or noisy intervals are easy to identify.

## Stage 4: Consecutive Scan Overlay

Generate overlays of 5 to 10 nearby scans, all drawn in their local frame:

```text
outputs/03_overlay_0000_0010.png
outputs/03_overlay_5000_5010.png
```

Check:

- Nearby scans mostly resemble each other.
- Motion between scans appears gradual.
- There are no wild frame-to-frame jumps.

## Stage 5: Keyframe Selection Preview

Pick keyframes every 5 to 10 scans, or when the scan changes enough.

Generate:

```text
outputs/04_keyframe_samples.png
```

Check:

- There are enough keyframes to capture turns.
- There are not so many keyframes that matching becomes unnecessarily slow.
- Sparse or bad scans are skipped or downweighted.

## Stage 6: First Odometry Preview

Run odometry on a small prefix of the dataset before trying the full file.

Generate:

```text
outputs/05_odom_first_1000.png
outputs/05_odom_first_3000.png
```

Check:

- The path is locally smooth.
- Straight aisle segments stay mostly straight.
- Turns happen where the scan structure changes.
- There are no sudden teleports or impossible jumps.

## Stage 7: Submap Matching Debug

For selected keyframes, save the current scan overlaid on the local submap after applying the estimated transform.

Generate:

```text
outputs/debug_match/keyframe_0120.png
outputs/debug_match/keyframe_0240.png
```

Check:

- The transformed current scan lands on existing walls, racks, or aisle boundaries.
- Poor matches are visibly poor and have low scores.
- Good matches are visually obvious.

## Stage 8: Heading and Structure Plot

Generate:

```text
outputs/06_heading_over_time.png
```

Check:

- Heading is stable in long aisles.
- Heading changes mostly at turns.
- There is no slow continuous heading drift during straight runs.

## Stage 9: Full Raw Trajectory

Generate:

```text
outputs/07_full_odometry_raw.png
```

Check:

- The overall route shape is recognizable.
- Loops may not close yet, but the route should not be completely tangled.
- Long aisles should not curl badly.

## Stage 10: Loop Closure Before and After

Generate:

```text
outputs/08_before_loop_closure.png
outputs/09_after_loop_closure.png
```

Check:

- Revisited areas move closer together after loop closure.
- Loop closure improves the path without destroying straight aisle sections.
- Bad loop closures are rejected instead of folding the map incorrectly.

## Stage 11: Final Aligned Render

Generate:

```text
outputs/10_final_trajectory.png
```

Check:

- The trajectory visually matches the large-scale route in `challenge/Output_Trajectory.png`.
- Final alignment only rotates, translates, uniformly scales, or mirrors the trajectory for rendering.
- The path shape, turn order, and relative segment lengths still come from the LiDAR solution.

## Suggested CLI Shape

Keep the command-line interface small and repeatable:

```bash
python main.py stats
python main.py plot-scans
python main.py range-image
python main.py odom-preview --limit 3000
python main.py full
```

This makes it easier to identify which stage introduced a problem: parsing, scan conversion, filtering, odometry, loop closure, alignment, or final rendering.
