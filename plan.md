# Plan

## Assumptions

- The source guidance in `CODEX.MD` applies: keep changes simple, make surgical edits, and verify each step.
- The challenge README is the authoritative data description.
- The target artifact is `challenge/Output_Trajectory.png`; the user's `output_traject.png` appears to refer to that file.
- The final goal is to reconstruct and render a trajectory whose shape and alignment are close to the purple reference path. The red numbered pins are useful visual checkpoints, but no separate pin-coordinate file is present.

## Data Facts

- Input file: `challenge/lidar.json`
- Format: NDJSON, one 360-degree LiDAR turn per line.
- Record shape: `[epochMs, [[distanceMm, intensity, angleQ6], ...]]`
- `angleQ6` is `degrees * 64`.
- Dataset size: 22,950 scans over about 2,293 seconds, approximately 10 Hz.
- Each scan has about 508 samples median, with about 353 valid nonzero ranges median.
- Valid distance range is roughly 0.7 m to 15.3 m, with a median near 2.7 m.

## Success Criteria

The output is good enough when:

1. The rendered trajectory follows the same large-scale route as `challenge/Output_Trajectory.png`.
2. Long straight aisle traversals stay straight and parallel instead of drifting or curling.
3. Turns occur in the same order and at comparable relative positions as the reference.
4. Looping areas in the lower-left, upper-left, and right side visually close instead of accumulating large drift.
5. The final plot can be produced repeatedly from the raw `challenge/lidar.json` data by running the project code.

## Best Approach

Use a small SLAM-style pipeline built around scan-to-submap matching, structural line features, and pose-graph cleanup. Do not rely on plain frame-to-frame ICP as the main method; the README explicitly warns that classic ICP may not work well, and the reference path contains long, repetitive aisle geometry where ICP can easily slide along parallel racks.

## Step 1: Load and Normalize the LiDAR

Create a loader that streams `challenge/lidar.json` line by line.

For each scan:

- Parse the timestamp and sample list.
- Drop zero-distance returns.
- Convert `angleQ6` to radians: `angle_rad = angleQ6 / 64 * pi / 180`.
- Convert polar measurements to local Cartesian points.
- Keep intensity available for optional filtering, but do not overfit to intensity early.
- Filter obvious outliers, for example extremely short noisy hits and isolated far returns.

Verify:

- Plot 10 to 20 individual scans in local coordinates.
- Confirm the scan orientation is consistent and walls/racks appear as lines.
- Confirm there is no accidental degree/radian or angle-wrap bug.

## Step 2: Build a Visualization Baseline

Before solving odometry, make quick diagnostic plots:

- Range image over time.
- Valid point count over time.
- A few consecutive scans overlaid in local coordinates.
- A naive "all scans at origin" plot to inspect recurring structural directions.

Verify:

- The dominant aisle/rack directions are visible.
- Empty or sparse intervals are identified.
- Bad scans are either rare or easy to filter.

## Step 3: Estimate Dominant Structure

Extract line-like structure from scans or small scan windows.

Recommended simple path:

- Downsample each scan to reduce noise.
- Use RANSAC or Hough-style line detection on local points.
- Track dominant line angles over time.
- Snap heading estimates toward the dominant parallel aisle directions when confidence is high.

This helps because the environment in the reference image is dominated by long parallel rows. Heading drift will be more damaging than small translation noise, so heading needs explicit constraints.

Verify:

- Plot dominant heading over time.
- Confirm heading changes occur mostly at turns.
- Confirm long aisle segments have stable heading.

## Step 4: Estimate Local Motion With Scan-to-Submap Matching

Use keyframes instead of every single scan.

Recommended starting point:

- Create a keyframe every 5 to 10 scans, or when the scan changes enough.
- Maintain a local occupancy grid or point submap from recent keyframes.
- For each new keyframe, search over a small `(dx, dy, dtheta)` window.
- Score candidate poses by matching scan points against a distance transform of the local submap.
- Seed the search with the previous motion estimate and the dominant-heading estimate.

This is more robust than consecutive ICP because each scan is matched against accumulated local structure instead of one noisy, partially overlapping scan.

Verify:

- Render partial trajectories every few minutes of data.
- Check that straight aisle runs remain parallel.
- Log match score and reject or downweight low-confidence updates.

## Step 5: Handle Degenerate Aisle Motion

Long aisles are ambiguous: scan matching may know the robot is between two parallel rows but not exactly how far along the aisle it moved.

Use conservative constraints:

- Smooth velocity between neighboring poses.
- Penalize sudden lateral jumps.
- Prefer heading continuity except near detected turns.
- Use scan signature correlation along the aisle direction to estimate longitudinal movement.
- Delay hard decisions when the match score is weak and let later loop closures correct drift.

Verify:

- Inspect the longest straight runs in the output image and compare them to the generated path.
- Check for sliding, reversed movement, and doubled-back artifacts.

## Step 6: Add Loop Closures and Pose-Graph Optimization

Build a pose graph after local odometry works.

Nodes:

- Keyframe poses.

Edges:

- Local odometry constraints between adjacent keyframes.
- Scan-to-submap constraints for nearby revisits.
- Loop closures when a current scan matches an earlier submap with a strong score.
- Optional heading constraints from dominant line directions.

Optimize with a robust least-squares solver. Keep this implementation small; `scipy.optimize.least_squares` is enough for an interview-quality solution unless a SLAM library is already available.

Verify:

- Before and after plots show loop closure improving global consistency.
- Bad loop closures are rejected by score, geometry agreement, or robust loss.
- The lower-left and right-side loops close without breaking straight aisle sections.

## Step 7: Align the Result to the Reference Image

The LiDAR solution will be in an arbitrary coordinate frame. For visual comparison to `challenge/Output_Trajectory.png`, apply a final 2D similarity transform only for rendering:

- Rotate.
- Translate.
- Uniformly scale if needed.
- Optionally mirror only if scan coordinate handedness proves opposite to the reference.

This transform should not be used to hide trajectory errors. It should only place the reconstructed path into the same visual frame as the reference image.

Verify:

- The transformed path overlays the target route shape.
- Turn order and relative segment lengths still come from the LiDAR solution, not from manual drawing.

## Step 8: Render the Final Output

Create a reproducible script that writes a final PNG.

The final plot should include:

- Purple reconstructed trajectory.
- A legend matching the reference style where practical.
- Optional red numbered checkpoint markers selected from key trajectory events, such as turns, pauses, loop closures, or manually chosen keyframes for comparison.

If no map image is available besides the target reference, do not claim to reconstruct the red building map. The core deliverable should be the trajectory reconstruction from LiDAR.

Verify:

- Running the script from a clean checkout regenerates the same output.
- The generated image is visually compared against `challenge/Output_Trajectory.png`.

## Implementation Order

1. Replace the starter `main.py` with a minimal command-line entry point.
2. Add a loader and scan conversion function.
3. Add diagnostic plotting.
4. Implement keyframe selection and local scan-to-submap matching.
5. Add structural heading estimation.
6. Add pose-graph cleanup and loop closure.
7. Add final alignment and rendering.
8. Tune only after each stage has plots that show what improved.

## Practical First Milestone

The first useful milestone is not the final perfect image. It is a plot showing a locally consistent trajectory over the first few minutes of data. Once that works, extend to the full dataset and add loop closures. This keeps the work testable and prevents tuning against the target image before the LiDAR pipeline is sound.
