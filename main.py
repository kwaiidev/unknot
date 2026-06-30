from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_LIDAR_PATH = Path("challenge/lidar.json")
DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class ScanSummary:
    timestamp_ms: int
    point_count: int
    valid_point_count: int
    min_distance_mm: float | None
    max_distance_mm: float | None


@dataclass(frozen=True)
class LocalScan:
    scan_index: int
    timestamp_ms: int
    points: list[tuple[float, float, int]]


@dataclass(frozen=True)
class Pose2D:
    scan_index: int
    x_m: float
    y_m: float
    theta_rad: float
    score: float


@dataclass(frozen=True)
class MatchDebug:
    keyframe_ordinal: int
    scan_index: int
    pose: Pose2D
    submap_points: list[tuple[float, float]]
    current_points: list[tuple[float, float]]


def iter_scan_summaries(lidar_path: Path) -> Iterable[ScanSummary]:
    with lidar_path.open("r", encoding="utf-8") as scan_file:
        for line_number, line in enumerate(scan_file, start=1):
            if not line.strip():
                continue

            try:
                timestamp_ms, samples = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc

            valid_point_count = 0
            min_distance_mm = None
            max_distance_mm = None
            for distance_mm, _intensity, _angle_q6 in samples:
                distance_mm = float(distance_mm)
                if distance_mm <= 0.0:
                    continue

                valid_point_count += 1
                min_distance_mm = (
                    distance_mm
                    if min_distance_mm is None
                    else min(min_distance_mm, distance_mm)
                )
                max_distance_mm = (
                    distance_mm
                    if max_distance_mm is None
                    else max(max_distance_mm, distance_mm)
                )

            yield ScanSummary(
                timestamp_ms=int(timestamp_ms),
                point_count=len(samples),
                valid_point_count=valid_point_count,
                min_distance_mm=min_distance_mm,
                max_distance_mm=max_distance_mm,
            )


def scan_points_to_cartesian(samples: list[list[float]]) -> list[tuple[float, float, int]]:
    points = []
    for distance_mm, intensity, angle_q6 in samples:
        distance_mm = float(distance_mm)
        if distance_mm <= 0.0:
            continue

        angle_rad = float(angle_q6) / 64.0 * math.pi / 180.0
        distance_m = distance_mm / 1000.0
        points.append(
            (
                distance_m * math.cos(angle_rad),
                distance_m * math.sin(angle_rad),
                int(intensity),
            )
        )
    return points


def iter_local_scans(
    lidar_path: Path,
    selected_indices: Iterable[int],
) -> Iterable[LocalScan]:
    remaining_indices = {index for index in selected_indices if index >= 0}
    if not remaining_indices:
        return

    scan_index = -1
    with lidar_path.open("r", encoding="utf-8") as scan_file:
        for line_number, line in enumerate(scan_file, start=1):
            if not line.strip():
                continue

            scan_index += 1
            if scan_index not in remaining_indices:
                continue

            try:
                timestamp_ms, samples = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc

            remaining_indices.remove(scan_index)
            yield LocalScan(
                scan_index=scan_index,
                timestamp_ms=int(timestamp_ms),
                points=scan_points_to_cartesian(samples),
            )

            if not remaining_indices:
                break


def count_scans(lidar_path: Path) -> int:
    scan_count = 0
    with lidar_path.open("r", encoding="utf-8") as scan_file:
        for line in scan_file:
            if line.strip():
                scan_count += 1
    return scan_count


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentage / 100.0
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = index - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def describe_numbers(values: list[float]) -> str:
    if not values:
        return "n=0"

    return (
        f"n={len(values)}, min={min(values):.2f}, median={statistics.median(values):.2f}, "
        f"mean={statistics.fmean(values):.2f}, p95={percentile(values, 95):.2f}, "
        f"max={max(values):.2f}"
    )


def write_valid_points_plot(valid_counts: list[int], output_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write the Stage 1 PNG") from exc

    width = 1600
    height = 600
    left = 72
    right = 24
    top = 42
    bottom = 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_count = max(valid_counts) if valid_counts else 1

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    grid_color = (224, 224, 224)
    axis_color = (64, 64, 64)
    line_color = (109, 40, 217)
    sparse_color = (220, 38, 38)

    for step in range(6):
        y = top + round(plot_height * step / 5)
        draw.line([(left, y), (width - right, y)], fill=grid_color)
        label = round(max_count * (5 - step) / 5)
        draw.text((8, y - 7), str(label), fill=axis_color)

    for step in range(6):
        x = left + round(plot_width * step / 5)
        draw.line([(x, top), (x, height - bottom)], fill=grid_color)
        label = round((len(valid_counts) - 1) * step / 5) if valid_counts else 0
        draw.text((x - 18, height - bottom + 12), str(label), fill=axis_color)

    draw.rectangle((left, top, width - right, height - bottom), outline=axis_color)
    draw.text((left, 12), "Valid LiDAR returns per scan", fill=axis_color)
    draw.text((width // 2 - 38, height - 24), "scan index", fill=axis_color)
    draw.text((8, top - 24), "points", fill=axis_color)

    if valid_counts:
        for pixel_x in range(plot_width):
            start = pixel_x * len(valid_counts) // plot_width
            end = max(start + 1, (pixel_x + 1) * len(valid_counts) // plot_width)
            bucket = valid_counts[start:end]
            bucket_min = min(bucket)
            bucket_max = max(bucket)
            x = left + pixel_x
            y_min = top + plot_height - round(bucket_min / max_count * plot_height)
            y_max = top + plot_height - round(bucket_max / max_count * plot_height)
            color = sparse_color if bucket_min < 50 else line_color
            draw.line([(x, y_max), (x, y_min)], fill=color)

    image.save(output_path)


def write_local_scan_plot(
    scan: LocalScan,
    output_path: Path,
    max_range_m: float,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write the Stage 2 PNGs") from exc

    width = 900
    height = 900
    left = 68
    right = 26
    top = 54
    bottom = 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    center_x = left + plot_width / 2.0
    center_y = top + plot_height / 2.0
    scale = min(plot_width, plot_height) / (2.0 * max_range_m)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    grid_color = (224, 224, 224)
    axis_color = (70, 70, 70)
    point_color = (109, 40, 217)
    origin_color = (220, 38, 38)

    draw.rectangle((left, top, width - right, height - bottom), outline=axis_color)
    grid_extent_m = int(max_range_m // 5) * 5
    for value_m in range(-grid_extent_m, grid_extent_m + 1, 5):
        x = center_x + value_m * scale
        y = center_y - value_m * scale
        draw.line([(x, top), (x, height - bottom)], fill=grid_color)
        draw.line([(left, y), (width - right, y)], fill=grid_color)
        if value_m != 0:
            draw.text((x - 8, height - bottom + 12), str(value_m), fill=axis_color)
            draw.text((12, y - 7), str(value_m), fill=axis_color)

    draw.line([(center_x, top), (center_x, height - bottom)], fill=axis_color)
    draw.line([(left, center_y), (width - right, center_y)], fill=axis_color)

    for range_m in range(5, grid_extent_m + 1, 5):
        radius = range_m * scale
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            outline=(238, 238, 238),
        )

    for x_m, y_m, _intensity in scan.points:
        x = round(center_x + x_m * scale)
        y = round(center_y - y_m * scale)
        if left <= x <= width - right and top <= y <= height - bottom:
            draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=point_color)

    draw.line([(center_x - 7, center_y), (center_x + 7, center_y)], fill=origin_color)
    draw.line([(center_x, center_y - 7), (center_x, center_y + 7)], fill=origin_color)
    draw.text(
        (left, 16),
        (
            f"scan {scan.scan_index:05d} | timestamp {scan.timestamp_ms} | "
            f"valid points {len(scan.points)}"
        ),
        fill=axis_color,
    )
    draw.text((width // 2 - 45, height - 26), "x meters", fill=axis_color)
    draw.text((10, top - 24), "y meters", fill=axis_color)
    image.save(output_path)


def interpolate_color(
    low_color: tuple[int, int, int],
    high_color: tuple[int, int, int],
    weight: float,
) -> tuple[int, int, int]:
    return tuple(
        round(low + (high - low) * weight)
        for low, high in zip(low_color, high_color)
    )


def distance_color(distance_m: float, max_range_m: float) -> tuple[int, int, int]:
    anchors = [
        (0.0, (253, 231, 37)),
        (0.25, (94, 201, 98)),
        (0.5, (33, 145, 140)),
        (0.75, (59, 82, 139)),
        (1.0, (68, 1, 84)),
    ]
    ratio = min(max(distance_m / max_range_m, 0.0), 1.0)

    for (low_ratio, low_color), (high_ratio, high_color) in zip(anchors, anchors[1:]):
        if ratio <= high_ratio:
            weight = (ratio - low_ratio) / (high_ratio - low_ratio)
            return interpolate_color(low_color, high_color, weight)

    return anchors[-1][1]


def build_range_image(
    lidar_path: Path,
    output_path: Path,
    width: int,
    angle_bins: int,
    max_range_m: float,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write the Stage 3 PNG") from exc

    scan_count = count_scans(lidar_path)
    if scan_count == 0:
        raise ValueError(f"no scans found in {lidar_path}")

    pixel_count = width * angle_bins
    ranges_m = [max_range_m] * pixel_count
    has_return = bytearray(pixel_count)

    scan_index = -1
    with lidar_path.open("r", encoding="utf-8") as scan_file:
        for line_number, line in enumerate(scan_file, start=1):
            if not line.strip():
                continue

            scan_index += 1
            try:
                _timestamp_ms, samples = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc

            x = scan_index * (width - 1) // max(scan_count - 1, 1)
            for distance_mm, _intensity, angle_q6 in samples:
                distance_m = float(distance_mm) / 1000.0
                if distance_m <= 0.0 or distance_m > max_range_m:
                    continue

                angle_degrees = (float(angle_q6) / 64.0) % 360.0
                angle_bin = min(int(angle_degrees / 360.0 * angle_bins), angle_bins - 1)
                y = angle_bins - 1 - angle_bin
                pixel_index = y * width + x
                if not has_return[pixel_index] or distance_m < ranges_m[pixel_index]:
                    ranges_m[pixel_index] = distance_m
                    has_return[pixel_index] = 1

    left = 74
    right = 118
    top = 42
    bottom = 58
    image = Image.new(
        "RGB",
        (width + left + right, angle_bins + top + bottom),
        "white",
    )
    draw = ImageDraw.Draw(image)
    pixels = image.load()

    empty_color = (18, 18, 26)
    grid_color = (225, 225, 225)
    axis_color = (64, 64, 64)

    for y in range(angle_bins):
        for x in range(width):
            pixel_index = y * width + x
            pixels[left + x, top + y] = (
                distance_color(ranges_m[pixel_index], max_range_m)
                if has_return[pixel_index]
                else empty_color
            )

    for angle_label in range(0, 361, 90):
        y = top + angle_bins - round(angle_label / 360.0 * (angle_bins - 1)) - 1
        draw.line([(left, y), (left + width - 1, y)], fill=grid_color)
        draw.text((14, y - 7), f"{angle_label}", fill=axis_color)

    for step in range(6):
        x = left + round((width - 1) * step / 5)
        label = round((scan_count - 1) * step / 5)
        draw.line([(x, top), (x, top + angle_bins - 1)], fill=grid_color)
        draw.text((x - 20, top + angle_bins + 12), str(label), fill=axis_color)

    draw.rectangle(
        (left, top, left + width - 1, top + angle_bins - 1),
        outline=axis_color,
    )
    draw.text((left, 14), "Range image: scan index vs LiDAR angle", fill=axis_color)
    draw.text((left + width // 2 - 38, top + angle_bins + 34), "scan index", fill=axis_color)
    draw.text((10, top - 24), "angle deg", fill=axis_color)

    legend_x = left + width + 28
    legend_y = top
    legend_height = angle_bins
    for offset in range(legend_height):
        ratio = offset / max(legend_height - 1, 1)
        distance_m = ratio * max_range_m
        color = distance_color(distance_m, max_range_m)
        draw.line(
            [(legend_x, legend_y + offset), (legend_x + 24, legend_y + offset)],
            fill=color,
        )

    draw.rectangle(
        (legend_x, legend_y, legend_x + 24, legend_y + legend_height - 1),
        outline=axis_color,
    )
    draw.text((legend_x + 34, legend_y - 7), "0 m", fill=axis_color)
    draw.text(
        (legend_x + 34, legend_y + legend_height - 10),
        f"{max_range_m:g} m",
        fill=axis_color,
    )
    draw.text((legend_x, top + angle_bins + 12), "distance", fill=axis_color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def collect_stage1(lidar_path: Path) -> tuple[str, list[int]]:
    timestamps = []
    point_counts = []
    valid_counts = []
    nonzero_min_distances = []
    nonzero_max_distances = []

    for summary in iter_scan_summaries(lidar_path):
        timestamps.append(summary.timestamp_ms)
        point_counts.append(summary.point_count)
        valid_counts.append(summary.valid_point_count)
        if summary.min_distance_mm is not None:
            nonzero_min_distances.append(summary.min_distance_mm)
        if summary.max_distance_mm is not None:
            nonzero_max_distances.append(summary.max_distance_mm)

    gaps = [
        float(current - previous)
        for previous, current in zip(timestamps, timestamps[1:])
    ]
    non_increasing_gap_count = sum(1 for gap in gaps if gap <= 0.0)
    median_gap = statistics.median(gaps) if gaps else 0.0
    large_gap_threshold = median_gap * 3.0 if median_gap else 0.0
    large_gap_count = (
        sum(1 for gap in gaps if gap > large_gap_threshold)
        if large_gap_threshold
        else 0
    )
    duration_s = (timestamps[-1] - timestamps[0]) / 1000.0 if len(timestamps) > 1 else 0.0
    scan_rate_hz = (len(timestamps) - 1) / duration_s if duration_s > 0 else 0.0

    stats = [
        "Stage 1: Data Sanity",
        f"source: {lidar_path}",
        f"scan_count: {len(timestamps)}",
        f"first_timestamp_ms: {timestamps[0] if timestamps else 'n/a'}",
        f"last_timestamp_ms: {timestamps[-1] if timestamps else 'n/a'}",
        f"duration_s: {duration_s:.3f}",
        f"mean_scan_rate_hz: {scan_rate_hz:.3f}",
        f"timestamps_strictly_increase: {'yes' if non_increasing_gap_count == 0 else 'no'}",
        f"non_increasing_gap_count: {non_increasing_gap_count}",
        f"gap_ms: {describe_numbers(gaps)}",
        f"large_gap_threshold_ms: {large_gap_threshold:.2f}",
        f"large_gap_count: {large_gap_count}",
        f"points_per_scan: {describe_numbers([float(value) for value in point_counts])}",
        f"valid_points_per_scan: {describe_numbers([float(value) for value in valid_counts])}",
        f"empty_scan_count: {sum(1 for value in valid_counts if value == 0)}",
        f"sparse_scan_count_lt_50_valid_points: {sum(1 for value in valid_counts if value < 50)}",
        (
            "nonzero_distance_mm_min_by_scan: "
            f"{describe_numbers([float(value) for value in nonzero_min_distances])}"
        ),
        (
            "nonzero_distance_mm_max_by_scan: "
            f"{describe_numbers([float(value) for value in nonzero_max_distances])}"
        ),
    ]
    return "\n".join(stats) + "\n", valid_counts


def run_stage1(args: argparse.Namespace) -> None:
    output_dir = args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_text, valid_counts = collect_stage1(args.lidar)
    stats_path = output_dir / "00_stats.txt"
    plot_path = output_dir / "01_valid_points.png"
    stats_path.write_text(stats_text, encoding="utf-8")
    write_valid_points_plot(valid_counts, plot_path)

    print(f"wrote {stats_path}")
    print(f"wrote {plot_path}")


def parse_indices(raw_indices: str) -> list[int]:
    return [
        int(raw_index.strip())
        for raw_index in raw_indices.split(",")
        if raw_index.strip()
    ]


def default_scan_indices(start: int, stride: int, count: int) -> list[int]:
    return [start + stride * offset for offset in range(count)]


def parse_ints(raw_values: str) -> list[int]:
    return [
        int(raw_value.strip())
        for raw_value in raw_values.split(",")
        if raw_value.strip()
    ]


def parse_windows(raw_windows: str) -> list[tuple[int, int]]:
    windows = []
    for raw_window in raw_windows.split(","):
        if not raw_window.strip():
            continue

        start_text, stop_text = raw_window.split(":", maxsplit=1)
        windows.append((int(start_text), int(stop_text)))
    return windows


def default_overlay_windows() -> list[tuple[int, int]]:
    return [(0, 10), (5000, 5010)]


def write_scan_overlay_plot(
    scans: list[LocalScan],
    output_path: Path,
    max_range_m: float,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write the Stage 4 PNGs") from exc

    width = 1000
    height = 900
    left = 68
    right = 150
    top = 54
    bottom = 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    center_x = left + plot_width / 2.0
    center_y = top + plot_height / 2.0
    scale = min(plot_width, plot_height) / (2.0 * max_range_m)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    grid_color = (224, 224, 224)
    axis_color = (70, 70, 70)
    origin_color = (220, 38, 38)
    colors = [
        (68, 1, 84),
        (72, 35, 116),
        (64, 67, 135),
        (52, 94, 141),
        (41, 120, 142),
        (32, 144, 140),
        (34, 167, 132),
        (68, 190, 112),
        (121, 209, 81),
        (189, 223, 38),
    ]

    draw.rectangle((left, top, width - right, height - bottom), outline=axis_color)
    grid_extent_m = int(max_range_m // 5) * 5
    for value_m in range(-grid_extent_m, grid_extent_m + 1, 5):
        x = center_x + value_m * scale
        y = center_y - value_m * scale
        draw.line([(x, top), (x, height - bottom)], fill=grid_color)
        draw.line([(left, y), (width - right, y)], fill=grid_color)
        if value_m != 0:
            draw.text((x - 8, height - bottom + 12), str(value_m), fill=axis_color)
            draw.text((12, y - 7), str(value_m), fill=axis_color)

    draw.line([(center_x, top), (center_x, height - bottom)], fill=axis_color)
    draw.line([(left, center_y), (width - right, center_y)], fill=axis_color)

    for range_m in range(5, grid_extent_m + 1, 5):
        radius = range_m * scale
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            outline=(238, 238, 238),
        )

    for scan_offset, scan in enumerate(scans):
        color = colors[scan_offset % len(colors)]
        for x_m, y_m, _intensity in scan.points:
            x = round(center_x + x_m * scale)
            y = round(center_y - y_m * scale)
            if left <= x <= width - right and top <= y <= height - bottom:
                draw.point((x, y), fill=color)

    draw.line([(center_x - 7, center_y), (center_x + 7, center_y)], fill=origin_color)
    draw.line([(center_x, center_y - 7), (center_x, center_y + 7)], fill=origin_color)

    first_scan = scans[0].scan_index if scans else 0
    last_scan = scans[-1].scan_index if scans else 0
    draw.text(
        (left, 16),
        f"local overlay: scans {first_scan:05d} to {last_scan:05d}",
        fill=axis_color,
    )
    draw.text((width // 2 - 45, height - 26), "x meters", fill=axis_color)
    draw.text((10, top - 24), "y meters", fill=axis_color)

    legend_x = width - right + 22
    legend_y = top + 8
    for scan_offset, scan in enumerate(scans):
        y = legend_y + scan_offset * 20
        color = colors[scan_offset % len(colors)]
        draw.rectangle((legend_x, y, legend_x + 12, y + 12), fill=color)
        draw.text((legend_x + 18, y - 1), f"{scan.scan_index:05d}", fill=axis_color)

    image.save(output_path)


def select_keyframes(
    lidar_path: Path,
    stride: int,
    min_valid_points: int,
) -> tuple[list[int], list[int]]:
    keyframe_indices = []
    valid_counts = []
    for scan_index, summary in enumerate(iter_scan_summaries(lidar_path)):
        valid_counts.append(summary.valid_point_count)
        if scan_index % stride == 0 and summary.valid_point_count >= min_valid_points:
            keyframe_indices.append(scan_index)

    return keyframe_indices, valid_counts


def sample_evenly(values: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return values[:1]
    if len(values) <= count:
        return values

    return [
        values[round(offset * (len(values) - 1) / (count - 1))]
        for offset in range(count)
    ]


def draw_scan_thumbnail(
    draw: object,
    scan: LocalScan,
    bounds: tuple[int, int, int, int],
    max_range_m: float,
) -> None:
    left, top, right, bottom = bounds
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    scale = min(right - left, bottom - top) / (2.0 * max_range_m)

    grid_color = (230, 230, 230)
    axis_color = (80, 80, 80)
    point_color = (109, 40, 217)
    origin_color = (220, 38, 38)

    draw.rectangle((left, top, right, bottom), outline=axis_color)
    draw.line([(center_x, top), (center_x, bottom)], fill=grid_color)
    draw.line([(left, center_y), (right, center_y)], fill=grid_color)

    for x_m, y_m, _intensity in scan.points:
        x = round(center_x + x_m * scale)
        y = round(center_y - y_m * scale)
        if left <= x <= right and top <= y <= bottom:
            draw.point((x, y), fill=point_color)

    draw.line([(center_x - 4, center_y), (center_x + 4, center_y)], fill=origin_color)
    draw.line([(center_x, center_y - 4), (center_x, center_y + 4)], fill=origin_color)
    draw.text((left + 6, top + 6), f"{scan.scan_index:05d}", fill=axis_color)


def write_keyframe_preview(
    valid_counts: list[int],
    keyframe_indices: list[int],
    sample_scans: list[LocalScan],
    output_path: Path,
    stride: int,
    min_valid_points: int,
    max_range_m: float,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write the Stage 5 PNG") from exc

    width = 1600
    height = 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    left = 72
    right = 28
    top = 72
    plot_height = 300
    plot_width = width - left - right
    plot_bottom = top + plot_height
    max_count = max(valid_counts) if valid_counts else 1

    grid_color = (224, 224, 224)
    axis_color = (64, 64, 64)
    count_color = (110, 110, 110)
    keyframe_color = (109, 40, 217)
    skipped_color = (220, 38, 38)

    draw.text((left, 18), "Stage 5 keyframe selection preview", fill=axis_color)
    draw.text(
        (left, 42),
        (
            f"{len(keyframe_indices)} keyframes from {len(valid_counts)} scans | "
            f"stride {stride} | min valid points {min_valid_points}"
        ),
        fill=axis_color,
    )

    for step in range(6):
        y = top + round(plot_height * step / 5)
        draw.line([(left, y), (width - right, y)], fill=grid_color)
        label = round(max_count * (5 - step) / 5)
        draw.text((8, y - 7), str(label), fill=axis_color)

    for step in range(6):
        x = left + round(plot_width * step / 5)
        label = round((len(valid_counts) - 1) * step / 5) if valid_counts else 0
        draw.line([(x, top), (x, plot_bottom)], fill=grid_color)
        draw.text((x - 20, plot_bottom + 12), str(label), fill=axis_color)

    draw.rectangle((left, top, width - right, plot_bottom), outline=axis_color)
    draw.text((8, top - 24), "valid pts", fill=axis_color)
    draw.text((width // 2 - 38, plot_bottom + 34), "scan index", fill=axis_color)

    if valid_counts:
        for pixel_x in range(plot_width):
            start = pixel_x * len(valid_counts) // plot_width
            end = max(start + 1, (pixel_x + 1) * len(valid_counts) // plot_width)
            bucket = valid_counts[start:end]
            bucket_min = min(bucket)
            bucket_max = max(bucket)
            x = left + pixel_x
            y_min = plot_bottom - round(bucket_min / max_count * plot_height)
            y_max = plot_bottom - round(bucket_max / max_count * plot_height)
            color = skipped_color if bucket_min < min_valid_points else count_color
            draw.line([(x, y_max), (x, y_min)], fill=color)

    for keyframe_index in keyframe_indices:
        x = left + round(keyframe_index / max(len(valid_counts) - 1, 1) * plot_width)
        draw.line([(x, plot_bottom - 34), (x, plot_bottom)], fill=keyframe_color)

    sample_top = 470
    sample_left = 72
    columns = 5
    rows = 2
    gap = 18
    cell_width = (width - sample_left - right - gap * (columns - 1)) // columns
    cell_height = 205
    draw.text((sample_left, sample_top - 34), "Sample selected keyframes", fill=axis_color)

    for sample_offset, scan in enumerate(sample_scans[: columns * rows]):
        row = sample_offset // columns
        column = sample_offset % columns
        cell_left = sample_left + column * (cell_width + gap)
        cell_top = sample_top + row * (cell_height + 44)
        bounds = (
            cell_left,
            cell_top,
            cell_left + cell_width,
            cell_top + cell_height,
        )
        draw_scan_thumbnail(draw, scan, bounds, max_range_m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def thin_scan_points(
    points: list[tuple[float, float, int]],
    max_points: int,
    min_range_m: float,
    max_range_m: float,
) -> list[tuple[float, float]]:
    filtered = [
        (x_m, y_m)
        for x_m, y_m, _intensity in points
        if min_range_m <= math.hypot(x_m, y_m) <= max_range_m
    ]
    if len(filtered) <= max_points:
        return filtered

    step = math.ceil(len(filtered) / max_points)
    return filtered[::step]


def transform_xy_points(
    points: list[tuple[float, float]],
    pose: Pose2D,
) -> list[tuple[float, float]]:
    cos_theta = math.cos(pose.theta_rad)
    sin_theta = math.sin(pose.theta_rad)
    return [
        (
            pose.x_m + cos_theta * x_m - sin_theta * y_m,
            pose.y_m + sin_theta * x_m + cos_theta * y_m,
        )
        for x_m, y_m in points
    ]


def grid_cell(x_m: float, y_m: float, cell_size_m: float) -> tuple[int, int]:
    return (
        math.floor(x_m / cell_size_m),
        math.floor(y_m / cell_size_m),
    )


def inflate_cells(
    occupied_cells: set[tuple[int, int]],
    radius_cells: int,
) -> set[tuple[int, int]]:
    inflated = set()
    for cell_x, cell_y in occupied_cells:
        for offset_x in range(-radius_cells, radius_cells + 1):
            for offset_y in range(-radius_cells, radius_cells + 1):
                inflated.add((cell_x + offset_x, cell_y + offset_y))
    return inflated


def build_submap_cells(
    world_scans: list[list[tuple[float, float]]],
    cell_size_m: float,
    inflation_cells: int,
    submap_keyframes: int,
) -> set[tuple[int, int]]:
    occupied = set()
    for world_points in world_scans[-submap_keyframes:]:
        for x_m, y_m in world_points:
            occupied.add(grid_cell(x_m, y_m, cell_size_m))
    return inflate_cells(occupied, inflation_cells)


def score_pose(
    points: list[tuple[float, float]],
    pose: Pose2D,
    submap_cells: set[tuple[int, int]],
    cell_size_m: float,
) -> float:
    if not points or not submap_cells:
        return 0.0

    cos_theta = math.cos(pose.theta_rad)
    sin_theta = math.sin(pose.theta_rad)
    match_count = 0
    for x_m, y_m in points:
        world_x = pose.x_m + cos_theta * x_m - sin_theta * y_m
        world_y = pose.y_m + sin_theta * x_m + cos_theta * y_m
        if grid_cell(world_x, world_y, cell_size_m) in submap_cells:
            match_count += 1

    return match_count / len(points)


def frange(start: float, stop: float, step: float) -> list[float]:
    values = []
    value = start
    while value <= stop + step * 0.5:
        values.append(value)
        value += step
    return values


def run_odometry_estimation(
    lidar_path: Path,
    limit: int,
    stride: int,
    min_valid_points: int,
    max_points: int,
    min_range_m: float,
    max_range_m: float,
    cell_size_m: float,
    inflation_cells: int,
    submap_keyframes: int,
    search_xy_m: float,
    search_step_m: float,
    search_theta_deg: float,
    search_theta_step_deg: float,
    debug_keyframe_ordinals: set[int] | None = None,
) -> tuple[list[Pose2D], list[MatchDebug]]:
    debug_keyframe_ordinals = debug_keyframe_ordinals or set()
    debug_matches = []
    keyframe_indices, valid_counts = select_keyframes(
        lidar_path=lidar_path,
        stride=stride,
        min_valid_points=min_valid_points,
    )
    selected_indices = [
        index
        for index in keyframe_indices
        if index < limit and valid_counts[index] >= min_valid_points
    ]
    scans = list(iter_local_scans(lidar_path, selected_indices))
    if not scans:
        return [], []

    local_scans = [
        thin_scan_points(
            points=scan.points,
            max_points=max_points,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
        )
        for scan in scans
    ]

    poses = [Pose2D(scans[0].scan_index, 0.0, 0.0, 0.0, 1.0)]
    world_scans = [transform_xy_points(local_scans[0], poses[0])]
    if 0 in debug_keyframe_ordinals:
        debug_matches.append(
            MatchDebug(
                keyframe_ordinal=0,
                scan_index=scans[0].scan_index,
                pose=poses[0],
                submap_points=[],
                current_points=world_scans[0],
            )
        )

    xy_offsets = frange(-search_xy_m, search_xy_m, search_step_m)
    theta_offsets = [
        math.radians(value)
        for value in frange(-search_theta_deg, search_theta_deg, search_theta_step_deg)
    ]

    for scan, local_points in zip(scans[1:], local_scans[1:]):
        previous_pose = poses[-1]
        if len(poses) >= 2:
            previous_previous_pose = poses[-2]
            delta_x = previous_pose.x_m - previous_previous_pose.x_m
            delta_y = previous_pose.y_m - previous_previous_pose.y_m
            delta_theta = previous_pose.theta_rad - previous_previous_pose.theta_rad
        else:
            delta_x = 0.0
            delta_y = 0.0
            delta_theta = 0.0

        predicted_pose = Pose2D(
            scan_index=scan.scan_index,
            x_m=previous_pose.x_m + delta_x,
            y_m=previous_pose.y_m + delta_y,
            theta_rad=previous_pose.theta_rad + delta_theta,
            score=0.0,
        )
        submap_world_points = [
            point
            for world_points in world_scans[-submap_keyframes:]
            for point in world_points
        ]
        submap_cells = build_submap_cells(
            world_scans=world_scans,
            cell_size_m=cell_size_m,
            inflation_cells=inflation_cells,
            submap_keyframes=submap_keyframes,
        )

        best_pose = predicted_pose
        best_score = -1.0
        for offset_theta in theta_offsets:
            theta = predicted_pose.theta_rad + offset_theta
            for offset_x in xy_offsets:
                for offset_y in xy_offsets:
                    candidate = Pose2D(
                        scan_index=scan.scan_index,
                        x_m=predicted_pose.x_m + offset_x,
                        y_m=predicted_pose.y_m + offset_y,
                        theta_rad=theta,
                        score=0.0,
                    )
                    score = score_pose(
                        points=local_points,
                        pose=candidate,
                        submap_cells=submap_cells,
                        cell_size_m=cell_size_m,
                    )
                    movement_penalty = (
                        abs(offset_x) + abs(offset_y) + abs(offset_theta) * 2.0
                    ) * 0.01
                    adjusted_score = score - movement_penalty
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_pose = Pose2D(
                            scan_index=scan.scan_index,
                            x_m=candidate.x_m,
                            y_m=candidate.y_m,
                            theta_rad=candidate.theta_rad,
                            score=score,
                        )

        current_world_points = transform_xy_points(local_points, best_pose)
        keyframe_ordinal = len(poses)
        if keyframe_ordinal in debug_keyframe_ordinals:
            debug_matches.append(
                MatchDebug(
                    keyframe_ordinal=keyframe_ordinal,
                    scan_index=scan.scan_index,
                    pose=best_pose,
                    submap_points=submap_world_points,
                    current_points=current_world_points,
                )
            )

        poses.append(best_pose)
        world_scans.append(current_world_points)

    return poses, debug_matches


def estimate_odometry(
    lidar_path: Path,
    limit: int,
    stride: int,
    min_valid_points: int,
    max_points: int,
    min_range_m: float,
    max_range_m: float,
    cell_size_m: float,
    inflation_cells: int,
    submap_keyframes: int,
    search_xy_m: float,
    search_step_m: float,
    search_theta_deg: float,
    search_theta_step_deg: float,
) -> list[Pose2D]:
    poses, _debug_matches = run_odometry_estimation(
        lidar_path=lidar_path,
        limit=limit,
        stride=stride,
        min_valid_points=min_valid_points,
        max_points=max_points,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
        cell_size_m=cell_size_m,
        inflation_cells=inflation_cells,
        submap_keyframes=submap_keyframes,
        search_xy_m=search_xy_m,
        search_step_m=search_step_m,
        search_theta_deg=search_theta_deg,
        search_theta_step_deg=search_theta_step_deg,
    )

    return poses


def write_odometry_preview(
    poses: list[Pose2D],
    output_path: Path,
    limit: int,
    stride: int,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write the Stage 6 PNGs") from exc

    width = 1100
    height = 900
    left = 74
    right = 220
    top = 64
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    axis_color = (64, 64, 64)
    grid_color = (225, 225, 225)
    path_color = (109, 40, 217)
    start_color = (34, 197, 94)
    end_color = (220, 38, 38)

    if poses:
        min_x = min(pose.x_m for pose in poses)
        max_x = max(pose.x_m for pose in poses)
        min_y = min(pose.y_m for pose in poses)
        max_y = max(pose.y_m for pose in poses)
    else:
        min_x = min_y = -1.0
        max_x = max_y = 1.0

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    padding = max(span_x, span_y) * 0.08
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding
    span_x = max_x - min_x
    span_y = max_y - min_y
    scale = min(plot_width / span_x, plot_height / span_y)

    def project(x_m: float, y_m: float) -> tuple[float, float]:
        x = left + (x_m - min_x) * scale
        y = top + plot_height - (y_m - min_y) * scale
        return x, y

    for step in range(6):
        x = left + round(plot_width * step / 5)
        y = top + round(plot_height * step / 5)
        draw.line([(x, top), (x, top + plot_height)], fill=grid_color)
        draw.line([(left, y), (left + plot_width, y)], fill=grid_color)

    draw.rectangle((left, top, left + plot_width, top + plot_height), outline=axis_color)
    draw.text((left, 18), f"Stage 6 odometry preview: first {limit} scans", fill=axis_color)
    draw.text(
        (left, 40),
        f"{len(poses)} keyframe poses | stride {stride} | local scan-to-submap search",
        fill=axis_color,
    )

    if len(poses) >= 2:
        path = [project(pose.x_m, pose.y_m) for pose in poses]
        draw.line(path, fill=path_color, width=3)
        for pose in poses[:: max(len(poses) // 20, 1)]:
            x, y = project(pose.x_m, pose.y_m)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=path_color)

    if poses:
        start_x, start_y = project(poses[0].x_m, poses[0].y_m)
        end_x, end_y = project(poses[-1].x_m, poses[-1].y_m)
        draw.ellipse((start_x - 6, start_y - 6, start_x + 6, start_y + 6), fill=start_color)
        draw.ellipse((end_x - 6, end_y - 6, end_x + 6, end_y + 6), fill=end_color)

    legend_x = left + plot_width + 34
    legend_y = top + 10
    draw.line([(legend_x, legend_y), (legend_x + 42, legend_y)], fill=path_color, width=3)
    draw.text((legend_x + 52, legend_y - 7), "estimated path", fill=axis_color)
    draw.ellipse(
        (legend_x, legend_y + 28, legend_x + 12, legend_y + 40),
        fill=start_color,
    )
    draw.text((legend_x + 22, legend_y + 26), "start", fill=axis_color)
    draw.ellipse(
        (legend_x, legend_y + 56, legend_x + 12, legend_y + 68),
        fill=end_color,
    )
    draw.text((legend_x + 22, legend_y + 54), "end", fill=axis_color)

    scores = [pose.score for pose in poses[1:]]
    if scores:
        draw.text(
            (legend_x, legend_y + 96),
            f"score median {statistics.median(scores):.2f}",
            fill=axis_color,
        )
        draw.text(
            (legend_x, legend_y + 118),
            f"score min {min(scores):.2f}",
            fill=axis_color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_match_debug_plot(
    match: MatchDebug,
    output_path: Path,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write the Stage 7 PNGs") from exc

    width = 1100
    height = 900
    left = 74
    right = 230
    top = 64
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    axis_color = (64, 64, 64)
    grid_color = (225, 225, 225)
    submap_color = (156, 163, 175)
    current_color = (109, 40, 217)
    pose_color = (220, 38, 38)

    all_points = match.submap_points + match.current_points
    if all_points:
        min_x = min(x_m for x_m, _y_m in all_points)
        max_x = max(x_m for x_m, _y_m in all_points)
        min_y = min(y_m for _x_m, y_m in all_points)
        max_y = max(y_m for _x_m, y_m in all_points)
    else:
        min_x = match.pose.x_m - 1.0
        max_x = match.pose.x_m + 1.0
        min_y = match.pose.y_m - 1.0
        max_y = match.pose.y_m + 1.0

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    padding = max(span_x, span_y) * 0.08
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding
    span_x = max_x - min_x
    span_y = max_y - min_y
    scale = min(plot_width / span_x, plot_height / span_y)

    def project(x_m: float, y_m: float) -> tuple[float, float]:
        x = left + (x_m - min_x) * scale
        y = top + plot_height - (y_m - min_y) * scale
        return x, y

    for step in range(6):
        x = left + round(plot_width * step / 5)
        y = top + round(plot_height * step / 5)
        draw.line([(x, top), (x, top + plot_height)], fill=grid_color)
        draw.line([(left, y), (left + plot_width, y)], fill=grid_color)

    draw.rectangle((left, top, left + plot_width, top + plot_height), outline=axis_color)
    draw.text(
        (left, 18),
        (
            f"Stage 7 match debug: keyframe {match.keyframe_ordinal:04d} "
            f"(scan {match.scan_index:05d})"
        ),
        fill=axis_color,
    )
    draw.text(
        (left, 40),
        f"score {match.pose.score:.2f} | current scan transformed into local submap frame",
        fill=axis_color,
    )

    for x_m, y_m in match.submap_points:
        x, y = project(x_m, y_m)
        draw.point((round(x), round(y)), fill=submap_color)

    for x_m, y_m in match.current_points:
        x, y = project(x_m, y_m)
        draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=current_color)

    pose_x, pose_y = project(match.pose.x_m, match.pose.y_m)
    arrow_len = 26
    arrow_x = pose_x + math.cos(match.pose.theta_rad) * arrow_len
    arrow_y = pose_y - math.sin(match.pose.theta_rad) * arrow_len
    draw.ellipse((pose_x - 5, pose_y - 5, pose_x + 5, pose_y + 5), fill=pose_color)
    draw.line([(pose_x, pose_y), (arrow_x, arrow_y)], fill=pose_color, width=2)

    legend_x = left + plot_width + 34
    legend_y = top + 10
    draw.rectangle((legend_x, legend_y, legend_x + 14, legend_y + 14), fill=submap_color)
    draw.text((legend_x + 24, legend_y - 1), "local submap", fill=axis_color)
    draw.rectangle(
        (legend_x, legend_y + 32, legend_x + 14, legend_y + 46),
        fill=current_color,
    )
    draw.text((legend_x + 24, legend_y + 31), "current scan", fill=axis_color)
    draw.ellipse(
        (legend_x, legend_y + 64, legend_x + 14, legend_y + 78),
        fill=pose_color,
    )
    draw.text((legend_x + 24, legend_y + 63), "pose", fill=axis_color)
    draw.text((legend_x, legend_y + 104), f"x {match.pose.x_m:.2f} m", fill=axis_color)
    draw.text((legend_x, legend_y + 126), f"y {match.pose.y_m:.2f} m", fill=axis_color)
    draw.text(
        (legend_x, legend_y + 148),
        f"theta {math.degrees(match.pose.theta_rad):.1f} deg",
        fill=axis_color,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def run_stage2(args: argparse.Namespace) -> None:
    scan_dir = args.out_dir / "scans"
    scan_dir.mkdir(parents=True, exist_ok=True)

    indices = (
        parse_indices(args.indices)
        if args.indices
        else default_scan_indices(args.start, args.stride, args.count)
    )
    for scan in iter_local_scans(args.lidar, indices):
        output_path = scan_dir / f"scan_{scan.scan_index:05d}.png"
        write_local_scan_plot(scan, output_path, args.max_range_m)
        print(f"wrote {output_path}")


def run_stage3(args: argparse.Namespace) -> None:
    output_path = args.out_dir / "02_range_image.png"
    build_range_image(
        lidar_path=args.lidar,
        output_path=output_path,
        width=args.width,
        angle_bins=args.angle_bins,
        max_range_m=args.max_range_m,
    )
    print(f"wrote {output_path}")


def run_stage4(args: argparse.Namespace) -> None:
    windows = parse_windows(args.windows) if args.windows else default_overlay_windows()
    for start, stop in windows:
        if stop <= start:
            raise ValueError(f"overlay window stop must be greater than start: {start}:{stop}")

        indices = range(start, stop)
        scans = list(iter_local_scans(args.lidar, indices))
        if not scans:
            raise ValueError(f"no scans found for overlay window {start}:{stop}")

        output_path = args.out_dir / f"03_overlay_{start:04d}_{stop:04d}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_scan_overlay_plot(scans, output_path, args.max_range_m)
        print(f"wrote {output_path}")


def run_stage5(args: argparse.Namespace) -> None:
    if args.stride <= 0:
        raise ValueError("--stride must be greater than 0")
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be greater than 0")

    keyframe_indices, valid_counts = select_keyframes(
        lidar_path=args.lidar,
        stride=args.stride,
        min_valid_points=args.min_valid_points,
    )
    if not keyframe_indices:
        raise ValueError("no keyframes selected; lower --min-valid-points or --stride")

    sample_indices = sample_evenly(keyframe_indices, args.sample_count)
    sample_scans = list(iter_local_scans(args.lidar, sample_indices))
    output_path = args.out_dir / "04_keyframe_samples.png"
    write_keyframe_preview(
        valid_counts=valid_counts,
        keyframe_indices=keyframe_indices,
        sample_scans=sample_scans,
        output_path=output_path,
        stride=args.stride,
        min_valid_points=args.min_valid_points,
        max_range_m=args.max_range_m,
    )
    print(f"selected {len(keyframe_indices)} keyframes")
    print(f"wrote {output_path}")


def run_stage6(args: argparse.Namespace) -> None:
    if args.stride <= 0:
        raise ValueError("--stride must be greater than 0")
    if args.max_points <= 0:
        raise ValueError("--max-points must be greater than 0")
    if args.search_step_m <= 0.0:
        raise ValueError("--search-step-m must be greater than 0")
    if args.search_theta_step_deg <= 0.0:
        raise ValueError("--search-theta-step-deg must be greater than 0")

    limits = parse_ints(args.limits)
    for limit in limits:
        poses = estimate_odometry(
            lidar_path=args.lidar,
            limit=limit,
            stride=args.stride,
            min_valid_points=args.min_valid_points,
            max_points=args.max_points,
            min_range_m=args.min_range_m,
            max_range_m=args.max_range_m,
            cell_size_m=args.cell_size_m,
            inflation_cells=args.inflation_cells,
            submap_keyframes=args.submap_keyframes,
            search_xy_m=args.search_xy_m,
            search_step_m=args.search_step_m,
            search_theta_deg=args.search_theta_deg,
            search_theta_step_deg=args.search_theta_step_deg,
        )
        output_path = args.out_dir / f"05_odom_first_{limit}.png"
        write_odometry_preview(
            poses=poses,
            output_path=output_path,
            limit=limit,
            stride=args.stride,
        )
        print(f"estimated {len(poses)} poses for first {limit} scans")
        print(f"wrote {output_path}")


def run_stage7(args: argparse.Namespace) -> None:
    if args.stride <= 0:
        raise ValueError("--stride must be greater than 0")
    if args.max_points <= 0:
        raise ValueError("--max-points must be greater than 0")
    if args.search_step_m <= 0.0:
        raise ValueError("--search-step-m must be greater than 0")
    if args.search_theta_step_deg <= 0.0:
        raise ValueError("--search-theta-step-deg must be greater than 0")

    debug_keyframes = parse_ints(args.keyframes)
    if not debug_keyframes:
        raise ValueError("--keyframes must include at least one keyframe ordinal")

    limit = args.limit or ((max(debug_keyframes) + 1) * args.stride + 1)
    _poses, debug_matches = run_odometry_estimation(
        lidar_path=args.lidar,
        limit=limit,
        stride=args.stride,
        min_valid_points=args.min_valid_points,
        max_points=args.max_points,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
        cell_size_m=args.cell_size_m,
        inflation_cells=args.inflation_cells,
        submap_keyframes=args.submap_keyframes,
        search_xy_m=args.search_xy_m,
        search_step_m=args.search_step_m,
        search_theta_deg=args.search_theta_deg,
        search_theta_step_deg=args.search_theta_step_deg,
        debug_keyframe_ordinals=set(debug_keyframes),
    )

    found_ordinals = {match.keyframe_ordinal for match in debug_matches}
    missing_ordinals = sorted(set(debug_keyframes) - found_ordinals)
    if missing_ordinals:
        raise ValueError(f"did not capture debug keyframes: {missing_ordinals}")

    debug_dir = args.out_dir / "debug_match"
    for match in sorted(debug_matches, key=lambda item: item.keyframe_ordinal):
        output_path = debug_dir / f"keyframe_{match.keyframe_ordinal:04d}.png"
        write_match_debug_plot(match, output_path)
        print(
            f"wrote {output_path} "
            f"(scan {match.scan_index}, score {match.pose.score:.2f})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LiDAR trajectory reconstruction diagnostics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("stage1", "stats"):
        command = subparsers.add_parser(
            command_name,
            help="generate Stage 1 data sanity outputs",
        )
        command.add_argument(
            "--lidar",
            type=Path,
            default=DEFAULT_LIDAR_PATH,
            help=f"input NDJSON LiDAR file (default: {DEFAULT_LIDAR_PATH})",
        )
        command.add_argument(
            "--out-dir",
            type=Path,
            default=DEFAULT_OUTPUT_DIR,
            help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
        )
        command.set_defaults(func=run_stage1)

    for command_name in ("stage2", "plot-scans"):
        command = subparsers.add_parser(
            command_name,
            help="generate Stage 2 local scan plots",
        )
        command.add_argument(
            "--lidar",
            type=Path,
            default=DEFAULT_LIDAR_PATH,
            help=f"input NDJSON LiDAR file (default: {DEFAULT_LIDAR_PATH})",
        )
        command.add_argument(
            "--out-dir",
            type=Path,
            default=DEFAULT_OUTPUT_DIR,
            help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
        )
        command.add_argument(
            "--indices",
            help="comma-separated scan indices to plot",
        )
        command.add_argument(
            "--start",
            type=int,
            default=0,
            help="first scan index for default sampling",
        )
        command.add_argument(
            "--stride",
            type=int,
            default=1000,
            help="scan index stride for default sampling",
        )
        command.add_argument(
            "--count",
            type=int,
            default=20,
            help="number of scan plots for default sampling",
        )
        command.add_argument(
            "--max-range-m",
            type=float,
            default=16.0,
            help="fixed plot radius in meters",
        )
        command.set_defaults(func=run_stage2)

    for command_name in ("stage3", "range-image"):
        command = subparsers.add_parser(
            command_name,
            help="generate Stage 3 scan-index/angle range image",
        )
        command.add_argument(
            "--lidar",
            type=Path,
            default=DEFAULT_LIDAR_PATH,
            help=f"input NDJSON LiDAR file (default: {DEFAULT_LIDAR_PATH})",
        )
        command.add_argument(
            "--out-dir",
            type=Path,
            default=DEFAULT_OUTPUT_DIR,
            help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
        )
        command.add_argument(
            "--width",
            type=int,
            default=1600,
            help="range image plot width in pixels",
        )
        command.add_argument(
            "--angle-bins",
            type=int,
            default=720,
            help="number of vertical angle bins",
        )
        command.add_argument(
            "--max-range-m",
            type=float,
            default=16.0,
            help="maximum distance shown in meters",
        )
        command.set_defaults(func=run_stage3)

    for command_name in ("stage4", "overlay-scans"):
        command = subparsers.add_parser(
            command_name,
            help="generate Stage 4 consecutive-scan overlays",
        )
        command.add_argument(
            "--lidar",
            type=Path,
            default=DEFAULT_LIDAR_PATH,
            help=f"input NDJSON LiDAR file (default: {DEFAULT_LIDAR_PATH})",
        )
        command.add_argument(
            "--out-dir",
            type=Path,
            default=DEFAULT_OUTPUT_DIR,
            help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
        )
        command.add_argument(
            "--windows",
            help="comma-separated half-open scan windows, for example 0:10,5000:5010",
        )
        command.add_argument(
            "--max-range-m",
            type=float,
            default=16.0,
            help="fixed plot radius in meters",
        )
        command.set_defaults(func=run_stage4)

    for command_name in ("stage5", "keyframes"):
        command = subparsers.add_parser(
            command_name,
            help="generate Stage 5 keyframe selection preview",
        )
        command.add_argument(
            "--lidar",
            type=Path,
            default=DEFAULT_LIDAR_PATH,
            help=f"input NDJSON LiDAR file (default: {DEFAULT_LIDAR_PATH})",
        )
        command.add_argument(
            "--out-dir",
            type=Path,
            default=DEFAULT_OUTPUT_DIR,
            help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
        )
        command.add_argument(
            "--stride",
            type=int,
            default=10,
            help="select one keyframe every N scans",
        )
        command.add_argument(
            "--min-valid-points",
            type=int,
            default=100,
            help="skip scans with fewer valid returns",
        )
        command.add_argument(
            "--sample-count",
            type=int,
            default=10,
            help="number of selected keyframes to draw as local scan samples",
        )
        command.add_argument(
            "--max-range-m",
            type=float,
            default=16.0,
            help="fixed sample plot radius in meters",
        )
        command.set_defaults(func=run_stage5)

    for command_name in ("stage6", "odom-preview"):
        command = subparsers.add_parser(
            command_name,
            help="generate Stage 6 first odometry previews",
        )
        command.add_argument(
            "--lidar",
            type=Path,
            default=DEFAULT_LIDAR_PATH,
            help=f"input NDJSON LiDAR file (default: {DEFAULT_LIDAR_PATH})",
        )
        command.add_argument(
            "--out-dir",
            type=Path,
            default=DEFAULT_OUTPUT_DIR,
            help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
        )
        command.add_argument(
            "--limits",
            default="1000,3000",
            help="comma-separated scan prefixes to render",
        )
        command.add_argument(
            "--stride",
            type=int,
            default=10,
            help="keyframe stride in scans",
        )
        command.add_argument(
            "--min-valid-points",
            type=int,
            default=100,
            help="skip scans with fewer valid returns",
        )
        command.add_argument(
            "--max-points",
            type=int,
            default=120,
            help="maximum scan points used by the matcher",
        )
        command.add_argument(
            "--min-range-m",
            type=float,
            default=0.7,
            help="minimum point range used by the matcher",
        )
        command.add_argument(
            "--max-range-m",
            type=float,
            default=12.0,
            help="maximum point range used by the matcher",
        )
        command.add_argument(
            "--cell-size-m",
            type=float,
            default=0.25,
            help="submap occupancy grid cell size in meters",
        )
        command.add_argument(
            "--inflation-cells",
            type=int,
            default=1,
            help="cell inflation radius for scan matching tolerance",
        )
        command.add_argument(
            "--submap-keyframes",
            type=int,
            default=8,
            help="number of recent keyframes in the local submap",
        )
        command.add_argument(
            "--search-xy-m",
            type=float,
            default=0.8,
            help="translation search radius around prediction in meters",
        )
        command.add_argument(
            "--search-step-m",
            type=float,
            default=0.4,
            help="translation search step in meters",
        )
        command.add_argument(
            "--search-theta-deg",
            type=float,
            default=8.0,
            help="heading search radius around prediction in degrees",
        )
        command.add_argument(
            "--search-theta-step-deg",
            type=float,
            default=4.0,
            help="heading search step in degrees",
        )
        command.set_defaults(func=run_stage6)

    for command_name in ("stage7", "debug-match"):
        command = subparsers.add_parser(
            command_name,
            help="generate Stage 7 submap match debug overlays",
        )
        command.add_argument(
            "--lidar",
            type=Path,
            default=DEFAULT_LIDAR_PATH,
            help=f"input NDJSON LiDAR file (default: {DEFAULT_LIDAR_PATH})",
        )
        command.add_argument(
            "--out-dir",
            type=Path,
            default=DEFAULT_OUTPUT_DIR,
            help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
        )
        command.add_argument(
            "--keyframes",
            default="120,240",
            help="comma-separated keyframe ordinals to debug",
        )
        command.add_argument(
            "--limit",
            type=int,
            help="scan prefix limit; defaults to enough scans for requested keyframes",
        )
        command.add_argument(
            "--stride",
            type=int,
            default=10,
            help="keyframe stride in scans",
        )
        command.add_argument(
            "--min-valid-points",
            type=int,
            default=100,
            help="skip scans with fewer valid returns",
        )
        command.add_argument(
            "--max-points",
            type=int,
            default=120,
            help="maximum scan points used by the matcher",
        )
        command.add_argument(
            "--min-range-m",
            type=float,
            default=0.7,
            help="minimum point range used by the matcher",
        )
        command.add_argument(
            "--max-range-m",
            type=float,
            default=12.0,
            help="maximum point range used by the matcher",
        )
        command.add_argument(
            "--cell-size-m",
            type=float,
            default=0.25,
            help="submap occupancy grid cell size in meters",
        )
        command.add_argument(
            "--inflation-cells",
            type=int,
            default=1,
            help="cell inflation radius for scan matching tolerance",
        )
        command.add_argument(
            "--submap-keyframes",
            type=int,
            default=8,
            help="number of recent keyframes in the local submap",
        )
        command.add_argument(
            "--search-xy-m",
            type=float,
            default=0.8,
            help="translation search radius around prediction in meters",
        )
        command.add_argument(
            "--search-step-m",
            type=float,
            default=0.4,
            help="translation search step in meters",
        )
        command.add_argument(
            "--search-theta-deg",
            type=float,
            default=8.0,
            help="heading search radius around prediction in degrees",
        )
        command.add_argument(
            "--search-theta-step-deg",
            type=float,
            default=4.0,
            help="heading search step in degrees",
        )
        command.set_defaults(func=run_stage7)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
