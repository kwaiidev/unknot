from __future__ import annotations

import argparse
import json
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
