from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .frontend.degeneracy import compute_degeneracy_metrics


DEFAULT_LIDAR_PATH = Path("challenge/lidar.json")
DEFAULT_OUTPUT_DIR = Path("outputs")
PoseCovariance = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
ScanPoint = tuple[float, float] | tuple[float, float, float, str] | tuple[float, float, float, str, float]


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
class MatchConfidence:
    best_score: float
    second_best_score: float
    score_gap: float
    normalized_score_gap: float
    score_variance: float
    near_best_count: int
    candidate_count: int


@dataclass(frozen=True)
class ScoreSurface:
    hessian: PoseCovariance
    information: PoseCovariance
    eigenvalues: tuple[float, float, float]
    weak_directions: tuple[str, ...]
    condition_number: float
    degenerate: bool


@dataclass(frozen=True)
class ScanScoreBreakdown:
    linear_score_fraction: float
    salient_score_fraction: float
    neutral_score_fraction: float
    linear_point_fraction: float
    salient_point_fraction: float
    neutral_point_fraction: float
    linear_weight_fraction: float
    salient_weight_fraction: float
    neutral_weight_fraction: float


@dataclass(frozen=True)
class DegeneracyDiagnostics:
    mode: str
    entropy: float
    linearity: float
    dominant_angle: float
    confidence: float
    yaw_bias_rad: float
    yaw_correction_applied: bool
    recovery_retry: bool


@dataclass(frozen=True)
class PoseHypothesis:
    pose: tuple[float, float, float]
    covariance: PoseCovariance | None
    score: float
    weight: float


@dataclass(frozen=True)
class HypothesisDiagnostics:
    active_count: int
    weight_entropy: float
    max_divergence: float
    max_translation_divergence_m: float
    max_heading_divergence_rad: float
    aisle_mode: bool
    ambiguity_gate: bool
    ambiguity_reason: str | None
    weights: tuple[float, ...]


@dataclass(frozen=True)
class AisleFrameDiagnostics:
    corridor_id: int | None
    axis_x: float
    axis_y: float
    raw_axis_x: float
    raw_axis_y: float
    confidence: float
    local_linearity: float
    corridor_confidence: float
    raw_to_stable_angle_rad: float
    raw_axis_change_rad: float
    stable_axis_change_rad: float
    axis_update_magnitude_rad: float
    ema_smoothing_angle_rad: float
    covariance_rotation_delta_rad: float
    jitter_reduction_rad: float
    corridor_updated: bool
    corridor_reset: bool
    corridor_frozen_reason: str | None
    along_motion_m: float
    cross_motion_m: float
    expected_along_m: float
    longitudinal_drift_m: float
    sigma_along_m: float
    sigma_cross_m: float
    sigma_theta_rad: float
    covariance_world: PoseCovariance
    covariance_relative: PoseCovariance | None = None
    anisotropic_applied: bool = False


@dataclass
class CorridorState:
    corridor_id: int
    axis: tuple[float, float]
    covariance_axis: float
    confidence: float
    last_update_frame: int
    ema_axis: tuple[float, float]
    stale_frames: int = 0
    last_position: tuple[float, float] | None = None
    previous_raw_axis: tuple[float, float] | None = None
    previous_stable_axis: tuple[float, float] | None = None


@dataclass(frozen=True)
class Pose2D:
    scan_index: int
    x_m: float
    y_m: float
    theta_rad: float
    score: float
    covariance: PoseCovariance | None = None
    confidence: MatchConfidence | None = None
    surface: ScoreSurface | None = None
    score_breakdown: ScanScoreBreakdown | None = None
    degeneracy: DegeneracyDiagnostics | None = None
    hypothesis_diagnostics: HypothesisDiagnostics | None = None
    aisle_frame: AisleFrameDiagnostics | None = None
    hypotheses: tuple[PoseHypothesis, ...] = ()
    accepted: bool = True


@dataclass(frozen=True)
class SearchStage:
    xy_radius_m: float
    xy_step_m: float
    theta_radius_rad: float
    theta_step_rad: float


@dataclass(frozen=True)
class ScoredCandidate:
    adjusted_score: float
    raw_score: float
    pose: Pose2D


@dataclass(frozen=True)
class SearchResult:
    best_candidate: ScoredCandidate | None
    candidates: list[ScoredCandidate]
    stage_count: int
    final_step_m: float
    final_theta_step_rad: float


@dataclass(frozen=True)
class HypothesisCandidate:
    candidate: ScoredCandidate
    predicted_pose: Pose2D
    previous_pose: Pose2D
    search_result: SearchResult
    confidence: MatchConfidence
    degeneracy: DegeneracyDiagnostics
    close_scores: bool
    parent_weight: float


@dataclass(frozen=True)
class MatchQualityConfig:
    confidence_threshold: float = 0.35
    score_gap_threshold: float = 0.0001
    near_best_score_percent: float = 0.5
    max_near_best_candidates: int = 60
    covariance_scaling: float = 0.02
    hessian_conditioning_threshold: float = 2000.0
    hessian_min_curvature: float = 0.001
    expanded_search_xy_m: float = 1.6
    expanded_search_theta_deg: float = 16.0
    feature_weighting_enabled: bool = True
    linear_feature_weight: float = 0.35
    salient_feature_weight: float = 2.0
    saliency_neighbor_count: int = 3
    saliency_linear_curvature_deg: float = 8.0
    saliency_corner_curvature_deg: float = 25.0
    saliency_gap_m: float = 0.45
    degeneracy_scoring_enabled: bool = True
    aisle_aligned_feature_weight: float = 0.1
    aisle_cross_feature_weight: float = 2.5
    aisle_yaw_lambda: float = 0.08
    aisle_theta_window_scale: float = 0.65
    yaw_bias_correction_alpha: float = 0.18
    yaw_bias_top_score_percent: float = 2.0
    aisle_persistence_frames: int = 8
    aisle_recovery_xy_m: float = 0.35
    aisle_recovery_theta_deg: float = 4.0
    hypothesis_count: int = 3
    hypothesis_weight_threshold: float = 0.01
    hypothesis_ambiguous_weight_threshold: float = 0.001
    hypothesis_score_temperature: float = 0.10
    hypothesis_ambiguous_temperature_scale: float = 2.0
    hypothesis_branch_score_percent: float = 2.0
    hypothesis_close_score_gap: float = 0.001
    hypothesis_near_best_count_threshold: int = 5
    hypothesis_divergence_threshold_m: float = 0.8
    hypothesis_theta_distance_weight_m: float = 1.0
    hypothesis_ambiguity_covariance_scale: float = 4.0
    loop_hypothesis_agreement_m: float = 0.75
    loop_hypothesis_agreement_theta_deg: float = 12.0
    loop_strong_score_gap: float = 0.04
    anisotropic_constraints_enabled: bool = True
    anisotropic_min_confidence: float = 0.35
    anisotropic_aisle_linearity_threshold: float = 3.5
    anisotropic_along_sigma_m: float = 0.08
    anisotropic_cross_sigma_m: float = 0.18
    anisotropic_theta_sigma_deg: float = 3.0
    anisotropic_scan_covariance_floor_m: float = 0.04
    anisotropic_drift_gain: float = 0.35
    corridor_frame_enabled: bool = True
    corridor_axis_alpha: float = 0.12
    corridor_ema_beta: float = 0.08
    corridor_update_min_confidence: float = 0.35
    corridor_update_min_linearity: float = 3.5
    corridor_max_hypothesis_entropy: float = 0.35
    corridor_reset_axis_deg: float = 35.0
    corridor_reset_position_gap_m: float = 3.0
    corridor_reset_low_confidence_frames: int = 12
    corridor_graph_enabled: bool = True
    corridor_segment_axis_threshold_deg: float = 25.0
    corridor_segment_entropy_threshold: float = 0.65
    corridor_segment_confidence_jump: float = 0.35
    corridor_segment_drift_reset_m: float = 0.8
    corridor_segment_min_poses: int = 2
    corridor_edge_support_required: int = 2
    corridor_similarity_low_confidence: float = 0.35
    corridor_graph_conflict_distance_m: float = 2.0
    corridor_graph_conflict_theta_deg: float = 35.0
    corridor_graph_opt_iterations: int = 20
    corridor_graph_opt_step: float = 0.15


@dataclass(frozen=True)
class ScanMatchDiagnostics:
    keyframe_ordinal: int
    scan_index: int
    x_m: float
    y_m: float
    theta_rad: float
    raw_score: float
    accepted: bool
    rejected_reason: str | None
    expanded_search: bool
    stage_count: int
    confidence: MatchConfidence
    surface: ScoreSurface | None
    covariance: PoseCovariance | None
    score_breakdown: ScanScoreBreakdown | None
    degeneracy: DegeneracyDiagnostics | None
    hypothesis: HypothesisDiagnostics | None
    aisle_frame: AisleFrameDiagnostics | None


@dataclass(frozen=True)
class MatchDebug:
    keyframe_ordinal: int
    scan_index: int
    pose: Pose2D
    submap_points: list[tuple[float, float]]
    current_points: list[tuple[float, float]]


@dataclass(frozen=True)
class OdometryResult:
    poses: list[Pose2D]
    debug_matches: list[MatchDebug]
    diagnostics: list[ScanMatchDiagnostics]


@dataclass(frozen=True)
class HeadingSample:
    scan_index: int
    angle_deg: float | None
    smoothed_deg: float | None
    confidence: float
    valid_points: int


@dataclass(frozen=True)
class LoopClosure:
    candidate_id: int
    from_ordinal: int
    to_ordinal: int
    raw_distance_m: float
    signature_distance: float


@dataclass(frozen=True)
class LoopConstraint:
    from_ordinal: int
    to_ordinal: int
    raw_distance_m: float
    signature_distance: float
    relative_x_m: float
    relative_y_m: float
    relative_theta_rad: float
    match_score: float
    score_gap: float


@dataclass(frozen=True)
class CorridorSegment:
    id: int
    start_pose: tuple[float, float, float]
    end_pose: tuple[float, float, float]
    axis: tuple[float, float]
    length_estimate: float
    signature: dict
    parent_poses: tuple[int, ...]


@dataclass(frozen=True)
class CorridorGraphEdge:
    from_segment: int
    to_segment: int
    edge_type: str
    relative_transform: tuple[float, float, float]
    confidence_score: float
    supporting_scan_indices: tuple[int, ...]
    support_count: int
    status: str


@dataclass(frozen=True)
class CorridorGraph:
    segments: tuple[CorridorSegment, ...]
    edges: tuple[CorridorGraphEdge, ...]
    pose_to_segment: dict[int, int]
    ambiguous_edges_count: int
    conflicting_loop_closures: int
    topological_inconsistencies: tuple[str, ...]


@dataclass(frozen=True)
class LoopClosureDiagnostics:
    candidate_id: int
    from_ordinal: int
    to_ordinal: int
    from_scan_index: int
    to_scan_index: int
    accepted: bool
    reason: str
    agreement_count: int
    from_hypotheses: int
    to_hypotheses: int
    from_ambiguous: bool
    to_ambiguous: bool
    match_score: float
    score_gap: float
    confidence: MatchConfidence | None
    hessian_eigenvalues: tuple[float, float, float] | None
    covariance: PoseCovariance | None
    from_corridor_id: int | None
    to_corridor_id: int | None
    corridor_signature_similarity: float | None
    from_degeneracy: dict
    to_degeneracy: dict
    predicted_relative_transform: tuple[float, float, float] | None
    matched_relative_transform: tuple[float, float, float] | None
    transform_residual_m: float | None
    transform_residual_theta_rad: float | None
    ambiguous: bool
    conflicting: bool
    raw_distance_m: float
    signature_distance: float


@dataclass(frozen=True)
class LoopAnalysisContext:
    pose_to_segment: dict[int, int]
    segments_by_id: dict[int, CorridorSegment]


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


def filter_scan_xy_points(
    points: list[tuple[float, float, int]],
    min_range_m: float,
    max_range_m: float,
) -> list[tuple[float, float]]:
    return [
        (x_m, y_m)
        for x_m, y_m, _intensity in points
        if min_range_m <= math.hypot(x_m, y_m) <= max_range_m
    ]


def classify_scan_point(
    points: list[tuple[float, float]],
    index: int,
    config: MatchQualityConfig,
) -> str:
    if len(points) <= 2:
        return "salient"

    neighbor_count = max(config.saliency_neighbor_count, 1)
    previous_index = max(index - neighbor_count, 0)
    next_index = min(index + neighbor_count, len(points) - 1)
    if previous_index == index or next_index == index:
        return "salient"

    x_m, y_m = points[index]
    previous_x_m, previous_y_m = points[previous_index]
    next_x_m, next_y_m = points[next_index]
    immediate_previous_gap_m = (
        math.hypot(x_m - points[index - 1][0], y_m - points[index - 1][1])
        if index > 0
        else math.inf
    )
    immediate_next_gap_m = (
        math.hypot(x_m - points[index + 1][0], y_m - points[index + 1][1])
        if index + 1 < len(points)
        else math.inf
    )
    if (
        immediate_previous_gap_m > config.saliency_gap_m
        or immediate_next_gap_m > config.saliency_gap_m
    ):
        return "salient"

    previous_dx = x_m - previous_x_m
    previous_dy = y_m - previous_y_m
    next_dx = next_x_m - x_m
    next_dy = next_y_m - y_m
    previous_length_m = math.hypot(previous_dx, previous_dy)
    next_length_m = math.hypot(next_dx, next_dy)
    if previous_length_m <= 1e-6 or next_length_m <= 1e-6:
        return "neutral"

    cosine = (
        (previous_dx * next_dx + previous_dy * next_dy)
        / (previous_length_m * next_length_m)
    )
    cosine = min(max(cosine, -1.0), 1.0)
    curvature_deg = math.degrees(math.acos(cosine))
    if curvature_deg >= config.saliency_corner_curvature_deg:
        return "salient"
    if curvature_deg <= config.saliency_linear_curvature_deg:
        return "linear"
    return "neutral"


def scan_point_local_angle(
    points: list[tuple[float, float]],
    index: int,
    config: MatchQualityConfig,
) -> float:
    if not points:
        return 0.0

    neighbor_count = max(config.saliency_neighbor_count, 1)
    previous_index = max(index - neighbor_count, 0)
    next_index = min(index + neighbor_count, len(points) - 1)
    x_m, y_m = points[index]
    if previous_index == next_index:
        return math.atan2(y_m, x_m) % math.pi

    previous_x_m, previous_y_m = points[previous_index]
    next_x_m, next_y_m = points[next_index]
    dx_m = next_x_m - previous_x_m
    dy_m = next_y_m - previous_y_m
    if math.hypot(dx_m, dy_m) <= 1e-6:
        return math.atan2(y_m, x_m) % math.pi
    return math.atan2(dy_m, dx_m) % math.pi


def build_scan_match_points(
    points: list[tuple[float, float, int]],
    max_points: int,
    min_range_m: float,
    max_range_m: float,
    config: MatchQualityConfig,
) -> list[ScanPoint]:
    filtered = filter_scan_xy_points(points, min_range_m, max_range_m)
    weighted_points = []
    for index, (x_m, y_m) in enumerate(filtered):
        feature_class = classify_scan_point(filtered, index, config)
        local_angle = scan_point_local_angle(filtered, index, config)
        if not config.feature_weighting_enabled:
            raw_weight = 1.0
        elif feature_class == "linear":
            raw_weight = config.linear_feature_weight
        elif feature_class == "salient":
            raw_weight = config.salient_feature_weight
        else:
            raw_weight = 1.0

        raw_weight = max(raw_weight, 0.0)
        weighted_points.append((x_m, y_m, raw_weight, feature_class, local_angle))

    if not weighted_points:
        return []

    selected_points = weighted_points
    if len(selected_points) > max_points:
        step = math.ceil(len(selected_points) / max_points)
        selected_points = selected_points[::step]

    selected_weight_total = sum(
        weight for _x_m, _y_m, weight, _feature_class, _angle in selected_points
    )
    mean_weight = selected_weight_total / len(selected_points)
    if mean_weight <= 0.0:
        mean_weight = 1.0
    return [
        (x_m, y_m, weight / mean_weight, feature_class, local_angle)
        for x_m, y_m, weight, feature_class, local_angle in selected_points
    ]


def scan_point_xy(point: ScanPoint) -> tuple[float, float]:
    return point[0], point[1]


def scan_point_weight(point: ScanPoint) -> float:
    return float(point[2]) if len(point) >= 3 else 1.0


def scan_point_feature_class(point: ScanPoint) -> str:
    return str(point[3]) if len(point) >= 4 else "neutral"


def scan_point_local_direction(point: ScanPoint) -> float | None:
    return float(point[4]) if len(point) >= 5 else None


def apply_aisle_degeneracy_weights(
    points: list[ScanPoint],
    dominant_angle: float,
    config: MatchQualityConfig,
) -> list[ScanPoint]:
    if not config.degeneracy_scoring_enabled:
        return points

    weighted_points = []
    raw_total = 0.0
    for point in points:
        x_m, y_m = scan_point_xy(point)
        feature_class = scan_point_feature_class(point)
        local_angle = scan_point_local_direction(point)
        base_weight = scan_point_weight(point)
        if local_angle is not None:
            aligned = abs(normalize_line_angle_delta_rad(local_angle - dominant_angle)) <= math.radians(10.0)
        else:
            aligned = False

        if aligned:
            weight = min(base_weight, config.aisle_aligned_feature_weight)
        elif feature_class == "linear":
            weight = max(base_weight, 1.0)
        else:
            weight = max(base_weight, config.aisle_cross_feature_weight)

        raw_total += weight
        weighted_points.append((x_m, y_m, max(weight, 0.0), feature_class, local_angle or 0.0))

    if not weighted_points:
        return []

    mean_weight = raw_total / len(weighted_points)
    if mean_weight <= 0.0:
        mean_weight = 1.0
    return [
        (x_m, y_m, weight / mean_weight, feature_class, local_angle)
        for x_m, y_m, weight, feature_class, local_angle in weighted_points
    ]


def transform_xy_points(
    points: list[ScanPoint],
    pose: Pose2D,
) -> list[tuple[float, float]]:
    cos_theta = math.cos(pose.theta_rad)
    sin_theta = math.sin(pose.theta_rad)
    return [
        (
            pose.x_m + cos_theta * x_m - sin_theta * y_m,
            pose.y_m + sin_theta * x_m + cos_theta * y_m,
        )
        for x_m, y_m in (scan_point_xy(point) for point in points)
    ]


def grid_cell(x_m: float, y_m: float, cell_size_m: float) -> tuple[int, int]:
    return (
        math.floor(x_m / cell_size_m),
        math.floor(y_m / cell_size_m),
    )


def normalize_angle_rad(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def normalize_line_angle_delta_rad(angle_rad: float) -> float:
    return (angle_rad + math.pi / 2.0) % math.pi - math.pi / 2.0


def build_submap_distance_lookup(
    world_scans: list[list[ScanPoint]],
    cell_size_m: float,
    inflation_cells: int,
    submap_keyframes: int,
) -> tuple[dict[tuple[int, int], float], float]:
    occupied = set()
    for world_points in world_scans[-submap_keyframes:]:
        for point in world_points:
            x_m, y_m = scan_point_xy(point)
            occupied.add(grid_cell(x_m, y_m, cell_size_m))

    radius_cells = max(inflation_cells + 2, 1)
    max_distance_m = radius_cells * cell_size_m
    offsets = []
    for offset_x in range(-radius_cells, radius_cells + 1):
        for offset_y in range(-radius_cells, radius_cells + 1):
            distance_m = math.hypot(offset_x, offset_y) * cell_size_m
            if distance_m <= max_distance_m:
                offsets.append((offset_x, offset_y, distance_m))

    distance_lookup = {}
    for cell_x, cell_y in occupied:
        for offset_x, offset_y, distance_m in offsets:
            cell = (cell_x + offset_x, cell_y + offset_y)
            previous_distance_m = distance_lookup.get(cell)
            if previous_distance_m is None or distance_m < previous_distance_m:
                distance_lookup[cell] = distance_m

    return distance_lookup, max_distance_m


def default_score_breakdown() -> ScanScoreBreakdown:
    return ScanScoreBreakdown(
        linear_score_fraction=0.0,
        salient_score_fraction=0.0,
        neutral_score_fraction=1.0,
        linear_point_fraction=0.0,
        salient_point_fraction=0.0,
        neutral_point_fraction=1.0,
        linear_weight_fraction=0.0,
        salient_weight_fraction=0.0,
        neutral_weight_fraction=1.0,
    )


def default_degeneracy_diagnostics() -> DegeneracyDiagnostics:
    return DegeneracyDiagnostics(
        mode="NORMAL",
        entropy=1.0,
        linearity=1.0,
        dominant_angle=0.0,
        confidence=0.0,
        yaw_bias_rad=0.0,
        yaw_correction_applied=False,
        recovery_retry=False,
    )


def default_pose_covariance() -> PoseCovariance:
    return (
        (1e-6, 0.0, 0.0),
        (0.0, 1e-6, 0.0),
        (0.0, 0.0, 1e-6),
    )


def default_hypothesis_diagnostics() -> HypothesisDiagnostics:
    return HypothesisDiagnostics(
        active_count=1,
        weight_entropy=0.0,
        max_divergence=0.0,
        max_translation_divergence_m=0.0,
        max_heading_divergence_rad=0.0,
        aisle_mode=False,
        ambiguity_gate=False,
        ambiguity_reason=None,
        weights=(1.0,),
    )


def normalize_vector_2d(axis: tuple[float, float]) -> tuple[float, float]:
    axis_x, axis_y = axis
    norm = max(math.hypot(axis_x, axis_y), 1e-9)
    return axis_x / norm, axis_y / norm


def align_axis_to_reference(
    axis: tuple[float, float],
    reference: tuple[float, float],
) -> tuple[float, float]:
    axis_x, axis_y = normalize_vector_2d(axis)
    reference_x, reference_y = normalize_vector_2d(reference)
    if axis_x * reference_x + axis_y * reference_y < 0.0:
        return -axis_x, -axis_y
    return axis_x, axis_y


def axis_angle_delta_rad(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    first_angle = math.atan2(first[1], first[0])
    second_angle = math.atan2(second[1], second[0])
    return abs(normalize_line_angle_delta_rad(first_angle - second_angle))


def blend_axes(
    reference: tuple[float, float],
    update: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    alpha = min(max(alpha, 0.0), 1.0)
    update = align_axis_to_reference(update, reference)
    ref_x, ref_y = normalize_vector_2d(reference)
    upd_x, upd_y = update
    return normalize_vector_2d(
        (
            (1.0 - alpha) * ref_x + alpha * upd_x,
            (1.0 - alpha) * ref_y + alpha * upd_y,
        )
    )


def estimate_aisle_frame(points: list[ScanPoint]) -> tuple[tuple[float, float], float, float]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for aisle-frame estimation") from exc

    if len(points) < 3:
        return (1.0, 0.0), 0.0, 1.0

    xy = np.array([scan_point_xy(point) for point in points], dtype=float)
    centered = xy - xy.mean(axis=0)
    covariance = centered.T @ centered / max(len(xy) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    large_index = int(np.argmax(eigenvalues))
    axis = eigenvectors[:, large_index]
    axis_x = float(axis[0])
    axis_y = float(axis[1])
    norm = max(math.hypot(axis_x, axis_y), 1e-9)
    axis_x /= norm
    axis_y /= norm
    if axis_x < 0.0:
        axis_x = -axis_x
        axis_y = -axis_y

    small = max(float(np.min(eigenvalues)), 1e-9)
    large = max(float(np.max(eigenvalues)), 1e-9)
    linearity = large / small
    confidence = min(max(math.log1p(linearity) / math.log1p(30.0), 0.0), 1.0)
    return (axis_x, axis_y), confidence, linearity


def rotate_axis_to_world(
    local_axis: tuple[float, float],
    theta_rad: float,
) -> tuple[float, float]:
    axis_x, axis_y = local_axis
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    world_x = cos_theta * axis_x - sin_theta * axis_y
    world_y = sin_theta * axis_x + cos_theta * axis_y
    norm = max(math.hypot(world_x, world_y), 1e-9)
    return world_x / norm, world_y / norm


def create_corridor_state(
    corridor_id: int,
    axis: tuple[float, float],
    confidence: float,
    frame_index: int,
    pose: Pose2D,
) -> CorridorState:
    axis = normalize_vector_2d(axis)
    return CorridorState(
        corridor_id=corridor_id,
        axis=axis,
        covariance_axis=max(1.0 - confidence, 1e-6),
        confidence=confidence,
        last_update_frame=frame_index,
        ema_axis=axis,
        stale_frames=0,
        last_position=(pose.x_m, pose.y_m),
        previous_raw_axis=axis,
        previous_stable_axis=axis,
    )


def update_corridor_axis(
    raw_axis: tuple[float, float],
    confidence: float,
    linearity: float,
    frame_index: int,
    pose: Pose2D,
    degeneracy_mode: str,
    hypothesis_diagnostics: HypothesisDiagnostics | None,
    accepted: bool,
    corridor_states: dict[int, CorridorState],
    active_corridor_id: int | None,
    next_corridor_id: int,
    config: MatchQualityConfig,
) -> tuple[
    tuple[float, float],
    int | None,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    bool,
    bool,
    str | None,
    int | None,
    int,
]:
    raw_axis = normalize_vector_2d(raw_axis)
    if not config.corridor_frame_enabled:
        return (
            raw_axis,
            None,
            confidence,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            False,
            False,
            "disabled",
            active_corridor_id,
            next_corridor_id,
        )

    state = (
        corridor_states.get(active_corridor_id)
        if active_corridor_id is not None
        else None
    )
    hypothesis_entropy = (
        hypothesis_diagnostics.weight_entropy
        if hypothesis_diagnostics is not None
        else 0.0
    )
    can_update = (
        accepted
        and confidence >= config.corridor_update_min_confidence
        and linearity >= config.corridor_update_min_linearity
        and hypothesis_entropy <= config.corridor_max_hypothesis_entropy
        and degeneracy_mode != "INTERSECTION"
    )
    reset = False
    frozen_reason = None

    if degeneracy_mode == "INTERSECTION" and state is not None:
        reset = True
        frozen_reason = "intersection"
        active_corridor_id = None
        state = None

    if state is not None and state.last_position is not None:
        travel_gap_m = math.hypot(
            pose.x_m - state.last_position[0],
            pose.y_m - state.last_position[1],
        )
        if travel_gap_m > config.corridor_reset_position_gap_m:
            reset = True
            frozen_reason = "position_gap"
            active_corridor_id = None
            state = None

    if state is not None and can_update:
        raw_delta_to_state = axis_angle_delta_rad(raw_axis, state.ema_axis)
        if raw_delta_to_state > math.radians(config.corridor_reset_axis_deg):
            reset = True
            frozen_reason = "axis_change"
            active_corridor_id = None
            state = None

    if state is None and can_update:
        state = create_corridor_state(
            corridor_id=next_corridor_id,
            axis=raw_axis,
            confidence=confidence,
            frame_index=frame_index,
            pose=pose,
        )
        corridor_states[state.corridor_id] = state
        active_corridor_id = state.corridor_id
        next_corridor_id += 1
        raw_to_stable = 0.0
        return (
            state.ema_axis,
            state.corridor_id,
            state.confidence,
            raw_to_stable,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            True,
            reset,
            "initialized" if not reset else frozen_reason,
            active_corridor_id,
            next_corridor_id,
        )

    if state is None:
        reason = frozen_reason
        if reason is None:
            if not accepted:
                reason = "rejected"
            elif degeneracy_mode == "INTERSECTION":
                reason = "intersection"
            elif confidence < config.corridor_update_min_confidence:
                reason = "low_confidence"
            elif linearity < config.corridor_update_min_linearity:
                reason = "low_linearity"
            elif hypothesis_entropy > config.corridor_max_hypothesis_entropy:
                reason = "hypothesis_entropy"
            else:
                reason = "no_corridor"
        return (
            raw_axis,
            None,
            confidence,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            False,
            reset,
            reason,
            active_corridor_id,
            next_corridor_id,
        )

    previous_axis = state.axis
    previous_ema_axis = state.ema_axis
    previous_raw_axis = state.previous_raw_axis or raw_axis
    previous_stable_axis = state.previous_stable_axis or state.ema_axis
    raw_axis = align_axis_to_reference(raw_axis, previous_ema_axis)
    raw_axis_change = axis_angle_delta_rad(raw_axis, previous_raw_axis)
    updated = False

    if can_update:
        alpha = min(max(config.corridor_axis_alpha * confidence, 0.0), 1.0)
        new_axis = blend_axes(previous_axis, raw_axis, alpha)
        state.axis = new_axis
        state.ema_axis = blend_axes(
            previous_ema_axis,
            new_axis,
            config.corridor_ema_beta,
        )
        state.confidence = 0.9 * state.confidence + 0.1 * confidence
        state.covariance_axis = max(1.0 - state.confidence, 1e-6)
        state.last_update_frame = frame_index
        state.stale_frames = 0
        state.last_position = (pose.x_m, pose.y_m)
        updated = True
        frozen_reason = None
    else:
        state.stale_frames += 1
        state.confidence *= 0.98
        if not accepted:
            frozen_reason = "rejected"
        elif degeneracy_mode == "INTERSECTION":
            frozen_reason = "intersection"
        elif confidence < config.corridor_update_min_confidence:
            frozen_reason = "low_confidence"
        elif linearity < config.corridor_update_min_linearity:
            frozen_reason = "low_linearity"
        elif hypothesis_entropy > config.corridor_max_hypothesis_entropy:
            frozen_reason = "hypothesis_entropy"
        else:
            frozen_reason = "gated"
        if state.stale_frames >= config.corridor_reset_low_confidence_frames:
            reset = True
            frozen_reason = "stale_corridor"
            active_corridor_id = None

    stable_axis = state.ema_axis
    axis_update_magnitude = axis_angle_delta_rad(previous_axis, state.axis)
    ema_smoothing_effect = axis_angle_delta_rad(state.axis, stable_axis)
    stable_axis_change = axis_angle_delta_rad(stable_axis, previous_stable_axis)
    raw_to_stable = axis_angle_delta_rad(raw_axis, stable_axis)
    jitter_reduction = raw_axis_change - stable_axis_change
    state.previous_raw_axis = raw_axis
    state.previous_stable_axis = stable_axis

    return (
        stable_axis,
        state.corridor_id,
        state.confidence,
        raw_to_stable,
        raw_axis_change,
        stable_axis_change,
        axis_update_magnitude,
        ema_smoothing_effect,
        stable_axis_change,
        jitter_reduction,
        updated,
        reset,
        frozen_reason,
        active_corridor_id,
        next_corridor_id,
    )


def build_aisle_frame_diagnostics(
    axis: tuple[float, float],
    raw_axis: tuple[float, float],
    confidence: float,
    linearity: float,
    corridor_id: int | None,
    corridor_confidence: float,
    raw_to_stable_angle_rad: float,
    raw_axis_change_rad: float,
    stable_axis_change_rad: float,
    axis_update_magnitude_rad: float,
    ema_smoothing_angle_rad: float,
    covariance_rotation_delta_rad: float,
    jitter_reduction_rad: float,
    corridor_updated: bool,
    corridor_reset: bool,
    corridor_frozen_reason: str | None,
    previous_pose: Pose2D,
    pose: Pose2D,
    expected_along_m: float,
    config: MatchQualityConfig,
) -> AisleFrameDiagnostics:
    axis_x, axis_y = normalize_vector_2d(axis)
    raw_axis_x, raw_axis_y = align_axis_to_reference(raw_axis, axis)
    delta_x_m = pose.x_m - previous_pose.x_m
    delta_y_m = pose.y_m - previous_pose.y_m
    along_motion_m = delta_x_m * axis_x + delta_y_m * axis_y
    cross_axis_x = -axis_y
    cross_axis_y = axis_x
    cross_motion_m = delta_x_m * cross_axis_x + delta_y_m * cross_axis_y
    longitudinal_drift_m = along_motion_m - expected_along_m
    sigma_floor_m = max(config.anisotropic_scan_covariance_floor_m, 1e-6)
    sigma_along_m = max(config.anisotropic_along_sigma_m, sigma_floor_m)
    sigma_cross_m = max(config.anisotropic_cross_sigma_m, sigma_floor_m)
    sigma_theta_rad = max(math.radians(config.anisotropic_theta_sigma_deg), 1e-9)
    if pose.covariance is not None:
        covariance = pose.covariance
        cross_variance = (
            cross_axis_x
            * (covariance[0][0] * cross_axis_x + covariance[0][1] * cross_axis_y)
            + cross_axis_y
            * (covariance[1][0] * cross_axis_x + covariance[1][1] * cross_axis_y)
        )
        sigma_cross_m = max(
            sigma_floor_m,
            min(config.anisotropic_cross_sigma_m, math.sqrt(max(cross_variance, 0.0))),
        )
        sigma_theta_rad = max(
            1e-9,
            min(sigma_theta_rad, math.sqrt(max(covariance[2][2], 0.0))),
        )
    covariance_world = aisle_covariance_world(
        axis_x=axis_x,
        axis_y=axis_y,
        sigma_along_m=sigma_along_m,
        sigma_cross_m=sigma_cross_m,
        sigma_theta_rad=sigma_theta_rad,
    )
    return AisleFrameDiagnostics(
        corridor_id=corridor_id,
        axis_x=axis_x,
        axis_y=axis_y,
        raw_axis_x=raw_axis_x,
        raw_axis_y=raw_axis_y,
        confidence=confidence,
        local_linearity=linearity,
        corridor_confidence=corridor_confidence,
        raw_to_stable_angle_rad=raw_to_stable_angle_rad,
        raw_axis_change_rad=raw_axis_change_rad,
        stable_axis_change_rad=stable_axis_change_rad,
        axis_update_magnitude_rad=axis_update_magnitude_rad,
        ema_smoothing_angle_rad=ema_smoothing_angle_rad,
        covariance_rotation_delta_rad=covariance_rotation_delta_rad,
        jitter_reduction_rad=jitter_reduction_rad,
        corridor_updated=corridor_updated,
        corridor_reset=corridor_reset,
        corridor_frozen_reason=corridor_frozen_reason,
        along_motion_m=along_motion_m,
        cross_motion_m=cross_motion_m,
        expected_along_m=expected_along_m,
        longitudinal_drift_m=longitudinal_drift_m,
        sigma_along_m=sigma_along_m,
        sigma_cross_m=sigma_cross_m,
        sigma_theta_rad=sigma_theta_rad,
        covariance_world=covariance_world,
        covariance_relative=None,
        anisotropic_applied=False,
    )


def should_apply_anisotropic_constraints(
    pose: Pose2D,
    config: MatchQualityConfig,
) -> bool:
    if not config.anisotropic_constraints_enabled or pose.aisle_frame is None:
        return False
    if pose.aisle_frame.confidence < config.anisotropic_min_confidence:
        return False

    aisle_like = (
        pose.degeneracy is not None and pose.degeneracy.mode == "AISLE"
    ) or pose.aisle_frame.local_linearity >= config.anisotropic_aisle_linearity_threshold
    if not aisle_like:
        return False

    if (
        pose.hypothesis_diagnostics is not None
        and pose.hypothesis_diagnostics.max_divergence
        > config.hypothesis_divergence_threshold_m
    ):
        return False
    return True


def finalize_aisle_frame_for_pose(
    frame: AisleFrameDiagnostics,
    previous_pose: Pose2D,
    pose: Pose2D,
    config: MatchQualityConfig,
    np: object,
) -> AisleFrameDiagnostics:
    pose_with_frame = Pose2D(
        scan_index=pose.scan_index,
        x_m=pose.x_m,
        y_m=pose.y_m,
        theta_rad=pose.theta_rad,
        score=pose.score,
        covariance=pose.covariance,
        confidence=pose.confidence,
        surface=pose.surface,
        score_breakdown=pose.score_breakdown,
        degeneracy=pose.degeneracy,
        hypothesis_diagnostics=pose.hypothesis_diagnostics,
        aisle_frame=frame,
        hypotheses=pose.hypotheses,
        accepted=pose.accepted,
    )
    anisotropic_applied = should_apply_anisotropic_constraints(
        pose_with_frame,
        config,
    )
    covariance_relative = None
    if anisotropic_applied:
        covariance_relative = covariance_to_tuple(
            relative_frame_covariance(
                covariance=frame.covariance_world,
                reference_pose=previous_pose,
                np=np,
            )
    )
    return AisleFrameDiagnostics(
        corridor_id=frame.corridor_id,
        axis_x=frame.axis_x,
        axis_y=frame.axis_y,
        raw_axis_x=frame.raw_axis_x,
        raw_axis_y=frame.raw_axis_y,
        confidence=frame.confidence,
        local_linearity=frame.local_linearity,
        corridor_confidence=frame.corridor_confidence,
        raw_to_stable_angle_rad=frame.raw_to_stable_angle_rad,
        raw_axis_change_rad=frame.raw_axis_change_rad,
        stable_axis_change_rad=frame.stable_axis_change_rad,
        axis_update_magnitude_rad=frame.axis_update_magnitude_rad,
        ema_smoothing_angle_rad=frame.ema_smoothing_angle_rad,
        covariance_rotation_delta_rad=frame.covariance_rotation_delta_rad,
        jitter_reduction_rad=frame.jitter_reduction_rad,
        corridor_updated=frame.corridor_updated,
        corridor_reset=frame.corridor_reset,
        corridor_frozen_reason=frame.corridor_frozen_reason,
        along_motion_m=frame.along_motion_m,
        cross_motion_m=frame.cross_motion_m,
        expected_along_m=frame.expected_along_m,
        longitudinal_drift_m=frame.longitudinal_drift_m,
        sigma_along_m=frame.sigma_along_m,
        sigma_cross_m=frame.sigma_cross_m,
        sigma_theta_rad=frame.sigma_theta_rad,
        covariance_world=frame.covariance_world,
        covariance_relative=covariance_relative,
        anisotropic_applied=anisotropic_applied,
    )


def anisotropic_relative_pose_delta(
    from_pose: Pose2D,
    to_pose: Pose2D,
    config: MatchQualityConfig,
) -> tuple[float, float, float]:
    if not should_apply_anisotropic_constraints(to_pose, config):
        return relative_pose_delta(from_pose, to_pose)
    if to_pose.aisle_frame is None:
        return relative_pose_delta(from_pose, to_pose)

    axis_x = to_pose.aisle_frame.axis_x
    axis_y = to_pose.aisle_frame.axis_y
    cross_x = -axis_y
    cross_y = axis_x
    corrected_along_m = (
        to_pose.aisle_frame.along_motion_m
        - config.anisotropic_drift_gain * to_pose.aisle_frame.longitudinal_drift_m
    )
    corrected_world_x = (
        axis_x * corrected_along_m
        + cross_x * to_pose.aisle_frame.cross_motion_m
    )
    corrected_world_y = (
        axis_y * corrected_along_m
        + cross_y * to_pose.aisle_frame.cross_motion_m
    )
    cos_theta = math.cos(from_pose.theta_rad)
    sin_theta = math.sin(from_pose.theta_rad)
    return (
        cos_theta * corrected_world_x + sin_theta * corrected_world_y,
        -sin_theta * corrected_world_x + cos_theta * corrected_world_y,
        normalize_angle_rad(to_pose.theta_rad - from_pose.theta_rad),
    )


def pose_to_hypothesis(pose: Pose2D, weight: float) -> PoseHypothesis:
    return PoseHypothesis(
        pose=(pose.x_m, pose.y_m, pose.theta_rad),
        covariance=pose.covariance,
        score=pose.score,
        weight=weight,
    )


def hypothesis_to_pose2d(
    hypothesis: PoseHypothesis,
    scan_index: int,
    score: float | None = None,
) -> Pose2D:
    x_m, y_m, theta_rad = hypothesis.pose
    return Pose2D(
        scan_index=scan_index,
        x_m=x_m,
        y_m=y_m,
        theta_rad=theta_rad,
        score=hypothesis.score if score is None else score,
        covariance=hypothesis.covariance,
    )


def propagate_hypothesis(
    hypothesis: PoseHypothesis,
    delta_x_m: float,
    delta_y_m: float,
    delta_theta_rad: float,
) -> PoseHypothesis:
    x_m, y_m, theta_rad = hypothesis.pose
    return PoseHypothesis(
        pose=(
            x_m + delta_x_m,
            y_m + delta_y_m,
            normalize_angle_rad(theta_rad + delta_theta_rad),
        ),
        covariance=hypothesis.covariance,
        score=hypothesis.score,
        weight=hypothesis.weight,
    )


def normalize_hypotheses(
    hypotheses: list[PoseHypothesis],
) -> list[PoseHypothesis]:
    total_weight = sum(max(hypothesis.weight, 0.0) for hypothesis in hypotheses)
    if total_weight <= 0.0:
        uniform_weight = 1.0 / max(len(hypotheses), 1)
        return [
            PoseHypothesis(
                pose=hypothesis.pose,
                covariance=hypothesis.covariance,
                score=hypothesis.score,
                weight=uniform_weight,
            )
            for hypothesis in hypotheses
        ]

    return [
        PoseHypothesis(
            pose=hypothesis.pose,
            covariance=hypothesis.covariance,
            score=hypothesis.score,
            weight=max(hypothesis.weight, 0.0) / total_weight,
        )
        for hypothesis in hypotheses
    ]


def hypothesis_pair_distance(
    first: PoseHypothesis,
    second: PoseHypothesis,
    theta_weight_m: float,
) -> tuple[float, float, float]:
    first_x_m, first_y_m, first_theta = first.pose
    second_x_m, second_y_m, second_theta = second.pose
    translation_m = math.hypot(first_x_m - second_x_m, first_y_m - second_y_m)
    heading_rad = abs(normalize_angle_rad(first_theta - second_theta))
    distance = math.hypot(translation_m, heading_rad * theta_weight_m)
    return distance, translation_m, heading_rad


def hypothesis_divergence(
    hypotheses: list[PoseHypothesis],
    theta_weight_m: float,
) -> tuple[float, float, float]:
    max_distance = 0.0
    max_translation_m = 0.0
    max_heading_rad = 0.0
    for first_index, first in enumerate(hypotheses):
        for second in hypotheses[first_index + 1 :]:
            distance, translation_m, heading_rad = hypothesis_pair_distance(
                first,
                second,
                theta_weight_m,
            )
            max_distance = max(max_distance, distance)
            max_translation_m = max(max_translation_m, translation_m)
            max_heading_rad = max(max_heading_rad, heading_rad)
    return max_distance, max_translation_m, max_heading_rad


def hypothesis_weight_entropy(hypotheses: list[PoseHypothesis]) -> float:
    if len(hypotheses) <= 1:
        return 0.0

    probabilities = [max(hypothesis.weight, 0.0) for hypothesis in hypotheses]
    total = sum(probabilities)
    if total <= 0.0:
        return 0.0

    entropy = 0.0
    for probability in probabilities:
        if probability <= 0.0:
            continue
        normalized = probability / total
        entropy -= normalized * math.log(normalized)
    return min(max(entropy / math.log(len(hypotheses)), 0.0), 1.0)


def build_hypothesis_diagnostics(
    hypotheses: list[PoseHypothesis],
    aisle_mode: bool,
    ambiguity_gate: bool,
    ambiguity_reason: str | None,
    config: MatchQualityConfig,
) -> HypothesisDiagnostics:
    max_distance, max_translation_m, max_heading_rad = hypothesis_divergence(
        hypotheses,
        theta_weight_m=config.hypothesis_theta_distance_weight_m,
    )
    return HypothesisDiagnostics(
        active_count=len(hypotheses),
        weight_entropy=hypothesis_weight_entropy(hypotheses),
        max_divergence=max_distance,
        max_translation_divergence_m=max_translation_m,
        max_heading_divergence_rad=max_heading_rad,
        aisle_mode=aisle_mode,
        ambiguity_gate=ambiguity_gate,
        ambiguity_reason=ambiguity_reason,
        weights=tuple(hypothesis.weight for hypothesis in hypotheses),
    )


def degeneracy_diagnostics_from_metrics(
    metrics: dict,
    yaw_bias_rad: float = 0.0,
    yaw_correction_applied: bool = False,
    recovery_retry: bool = False,
) -> DegeneracyDiagnostics:
    return DegeneracyDiagnostics(
        mode=str(metrics.get("mode", "NORMAL")),
        entropy=float(metrics.get("entropy", 1.0)),
        linearity=float(metrics.get("linearity", 1.0)),
        dominant_angle=float(metrics.get("dominant_angle", 0.0)),
        confidence=float(metrics.get("confidence", 0.0)),
        yaw_bias_rad=yaw_bias_rad,
        yaw_correction_applied=yaw_correction_applied,
        recovery_retry=recovery_retry,
    )


def score_pose_details(
    points: list[ScanPoint],
    pose: Pose2D,
    submap_distances: dict[tuple[int, int], float],
    cell_size_m: float,
    max_distance_m: float,
) -> tuple[float, ScanScoreBreakdown]:
    if not points or not submap_distances:
        return 0.0, default_score_breakdown()

    cos_theta = math.cos(pose.theta_rad)
    sin_theta = math.sin(pose.theta_rad)
    total_distance_m = 0.0
    total_weight = 0.0
    matched_weight = 0.0
    point_counts = {"linear": 0.0, "salient": 0.0, "neutral": 0.0}
    weight_totals = {"linear": 0.0, "salient": 0.0, "neutral": 0.0}
    score_totals = {"linear": 0.0, "salient": 0.0, "neutral": 0.0}
    for point in points:
        x_m, y_m = scan_point_xy(point)
        weight = max(scan_point_weight(point), 0.0)
        feature_class = scan_point_feature_class(point)
        if feature_class not in point_counts:
            feature_class = "neutral"

        world_x = pose.x_m + cos_theta * x_m - sin_theta * y_m
        world_y = pose.y_m + sin_theta * x_m + cos_theta * y_m
        distance_m = submap_distances.get(
            grid_cell(world_x, world_y, cell_size_m),
            max_distance_m,
        )
        total_weight += weight
        total_distance_m += distance_m * weight
        point_counts[feature_class] += 1.0
        weight_totals[feature_class] += weight
        unmatched_penalty_m = 0.0
        if distance_m < max_distance_m:
            matched_weight += weight
        else:
            unmatched_penalty_m = max_distance_m * 0.7

        score_totals[feature_class] += weight / (1.0 + distance_m + unmatched_penalty_m)

    if total_weight <= 0.0:
        return 0.0, default_score_breakdown()

    mean_distance_m = total_distance_m / total_weight
    unmatched_ratio = 1.0 - matched_weight / total_weight
    distance_error = mean_distance_m + unmatched_ratio * max_distance_m * 0.7
    score = 1.0 / (1.0 + distance_error)
    total_score_evidence = sum(score_totals.values())
    if total_score_evidence <= 0.0:
        total_score_evidence = 1.0

    point_count = float(len(points))
    breakdown = ScanScoreBreakdown(
        linear_score_fraction=score_totals["linear"] / total_score_evidence,
        salient_score_fraction=score_totals["salient"] / total_score_evidence,
        neutral_score_fraction=score_totals["neutral"] / total_score_evidence,
        linear_point_fraction=point_counts["linear"] / point_count,
        salient_point_fraction=point_counts["salient"] / point_count,
        neutral_point_fraction=point_counts["neutral"] / point_count,
        linear_weight_fraction=weight_totals["linear"] / total_weight,
        salient_weight_fraction=weight_totals["salient"] / total_weight,
        neutral_weight_fraction=weight_totals["neutral"] / total_weight,
    )
    return score, breakdown


def score_pose(
    points: list[ScanPoint],
    pose: Pose2D,
    submap_distances: dict[tuple[int, int], float],
    cell_size_m: float,
    max_distance_m: float,
) -> float:
    score, _breakdown = score_pose_details(
        points=points,
        pose=pose,
        submap_distances=submap_distances,
        cell_size_m=cell_size_m,
        max_distance_m=max_distance_m,
    )
    return score


def frange(start: float, stop: float, step: float) -> list[float]:
    values = []
    value = start
    while value <= stop + step * 0.5:
        values.append(value)
        value += step
    return values


def covariance_to_tuple(matrix: object) -> PoseCovariance:
    return (
        (
            float(matrix[0][0]),
            float(matrix[0][1]),
            float(matrix[0][2]),
        ),
        (
            float(matrix[1][0]),
            float(matrix[1][1]),
            float(matrix[1][2]),
        ),
        (
            float(matrix[2][0]),
            float(matrix[2][1]),
            float(matrix[2][2]),
        ),
    )


def default_match_confidence() -> MatchConfidence:
    return MatchConfidence(
        best_score=1.0,
        second_best_score=0.0,
        score_gap=1.0,
        normalized_score_gap=1.0,
        score_variance=0.0,
        near_best_count=1,
        candidate_count=1,
    )


def build_search_stages(
    search_xy_m: float,
    search_step_m: float,
    search_theta_deg: float,
    search_theta_step_deg: float,
) -> list[SearchStage]:
    theta_radius_rad = math.radians(search_theta_deg)
    theta_step_rad = math.radians(search_theta_step_deg)
    coarse_xy_step_m = max(search_step_m * 2.0, search_step_m)
    coarse_theta_step_rad = max(theta_step_rad * 2.0, theta_step_rad)
    fine_xy_step_m = max(search_step_m * 0.5, 1e-6)
    fine_theta_step_rad = max(theta_step_rad * 0.5, 1e-9)

    return [
        SearchStage(
            xy_radius_m=search_xy_m,
            xy_step_m=coarse_xy_step_m,
            theta_radius_rad=theta_radius_rad,
            theta_step_rad=coarse_theta_step_rad,
        ),
        SearchStage(
            xy_radius_m=min(search_xy_m, max(coarse_xy_step_m, search_step_m)),
            xy_step_m=search_step_m,
            theta_radius_rad=min(theta_radius_rad, max(coarse_theta_step_rad, theta_step_rad)),
            theta_step_rad=theta_step_rad,
        ),
        SearchStage(
            xy_radius_m=min(search_xy_m, max(search_step_m, fine_xy_step_m)),
            xy_step_m=fine_xy_step_m,
            theta_radius_rad=min(theta_radius_rad, max(theta_step_rad, fine_theta_step_rad)),
            theta_step_rad=fine_theta_step_rad,
        ),
    ]


def evaluate_scan_match_candidate(
    points: list[ScanPoint],
    candidate: Pose2D,
    predicted_pose: Pose2D,
    previous_pose: Pose2D,
    submap_distances: dict[tuple[int, int], float],
    cell_size_m: float,
    max_distance_m: float,
    expected_theta: float | None,
    heading_weight: float,
    degeneracy_mode: str,
    theta_prior: float,
    yaw_lambda: float,
) -> ScoredCandidate:
    raw_score = score_pose(
        points=points,
        pose=candidate,
        submap_distances=submap_distances,
        cell_size_m=cell_size_m,
        max_distance_m=max_distance_m,
    )
    offset_x = candidate.x_m - predicted_pose.x_m
    offset_y = candidate.y_m - predicted_pose.y_m
    offset_theta = normalize_angle_rad(candidate.theta_rad - predicted_pose.theta_rad)
    movement_penalty = (
        (abs(offset_x) + abs(offset_y)) * 0.04
        + abs(offset_theta) * 0.2
    )
    move_x = candidate.x_m - previous_pose.x_m
    move_y = candidate.y_m - previous_pose.y_m
    previous_cos = math.cos(previous_pose.theta_rad)
    previous_sin = math.sin(previous_pose.theta_rad)
    forward_m = previous_cos * move_x + previous_sin * move_y
    lateral_m = -previous_sin * move_x + previous_cos * move_y
    aisle_motion_penalty = abs(lateral_m) * 0.08
    if forward_m < 0.0:
        aisle_motion_penalty += abs(forward_m) * 0.03
    heading_penalty = (
        heading_weight * abs(normalize_angle_rad(candidate.theta_rad - expected_theta))
        if expected_theta is not None
        else 0.0
    )
    yaw_penalty = (
        yaw_lambda * normalize_angle_rad(candidate.theta_rad - theta_prior) ** 2
        if degeneracy_mode == "AISLE"
        else 0.0
    )
    return ScoredCandidate(
        adjusted_score=(
            raw_score
            - movement_penalty
            - aisle_motion_penalty
            - heading_penalty
            - yaw_penalty
        ),
        raw_score=raw_score,
        pose=Pose2D(
            scan_index=candidate.scan_index,
            x_m=candidate.x_m,
            y_m=candidate.y_m,
            theta_rad=candidate.theta_rad,
            score=raw_score,
        ),
    )


def hierarchical_scan_match(
    points: list[ScanPoint],
    predicted_pose: Pose2D,
    previous_pose: Pose2D,
    submap_distances: dict[tuple[int, int], float],
    cell_size_m: float,
    max_distance_m: float,
    search_xy_m: float,
    search_step_m: float,
    search_theta_deg: float,
    search_theta_step_deg: float,
    expected_theta: float | None,
    heading_weight: float,
    degeneracy_mode: str = "NORMAL",
    theta_prior: float | None = None,
    yaw_lambda: float = 0.0,
) -> SearchResult:
    stages = build_search_stages(
        search_xy_m=search_xy_m,
        search_step_m=search_step_m,
        search_theta_deg=search_theta_deg,
        search_theta_step_deg=search_theta_step_deg,
    )
    search_theta_rad = math.radians(search_theta_deg)
    center_pose = predicted_pose
    theta_prior = predicted_pose.theta_rad if theta_prior is None else theta_prior
    best_candidate = None
    candidates = []
    seen = set()

    for stage in stages:
        stage_best = None
        xy_offsets = frange(-stage.xy_radius_m, stage.xy_radius_m, stage.xy_step_m)
        theta_offsets = frange(
            -stage.theta_radius_rad,
            stage.theta_radius_rad,
            stage.theta_step_rad,
        )
        for offset_theta in theta_offsets:
            theta = center_pose.theta_rad + offset_theta
            total_theta_offset = normalize_angle_rad(theta - predicted_pose.theta_rad)
            if abs(total_theta_offset) > search_theta_rad + stage.theta_step_rad * 0.5:
                continue
            for offset_x in xy_offsets:
                x_m = center_pose.x_m + offset_x
                total_x_offset = x_m - predicted_pose.x_m
                if abs(total_x_offset) > search_xy_m + stage.xy_step_m * 0.5:
                    continue
                for offset_y in xy_offsets:
                    y_m = center_pose.y_m + offset_y
                    total_y_offset = y_m - predicted_pose.y_m
                    if abs(total_y_offset) > search_xy_m + stage.xy_step_m * 0.5:
                        continue

                    key = (
                        round(x_m, 6),
                        round(y_m, 6),
                        round(normalize_angle_rad(theta), 8),
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    candidate = Pose2D(
                        scan_index=predicted_pose.scan_index,
                        x_m=x_m,
                        y_m=y_m,
                        theta_rad=theta,
                        score=0.0,
                    )
                    scored_candidate = evaluate_scan_match_candidate(
                        points=points,
                        candidate=candidate,
                        predicted_pose=predicted_pose,
                        previous_pose=previous_pose,
                        submap_distances=submap_distances,
                        cell_size_m=cell_size_m,
                        max_distance_m=max_distance_m,
                        expected_theta=expected_theta,
                        heading_weight=heading_weight,
                        degeneracy_mode=degeneracy_mode,
                        theta_prior=theta_prior,
                        yaw_lambda=yaw_lambda,
                    )
                    candidates.append(scored_candidate)
                    if (
                        stage_best is None
                        or scored_candidate.adjusted_score > stage_best.adjusted_score
                    ):
                        stage_best = scored_candidate
                    if (
                        best_candidate is None
                        or scored_candidate.adjusted_score > best_candidate.adjusted_score
                    ):
                        best_candidate = scored_candidate

        if stage_best is not None:
            center_pose = stage_best.pose

    final_stage = stages[-1]
    return SearchResult(
        best_candidate=best_candidate,
        candidates=candidates,
        stage_count=len(stages),
        final_step_m=final_stage.xy_step_m,
        final_theta_step_rad=final_stage.theta_step_rad,
    )


def compute_match_confidence(
    candidates: list[ScoredCandidate],
    near_best_score_percent: float,
) -> MatchConfidence:
    if not candidates:
        return MatchConfidence(
            best_score=0.0,
            second_best_score=0.0,
            score_gap=0.0,
            normalized_score_gap=0.0,
            score_variance=0.0,
            near_best_count=0,
            candidate_count=0,
        )

    scores = sorted(
        (candidate.adjusted_score for candidate in candidates),
        reverse=True,
    )
    best_score = scores[0]
    second_best_score = scores[1] if len(scores) > 1 else 0.0
    score_gap = best_score - second_best_score
    normalized_score_gap = score_gap / max(abs(best_score), 1e-9)
    score_variance = statistics.pvariance(scores) if len(scores) > 1 else 0.0
    # Repetitive aisles often create several nearly equal peaks. Counting the
    # candidates inside a small band around the best score exposes that
    # ambiguity even when the winning score itself is high.
    near_best_tolerance = abs(best_score) * near_best_score_percent / 100.0
    near_best_count = sum(
        1 for score in scores if best_score - score <= near_best_tolerance
    )
    return MatchConfidence(
        best_score=best_score,
        second_best_score=second_best_score,
        score_gap=score_gap,
        normalized_score_gap=normalized_score_gap,
        score_variance=score_variance,
        near_best_count=near_best_count,
        candidate_count=len(scores),
    )


def estimate_yaw_bias(
    search_result: SearchResult,
    best_pose: Pose2D,
    top_score_percent: float,
) -> float:
    if not search_result.candidates:
        return 0.0

    best_score = max(candidate.adjusted_score for candidate in search_result.candidates)
    min_score = min(candidate.adjusted_score for candidate in search_result.candidates)
    score_span = max(best_score - min_score, 1e-9)
    tolerance = max(abs(best_score) * top_score_percent / 100.0, score_span * 0.02)
    weighted_total = 0.0
    weight_total = 0.0
    for candidate in search_result.candidates:
        if best_score - candidate.adjusted_score > tolerance:
            continue

        score_weight = max(candidate.adjusted_score - min_score, 0.0) + 1e-9
        theta_offset = normalize_angle_rad(candidate.pose.theta_rad - best_pose.theta_rad)
        weighted_total += theta_offset * score_weight
        weight_total += score_weight

    if weight_total <= 0.0:
        return 0.0
    return weighted_total / weight_total


def corrected_search_candidate(
    search_result: SearchResult,
    points: list[ScanPoint],
    predicted_pose: Pose2D,
    previous_pose: Pose2D,
    submap_distances: dict[tuple[int, int], float],
    cell_size_m: float,
    max_distance_m: float,
    expected_theta: float | None,
    heading_weight: float,
    degeneracy_mode: str,
    config: MatchQualityConfig,
) -> tuple[ScoredCandidate | None, float, bool]:
    if search_result.best_candidate is None:
        return None, 0.0, False

    best_candidate = search_result.best_candidate
    if (
        degeneracy_mode != "AISLE"
        or config.yaw_bias_correction_alpha <= 0.0
        or not search_result.candidates
    ):
        return best_candidate, 0.0, False

    yaw_bias = estimate_yaw_bias(
        search_result=search_result,
        best_pose=best_candidate.pose,
        top_score_percent=config.yaw_bias_top_score_percent,
    )
    correction = config.yaw_bias_correction_alpha * yaw_bias
    if abs(correction) <= 1e-6:
        return best_candidate, yaw_bias, False

    corrected_pose = Pose2D(
        scan_index=best_candidate.pose.scan_index,
        x_m=best_candidate.pose.x_m,
        y_m=best_candidate.pose.y_m,
        theta_rad=best_candidate.pose.theta_rad - correction,
        score=0.0,
    )
    return (
        evaluate_scan_match_candidate(
            points=points,
            candidate=corrected_pose,
            predicted_pose=predicted_pose,
            previous_pose=previous_pose,
            submap_distances=submap_distances,
            cell_size_m=cell_size_m,
            max_distance_m=max_distance_m,
            expected_theta=expected_theta,
            heading_weight=heading_weight,
            degeneracy_mode=degeneracy_mode,
            theta_prior=predicted_pose.theta_rad,
            yaw_lambda=config.aisle_yaw_lambda,
        ),
        yaw_bias,
        True,
    )


def distinct_search_candidates(
    search_result: SearchResult,
    primary_candidate: ScoredCandidate,
    max_count: int,
    score_percent: float,
) -> list[ScoredCandidate]:
    if max_count <= 1:
        return [primary_candidate]

    sorted_candidates = sorted(
        search_result.candidates,
        key=lambda candidate: candidate.adjusted_score,
        reverse=True,
    )
    score_tolerance = abs(primary_candidate.adjusted_score) * score_percent / 100.0
    selected = [primary_candidate]
    distinct_xy_m = max(search_result.final_step_m * 1.5, 1e-6)
    distinct_theta_rad = max(search_result.final_theta_step_rad * 1.5, 1e-9)

    for candidate in sorted_candidates:
        if len(selected) >= max_count:
            break
        if primary_candidate.adjusted_score - candidate.adjusted_score > score_tolerance:
            break
        if candidate is primary_candidate:
            continue
        if all(
            math.hypot(
                candidate.pose.x_m - selected_candidate.pose.x_m,
                candidate.pose.y_m - selected_candidate.pose.y_m,
            )
            >= distinct_xy_m
            or abs(
                normalize_angle_rad(
                    candidate.pose.theta_rad - selected_candidate.pose.theta_rad
                )
            )
            >= distinct_theta_rad
            for selected_candidate in selected
        ):
            selected.append(candidate)

    return selected


def scale_pose_covariance(
    covariance: PoseCovariance,
    scale: float,
) -> PoseCovariance:
    return (
        tuple(float(value) * scale for value in covariance[0]),
        tuple(float(value) * scale for value in covariance[1]),
        tuple(float(value) * scale for value in covariance[2]),
    )


def aisle_covariance_world(
    axis_x: float,
    axis_y: float,
    sigma_along_m: float,
    sigma_cross_m: float,
    sigma_theta_rad: float,
) -> PoseCovariance:
    along_variance = sigma_along_m * sigma_along_m
    cross_variance = sigma_cross_m * sigma_cross_m
    theta_variance = sigma_theta_rad * sigma_theta_rad
    cross_x = -axis_y
    cross_y = axis_x
    return (
        (
            axis_x * axis_x * along_variance + cross_x * cross_x * cross_variance,
            axis_x * axis_y * along_variance + cross_x * cross_y * cross_variance,
            0.0,
        ),
        (
            axis_y * axis_x * along_variance + cross_y * cross_x * cross_variance,
            axis_y * axis_y * along_variance + cross_y * cross_y * cross_variance,
            0.0,
        ),
        (0.0, 0.0, theta_variance),
    )


def relative_frame_covariance(
    covariance: PoseCovariance,
    reference_pose: Pose2D,
    np: object,
) -> object:
    matrix = np.array(covariance, dtype=float)
    cos_theta = math.cos(reference_pose.theta_rad)
    sin_theta = math.sin(reference_pose.theta_rad)
    world_to_reference = np.array(
        [
            [cos_theta, sin_theta, 0.0],
            [-sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return world_to_reference @ matrix @ world_to_reference.T


def estimate_score_surface(
    points: list[ScanPoint],
    pose: Pose2D,
    submap_distances: dict[tuple[int, int], float],
    cell_size_m: float,
    max_distance_m: float,
    step_m: float,
    theta_step_rad: float,
    config: MatchQualityConfig,
    confidence: MatchConfidence,
) -> tuple[ScoreSurface, PoseCovariance]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for scan-match covariance estimation") from exc

    steps = np.array(
        [max(step_m, 1e-6), max(step_m, 1e-6), max(theta_step_rad, 1e-9)],
        dtype=float,
    )

    def shifted_score(offsets: object) -> float:
        offset_x, offset_y, offset_theta = offsets
        return score_pose(
            points=points,
            pose=Pose2D(
                scan_index=pose.scan_index,
                x_m=pose.x_m + float(offset_x),
                y_m=pose.y_m + float(offset_y),
                theta_rad=pose.theta_rad + float(offset_theta),
                score=0.0,
            ),
            submap_distances=submap_distances,
            cell_size_m=cell_size_m,
            max_distance_m=max_distance_m,
        )

    center_score = shifted_score(np.zeros(3, dtype=float))
    hessian = np.zeros((3, 3), dtype=float)
    for row in range(3):
        offset = np.zeros(3, dtype=float)
        offset[row] = steps[row]
        hessian[row, row] = (
            shifted_score(offset)
            - 2.0 * center_score
            + shifted_score(-offset)
        ) / (steps[row] ** 2)

    for row in range(3):
        for column in range(row + 1, 3):
            row_offset = np.zeros(3, dtype=float)
            column_offset = np.zeros(3, dtype=float)
            row_offset[row] = steps[row]
            column_offset[column] = steps[column]
            hessian[row, column] = (
                shifted_score(row_offset + column_offset)
                - shifted_score(row_offset - column_offset)
                - shifted_score(-row_offset + column_offset)
                + shifted_score(-row_offset - column_offset)
            ) / (4.0 * steps[row] * steps[column])
            hessian[column, row] = hessian[row, column]

    # The matcher maximizes score. Around a well-constrained local maximum the
    # Hessian is negative definite, so -H is a local information matrix. Small
    # eigenvalues mean a direction can move without changing score much, which
    # is exactly the warehouse-aisle degeneracy we want GTSAM to trust less.
    information = -0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    max_eigenvalue = float(np.max(eigenvalues)) if len(eigenvalues) else 0.0
    min_eigenvalue = float(np.min(eigenvalues)) if len(eigenvalues) else 0.0
    if min_eigenvalue <= 0.0 or max_eigenvalue <= 0.0:
        condition_number = math.inf
    else:
        condition_number = max_eigenvalue / min_eigenvalue

    weak_cutoff = max(
        config.hessian_min_curvature,
        max_eigenvalue / max(config.hessian_conditioning_threshold, 1.0),
    )
    axes = ("x", "y", "theta")
    weak_directions = []
    for index, eigenvalue in enumerate(eigenvalues):
        if float(eigenvalue) <= weak_cutoff:
            axis = axes[int(np.argmax(np.abs(eigenvectors[:, index])))]
            if axis not in weak_directions:
                weak_directions.append(axis)

    degenerate = (
        not math.isfinite(condition_number)
        or condition_number > config.hessian_conditioning_threshold
        or min_eigenvalue < config.hessian_min_curvature
    )

    regularized_eigenvalues = np.maximum(eigenvalues, config.hessian_min_curvature)
    covariance = (
        eigenvectors
        @ np.diag(1.0 / regularized_eigenvalues)
        @ eigenvectors.T
        * config.covariance_scaling
    )
    near_best_ratio = confidence.near_best_count / max(confidence.candidate_count, 1)
    ambiguity_scale = 1.0 + min(9.0, near_best_ratio * 9.0)
    if config.score_gap_threshold > 0.0:
        gap_deficit = max(
            0.0,
            (config.score_gap_threshold - confidence.score_gap)
            / config.score_gap_threshold,
        )
        ambiguity_scale *= 1.0 + gap_deficit * 4.0
    covariance *= ambiguity_scale

    surface = ScoreSurface(
        hessian=covariance_to_tuple(hessian),
        information=covariance_to_tuple(information),
        eigenvalues=(
            float(eigenvalues[0]),
            float(eigenvalues[1]),
            float(eigenvalues[2]),
        ),
        weak_directions=tuple(weak_directions),
        condition_number=float(condition_number),
        degenerate=degenerate,
    )
    return surface, covariance_to_tuple(covariance)


def reject_match_reason(
    confidence: MatchConfidence,
    surface: ScoreSurface,
    config: MatchQualityConfig,
) -> str | None:
    if confidence.best_score < config.confidence_threshold:
        return "low_confidence"
    if confidence.score_gap < config.score_gap_threshold:
        return "small_score_gap"
    if confidence.near_best_count > config.max_near_best_candidates:
        return "ambiguous_score_landscape"
    if surface.degenerate:
        return "degenerate_score_surface"
    return None


def attach_match_quality(
    candidate: ScoredCandidate,
    confidence: MatchConfidence,
    surface: ScoreSurface,
    covariance: PoseCovariance,
    score_breakdown: ScanScoreBreakdown,
    degeneracy: DegeneracyDiagnostics,
    accepted: bool,
    hypothesis_diagnostics: HypothesisDiagnostics | None = None,
    aisle_frame: AisleFrameDiagnostics | None = None,
    hypotheses: tuple[PoseHypothesis, ...] = (),
) -> Pose2D:
    return Pose2D(
        scan_index=candidate.pose.scan_index,
        x_m=candidate.pose.x_m,
        y_m=candidate.pose.y_m,
        theta_rad=candidate.pose.theta_rad,
        score=candidate.raw_score,
        covariance=covariance,
        confidence=confidence,
        surface=surface,
        score_breakdown=score_breakdown,
        degeneracy=degeneracy,
        hypothesis_diagnostics=hypothesis_diagnostics,
        aisle_frame=aisle_frame,
        hypotheses=hypotheses,
        accepted=accepted,
    )


def finalize_hypothesis_candidate(
    update: HypothesisCandidate,
    points: list[ScanPoint],
    submap_distances: dict[tuple[int, int], float],
    cell_size_m: float,
    max_distance_m: float,
    search_xy_m: float,
    search_step_m: float,
    search_theta_deg: float,
    search_theta_step_deg: float,
    expected_theta: float | None,
    heading_weight: float,
    scoring_mode: str,
    theta_prior: float,
    yaw_lambda: float,
    ambiguity_gate: bool,
    config: MatchQualityConfig,
    hypothesis_diagnostics: HypothesisDiagnostics,
    hypotheses: tuple[PoseHypothesis, ...],
) -> tuple[
    Pose2D,
    MatchConfidence,
    ScoreSurface | None,
    PoseCovariance | None,
    str | None,
    bool,
    int,
    SearchResult,
]:
    confidence = update.confidence
    search_result = update.search_result
    selected_candidate = update.candidate
    expanded_search = False
    surface = None
    covariance = None
    score_breakdown = default_score_breakdown()

    _score, score_breakdown = score_pose_details(
        points=points,
        pose=selected_candidate.pose,
        submap_distances=submap_distances,
        cell_size_m=cell_size_m,
        max_distance_m=max_distance_m,
    )
    surface, covariance = estimate_score_surface(
        points=points,
        pose=selected_candidate.pose,
        submap_distances=submap_distances,
        cell_size_m=cell_size_m,
        max_distance_m=max_distance_m,
        step_m=search_result.final_step_m,
        theta_step_rad=search_result.final_theta_step_rad,
        config=config,
        confidence=confidence,
    )
    rejected_reason = reject_match_reason(confidence, surface, config)

    expanded_possible = (
        config.expanded_search_xy_m > search_xy_m
        or config.expanded_search_theta_deg > search_theta_deg
    )
    if rejected_reason is not None and expanded_possible:
        expanded_search = True
        expanded_result = hierarchical_scan_match(
            points=points,
            predicted_pose=update.predicted_pose,
            previous_pose=update.previous_pose,
            submap_distances=submap_distances,
            cell_size_m=cell_size_m,
            max_distance_m=max_distance_m,
            search_xy_m=max(config.expanded_search_xy_m, search_xy_m),
            search_step_m=search_step_m,
            search_theta_deg=max(config.expanded_search_theta_deg, search_theta_deg),
            search_theta_step_deg=search_theta_step_deg,
            expected_theta=expected_theta,
            heading_weight=heading_weight,
            degeneracy_mode=scoring_mode,
            theta_prior=theta_prior,
            yaw_lambda=yaw_lambda,
        )
        expanded_confidence = compute_match_confidence(
            expanded_result.candidates,
            config.near_best_score_percent,
        )
        expanded_candidate, yaw_bias_rad, yaw_correction_applied = corrected_search_candidate(
            search_result=expanded_result,
            points=points,
            predicted_pose=update.predicted_pose,
            previous_pose=update.previous_pose,
            submap_distances=submap_distances,
            cell_size_m=cell_size_m,
            max_distance_m=max_distance_m,
            expected_theta=expected_theta,
            heading_weight=heading_weight,
            degeneracy_mode=scoring_mode,
            config=config,
        )
        if expanded_candidate is not None:
            confidence = expanded_confidence
            search_result = expanded_result
            selected_candidate = expanded_candidate
            update = HypothesisCandidate(
                candidate=expanded_candidate,
                predicted_pose=update.predicted_pose,
                previous_pose=update.previous_pose,
                search_result=expanded_result,
                confidence=expanded_confidence,
                degeneracy=DegeneracyDiagnostics(
                    mode=update.degeneracy.mode,
                    entropy=update.degeneracy.entropy,
                    linearity=update.degeneracy.linearity,
                    dominant_angle=update.degeneracy.dominant_angle,
                    confidence=update.degeneracy.confidence,
                    yaw_bias_rad=yaw_bias_rad,
                    yaw_correction_applied=yaw_correction_applied,
                    recovery_retry=update.degeneracy.recovery_retry,
                ),
                close_scores=update.close_scores,
                parent_weight=update.parent_weight,
            )
            _score, score_breakdown = score_pose_details(
                points=points,
                pose=selected_candidate.pose,
                submap_distances=submap_distances,
                cell_size_m=cell_size_m,
                max_distance_m=max_distance_m,
            )
            surface, covariance = estimate_score_surface(
                points=points,
                pose=selected_candidate.pose,
                submap_distances=submap_distances,
                cell_size_m=cell_size_m,
                max_distance_m=max_distance_m,
                step_m=search_result.final_step_m,
                theta_step_rad=search_result.final_theta_step_rad,
                config=config,
                confidence=confidence,
            )
            rejected_reason = reject_match_reason(confidence, surface, config)

    if ambiguity_gate:
        covariance = scale_pose_covariance(
            covariance,
            config.hypothesis_ambiguity_covariance_scale,
        )

    best_pose = attach_match_quality(
        candidate=selected_candidate,
        confidence=confidence,
        surface=surface,
        covariance=covariance,
        score_breakdown=score_breakdown,
        degeneracy=update.degeneracy,
        accepted=rejected_reason is None,
        hypothesis_diagnostics=hypothesis_diagnostics,
        hypotheses=hypotheses,
    )
    return (
        best_pose,
        confidence,
        surface,
        covariance,
        rejected_reason,
        expanded_search,
        search_result.stage_count,
        search_result,
    )


def diagnostics_for_pose(
    keyframe_ordinal: int,
    pose: Pose2D,
    confidence: MatchConfidence,
    surface: ScoreSurface | None,
    covariance: PoseCovariance | None,
    rejected_reason: str | None,
    expanded_search: bool,
    stage_count: int,
) -> ScanMatchDiagnostics:
    return ScanMatchDiagnostics(
        keyframe_ordinal=keyframe_ordinal,
        scan_index=pose.scan_index,
        x_m=pose.x_m,
        y_m=pose.y_m,
        theta_rad=pose.theta_rad,
        raw_score=pose.score,
        accepted=rejected_reason is None,
        rejected_reason=rejected_reason,
        expanded_search=expanded_search,
        stage_count=stage_count,
        confidence=confidence,
        surface=surface,
        covariance=covariance,
        score_breakdown=pose.score_breakdown,
        degeneracy=pose.degeneracy,
        hypothesis=pose.hypothesis_diagnostics,
        aisle_frame=pose.aisle_frame,
    )


def write_jsonl_records(records: list[object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(json_safe(record), sort_keys=True, allow_nan=False)
        for record in records
    ]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_match_diagnostics(
    diagnostics: list[ScanMatchDiagnostics],
    output_path: Path,
) -> None:
    write_jsonl_records([asdict(diagnostic) for diagnostic in diagnostics], output_path)


def write_loop_closure_diagnostics(
    diagnostics: list[LoopClosureDiagnostics],
    output_path: Path,
) -> None:
    write_jsonl_records([asdict(diagnostic) for diagnostic in diagnostics], output_path)


def loop_distribution(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    sorted_values = sorted(values)

    def percentile(ratio: float) -> float:
        index = min(
            max(round((len(sorted_values) - 1) * ratio), 0),
            len(sorted_values) - 1,
        )
        return sorted_values[index]

    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p25": percentile(0.25),
        "median": statistics.median(sorted_values),
        "p75": percentile(0.75),
        "max": sorted_values[-1],
    }


def loop_candidate_summary(diagnostic: LoopClosureDiagnostics) -> dict:
    return {
        "candidate_id": diagnostic.candidate_id,
        "from_ordinal": diagnostic.from_ordinal,
        "to_ordinal": diagnostic.to_ordinal,
        "from_scan_index": diagnostic.from_scan_index,
        "to_scan_index": diagnostic.to_scan_index,
        "accepted": diagnostic.accepted,
        "reason": diagnostic.reason,
        "match_score": diagnostic.match_score,
        "score_gap": diagnostic.score_gap,
        "normalized_score_gap": (
            diagnostic.confidence.normalized_score_gap
            if diagnostic.confidence is not None
            else None
        ),
        "near_best_count": (
            diagnostic.confidence.near_best_count
            if diagnostic.confidence is not None
            else None
        ),
        "corridor_signature_similarity": diagnostic.corridor_signature_similarity,
        "mht_agreement": diagnostic.agreement_count,
        "ambiguous": diagnostic.ambiguous,
        "conflicting": diagnostic.conflicting,
        "transform_residual_m": diagnostic.transform_residual_m,
        "transform_residual_theta_deg": (
            math.degrees(diagnostic.transform_residual_theta_rad)
            if diagnostic.transform_residual_theta_rad is not None
            else None
        ),
    }


def loop_candidate_rank_score(
    diagnostic: LoopClosureDiagnostics,
    strategy: str,
) -> float:
    confidence = diagnostic.confidence
    normalized_gap = confidence.normalized_score_gap if confidence is not None else 0.0
    near_best_count = confidence.near_best_count if confidence is not None else 0
    near_best_penalty = min(near_best_count / 100.0, 1.0)
    corridor_similarity = diagnostic.corridor_signature_similarity
    if corridor_similarity is None:
        corridor_similarity = 0.5
    mht_support = min(diagnostic.agreement_count / 2.0, 1.0)
    ambiguity_penalty = 0.15 if diagnostic.ambiguous else 0.0
    conflict_penalty = 0.5 if diagnostic.conflicting else 0.0

    current_score = diagnostic.match_score
    confidence_score = (
        diagnostic.match_score
        + min(diagnostic.score_gap, 0.25)
        + min(normalized_gap, 0.25)
        - near_best_penalty
    )
    corridor_score = confidence_score + 0.35 * corridor_similarity - conflict_penalty
    mht_score = confidence_score + 0.35 * mht_support - ambiguity_penalty
    hybrid_score = (
        confidence_score
        + 0.25 * corridor_similarity
        + 0.25 * mht_support
        - ambiguity_penalty
        - conflict_penalty
    )

    if strategy == "current":
        return current_score
    if strategy == "confidence_weighted":
        return confidence_score
    if strategy == "corridor_aware":
        return corridor_score
    if strategy == "mht_aware":
        return mht_score
    if strategy == "hybrid":
        return hybrid_score
    raise ValueError(f"unknown loop ranking strategy: {strategy}")


def loop_density_bins(
    diagnostics: list[LoopClosureDiagnostics],
    pose_count: int,
    bin_count: int = 20,
) -> list[dict]:
    if pose_count <= 0:
        return []
    bins = [
        {"start_ordinal": 0, "end_ordinal": 0, "candidate_count": 0}
        for _index in range(bin_count)
    ]
    for index in range(bin_count):
        bins[index]["start_ordinal"] = round(index * pose_count / bin_count)
        bins[index]["end_ordinal"] = round((index + 1) * pose_count / bin_count)
    for diagnostic in diagnostics:
        index = min(
            int(diagnostic.to_ordinal / max(pose_count, 1) * bin_count),
            bin_count - 1,
        )
        bins[index]["candidate_count"] += 1
    return bins


def build_loop_closure_analysis_report(
    diagnostics: list[LoopClosureDiagnostics],
    pose_count: int,
) -> dict:
    accepted = [diagnostic for diagnostic in diagnostics if diagnostic.accepted]
    rejected = [diagnostic for diagnostic in diagnostics if not diagnostic.accepted]
    reason_histogram: dict[str, int] = {}
    for diagnostic in diagnostics:
        reason_histogram[diagnostic.reason] = reason_histogram.get(diagnostic.reason, 0) + 1

    corridor_similarities = [
        diagnostic.corridor_signature_similarity
        for diagnostic in diagnostics
        if diagnostic.corridor_signature_similarity is not None
    ]
    gate_percentages = {
        reason: count / max(len(diagnostics), 1)
        for reason, count in sorted(reason_histogram.items())
        if not reason.startswith("accepted")
    }

    accepted_count = len(accepted)
    current_ids = {diagnostic.candidate_id for diagnostic in accepted}
    ranking_payload = {}
    for strategy in (
        "current",
        "confidence_weighted",
        "corridor_aware",
        "mht_aware",
        "hybrid",
    ):
        if strategy == "current":
            selected = accepted
        else:
            ranked = sorted(
                diagnostics,
                key=lambda diagnostic: loop_candidate_rank_score(diagnostic, strategy),
                reverse=True,
            )
            selected = ranked[:accepted_count]
        selected_ids = {diagnostic.candidate_id for diagnostic in selected}
        ranking_payload[strategy] = {
            "top_candidate_ids": [diagnostic.candidate_id for diagnostic in selected],
            "would_add": sorted(selected_ids - current_ids),
            "would_remove": sorted(current_ids - selected_ids),
            "top_candidates": [
                loop_candidate_summary(diagnostic) for diagnostic in selected[:10]
            ],
        }

    report = {
        "summary_statistics": {
            "evaluated_candidates": len(diagnostics),
            "accepted_candidates": accepted_count,
            "rejected_candidates": len(rejected),
            "true_acceptance_rate": accepted_count / max(len(diagnostics), 1),
            "rejection_reasons_histogram": reason_histogram,
            "gate_rejection_percentages": gate_percentages,
            "average_corridor_similarity": (
                statistics.fmean(corridor_similarities)
                if corridor_similarities
                else None
            ),
            "score_gap_distribution": loop_distribution(
                [diagnostic.score_gap for diagnostic in diagnostics]
            ),
            "normalized_score_gap_distribution": loop_distribution(
                [
                    diagnostic.confidence.normalized_score_gap
                    for diagnostic in diagnostics
                    if diagnostic.confidence is not None
                ]
            ),
            "match_score_distribution": loop_distribution(
                [diagnostic.match_score for diagnostic in diagnostics]
            ),
            "candidate_density_over_trajectory": loop_density_bins(
                diagnostics,
                pose_count,
            ),
        },
        "rankings": ranking_payload,
        "highest_scoring_rejected": [
            loop_candidate_summary(diagnostic)
            for diagnostic in sorted(
                rejected,
                key=lambda diagnostic: diagnostic.match_score,
                reverse=True,
            )[:10]
        ],
        "lowest_confidence_accepted": [
            loop_candidate_summary(diagnostic)
            for diagnostic in sorted(
                accepted,
                key=lambda diagnostic: (
                    diagnostic.confidence.normalized_score_gap
                    if diagnostic.confidence is not None
                    else 0.0,
                    diagnostic.score_gap,
                    diagnostic.match_score,
                ),
            )[:10]
        ],
        "ambiguous_candidates": [
            loop_candidate_summary(diagnostic)
            for diagnostic in sorted(
                [diagnostic for diagnostic in diagnostics if diagnostic.ambiguous],
                key=lambda diagnostic: diagnostic.match_score,
                reverse=True,
            )[:20]
        ],
        "conflicting_candidates": [
            loop_candidate_summary(diagnostic)
            for diagnostic in sorted(
                [diagnostic for diagnostic in diagnostics if diagnostic.conflicting],
                key=lambda diagnostic: diagnostic.transform_residual_m or 0.0,
                reverse=True,
            )[:20]
        ],
    }
    return report


def write_loop_closure_analysis_json(
    diagnostics: list[LoopClosureDiagnostics],
    poses: list[Pose2D],
    output_path: Path,
) -> dict:
    report = build_loop_closure_analysis_report(diagnostics, len(poses))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def write_loop_closure_analysis_markdown(
    report: dict,
    output_path: Path,
) -> None:
    summary = report["summary_statistics"]
    lines = [
        "# Loop Closure Analysis",
        "",
        f"- evaluated candidates: {summary['evaluated_candidates']}",
        f"- accepted candidates: {summary['accepted_candidates']}",
        f"- true acceptance rate: {summary['true_acceptance_rate']:.3f}",
        f"- average corridor similarity: {summary['average_corridor_similarity']}",
        f"- rejection reasons: {summary['rejection_reasons_histogram']}",
        "",
        "## Highest-Scoring Rejected",
    ]
    for candidate in report["highest_scoring_rejected"]:
        lines.append(
            f"- #{candidate['candidate_id']} {candidate['from_ordinal']}->{candidate['to_ordinal']} "
            f"score={candidate['match_score']:.3f} gap={candidate['score_gap']:.3f} "
            f"reason={candidate['reason']}"
        )
    lines.append("")
    lines.append("## Lowest-Confidence Accepted")
    for candidate in report["lowest_confidence_accepted"]:
        normalized_gap = candidate["normalized_score_gap"]
        normalized_text = "none" if normalized_gap is None else f"{normalized_gap:.4f}"
        lines.append(
            f"- #{candidate['candidate_id']} {candidate['from_ordinal']}->{candidate['to_ordinal']} "
            f"score={candidate['match_score']:.3f} norm_gap={normalized_text} "
            f"reason={candidate['reason']}"
        )
    lines.append("")
    lines.append("## Offline Re-Ranking Changes")
    for strategy, payload in report["rankings"].items():
        lines.append(
            f"- {strategy}: add {payload['would_add']} remove {payload['would_remove']}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_loop_closure_analysis_plot(
    poses: list[Pose2D],
    diagnostics: list[LoopClosureDiagnostics],
    output_path: Path,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write loop closure analysis plots") from exc

    width = 1300
    height = 950
    left = 70
    right = 260
    top = 70
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    axis_color = (64, 64, 64)
    grid_color = (226, 226, 226)
    path_color = (30, 64, 175)
    accepted_color = (22, 163, 74)
    rejected_color = (220, 38, 38)
    ambiguous_color = (234, 179, 8)
    conflict_color = (127, 29, 29)

    points = [(pose.x_m, pose.y_m) for pose in poses]
    if points:
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
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
    scale = min(plot_width / max(max_x - min_x, 1.0), plot_height / max(max_y - min_y, 1.0))

    def project(point: tuple[float, float]) -> tuple[float, float]:
        return (
            left + (point[0] - min_x) * scale,
            top + plot_height - (point[1] - min_y) * scale,
        )

    for step in range(6):
        x = left + round(plot_width * step / 5)
        y = top + round(plot_height * step / 5)
        draw.line([(x, top), (x, top + plot_height)], fill=grid_color)
        draw.line([(left, y), (left + plot_width, y)], fill=grid_color)
    draw.rectangle((left, top, left + plot_width, top + plot_height), outline=axis_color)
    draw.text((left, 24), "Loop closure candidate analysis", fill=axis_color)
    draw.text(
        (left, 46),
        f"{len(diagnostics)} evaluated candidates; labels are candidate_id values in JSON",
        fill=axis_color,
    )

    if len(points) >= 2:
        projected_path = [project(point) for point in points]
        draw.line(projected_path, fill=path_color, width=2)

    for diagnostic in diagnostics:
        if diagnostic.from_ordinal >= len(poses) or diagnostic.to_ordinal >= len(poses):
            continue
        from_point = project((poses[diagnostic.from_ordinal].x_m, poses[diagnostic.from_ordinal].y_m))
        to_point = project((poses[diagnostic.to_ordinal].x_m, poses[diagnostic.to_ordinal].y_m))
        color = accepted_color if diagnostic.accepted else rejected_color
        width_px = 2 if diagnostic.accepted else 1
        if diagnostic.ambiguous:
            color = ambiguous_color
            width_px = max(width_px, 2)
        if diagnostic.conflicting:
            color = conflict_color
            width_px = 3
        draw.line([from_point, to_point], fill=color, width=width_px)
        mid_x = (from_point[0] + to_point[0]) * 0.5
        mid_y = (from_point[1] + to_point[1]) * 0.5
        draw.text((mid_x + 3, mid_y - 7), str(diagnostic.candidate_id), fill=color)

    legend_x = left + plot_width + 34
    legend_y = top
    draw.line([(legend_x, legend_y), (legend_x + 34, legend_y)], fill=path_color, width=2)
    draw.text((legend_x + 44, legend_y - 7), "pose graph", fill=axis_color)
    draw.line([(legend_x, legend_y + 30), (legend_x + 34, legend_y + 30)], fill=accepted_color, width=2)
    draw.text((legend_x + 44, legend_y + 23), "accepted", fill=axis_color)
    draw.line([(legend_x, legend_y + 60), (legend_x + 34, legend_y + 60)], fill=rejected_color, width=2)
    draw.text((legend_x + 44, legend_y + 53), "rejected", fill=axis_color)
    draw.line([(legend_x, legend_y + 90), (legend_x + 34, legend_y + 90)], fill=ambiguous_color, width=2)
    draw.text((legend_x + 44, legend_y + 83), "ambiguous", fill=axis_color)
    draw.line([(legend_x, legend_y + 120), (legend_x + 34, legend_y + 120)], fill=conflict_color, width=3)
    draw.text((legend_x + 44, legend_y + 113), "conflicting", fill=axis_color)
    draw.text((legend_x, legend_y + 166), f"accepted {sum(d.accepted for d in diagnostics)}", fill=axis_color)
    draw.text((legend_x, legend_y + 190), f"ambiguous {sum(d.ambiguous for d in diagnostics)}", fill=axis_color)
    draw.text((legend_x, legend_y + 214), f"conflicting {sum(d.conflicting for d in diagnostics)}", fill=axis_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_loop_closure_analysis_outputs(
    diagnostics: list[LoopClosureDiagnostics],
    poses: list[Pose2D],
    args: argparse.Namespace,
    label: str,
) -> None:
    json_path = args.out_dir / f"loop_closure_analysis_{label}.json"
    markdown_path = args.out_dir / f"loop_closure_report_{label}.md"
    plot_path = args.out_dir / f"loop_closure_candidates_{label}.png"
    report = write_loop_closure_analysis_json(diagnostics, poses, json_path)
    write_loop_closure_analysis_markdown(report, markdown_path)
    write_loop_closure_analysis_plot(poses, diagnostics, plot_path)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    print(f"wrote {plot_path}")


def write_aisle_frame_diagnostics_plot(
    diagnostics: list[ScanMatchDiagnostics],
    output_path: Path,
    drift_gain: float,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write aisle-frame diagnostics") from exc

    frames = [
        diagnostic.aisle_frame
        for diagnostic in diagnostics
        if diagnostic.aisle_frame is not None
    ]
    if not frames:
        return

    width = 1200
    height = 760
    left = 72
    top = 64
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    axis_color = (64, 64, 64)
    grid_color = (226, 226, 226)
    before_color = (126, 34, 206)
    after_color = (34, 197, 94)
    ellipse_color = (37, 99, 235)

    draw.text((left, 20), "Aisle-frame anisotropic diagnostics", fill=axis_color)

    drift_before = [frame.longitudinal_drift_m for frame in frames]
    drift_after = [
        frame.longitudinal_drift_m
        * (1.0 - drift_gain if frame.anisotropic_applied else 1.0)
        for frame in frames
    ]
    max_abs_drift = max(
        max(abs(value) for value in drift_before),
        max(abs(value) for value in drift_after),
        1e-6,
    )
    bin_count = 32
    histogram_left = left
    histogram_top = top
    histogram_width = 520
    histogram_height = 280

    def histogram(values: list[float]) -> list[int]:
        counts = [0] * bin_count
        for value in values:
            ratio = (value + max_abs_drift) / (2.0 * max_abs_drift)
            index = min(max(int(ratio * bin_count), 0), bin_count - 1)
            counts[index] += 1
        return counts

    before_counts = histogram(drift_before)
    after_counts = histogram(drift_after)
    max_count = max(max(before_counts), max(after_counts), 1)
    draw.rectangle(
        (
            histogram_left,
            histogram_top,
            histogram_left + histogram_width,
            histogram_top + histogram_height,
        ),
        outline=axis_color,
    )
    for step in range(5):
        y = histogram_top + round(histogram_height * step / 4)
        draw.line(
            [(histogram_left, y), (histogram_left + histogram_width, y)],
            fill=grid_color,
        )
    zero_x = histogram_left + histogram_width // 2
    draw.line(
        [(zero_x, histogram_top), (zero_x, histogram_top + histogram_height)],
        fill=(180, 180, 180),
    )
    bar_width = histogram_width / bin_count
    for index, (before_count, after_count) in enumerate(zip(before_counts, after_counts)):
        x0 = histogram_left + index * bar_width
        x1 = histogram_left + (index + 0.45) * bar_width
        x2 = histogram_left + (index + 0.55) * bar_width
        x3 = histogram_left + (index + 1.0) * bar_width
        before_h = before_count / max_count * histogram_height
        after_h = after_count / max_count * histogram_height
        draw.rectangle(
            (
                x0,
                histogram_top + histogram_height - before_h,
                x1,
                histogram_top + histogram_height,
            ),
            fill=before_color,
        )
        draw.rectangle(
            (
                x2,
                histogram_top + histogram_height - after_h,
                x3,
                histogram_top + histogram_height,
            ),
            fill=after_color,
        )
    draw.text(
        (histogram_left, histogram_top - 26),
        "longitudinal drift before vs after regularization",
        fill=axis_color,
    )
    draw.text(
        (histogram_left, histogram_top + histogram_height + 12),
        f"-{max_abs_drift:.2f} m",
        fill=axis_color,
    )
    draw.text(
        (zero_x - 10, histogram_top + histogram_height + 12),
        "0",
        fill=axis_color,
    )
    draw.text(
        (histogram_left + histogram_width - 52, histogram_top + histogram_height + 12),
        f"{max_abs_drift:.2f} m",
        fill=axis_color,
    )
    draw.rectangle((left, top + 324, left + 16, top + 340), fill=before_color)
    draw.text((left + 24, top + 322), "before", fill=axis_color)
    draw.rectangle((left + 110, top + 324, left + 126, top + 340), fill=after_color)
    draw.text((left + 134, top + 322), "after", fill=axis_color)

    ellipse_left = 690
    ellipse_top = top
    ellipse_width = 410
    ellipse_height = 360
    draw.rectangle(
        (
            ellipse_left,
            ellipse_top,
            ellipse_left + ellipse_width,
            ellipse_top + ellipse_height,
        ),
        outline=axis_color,
    )
    draw.text(
        (ellipse_left, ellipse_top - 26),
        "sample covariance ellipses in aisle frame",
        fill=axis_color,
    )
    applied_frames = [frame for frame in frames if frame.anisotropic_applied]
    if applied_frames:
        max_sigma = max(
            max(frame.sigma_along_m, frame.sigma_cross_m)
            for frame in applied_frames
        )
        sample_count = min(len(applied_frames), 40)
        for offset, frame in enumerate(
            applied_frames[:: max(len(applied_frames) // sample_count, 1)]
        ):
            column = offset % 8
            row = offset // 8
            center_x = ellipse_left + 36 + column * 48
            center_y = ellipse_top + 42 + row * 58
            scale = 22.0 / max(max_sigma, 1e-6)
            radius_x = max(frame.sigma_along_m * scale, 2.0)
            radius_y = max(frame.sigma_cross_m * scale, 2.0)
            draw.ellipse(
                (
                    center_x - radius_x,
                    center_y - radius_y,
                    center_x + radius_x,
                    center_y + radius_y,
                ),
                outline=ellipse_color,
            )
    else:
        draw.text(
            (ellipse_left + 28, ellipse_top + 154),
            "no anisotropic factors passed the gate",
            fill=axis_color,
        )

    stats_top = 470
    applied_count = len(applied_frames)
    draw.text((left, stats_top), f"frames: {len(frames)}", fill=axis_color)
    draw.text((left, stats_top + 24), f"anisotropic factors: {applied_count}", fill=axis_color)
    draw.text(
        (left, stats_top + 48),
        f"median confidence: {statistics.median(frame.confidence for frame in frames):.2f}",
        fill=axis_color,
    )
    draw.text(
        (left, stats_top + 72),
        f"median linearity: {statistics.median(frame.local_linearity for frame in frames):.2f}",
        fill=axis_color,
    )
    draw.text(
        (left, stats_top + 96),
        f"median abs drift before: {statistics.median(abs(value) for value in drift_before):.3f} m",
        fill=axis_color,
    )
    draw.text(
        (left, stats_top + 120),
        f"median abs drift after: {statistics.median(abs(value) for value in drift_after):.3f} m",
        fill=axis_color,
    )
    draw.text(
        (left, stats_top + 144),
        f"median axis jitter reduction: {math.degrees(statistics.median(frame.jitter_reduction_rad for frame in frames)):.2f} deg",
        fill=axis_color,
    )

    axis_hist_left = 690
    axis_hist_top = 470
    axis_hist_width = 410
    axis_hist_height = 210
    raw_axis_changes = [abs(frame.raw_axis_change_rad) for frame in frames]
    stable_axis_changes = [abs(frame.stable_axis_change_rad) for frame in frames]
    max_axis_change = max(
        max(raw_axis_changes),
        max(stable_axis_changes),
        math.radians(1.0),
    )

    def angle_histogram(values: list[float]) -> list[int]:
        counts = [0] * bin_count
        for value in values:
            index = min(int(value / max_axis_change * bin_count), bin_count - 1)
            counts[index] += 1
        return counts

    raw_counts = angle_histogram(raw_axis_changes)
    stable_counts = angle_histogram(stable_axis_changes)
    max_axis_count = max(max(raw_counts), max(stable_counts), 1)
    draw.rectangle(
        (
            axis_hist_left,
            axis_hist_top,
            axis_hist_left + axis_hist_width,
            axis_hist_top + axis_hist_height,
        ),
        outline=axis_color,
    )
    draw.text(
        (axis_hist_left, axis_hist_top - 26),
        "axis angular change per frame",
        fill=axis_color,
    )
    axis_bar_width = axis_hist_width / bin_count
    for index, (raw_count, stable_count) in enumerate(zip(raw_counts, stable_counts)):
        x0 = axis_hist_left + index * axis_bar_width
        x1 = axis_hist_left + (index + 0.45) * axis_bar_width
        x2 = axis_hist_left + (index + 0.55) * axis_bar_width
        x3 = axis_hist_left + (index + 1.0) * axis_bar_width
        raw_h = raw_count / max_axis_count * axis_hist_height
        stable_h = stable_count / max_axis_count * axis_hist_height
        draw.rectangle(
            (
                x0,
                axis_hist_top + axis_hist_height - raw_h,
                x1,
                axis_hist_top + axis_hist_height,
            ),
            fill=before_color,
        )
        draw.rectangle(
            (
                x2,
                axis_hist_top + axis_hist_height - stable_h,
                x3,
                axis_hist_top + axis_hist_height,
            ),
            fill=after_color,
        )
    draw.text(
        (axis_hist_left, axis_hist_top + axis_hist_height + 12),
        "0 deg",
        fill=axis_color,
    )
    draw.text(
        (
            axis_hist_left + axis_hist_width - 72,
            axis_hist_top + axis_hist_height + 12,
        ),
        f"{math.degrees(max_axis_change):.1f} deg",
        fill=axis_color,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def print_loop_closure_diagnostics_summary(
    diagnostics: list[LoopClosureDiagnostics],
    output_path: Path | None,
) -> None:
    if not diagnostics:
        print("loop closure safety: no candidates evaluated")
        return

    accepted_count = sum(1 for diagnostic in diagnostics if diagnostic.accepted)
    reason_counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        reason_counts[diagnostic.reason] = reason_counts.get(diagnostic.reason, 0) + 1
    reason_summary = ", ".join(
        f"{reason} {count}" for reason, count in sorted(reason_counts.items())
    )
    print(
        "loop closure safety: "
        f"{accepted_count} accepted, {len(diagnostics) - accepted_count} deferred/rejected "
        f"({reason_summary})"
    )
    if output_path is not None:
        print(f"wrote {output_path}")


def print_match_diagnostics_summary(
    diagnostics: list[ScanMatchDiagnostics],
    output_path: Path | None,
) -> None:
    accepted_count = sum(1 for diagnostic in diagnostics if diagnostic.accepted)
    rejected_count = len(diagnostics) - accepted_count
    expanded_count = sum(1 for diagnostic in diagnostics if diagnostic.expanded_search)
    degenerate_count = sum(
        1
        for diagnostic in diagnostics
        if diagnostic.surface is not None and diagnostic.surface.degenerate
    )
    print(
        "match diagnostics: "
        f"{accepted_count} accepted, {rejected_count} rejected, "
        f"{expanded_count} expanded retries, {degenerate_count} degenerate surfaces"
    )
    score_breakdowns = [
        diagnostic.score_breakdown
        for diagnostic in diagnostics
        if diagnostic.score_breakdown is not None
    ]
    if score_breakdowns:
        print(
            "score contribution median: "
            f"linear {statistics.median(item.linear_score_fraction for item in score_breakdowns):.2f}, "
            f"salient {statistics.median(item.salient_score_fraction for item in score_breakdowns):.2f}"
        )
    degeneracy_values = [
        diagnostic.degeneracy
        for diagnostic in diagnostics
        if diagnostic.degeneracy is not None
    ]
    if degeneracy_values:
        aisle_fraction = sum(
            1 for degeneracy in degeneracy_values if degeneracy.mode == "AISLE"
        ) / len(degeneracy_values)
        recovery_count = sum(1 for degeneracy in degeneracy_values if degeneracy.recovery_retry)
        yaw_drift_deg = math.degrees(
            normalize_angle_rad(diagnostics[-1].theta_rad - diagnostics[0].theta_rad)
        ) if len(diagnostics) >= 2 else 0.0
        print(
            "degeneracy summary: "
            f"aisle {aisle_fraction:.1%}, yaw drift {yaw_drift_deg:.2f} deg, "
            f"recovery retries {recovery_count}"
        )
    hypothesis_values = [
        diagnostic.hypothesis
        for diagnostic in diagnostics
        if diagnostic.hypothesis is not None
    ]
    if hypothesis_values:
        ambiguity_fraction = sum(
            1 for hypothesis in hypothesis_values if hypothesis.ambiguity_gate
        ) / len(hypothesis_values)
        print(
            "hypothesis summary: "
            f"active median {statistics.median(item.active_count for item in hypothesis_values):.1f}, "
            f"entropy median {statistics.median(item.weight_entropy for item in hypothesis_values):.2f}, "
            f"divergence median {statistics.median(item.max_divergence for item in hypothesis_values):.2f} m, "
            f"ambiguity {ambiguity_fraction:.1%}"
        )
    aisle_frames = [
        diagnostic.aisle_frame
        for diagnostic in diagnostics
        if diagnostic.aisle_frame is not None
    ]
    if aisle_frames:
        anisotropic_count = sum(
            1 for frame in aisle_frames if frame.anisotropic_applied
        )
        corridor_updates = sum(1 for frame in aisle_frames if frame.corridor_updated)
        corridor_resets = sum(1 for frame in aisle_frames if frame.corridor_reset)
        print(
            "aisle-frame summary: "
            f"confidence median {statistics.median(frame.confidence for frame in aisle_frames):.2f}, "
            f"linearity median {statistics.median(frame.local_linearity for frame in aisle_frames):.2f}, "
            f"along drift median {statistics.median(frame.longitudinal_drift_m for frame in aisle_frames):.3f} m, "
            f"anisotropic factors {anisotropic_count}, "
            f"corridor updates {corridor_updates}, resets {corridor_resets}, "
            f"axis jitter reduction median {math.degrees(statistics.median(frame.jitter_reduction_rad for frame in aisle_frames)):.2f} deg"
        )
    if output_path is not None:
        print(f"wrote {output_path}")


def write_score_heatmap(
    points: list[ScanPoint],
    pose: Pose2D,
    submap_distances: dict[tuple[int, int], float],
    cell_size_m: float,
    max_distance_m: float,
    output_path: Path,
    radius_m: float,
    step_m: float,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write score heatmaps") from exc

    offsets = frange(-radius_m, radius_m, max(step_m, 1e-6))
    if not offsets:
        return

    scores = []
    for y_offset in reversed(offsets):
        row = []
        for x_offset in offsets:
            row.append(
                score_pose(
                    points=points,
                    pose=Pose2D(
                        scan_index=pose.scan_index,
                        x_m=pose.x_m + x_offset,
                        y_m=pose.y_m + y_offset,
                        theta_rad=pose.theta_rad,
                        score=0.0,
                    ),
                    submap_distances=submap_distances,
                    cell_size_m=cell_size_m,
                    max_distance_m=max_distance_m,
                )
            )
        scores.append(row)

    flat_scores = [score for row in scores for score in row]
    min_score = min(flat_scores)
    max_score = max(flat_scores)
    score_span = max(max_score - min_score, 1e-9)
    cell_px = max(3, min(16, 720 // max(len(offsets), 1)))
    left = 70
    top = 60
    right = 180
    bottom = 56
    heatmap_size = len(offsets) * cell_px
    width = left + heatmap_size + right
    height = top + heatmap_size + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    low_color = (49, 54, 149)
    high_color = (253, 231, 37)
    axis_color = (64, 64, 64)

    for row_index, row in enumerate(scores):
        for column_index, score in enumerate(row):
            weight = (score - min_score) / score_span
            color = interpolate_color(low_color, high_color, weight)
            x = left + column_index * cell_px
            y = top + row_index * cell_px
            draw.rectangle((x, y, x + cell_px, y + cell_px), fill=color)

    center_index = len(offsets) // 2
    center_x = left + center_index * cell_px + cell_px // 2
    center_y = top + center_index * cell_px + cell_px // 2
    draw.line([(center_x - 8, center_y), (center_x + 8, center_y)], fill=(220, 38, 38))
    draw.line([(center_x, center_y - 8), (center_x, center_y + 8)], fill=(220, 38, 38))
    draw.rectangle((left, top, left + heatmap_size, top + heatmap_size), outline=axis_color)
    draw.text(
        (left, 18),
        f"Score heatmap scan {pose.scan_index:05d} at theta {math.degrees(pose.theta_rad):.1f} deg",
        fill=axis_color,
    )
    draw.text(
        (left, 38),
        f"score {min_score:.3f} to {max_score:.3f} | radius {radius_m:.2f} m",
        fill=axis_color,
    )
    draw.text((left + heatmap_size // 2 - 24, height - 32), "x offset", fill=axis_color)
    draw.text((12, top + heatmap_size // 2 - 8), "y offset", fill=axis_color)

    legend_x = left + heatmap_size + 34
    legend_y = top
    legend_height = heatmap_size
    for offset in range(legend_height):
        ratio = 1.0 - offset / max(legend_height - 1, 1)
        color = interpolate_color(low_color, high_color, ratio)
        draw.line(
            [(legend_x, legend_y + offset), (legend_x + 24, legend_y + offset)],
            fill=color,
        )
    draw.rectangle(
        (legend_x, legend_y, legend_x + 24, legend_y + legend_height - 1),
        outline=axis_color,
    )
    draw.text((legend_x + 34, legend_y - 7), f"{max_score:.3f}", fill=axis_color)
    draw.text(
        (legend_x + 34, legend_y + legend_height - 10),
        f"{min_score:.3f}",
        fill=axis_color,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


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
    heading_weight: float,
    heading_min_confidence: float,
    max_step_m: float,
    debug_keyframe_ordinals: set[int] | None = None,
    match_quality: MatchQualityConfig | None = None,
    diagnostics: list[ScanMatchDiagnostics] | None = None,
    score_heatmap_dir: Path | None = None,
) -> tuple[list[Pose2D], list[MatchDebug]]:
    match_quality = match_quality or MatchQualityConfig()
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
        build_scan_match_points(
            points=scan.points,
            max_points=max_points,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            config=match_quality,
        )
        for scan in scans
    ]
    local_aisle_frames = [estimate_aisle_frame(points) for points in local_scans]
    line_angles = [
        dominant_structure_angle(
            points=scan.points,
            bin_count=72,
            min_segment_m=0.05,
            max_segment_m=1.2,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
        )
        for scan in scans
    ]

    corridor_states: dict[int, CorridorState] = {}
    active_corridor_id = None
    next_corridor_id = 0
    initial_confidence = default_match_confidence()
    initial_axis, initial_aisle_confidence, initial_aisle_linearity = local_aisle_frames[0]
    initial_hypotheses = [
        PoseHypothesis(
            pose=(0.0, 0.0, 0.0),
            covariance=default_pose_covariance(),
            score=1.0,
            weight=1.0,
        )
    ]
    initial_hypothesis_diagnostics = default_hypothesis_diagnostics()
    initial_pose = Pose2D(
        scan_index=scans[0].scan_index,
        x_m=0.0,
        y_m=0.0,
        theta_rad=0.0,
        score=1.0,
    )
    initial_raw_axis = rotate_axis_to_world(initial_axis, initial_pose.theta_rad)
    (
        initial_stable_axis,
        initial_corridor_id,
        initial_corridor_confidence,
        initial_raw_to_stable,
        initial_raw_axis_change,
        initial_stable_axis_change,
        initial_axis_update,
        initial_ema_smoothing,
        initial_covariance_rotation,
        initial_jitter_reduction,
        initial_corridor_updated,
        initial_corridor_reset,
        initial_corridor_reason,
        active_corridor_id,
        next_corridor_id,
    ) = update_corridor_axis(
        raw_axis=initial_raw_axis,
        confidence=initial_aisle_confidence,
        linearity=initial_aisle_linearity,
        frame_index=0,
        pose=initial_pose,
        degeneracy_mode="NORMAL",
        hypothesis_diagnostics=initial_hypothesis_diagnostics,
        accepted=True,
        corridor_states=corridor_states,
        active_corridor_id=active_corridor_id,
        next_corridor_id=next_corridor_id,
        config=match_quality,
    )
    initial_aisle_frame = build_aisle_frame_diagnostics(
        axis=initial_stable_axis,
        raw_axis=initial_raw_axis,
        confidence=initial_aisle_confidence,
        linearity=initial_aisle_linearity,
        corridor_id=initial_corridor_id,
        corridor_confidence=initial_corridor_confidence,
        raw_to_stable_angle_rad=initial_raw_to_stable,
        raw_axis_change_rad=initial_raw_axis_change,
        stable_axis_change_rad=initial_stable_axis_change,
        axis_update_magnitude_rad=initial_axis_update,
        ema_smoothing_angle_rad=initial_ema_smoothing,
        covariance_rotation_delta_rad=initial_covariance_rotation,
        jitter_reduction_rad=initial_jitter_reduction,
        corridor_updated=initial_corridor_updated,
        corridor_reset=initial_corridor_reset,
        corridor_frozen_reason=initial_corridor_reason,
        previous_pose=initial_pose,
        pose=initial_pose,
        expected_along_m=0.0,
        config=match_quality,
    )
    poses = [
        Pose2D(
            scan_index=scans[0].scan_index,
            x_m=0.0,
            y_m=0.0,
            theta_rad=0.0,
            score=1.0,
            confidence=initial_confidence,
            score_breakdown=default_score_breakdown(),
            degeneracy=default_degeneracy_diagnostics(),
            hypothesis_diagnostics=initial_hypothesis_diagnostics,
            aisle_frame=initial_aisle_frame,
            hypotheses=tuple(initial_hypotheses),
        )
    ]
    hypotheses = initial_hypotheses
    world_scans = [transform_xy_points(local_scans[0], poses[0])]
    if diagnostics is not None:
        diagnostics.append(
            diagnostics_for_pose(
                keyframe_ordinal=0,
                pose=poses[0],
                confidence=initial_confidence,
                surface=None,
                covariance=None,
                rejected_reason=None,
                expanded_search=False,
                stage_count=0,
            )
        )
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

    import numpy as np

    aisle_streak = 0
    for keyframe_ordinal, (scan, local_points) in enumerate(
        zip(scans[1:], local_scans[1:]),
        start=1,
    ):
        previous_pose = poses[-1]
        if len(poses) >= 2:
            previous_previous_pose = poses[-2]
            delta_x = previous_pose.x_m - previous_previous_pose.x_m
            delta_y = previous_pose.y_m - previous_previous_pose.y_m
            delta_theta = normalize_angle_rad(
                previous_pose.theta_rad - previous_previous_pose.theta_rad
            )
            if max_step_m > 0.0:
                step_m = math.hypot(delta_x, delta_y)
                if step_m > max_step_m:
                    scale = max_step_m / step_m
                    delta_x *= scale
                    delta_y *= scale
            max_delta_theta = math.radians(search_theta_deg)
            if abs(delta_theta) > max_delta_theta:
                delta_theta = math.copysign(max_delta_theta, delta_theta)
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
        submap_distances, max_distance_m = build_submap_distance_lookup(
            world_scans=world_scans,
            cell_size_m=cell_size_m,
            inflation_cells=inflation_cells,
            submap_keyframes=submap_keyframes,
        )
        previous_line_angle, previous_line_confidence = line_angles[keyframe_ordinal - 1]
        current_line_angle, current_line_confidence = line_angles[keyframe_ordinal]
        line_delta = None
        if (
            previous_line_angle is not None
            and current_line_angle is not None
            and previous_line_confidence >= heading_min_confidence
            and current_line_confidence >= heading_min_confidence
        ):
            line_delta = normalize_line_angle_delta_rad(
                current_line_angle - previous_line_angle
            )

        degeneracy_points = np.array(
            [scan_point_xy(point) for point in local_points],
            dtype=float,
        )
        local_aisle_axis, aisle_confidence, aisle_linearity = local_aisle_frames[
            keyframe_ordinal
        ]
        degeneracy_metrics = compute_degeneracy_metrics(degeneracy_points)
        degeneracy_mode = str(degeneracy_metrics.get("mode", "NORMAL"))
        scoring_mode = (
            degeneracy_mode
            if match_quality.degeneracy_scoring_enabled
            else "NORMAL"
        )
        if scoring_mode == "AISLE":
            aisle_streak += 1
            score_points = apply_aisle_degeneracy_weights(
                points=local_points,
                dominant_angle=float(degeneracy_metrics.get("dominant_angle", 0.0)),
                config=match_quality,
            )
            active_search_theta_deg = max(
                search_theta_step_deg,
                search_theta_deg * match_quality.aisle_theta_window_scale,
            )
            active_yaw_lambda = match_quality.aisle_yaw_lambda
        else:
            aisle_streak = 0
            score_points = local_points
            active_search_theta_deg = search_theta_deg
            active_yaw_lambda = 0.0

        candidate_updates: list[HypothesisCandidate] = []
        candidate_log_weights: list[float] = []
        for hypothesis in hypotheses:
            previous_hypothesis_pose = hypothesis_to_pose2d(
                hypothesis,
                scan_index=previous_pose.scan_index,
            )
            propagated_hypothesis = propagate_hypothesis(
                hypothesis=hypothesis,
                delta_x_m=delta_x,
                delta_y_m=delta_y,
                delta_theta_rad=delta_theta,
            )
            predicted_hypothesis_pose = hypothesis_to_pose2d(
                propagated_hypothesis,
                scan_index=scan.scan_index,
                score=0.0,
            )
            hypothesis_expected_theta = (
                None
                if line_delta is None
                else previous_hypothesis_pose.theta_rad - line_delta
            )
            search_result = hierarchical_scan_match(
                points=score_points,
                predicted_pose=predicted_hypothesis_pose,
                previous_pose=previous_hypothesis_pose,
                submap_distances=submap_distances,
                cell_size_m=cell_size_m,
                max_distance_m=max_distance_m,
                search_xy_m=search_xy_m,
                search_step_m=search_step_m,
                search_theta_deg=active_search_theta_deg,
                search_theta_step_deg=search_theta_step_deg,
                expected_theta=hypothesis_expected_theta,
                heading_weight=heading_weight,
                degeneracy_mode=scoring_mode,
                theta_prior=predicted_hypothesis_pose.theta_rad,
                yaw_lambda=active_yaw_lambda,
            )
            recovery_retry = False
            if (
                scoring_mode == "AISLE"
                and aisle_streak > match_quality.aisle_persistence_frames
            ):
                recovery_retry = True
                recovery_result = hierarchical_scan_match(
                    points=score_points,
                    predicted_pose=predicted_hypothesis_pose,
                    previous_pose=previous_hypothesis_pose,
                    submap_distances=submap_distances,
                    cell_size_m=cell_size_m,
                    max_distance_m=max_distance_m,
                    search_xy_m=search_xy_m + match_quality.aisle_recovery_xy_m,
                    search_step_m=search_step_m,
                    search_theta_deg=(
                        active_search_theta_deg + match_quality.aisle_recovery_theta_deg
                    ),
                    search_theta_step_deg=search_theta_step_deg,
                    expected_theta=hypothesis_expected_theta,
                    heading_weight=heading_weight,
                    degeneracy_mode=scoring_mode,
                    theta_prior=predicted_hypothesis_pose.theta_rad,
                    yaw_lambda=active_yaw_lambda,
                )
                if (
                    search_result.best_candidate is None
                    or (
                        recovery_result.best_candidate is not None
                        and recovery_result.best_candidate.adjusted_score
                        > search_result.best_candidate.adjusted_score
                    )
                ):
                    search_result = recovery_result

            confidence = compute_match_confidence(
                search_result.candidates,
                match_quality.near_best_score_percent,
            )
            best_candidate, yaw_bias_rad, yaw_correction_applied = corrected_search_candidate(
                search_result=search_result,
                points=score_points,
                predicted_pose=predicted_hypothesis_pose,
                previous_pose=previous_hypothesis_pose,
                submap_distances=submap_distances,
                cell_size_m=cell_size_m,
                max_distance_m=max_distance_m,
                expected_theta=hypothesis_expected_theta,
                heading_weight=heading_weight,
                degeneracy_mode=scoring_mode,
                config=match_quality,
            )
            if best_candidate is None:
                continue

            close_scores = (
                confidence.normalized_score_gap <= match_quality.hypothesis_close_score_gap
                or confidence.near_best_count
                >= match_quality.hypothesis_near_best_count_threshold
            )
            branch_count = (
                match_quality.hypothesis_count
                if scoring_mode == "AISLE" or close_scores
                else 1
            )
            branch_candidates = distinct_search_candidates(
                search_result=search_result,
                primary_candidate=best_candidate,
                max_count=branch_count,
                score_percent=match_quality.hypothesis_branch_score_percent,
            )
            temperature = match_quality.hypothesis_score_temperature
            if scoring_mode == "AISLE" or close_scores:
                temperature *= match_quality.hypothesis_ambiguous_temperature_scale
            temperature = max(temperature, 1e-9)
            degeneracy_diagnostics = degeneracy_diagnostics_from_metrics(
                metrics=degeneracy_metrics,
                yaw_bias_rad=yaw_bias_rad,
                yaw_correction_applied=yaw_correction_applied,
                recovery_retry=recovery_retry,
            )
            for branch_candidate in branch_candidates:
                candidate_updates.append(
                    HypothesisCandidate(
                        candidate=branch_candidate,
                        predicted_pose=predicted_hypothesis_pose,
                        previous_pose=previous_hypothesis_pose,
                        search_result=search_result,
                        confidence=confidence,
                        degeneracy=degeneracy_diagnostics,
                        close_scores=close_scores,
                        parent_weight=hypothesis.weight,
                    )
                )
                candidate_log_weights.append(
                    math.log(max(hypothesis.weight, 1e-12))
                    + branch_candidate.adjusted_score / temperature
                )

        expanded_search = False
        rejected_reason = "no_candidates"
        confidence = MatchConfidence(
            best_score=0.0,
            second_best_score=0.0,
            score_gap=0.0,
            normalized_score_gap=0.0,
            score_variance=0.0,
            near_best_count=0,
            candidate_count=0,
        )
        surface = None
        covariance = None
        search_result = SearchResult(
            best_candidate=None,
            candidates=[],
            stage_count=0,
            final_step_m=search_step_m,
            final_theta_step_rad=math.radians(search_theta_step_deg),
        )
        hypothesis_diagnostics = build_hypothesis_diagnostics(
            hypotheses=hypotheses,
            aisle_mode=scoring_mode == "AISLE",
            ambiguity_gate=scoring_mode == "AISLE",
            ambiguity_reason="aisle" if scoring_mode == "AISLE" else None,
            config=match_quality,
        )
        best_pose = Pose2D(
            scan_index=scan.scan_index,
            x_m=predicted_pose.x_m,
            y_m=predicted_pose.y_m,
            theta_rad=predicted_pose.theta_rad,
            score=0.0,
            score_breakdown=default_score_breakdown(),
            degeneracy=degeneracy_diagnostics_from_metrics(degeneracy_metrics),
            hypothesis_diagnostics=hypothesis_diagnostics,
            hypotheses=tuple(hypotheses),
            accepted=False,
        )

        if candidate_updates:
            max_log_weight = max(candidate_log_weights)
            raw_weights = [
                math.exp(log_weight - max_log_weight)
                for log_weight in candidate_log_weights
            ]
            total_raw_weight = max(sum(raw_weights), 1e-12)
            weighted_updates = []
            for update, raw_weight in zip(candidate_updates, raw_weights):
                normalized_weight = raw_weight / total_raw_weight
                weighted_updates.append(
                    (
                        update,
                        PoseHypothesis(
                            pose=(
                                update.candidate.pose.x_m,
                                update.candidate.pose.y_m,
                                normalize_angle_rad(update.candidate.pose.theta_rad),
                            ),
                            covariance=None,
                            score=update.candidate.raw_score,
                            weight=normalized_weight,
                        ),
                    )
                )

            merged_updates: list[tuple[HypothesisCandidate, PoseHypothesis]] = []
            merge_xy_m = max(search_step_m * 0.5, 1e-6)
            merge_theta_rad = max(math.radians(search_theta_step_deg) * 0.5, 1e-9)
            for update, hypothesis in sorted(
                weighted_updates,
                key=lambda item: item[1].weight,
                reverse=True,
            ):
                merged_index = None
                for index, (_existing_update, existing_hypothesis) in enumerate(
                    merged_updates
                ):
                    _distance, translation_m, heading_rad = hypothesis_pair_distance(
                        hypothesis,
                        existing_hypothesis,
                        theta_weight_m=match_quality.hypothesis_theta_distance_weight_m,
                    )
                    if translation_m <= merge_xy_m and heading_rad <= merge_theta_rad:
                        merged_index = index
                        break
                if merged_index is None:
                    merged_updates.append((update, hypothesis))
                    continue

                existing_update, existing_hypothesis = merged_updates[merged_index]
                combined_weight = existing_hypothesis.weight + hypothesis.weight
                kept_update = (
                    update
                    if update.candidate.adjusted_score
                    > existing_update.candidate.adjusted_score
                    else existing_update
                )
                kept_hypothesis = (
                    hypothesis
                    if update is kept_update
                    else existing_hypothesis
                )
                merged_updates[merged_index] = (
                    kept_update,
                    PoseHypothesis(
                        pose=kept_hypothesis.pose,
                        covariance=kept_hypothesis.covariance,
                        score=kept_hypothesis.score,
                        weight=combined_weight,
                    ),
                )
            weighted_updates = [
                (update, hypothesis)
                for (update, _old_hypothesis), hypothesis in zip(
                    merged_updates,
                    normalize_hypotheses(
                        [hypothesis for _update, hypothesis in merged_updates]
                    ),
                )
            ]

            preliminary_hypotheses = [
                hypothesis for _update, hypothesis in weighted_updates
            ]
            max_divergence, _max_translation_m, _max_heading_rad = hypothesis_divergence(
                preliminary_hypotheses,
                theta_weight_m=match_quality.hypothesis_theta_distance_weight_m,
            )
            ambiguity_reasons = []
            if scoring_mode == "AISLE":
                ambiguity_reasons.append("aisle")
            if any(update.close_scores for update, _hypothesis in weighted_updates):
                ambiguity_reasons.append("close_scores")
            divergence_ambiguous = (
                max_divergence > match_quality.hypothesis_divergence_threshold_m
            )
            if divergence_ambiguous:
                ambiguity_reasons.append("hypothesis_divergence")
            ambiguity_gate = bool(ambiguity_reasons)
            covariance_ambiguity_gate = scoring_mode == "AISLE" or divergence_ambiguous

            weighted_updates.sort(key=lambda item: item[1].weight, reverse=True)
            prune_threshold = (
                match_quality.hypothesis_ambiguous_weight_threshold
                if ambiguity_gate
                else match_quality.hypothesis_weight_threshold
            )
            min_keep = 2 if ambiguity_gate and match_quality.hypothesis_count > 1 else 1
            kept_updates = []
            for index, item in enumerate(weighted_updates):
                if len(kept_updates) >= match_quality.hypothesis_count:
                    break
                if index < min_keep or item[1].weight >= prune_threshold:
                    kept_updates.append(item)
            if not kept_updates:
                kept_updates = weighted_updates[:1]

            normalized_kept_hypotheses = normalize_hypotheses(
                [hypothesis for _update, hypothesis in kept_updates]
            )
            kept_updates = [
                (update, hypothesis)
                for (update, _old_hypothesis), hypothesis in zip(
                    kept_updates,
                    normalized_kept_hypotheses,
                )
            ]
            hypothesis_diagnostics = build_hypothesis_diagnostics(
                hypotheses=[hypothesis for _update, hypothesis in kept_updates],
                aisle_mode=scoring_mode == "AISLE",
                ambiguity_gate=ambiguity_gate,
                ambiguity_reason=",".join(ambiguity_reasons) if ambiguity_reasons else None,
                config=match_quality,
            )
            selected_update = max(
                kept_updates,
                key=lambda item: item[1].weight,
            )[0]
            selected_expected_theta = (
                None
                if line_delta is None
                else selected_update.previous_pose.theta_rad - line_delta
            )
            (
                best_pose,
                confidence,
                surface,
                covariance,
                rejected_reason,
                expanded_search,
                _stage_count,
                search_result,
            ) = finalize_hypothesis_candidate(
                update=selected_update,
                points=score_points,
                submap_distances=submap_distances,
                cell_size_m=cell_size_m,
                max_distance_m=max_distance_m,
                search_xy_m=search_xy_m,
                search_step_m=search_step_m,
                search_theta_deg=active_search_theta_deg,
                search_theta_step_deg=search_theta_step_deg,
                expected_theta=selected_expected_theta,
                heading_weight=heading_weight,
                scoring_mode=scoring_mode,
                theta_prior=selected_update.predicted_pose.theta_rad,
                yaw_lambda=active_yaw_lambda,
                ambiguity_gate=covariance_ambiguity_gate,
                config=match_quality,
                hypothesis_diagnostics=hypothesis_diagnostics,
                hypotheses=tuple(hypothesis for _update, hypothesis in kept_updates),
            )
            if rejected_reason is None:
                updated_hypotheses = []
                for update, hypothesis in kept_updates:
                    if update is selected_update:
                        updated_hypotheses.append(
                            PoseHypothesis(
                                pose=(best_pose.x_m, best_pose.y_m, best_pose.theta_rad),
                                covariance=best_pose.covariance,
                                score=best_pose.score,
                                weight=hypothesis.weight,
                            )
                        )
                    else:
                        updated_hypotheses.append(hypothesis)
                hypotheses = normalize_hypotheses(updated_hypotheses)
                hypothesis_diagnostics = build_hypothesis_diagnostics(
                    hypotheses=hypotheses,
                    aisle_mode=scoring_mode == "AISLE",
                    ambiguity_gate=ambiguity_gate,
                    ambiguity_reason=(
                        ",".join(ambiguity_reasons) if ambiguity_reasons else None
                    ),
                    config=match_quality,
                )
                best_pose = Pose2D(
                    scan_index=best_pose.scan_index,
                    x_m=best_pose.x_m,
                    y_m=best_pose.y_m,
                    theta_rad=best_pose.theta_rad,
                    score=best_pose.score,
                    covariance=best_pose.covariance,
                    confidence=best_pose.confidence,
                    surface=best_pose.surface,
                    score_breakdown=best_pose.score_breakdown,
                    degeneracy=best_pose.degeneracy,
                    hypothesis_diagnostics=hypothesis_diagnostics,
                    aisle_frame=best_pose.aisle_frame,
                    hypotheses=tuple(hypotheses),
                    accepted=best_pose.accepted,
                )

        raw_world_axis = rotate_axis_to_world(local_aisle_axis, best_pose.theta_rad)
        (
            stable_world_axis,
            corridor_id,
            corridor_confidence,
            raw_to_stable_angle_rad,
            raw_axis_change_rad,
            stable_axis_change_rad,
            axis_update_magnitude_rad,
            ema_smoothing_angle_rad,
            covariance_rotation_delta_rad,
            jitter_reduction_rad,
            corridor_updated,
            corridor_reset,
            corridor_frozen_reason,
            active_corridor_id,
            next_corridor_id,
        ) = update_corridor_axis(
            raw_axis=raw_world_axis,
            confidence=aisle_confidence,
            linearity=aisle_linearity,
            frame_index=keyframe_ordinal,
            pose=best_pose,
            degeneracy_mode=degeneracy_mode,
            hypothesis_diagnostics=best_pose.hypothesis_diagnostics,
            accepted=rejected_reason is None,
            corridor_states=corridor_states,
            active_corridor_id=active_corridor_id,
            next_corridor_id=next_corridor_id,
            config=match_quality,
        )
        predicted_axis_x, predicted_axis_y = stable_world_axis
        expected_delta_x_m = predicted_pose.x_m - previous_pose.x_m
        expected_delta_y_m = predicted_pose.y_m - previous_pose.y_m
        expected_along_m = (
            expected_delta_x_m * predicted_axis_x
            + expected_delta_y_m * predicted_axis_y
        )
        aisle_frame = build_aisle_frame_diagnostics(
            axis=stable_world_axis,
            raw_axis=raw_world_axis,
            confidence=aisle_confidence,
            linearity=aisle_linearity,
            corridor_id=corridor_id,
            corridor_confidence=corridor_confidence,
            raw_to_stable_angle_rad=raw_to_stable_angle_rad,
            raw_axis_change_rad=raw_axis_change_rad,
            stable_axis_change_rad=stable_axis_change_rad,
            axis_update_magnitude_rad=axis_update_magnitude_rad,
            ema_smoothing_angle_rad=ema_smoothing_angle_rad,
            covariance_rotation_delta_rad=covariance_rotation_delta_rad,
            jitter_reduction_rad=jitter_reduction_rad,
            corridor_updated=corridor_updated,
            corridor_reset=corridor_reset,
            corridor_frozen_reason=corridor_frozen_reason,
            previous_pose=previous_pose,
            pose=best_pose,
            expected_along_m=expected_along_m,
            config=match_quality,
        )
        aisle_frame = finalize_aisle_frame_for_pose(
            frame=aisle_frame,
            previous_pose=previous_pose,
            pose=best_pose,
            config=match_quality,
            np=np,
        )
        best_pose = Pose2D(
            scan_index=best_pose.scan_index,
            x_m=best_pose.x_m,
            y_m=best_pose.y_m,
            theta_rad=best_pose.theta_rad,
            score=best_pose.score,
            covariance=best_pose.covariance,
            confidence=best_pose.confidence,
            surface=best_pose.surface,
            score_breakdown=best_pose.score_breakdown,
            degeneracy=best_pose.degeneracy,
            hypothesis_diagnostics=best_pose.hypothesis_diagnostics,
            aisle_frame=aisle_frame,
            hypotheses=best_pose.hypotheses,
            accepted=best_pose.accepted,
        )

        current_world_points = transform_xy_points(local_points, best_pose)
        if score_heatmap_dir is not None and search_result.best_candidate is not None:
            write_score_heatmap(
                points=score_points,
                pose=best_pose,
                submap_distances=submap_distances,
                cell_size_m=cell_size_m,
                max_distance_m=max_distance_m,
                output_path=score_heatmap_dir / f"scan_{scan.scan_index:05d}.png",
                radius_m=max(search_step_m * 4.0, cell_size_m * 4.0),
                step_m=max(search_result.final_step_m, cell_size_m * 0.5),
            )

        if diagnostics is not None:
            diagnostics.append(
                diagnostics_for_pose(
                    keyframe_ordinal=keyframe_ordinal,
                    pose=best_pose,
                    confidence=confidence,
                    surface=surface,
                    covariance=covariance,
                    rejected_reason=rejected_reason,
                    expanded_search=expanded_search,
                    stage_count=search_result.stage_count,
                )
            )

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

        if rejected_reason is None:
            poses.append(best_pose)
            world_scans.append(current_world_points)

    return poses, debug_matches


def estimate_odometry_detailed(
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
    heading_weight: float,
    heading_min_confidence: float,
    max_step_m: float,
    debug_keyframe_ordinals: set[int] | None = None,
    match_quality: MatchQualityConfig | None = None,
    score_heatmap_dir: Path | None = None,
) -> OdometryResult:
    diagnostics = []
    poses, debug_matches = run_odometry_estimation(
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
        heading_weight=heading_weight,
        heading_min_confidence=heading_min_confidence,
        max_step_m=max_step_m,
        debug_keyframe_ordinals=debug_keyframe_ordinals,
        match_quality=match_quality,
        diagnostics=diagnostics,
        score_heatmap_dir=score_heatmap_dir,
    )
    return OdometryResult(
        poses=poses,
        debug_matches=debug_matches,
        diagnostics=diagnostics,
    )


def write_odometry_preview(
    poses: list[Pose2D],
    output_path: Path,
    limit: int,
    stride: int,
    title: str | None = None,
    subtitle: str | None = None,
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
    title = title or f"Stage 6 odometry preview: first {limit} scans"
    subtitle = subtitle or (
        f"{len(poses)} keyframe poses | stride {stride} | local scan-to-submap search"
    )
    draw.text((left, 18), title, fill=axis_color)
    draw.text(
        (left, 40),
        subtitle,
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


def dominant_structure_angle(
    points: list[tuple[float, float, int]],
    bin_count: int,
    min_segment_m: float,
    max_segment_m: float,
    min_range_m: float,
    max_range_m: float,
) -> tuple[float | None, float]:
    bins = [0.0] * bin_count
    previous_point = None
    valid_points = [
        (x_m, y_m)
        for x_m, y_m, _intensity in points
        if min_range_m <= math.hypot(x_m, y_m) <= max_range_m
    ]

    for point in valid_points:
        if previous_point is not None:
            dx = point[0] - previous_point[0]
            dy = point[1] - previous_point[1]
            segment_length = math.hypot(dx, dy)
            if min_segment_m <= segment_length <= max_segment_m:
                angle_rad = math.atan2(dy, dx) % math.pi
                bin_index = min(int(angle_rad / math.pi * bin_count), bin_count - 1)
                bins[bin_index] += segment_length
        previous_point = point

    total_weight = sum(bins)
    if total_weight <= 0.0:
        return None, 0.0

    best_index = max(range(bin_count), key=lambda index: bins[index])
    confidence = bins[best_index] / total_weight
    angle_rad = (best_index + 0.5) / bin_count * math.pi
    return angle_rad, confidence


def median_or_none(values: list[float | None]) -> float | None:
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return statistics.median(present_values)


def compute_heading_samples(
    lidar_path: Path,
    stride: int,
    min_valid_points: int,
    bin_count: int,
    smoothing_window: int,
    min_segment_m: float,
    max_segment_m: float,
    min_range_m: float,
    max_range_m: float,
) -> list[HeadingSample]:
    keyframe_indices, valid_counts = select_keyframes(
        lidar_path=lidar_path,
        stride=stride,
        min_valid_points=min_valid_points,
    )
    scans = list(iter_local_scans(lidar_path, keyframe_indices))

    angle_values = []
    confidences = []
    for scan in scans:
        angle_rad, confidence = dominant_structure_angle(
            points=scan.points,
            bin_count=bin_count,
            min_segment_m=min_segment_m,
            max_segment_m=max_segment_m,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
        )
        if angle_rad is None:
            angle_values.append(None)
            confidences.append(confidence)
            continue

        angle_deg = (math.degrees(angle_rad) + 90.0) % 180.0 - 90.0
        angle_values.append(angle_deg)
        confidences.append(confidence)

    half_window = max(smoothing_window // 2, 0)
    smoothed_angles = []
    for index in range(len(angle_values)):
        start = max(0, index - half_window)
        stop = min(len(angle_values), index + half_window + 1)
        smoothed_angles.append(median_or_none(angle_values[start:stop]))

    return [
        HeadingSample(
            scan_index=scan.scan_index,
            angle_deg=angle_deg,
            smoothed_deg=smoothed_deg,
            confidence=confidence,
            valid_points=valid_counts[scan.scan_index],
        )
        for scan, angle_deg, smoothed_deg, confidence in zip(
            scans,
            angle_values,
            smoothed_angles,
            confidences,
        )
    ]


def write_heading_plot(
    samples: list[HeadingSample],
    output_path: Path,
    stride: int,
    turn_threshold_deg: float,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write the Stage 8 PNG") from exc

    width = 1600
    height = 820
    left = 82
    right = 220
    top = 70
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    axis_color = (64, 64, 64)
    grid_color = (225, 225, 225)
    raw_color = (190, 190, 190)
    smooth_color = (109, 40, 217)
    turn_color = (220, 38, 38)
    confidence_color = (22, 163, 74)

    present_angles = [sample.smoothed_deg for sample in samples if sample.smoothed_deg is not None]
    if present_angles:
        min_angle = min(present_angles)
        max_angle = max(present_angles)
    else:
        min_angle = -90.0
        max_angle = 90.0

    angle_padding = max((max_angle - min_angle) * 0.08, 10.0)
    min_angle -= angle_padding
    max_angle += angle_padding
    angle_span = max(max_angle - min_angle, 1.0)
    max_scan_index = samples[-1].scan_index if samples else 1

    def project(scan_index: int, angle_deg: float) -> tuple[float, float]:
        x = left + scan_index / max(max_scan_index, 1) * plot_width
        y = top + plot_height - (angle_deg - min_angle) / angle_span * plot_height
        return x, y

    for step in range(6):
        x = left + round(plot_width * step / 5)
        y = top + round(plot_height * step / 5)
        draw.line([(x, top), (x, top + plot_height)], fill=grid_color)
        draw.line([(left, y), (left + plot_width, y)], fill=grid_color)

        scan_label = round(max_scan_index * step / 5)
        angle_label = max_angle - angle_span * step / 5
        draw.text((x - 22, top + plot_height + 12), str(scan_label), fill=axis_color)
        draw.text((10, y - 7), f"{angle_label:.0f}", fill=axis_color)

    draw.rectangle((left, top, left + plot_width, top + plot_height), outline=axis_color)
    draw.text((left, 18), "Stage 8 heading and structure plot", fill=axis_color)
    draw.text(
        (left, 42),
        (
            f"{len(samples)} sampled scans | stride {stride} | "
            "dominant local line angle"
        ),
        fill=axis_color,
    )
    draw.text((10, top - 24), "angle deg", fill=axis_color)
    draw.text((left + plot_width // 2 - 38, top + plot_height + 36), "scan index", fill=axis_color)

    raw_points = [
        project(sample.scan_index, sample.angle_deg)
        for sample in samples
        if sample.angle_deg is not None
    ]
    if len(raw_points) >= 2:
        draw.line(raw_points, fill=raw_color, width=1)

    smooth_points = [
        project(sample.scan_index, sample.smoothed_deg)
        for sample in samples
        if sample.smoothed_deg is not None
    ]
    if len(smooth_points) >= 2:
        draw.line(smooth_points, fill=smooth_color, width=3)

    turn_count = 0
    previous_sample = None
    for sample in samples:
        if (
            previous_sample is not None
            and sample.smoothed_deg is not None
            and previous_sample.smoothed_deg is not None
        ):
            delta = abs(sample.smoothed_deg - previous_sample.smoothed_deg)
            if delta >= turn_threshold_deg:
                x, _y = project(sample.scan_index, sample.smoothed_deg)
                draw.line([(x, top), (x, top + plot_height)], fill=turn_color)
                turn_count += 1
        if sample.smoothed_deg is not None:
            previous_sample = sample

    confidence_top = top + plot_height - 100
    confidence_points = [
        (
            left + sample.scan_index / max(max_scan_index, 1) * plot_width,
            confidence_top - sample.confidence * 80,
        )
        for sample in samples
    ]
    if len(confidence_points) >= 2:
        draw.line(confidence_points, fill=confidence_color, width=2)

    legend_x = left + plot_width + 34
    legend_y = top + 10
    draw.line([(legend_x, legend_y), (legend_x + 42, legend_y)], fill=smooth_color, width=3)
    draw.text((legend_x + 52, legend_y - 7), "smoothed angle", fill=axis_color)
    draw.line(
        [(legend_x, legend_y + 30), (legend_x + 42, legend_y + 30)],
        fill=raw_color,
        width=1,
    )
    draw.text((legend_x + 52, legend_y + 23), "raw angle", fill=axis_color)
    draw.line(
        [(legend_x, legend_y + 60), (legend_x + 42, legend_y + 60)],
        fill=confidence_color,
        width=2,
    )
    draw.text((legend_x + 52, legend_y + 53), "confidence", fill=axis_color)
    draw.line(
        [(legend_x, legend_y + 90), (legend_x + 42, legend_y + 90)],
        fill=turn_color,
        width=2,
    )
    draw.text((legend_x + 52, legend_y + 83), "turn marker", fill=axis_color)
    draw.text((legend_x, legend_y + 126), f"turn markers {turn_count}", fill=axis_color)
    if samples:
        confidence_values = [sample.confidence for sample in samples]
        draw.text(
            (legend_x, legend_y + 148),
            f"confidence median {statistics.median(confidence_values):.2f}",
            fill=axis_color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_scan_signature(
    scan: LocalScan,
    bin_count: int,
    max_range_m: float,
) -> list[float]:
    signature = [1.0] * bin_count
    for x_m, y_m, _intensity in scan.points:
        distance_m = math.hypot(x_m, y_m)
        if distance_m <= 0.0 or distance_m > max_range_m:
            continue

        angle_rad = math.atan2(y_m, x_m) % (2.0 * math.pi)
        bin_index = min(int(angle_rad / (2.0 * math.pi) * bin_count), bin_count - 1)
        signature[bin_index] = min(signature[bin_index], distance_m / max_range_m)

    return signature


def shifted_signature_distance(
    first: list[float],
    second: list[float],
    shift: int,
    min_overlap_bins: int,
) -> float:
    overlap_count = 0
    total_distance = 0.0
    mismatch_count = 0
    count = len(first)
    for index, first_value in enumerate(first):
        second_value = second[(index + shift) % count]
        first_has_return = first_value < 1.0
        second_has_return = second_value < 1.0
        if first_has_return and second_has_return:
            overlap_count += 1
            total_distance += abs(first_value - second_value)
        elif first_has_return != second_has_return:
            mismatch_count += 1

    if overlap_count < min_overlap_bins:
        return 1.0

    return total_distance / overlap_count + mismatch_count / count * 0.05


def scan_signature_distance(
    first: list[float],
    second: list[float],
    min_overlap_bins: int,
) -> float:
    shift_candidates = [
        0,
        len(first) // 4,
        len(first) // 2,
        len(first) * 3 // 4,
    ]
    return min(
        shifted_signature_distance(first, second, shift, min_overlap_bins)
        for shift in shift_candidates
    )


def relative_pose_delta(
    from_pose: Pose2D,
    to_pose: Pose2D,
) -> tuple[float, float, float]:
    delta_x = to_pose.x_m - from_pose.x_m
    delta_y = to_pose.y_m - from_pose.y_m
    cos_theta = math.cos(from_pose.theta_rad)
    sin_theta = math.sin(from_pose.theta_rad)
    return (
        cos_theta * delta_x + sin_theta * delta_y,
        -sin_theta * delta_x + cos_theta * delta_y,
        normalize_angle_rad(to_pose.theta_rad - from_pose.theta_rad),
    )


def compute_aisle_forward_uncertainty_scale(
    from_pose: Pose2D,
    to_pose: Pose2D,
    max_scale: float,
) -> float:
    if max_scale <= 1.0:
        return 1.0

    forward_m, lateral_m, delta_theta = relative_pose_delta(from_pose, to_pose)
    travel_m = abs(forward_m) + abs(lateral_m)
    if travel_m <= 1e-6:
        return 1.0

    forward_ratio = abs(forward_m) / travel_m
    straight_ratio = max(0.0, 1.0 - abs(delta_theta) / math.radians(12.0))
    return 1.0 + (max_scale - 1.0) * forward_ratio * straight_ratio


def odometry_factor_covariance(
    pose_covariance: PoseCovariance,
    reference_pose: Pose2D,
    pose: Pose2D,
    odometry_weight: float,
    forward_scale: float,
    config: MatchQualityConfig,
    np: object,
) -> object:
    if should_apply_anisotropic_constraints(pose, config) and pose.aisle_frame is not None:
        relative_covariance = relative_frame_covariance(
            covariance=pose.aisle_frame.covariance_world,
            reference_pose=reference_pose,
            np=np,
        )
    else:
        relative_covariance = relative_frame_covariance(
            covariance=pose_covariance,
            reference_pose=reference_pose,
            np=np,
        )
        scale = np.diag(np.array([forward_scale, 1.0, 1.0], dtype=float))
        relative_covariance = scale @ relative_covariance @ scale
    relative_covariance *= 1.0 / (odometry_weight * odometry_weight)
    relative_covariance = 0.5 * (relative_covariance + relative_covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(relative_covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-9)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def find_loop_closures(
    lidar_path: Path,
    poses: list[Pose2D],
    signature_bins: int,
    signature_max_range_m: float,
    min_separation: int,
    max_raw_distance_m: float,
    max_signature_distance: float,
    min_overlap_bins: int,
    max_closures: int,
) -> list[LoopClosure]:
    if max_closures <= 0:
        return []

    scan_indices = [pose.scan_index for pose in poses]
    scans = list(iter_local_scans(lidar_path, scan_indices))
    signatures = [
        build_scan_signature(
            scan=scan,
            bin_count=signature_bins,
            max_range_m=signature_max_range_m,
        )
        for scan in scans
    ]

    candidates = []
    for to_ordinal, to_pose in enumerate(poses):
        for from_ordinal in range(0, to_ordinal - min_separation):
            from_pose = poses[from_ordinal]
            raw_distance_m = math.hypot(
                to_pose.x_m - from_pose.x_m,
                to_pose.y_m - from_pose.y_m,
            )
            if raw_distance_m > max_raw_distance_m:
                continue

            signature_distance = scan_signature_distance(
                signatures[from_ordinal],
                signatures[to_ordinal],
                min_overlap_bins=min_overlap_bins,
            )
            if signature_distance > max_signature_distance:
                continue

            candidates.append(
                LoopClosure(
                    candidate_id=-1,
                    from_ordinal=from_ordinal,
                    to_ordinal=to_ordinal,
                    raw_distance_m=raw_distance_m,
                    signature_distance=signature_distance,
                )
            )

    candidates.sort(key=lambda closure: closure.signature_distance)
    selected = []
    used_pairs = set()
    for candidate in candidates:
        from_bucket = candidate.from_ordinal // 10
        to_bucket = candidate.to_ordinal // 10
        bucket_pair = (from_bucket, to_bucket)
        if bucket_pair in used_pairs:
            continue

        selected.append(
            LoopClosure(
                candidate_id=len(selected),
                from_ordinal=candidate.from_ordinal,
                to_ordinal=candidate.to_ordinal,
                raw_distance_m=candidate.raw_distance_m,
                signature_distance=candidate.signature_distance,
            )
        )
        used_pairs.add(bucket_pair)
        if len(selected) >= max_closures:
            break

    return selected


def pose_loop_hypotheses(pose: Pose2D) -> tuple[PoseHypothesis, ...]:
    if pose.hypotheses:
        return pose.hypotheses
    return (pose_to_hypothesis(pose, 1.0),)


def pose_has_loop_ambiguity(pose: Pose2D) -> bool:
    if pose.hypothesis_diagnostics is not None and pose.hypothesis_diagnostics.ambiguity_gate:
        return True
    return pose.degeneracy is not None and pose.degeneracy.mode == "AISLE"


def loop_pose_degeneracy_state(pose: Pose2D) -> dict:
    degeneracy_mode = pose.degeneracy.mode if pose.degeneracy is not None else "NORMAL"
    return {
        "mode": degeneracy_mode,
        "entropy": pose.degeneracy.entropy if pose.degeneracy is not None else None,
        "linearity": pose.degeneracy.linearity if pose.degeneracy is not None else None,
        "dominant_angle": (
            pose.degeneracy.dominant_angle if pose.degeneracy is not None else None
        ),
        "confidence": pose.degeneracy.confidence if pose.degeneracy is not None else None,
        "mht_entropy": (
            pose.hypothesis_diagnostics.weight_entropy
            if pose.hypothesis_diagnostics is not None
            else None
        ),
        "mht_ambiguity": (
            pose.hypothesis_diagnostics.ambiguity_gate
            if pose.hypothesis_diagnostics is not None
            else False
        ),
        "aisle_corridor_id": (
            pose.aisle_frame.corridor_id if pose.aisle_frame is not None else None
        ),
    }


def build_loop_analysis_context(
    poses: list[Pose2D],
    config: MatchQualityConfig,
) -> LoopAnalysisContext:
    segments, pose_to_segment = build_corridor_segments(poses, config)
    return LoopAnalysisContext(
        pose_to_segment=pose_to_segment,
        segments_by_id={segment.id: segment for segment in segments},
    )


def loop_corridor_similarity(
    from_segment_id: int | None,
    to_segment_id: int | None,
    context: LoopAnalysisContext | None,
) -> float | None:
    if (
        context is None
        or from_segment_id is None
        or to_segment_id is None
    ):
        return None
    if from_segment_id == to_segment_id:
        return 1.0
    from_segment = context.segments_by_id.get(from_segment_id)
    to_segment = context.segments_by_id.get(to_segment_id)
    if from_segment is None or to_segment is None:
        return None
    return corridor_segment_similarity(from_segment, to_segment)


def loop_hypothesis_agreement_count(
    closure: LoopClosure,
    poses: list[Pose2D],
    candidate: Pose2D,
    config: MatchQualityConfig,
) -> int:
    from_pose = poses[closure.from_ordinal]
    to_pose = poses[closure.to_ordinal]
    agreement_count = 0
    theta_threshold_rad = math.radians(config.loop_hypothesis_agreement_theta_deg)
    for from_hypothesis in pose_loop_hypotheses(from_pose):
        from_hypothesis_pose = hypothesis_to_pose2d(
            from_hypothesis,
            scan_index=from_pose.scan_index,
        )
        for to_hypothesis in pose_loop_hypotheses(to_pose):
            to_hypothesis_pose = hypothesis_to_pose2d(
                to_hypothesis,
                scan_index=to_pose.scan_index,
            )
            relative_x_m, relative_y_m, relative_theta_rad = relative_pose_delta(
                from_hypothesis_pose,
                to_hypothesis_pose,
            )
            if (
                math.hypot(
                    relative_x_m - candidate.x_m,
                    relative_y_m - candidate.y_m,
                )
                <= config.loop_hypothesis_agreement_m
                and abs(normalize_angle_rad(relative_theta_rad - candidate.theta_rad))
                <= theta_threshold_rad
            ):
                agreement_count += 1
    return agreement_count


def append_loop_closure_diagnostic(
    diagnostics: list[LoopClosureDiagnostics] | None,
    closure: LoopClosure,
    poses: list[Pose2D],
    accepted: bool,
    reason: str,
    match_quality: MatchQualityConfig,
    analysis_context: LoopAnalysisContext | None = None,
    agreement_count: int = 0,
    match_score: float = 0.0,
    score_gap: float = 0.0,
    confidence: MatchConfidence | None = None,
    surface: ScoreSurface | None = None,
    covariance: PoseCovariance | None = None,
    predicted_relative_transform: tuple[float, float, float] | None = None,
    matched_relative_transform: tuple[float, float, float] | None = None,
) -> None:
    if diagnostics is None:
        return

    from_pose = poses[closure.from_ordinal]
    to_pose = poses[closure.to_ordinal]
    if predicted_relative_transform is None:
        predicted_relative_transform = relative_pose_delta(from_pose, to_pose)
    residual_m = None
    residual_theta_rad = None
    if predicted_relative_transform is not None and matched_relative_transform is not None:
        residual_m = math.hypot(
            matched_relative_transform[0] - predicted_relative_transform[0],
            matched_relative_transform[1] - predicted_relative_transform[1],
        )
        residual_theta_rad = abs(
            normalize_angle_rad(
                matched_relative_transform[2] - predicted_relative_transform[2]
            )
        )

    from_corridor_id = (
        analysis_context.pose_to_segment.get(closure.from_ordinal)
        if analysis_context is not None
        else None
    )
    to_corridor_id = (
        analysis_context.pose_to_segment.get(closure.to_ordinal)
        if analysis_context is not None
        else None
    )
    signature_similarity = loop_corridor_similarity(
        from_corridor_id,
        to_corridor_id,
        analysis_context,
    )
    confidence_ambiguous = (
        confidence is not None
        and (
            confidence.near_best_count > match_quality.max_near_best_candidates
            or confidence.normalized_score_gap < match_quality.score_gap_threshold
        )
    )
    ambiguous = (
        confidence_ambiguous
        or pose_has_loop_ambiguity(from_pose)
        or pose_has_loop_ambiguity(to_pose)
        or reason in {"small_loop_score_gap", "deferred_ambiguous_loop"}
    )
    conflicting = (
        residual_m is not None
        and residual_theta_rad is not None
        and (
            residual_m > match_quality.corridor_graph_conflict_distance_m
            or residual_theta_rad
            > math.radians(match_quality.corridor_graph_conflict_theta_deg)
        )
    )
    diagnostics.append(
        LoopClosureDiagnostics(
            candidate_id=closure.candidate_id,
            from_ordinal=closure.from_ordinal,
            to_ordinal=closure.to_ordinal,
            from_scan_index=from_pose.scan_index,
            to_scan_index=to_pose.scan_index,
            accepted=accepted,
            reason=reason,
            agreement_count=agreement_count,
            from_hypotheses=len(pose_loop_hypotheses(from_pose)),
            to_hypotheses=len(pose_loop_hypotheses(to_pose)),
            from_ambiguous=pose_has_loop_ambiguity(from_pose),
            to_ambiguous=pose_has_loop_ambiguity(to_pose),
            match_score=match_score,
            score_gap=score_gap,
            confidence=confidence,
            hessian_eigenvalues=surface.eigenvalues if surface is not None else None,
            covariance=covariance,
            from_corridor_id=from_corridor_id,
            to_corridor_id=to_corridor_id,
            corridor_signature_similarity=signature_similarity,
            from_degeneracy=loop_pose_degeneracy_state(from_pose),
            to_degeneracy=loop_pose_degeneracy_state(to_pose),
            predicted_relative_transform=predicted_relative_transform,
            matched_relative_transform=matched_relative_transform,
            transform_residual_m=residual_m,
            transform_residual_theta_rad=residual_theta_rad,
            ambiguous=ambiguous,
            conflicting=conflicting,
            raw_distance_m=closure.raw_distance_m,
            signature_distance=closure.signature_distance,
        )
    )


def loop_closure_safety_decision(
    closure: LoopClosure,
    poses: list[Pose2D],
    candidate: Pose2D,
    match_score: float,
    score_gap: float,
    config: MatchQualityConfig,
) -> tuple[bool, str, int]:
    from_ambiguous = pose_has_loop_ambiguity(poses[closure.from_ordinal])
    to_ambiguous = pose_has_loop_ambiguity(poses[closure.to_ordinal])
    agreement_count = loop_hypothesis_agreement_count(
        closure=closure,
        poses=poses,
        candidate=candidate,
        config=config,
    )
    if not from_ambiguous and not to_ambiguous:
        return True, "accepted_unambiguous", agreement_count
    if agreement_count >= 2:
        return True, "accepted_hypothesis_agreement", agreement_count
    if score_gap >= config.loop_strong_score_gap:
        return True, "accepted_strong_margin", agreement_count
    return False, "deferred_ambiguous_loop", agreement_count


def match_loop_constraint(
    closure: LoopClosure,
    poses: list[Pose2D],
    local_points_by_ordinal: list[list[ScanPoint]],
    cell_size_m: float,
    inflation_cells: int,
    search_xy_m: float,
    search_step_m: float,
    search_theta_deg: float,
    search_theta_step_deg: float,
    min_match_score: float,
    min_score_gap: float,
    match_quality: MatchQualityConfig,
    analysis_context: LoopAnalysisContext | None = None,
    diagnostics: list[LoopClosureDiagnostics] | None = None,
) -> LoopConstraint | None:
    from_points = local_points_by_ordinal[closure.from_ordinal]
    to_points = local_points_by_ordinal[closure.to_ordinal]
    from_pose = poses[closure.from_ordinal]
    to_pose = poses[closure.to_ordinal]
    predicted_x_m, predicted_y_m, predicted_theta_rad = relative_pose_delta(
        from_pose,
        to_pose,
    )
    predicted_relative_transform = (
        predicted_x_m,
        predicted_y_m,
        predicted_theta_rad,
    )
    if not from_points or not to_points:
        append_loop_closure_diagnostic(
            diagnostics,
            closure,
            poses,
            accepted=False,
            reason="missing_scan_points",
            match_quality=match_quality,
            analysis_context=analysis_context,
            predicted_relative_transform=predicted_relative_transform,
        )
        return None

    submap_distances, max_distance_m = build_submap_distance_lookup(
        world_scans=[from_points],
        cell_size_m=cell_size_m,
        inflation_cells=inflation_cells,
        submap_keyframes=1,
    )
    xy_offsets = frange(-search_xy_m, search_xy_m, search_step_m)
    theta_offsets = [
        math.radians(value)
        for value in frange(-search_theta_deg, search_theta_deg, search_theta_step_deg)
    ]

    scored_candidates = []
    for offset_theta in theta_offsets:
        theta = predicted_theta_rad + offset_theta
        for offset_x in xy_offsets:
            for offset_y in xy_offsets:
                candidate = Pose2D(
                    scan_index=to_pose.scan_index,
                    x_m=predicted_x_m + offset_x,
                    y_m=predicted_y_m + offset_y,
                    theta_rad=theta,
                    score=0.0,
                )
                score = score_pose(
                    points=to_points,
                    pose=candidate,
                    submap_distances=submap_distances,
                    cell_size_m=cell_size_m,
                    max_distance_m=max_distance_m,
                )
                scored_candidates.append((score, candidate))

    if not scored_candidates:
        append_loop_closure_diagnostic(
            diagnostics,
            closure,
            poses,
            accepted=False,
            reason="no_loop_candidates",
            match_quality=match_quality,
            analysis_context=analysis_context,
            predicted_relative_transform=predicted_relative_transform,
        )
        return None

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate = scored_candidates[0]
    confidence = compute_match_confidence(
        [
            ScoredCandidate(
                adjusted_score=score,
                raw_score=score,
                pose=candidate,
            )
            for score, candidate in scored_candidates
        ],
        match_quality.near_best_score_percent,
    )
    distinct_xy_m = search_step_m * 1.5
    distinct_theta_rad = math.radians(search_theta_step_deg) * 1.5
    second_score = 0.0
    for score, candidate in scored_candidates[1:]:
        if (
            math.hypot(
                candidate.x_m - best_candidate.x_m,
                candidate.y_m - best_candidate.y_m,
            )
            >= distinct_xy_m
            or abs(normalize_angle_rad(candidate.theta_rad - best_candidate.theta_rad))
            >= distinct_theta_rad
        ):
            second_score = score
            break

    score_gap = best_score - second_score
    surface, covariance = estimate_score_surface(
        points=to_points,
        pose=best_candidate,
        submap_distances=submap_distances,
        cell_size_m=cell_size_m,
        max_distance_m=max_distance_m,
        step_m=search_step_m,
        theta_step_rad=math.radians(search_theta_step_deg),
        config=match_quality,
        confidence=confidence,
    )
    matched_relative_transform = (
        best_candidate.x_m,
        best_candidate.y_m,
        normalize_angle_rad(best_candidate.theta_rad),
    )
    agreement_count = loop_hypothesis_agreement_count(
        closure=closure,
        poses=poses,
        candidate=best_candidate,
        config=match_quality,
    )
    if best_score < min_match_score:
        append_loop_closure_diagnostic(
            diagnostics,
            closure,
            poses,
            accepted=False,
            reason="low_loop_match_score",
            match_quality=match_quality,
            analysis_context=analysis_context,
            agreement_count=agreement_count,
            match_score=best_score,
            score_gap=score_gap,
            confidence=confidence,
            surface=surface,
            covariance=covariance,
            predicted_relative_transform=predicted_relative_transform,
            matched_relative_transform=matched_relative_transform,
        )
        return None
    if score_gap < min_score_gap:
        append_loop_closure_diagnostic(
            diagnostics,
            closure,
            poses,
            accepted=False,
            reason="small_loop_score_gap",
            match_quality=match_quality,
            analysis_context=analysis_context,
            agreement_count=agreement_count,
            match_score=best_score,
            score_gap=score_gap,
            confidence=confidence,
            surface=surface,
            covariance=covariance,
            predicted_relative_transform=predicted_relative_transform,
            matched_relative_transform=matched_relative_transform,
        )
        return None

    accepted, reason, agreement_count = loop_closure_safety_decision(
        closure=closure,
        poses=poses,
        candidate=best_candidate,
        match_score=best_score,
        score_gap=score_gap,
        config=match_quality,
    )
    append_loop_closure_diagnostic(
        diagnostics,
        closure,
        poses,
        accepted=accepted,
        reason=reason,
        match_quality=match_quality,
        analysis_context=analysis_context,
        agreement_count=agreement_count,
        match_score=best_score,
        score_gap=score_gap,
        confidence=confidence,
        surface=surface,
        covariance=covariance,
        predicted_relative_transform=predicted_relative_transform,
        matched_relative_transform=matched_relative_transform,
    )
    if not accepted:
        return None

    return LoopConstraint(
        from_ordinal=closure.from_ordinal,
        to_ordinal=closure.to_ordinal,
        raw_distance_m=closure.raw_distance_m,
        signature_distance=closure.signature_distance,
        relative_x_m=best_candidate.x_m,
        relative_y_m=best_candidate.y_m,
        relative_theta_rad=normalize_angle_rad(best_candidate.theta_rad),
        match_score=best_score,
        score_gap=score_gap,
    )


def build_loop_constraints(
    lidar_path: Path,
    poses: list[Pose2D],
    closures: list[LoopClosure],
    max_points: int,
    min_range_m: float,
    max_range_m: float,
    cell_size_m: float,
    inflation_cells: int,
    search_xy_m: float,
    search_step_m: float,
    search_theta_deg: float,
    search_theta_step_deg: float,
    min_match_score: float,
    min_score_gap: float,
    match_quality: MatchQualityConfig | None = None,
    diagnostics: list[LoopClosureDiagnostics] | None = None,
) -> list[LoopConstraint]:
    if not closures:
        return []

    match_quality = match_quality or MatchQualityConfig()
    scan_indices = [pose.scan_index for pose in poses]
    local_points_by_scan_index = {
        scan.scan_index: build_scan_match_points(
            points=scan.points,
            max_points=max_points,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            config=match_quality,
        )
        for scan in iter_local_scans(lidar_path, scan_indices)
    }
    local_points_by_ordinal = [
        local_points_by_scan_index.get(pose.scan_index, []) for pose in poses
    ]
    analysis_context = build_loop_analysis_context(poses, match_quality)

    constraints = []
    for closure in closures:
        constraint = match_loop_constraint(
            closure=closure,
            poses=poses,
            local_points_by_ordinal=local_points_by_ordinal,
            cell_size_m=cell_size_m,
            inflation_cells=inflation_cells,
            search_xy_m=search_xy_m,
            search_step_m=search_step_m,
            search_theta_deg=search_theta_deg,
            search_theta_step_deg=search_theta_step_deg,
            min_match_score=min_match_score,
            min_score_gap=min_score_gap,
            match_quality=match_quality,
            analysis_context=analysis_context,
            diagnostics=diagnostics,
        )
        if constraint is not None:
            constraints.append(constraint)

    return constraints


def write_loop_replay_score_surface(
    records: list[dict],
    output_path: Path,
) -> None:
    sorted_records = sorted(
        records,
        key=lambda record: record["score"],
        reverse=True,
    )
    write_jsonl_records(
        [
            {**record, "rank": rank}
            for rank, record in enumerate(sorted_records, start=1)
        ],
        output_path,
    )


def replay_loop_closure_candidate(
    args: argparse.Namespace,
    poses: list[Pose2D],
    candidate_closures: list[LoopClosure],
    match_quality: MatchQualityConfig,
    label: str,
) -> None:
    candidate_id = args.replay_loop_closure
    if candidate_id is None:
        return
    selected = next(
        (
            closure
            for closure in candidate_closures
            if closure.candidate_id == candidate_id
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"--replay-loop-closure {candidate_id} was not evaluated")

    from_pose = poses[selected.from_ordinal]
    to_pose = poses[selected.to_ordinal]
    scan_indices = [from_pose.scan_index, to_pose.scan_index]
    scans = {scan.scan_index: scan for scan in iter_local_scans(args.lidar, scan_indices)}
    from_scan = scans.get(from_pose.scan_index)
    to_scan = scans.get(to_pose.scan_index)
    from_points = (
        build_scan_match_points(
            points=from_scan.points,
            max_points=args.max_points,
            min_range_m=args.min_range_m,
            max_range_m=args.max_range_m,
            config=match_quality,
        )
        if from_scan is not None
        else []
    )
    to_points = (
        build_scan_match_points(
            points=to_scan.points,
            max_points=args.max_points,
            min_range_m=args.min_range_m,
            max_range_m=args.max_range_m,
            config=match_quality,
        )
        if to_scan is not None
        else []
    )
    predicted_relative_transform = relative_pose_delta(from_pose, to_pose)
    diagnostics: list[LoopClosureDiagnostics] = []
    analysis_context = build_loop_analysis_context(poses, match_quality)
    score_records: list[dict] = []

    if not from_points or not to_points:
        append_loop_closure_diagnostic(
            diagnostics,
            selected,
            poses,
            accepted=False,
            reason="missing_scan_points",
            match_quality=match_quality,
            analysis_context=analysis_context,
            predicted_relative_transform=predicted_relative_transform,
        )
    else:
        submap_distances, max_distance_m = build_submap_distance_lookup(
            world_scans=[from_points],
            cell_size_m=args.cell_size_m,
            inflation_cells=args.inflation_cells,
            submap_keyframes=1,
        )
        xy_offsets = frange(
            -args.loop_match_search_xy_m,
            args.loop_match_search_xy_m,
            args.loop_match_search_step_m,
        )
        theta_offsets = [
            math.radians(value)
            for value in frange(
                -args.loop_match_search_theta_deg,
                args.loop_match_search_theta_deg,
                args.loop_match_search_theta_step_deg,
            )
        ]
        scored_candidates: list[tuple[float, Pose2D]] = []
        for offset_theta in theta_offsets:
            theta = predicted_relative_transform[2] + offset_theta
            for offset_x in xy_offsets:
                for offset_y in xy_offsets:
                    candidate = Pose2D(
                        scan_index=to_pose.scan_index,
                        x_m=predicted_relative_transform[0] + offset_x,
                        y_m=predicted_relative_transform[1] + offset_y,
                        theta_rad=theta,
                        score=0.0,
                    )
                    score = score_pose(
                        points=to_points,
                        pose=candidate,
                        submap_distances=submap_distances,
                        cell_size_m=args.cell_size_m,
                        max_distance_m=max_distance_m,
                    )
                    scored_candidates.append((score, candidate))
                    score_records.append(
                        {
                            "candidate_id": selected.candidate_id,
                            "offset_x_m": offset_x,
                            "offset_y_m": offset_y,
                            "offset_theta_rad": offset_theta,
                            "x_m": candidate.x_m,
                            "y_m": candidate.y_m,
                            "theta_rad": candidate.theta_rad,
                            "score": score,
                        }
                    )

        if not scored_candidates:
            append_loop_closure_diagnostic(
                diagnostics,
                selected,
                poses,
                accepted=False,
                reason="no_loop_candidates",
                match_quality=match_quality,
                analysis_context=analysis_context,
                predicted_relative_transform=predicted_relative_transform,
            )
        else:
            scored_candidates.sort(key=lambda item: item[0], reverse=True)
            best_score, best_candidate = scored_candidates[0]
            confidence = compute_match_confidence(
                [
                    ScoredCandidate(
                        adjusted_score=score,
                        raw_score=score,
                        pose=candidate,
                    )
                    for score, candidate in scored_candidates
                ],
                match_quality.near_best_score_percent,
            )
            distinct_xy_m = args.loop_match_search_step_m * 1.5
            distinct_theta_rad = math.radians(args.loop_match_search_theta_step_deg) * 1.5
            second_score = 0.0
            for score, candidate in scored_candidates[1:]:
                if (
                    math.hypot(
                        candidate.x_m - best_candidate.x_m,
                        candidate.y_m - best_candidate.y_m,
                    )
                    >= distinct_xy_m
                    or abs(normalize_angle_rad(candidate.theta_rad - best_candidate.theta_rad))
                    >= distinct_theta_rad
                ):
                    second_score = score
                    break
            score_gap = best_score - second_score
            surface, covariance = estimate_score_surface(
                points=to_points,
                pose=best_candidate,
                submap_distances=submap_distances,
                cell_size_m=args.cell_size_m,
                max_distance_m=max_distance_m,
                step_m=args.loop_match_search_step_m,
                theta_step_rad=math.radians(args.loop_match_search_theta_step_deg),
                config=match_quality,
                confidence=confidence,
            )
            agreement_count = loop_hypothesis_agreement_count(
                closure=selected,
                poses=poses,
                candidate=best_candidate,
                config=match_quality,
            )
            accepted = False
            reason = "low_loop_match_score"
            if best_score >= args.min_loop_match_score:
                if score_gap < args.min_loop_match_score_gap:
                    reason = "small_loop_score_gap"
                else:
                    accepted, reason, agreement_count = loop_closure_safety_decision(
                        closure=selected,
                        poses=poses,
                        candidate=best_candidate,
                        match_score=best_score,
                        score_gap=score_gap,
                        config=match_quality,
                    )
            append_loop_closure_diagnostic(
                diagnostics,
                selected,
                poses,
                accepted=accepted,
                reason=reason,
                match_quality=match_quality,
                analysis_context=analysis_context,
                agreement_count=agreement_count,
                match_score=best_score,
                score_gap=score_gap,
                confidence=confidence,
                surface=surface,
                covariance=covariance,
                predicted_relative_transform=predicted_relative_transform,
                matched_relative_transform=(
                    best_candidate.x_m,
                    best_candidate.y_m,
                    normalize_angle_rad(best_candidate.theta_rad),
                ),
            )

    diagnostic = diagnostics[0] if diagnostics else None
    from_segment_id = (
        analysis_context.pose_to_segment.get(selected.from_ordinal)
        if analysis_context is not None
        else None
    )
    to_segment_id = (
        analysis_context.pose_to_segment.get(selected.to_ordinal)
        if analysis_context is not None
        else None
    )
    replay_path = args.out_dir / f"loop_replay_{label}_{candidate_id}.json"
    score_surface_path = (
        args.out_dir / f"loop_replay_{label}_{candidate_id}_score_surface.jsonl"
    )
    payload = {
        "candidate_id": candidate_id,
        "source_pose_id": selected.from_ordinal,
        "target_pose_id": selected.to_ordinal,
        "source_scan_index": from_pose.scan_index,
        "target_scan_index": to_pose.scan_index,
        "submap": {
            "source_point_count": len(from_points),
            "target_point_count": len(to_points),
            "cell_size_m": args.cell_size_m,
            "inflation_cells": args.inflation_cells,
        },
        "diagnostic": asdict(diagnostic) if diagnostic is not None else None,
        "corridor_graph_context": {
            "source_corridor_id": from_segment_id,
            "target_corridor_id": to_segment_id,
            "signature_similarity": (
                diagnostic.corridor_signature_similarity
                if diagnostic is not None
                else None
            ),
            "source_segment": (
                asdict(analysis_context.segments_by_id[from_segment_id])
                if from_segment_id in analysis_context.segments_by_id
                else None
            ),
            "target_segment": (
                asdict(analysis_context.segments_by_id[to_segment_id])
                if to_segment_id in analysis_context.segments_by_id
                else None
            ),
        },
        "score_surface_file": str(score_surface_path),
    }
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    write_loop_replay_score_surface(score_records, score_surface_path)
    print(f"replayed loop closure candidate {candidate_id}")
    print(f"wrote {replay_path}")
    print(f"wrote {score_surface_path}")


def solve_loop_closure_graph(
    poses: list[Pose2D],
    closures: list[LoopConstraint],
    loop_weight: float,
    odometry_weight: float,
    anchor_weight: float,
    aisle_forward_uncertainty_scale: float,
    match_quality: MatchQualityConfig | None = None,
) -> list[Pose2D]:
    try:
        import numpy as np
        import gtsam
    except ImportError as exc:
        raise RuntimeError(
            "GTSAM and NumPy are required for Stage 10/11 loop closure. "
            "Install them in the active Python environment with "
            "`python3 -m pip install gtsam numpy`."
        ) from exc

    match_quality = match_quality or MatchQualityConfig()
    pose_count = len(poses)
    if pose_count == 0:
        return []

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    anchor_sigma = 1.0 / anchor_weight
    odometry_translation_sigma = 1.0 / odometry_weight
    odometry_theta_sigma = math.radians(5.0) / odometry_weight
    loop_translation_sigma = 1.0 / loop_weight
    loop_theta_sigma = math.radians(45.0)

    anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([anchor_sigma, anchor_sigma, anchor_sigma], dtype=float)
    )
    graph.add(
        gtsam.PriorFactorPose2(
            0,
            gtsam.Pose2(poses[0].x_m, poses[0].y_m, poses[0].theta_rad),
            anchor_noise,
        )
    )

    for ordinal, pose in enumerate(poses):
        initial.insert(ordinal, gtsam.Pose2(pose.x_m, pose.y_m, pose.theta_rad))

    for ordinal in range(1, pose_count):
        previous_pose = poses[ordinal - 1]
        pose = poses[ordinal]
        relative_x_m, relative_y_m, relative_theta_rad = anisotropic_relative_pose_delta(
            previous_pose,
            pose,
            match_quality,
        )
        forward_scale = compute_aisle_forward_uncertainty_scale(
            previous_pose,
            pose,
            aisle_forward_uncertainty_scale,
        )
        if pose.covariance is not None:
            odometry_noise = gtsam.noiseModel.Gaussian.Covariance(
                odometry_factor_covariance(
                    pose_covariance=pose.covariance,
                    reference_pose=previous_pose,
                    pose=pose,
                    odometry_weight=odometry_weight,
                    forward_scale=forward_scale,
                    config=match_quality,
                    np=np,
                )
            )
        else:
            odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array(
                    [
                        odometry_translation_sigma * forward_scale,
                        odometry_translation_sigma,
                        odometry_theta_sigma,
                    ],
                    dtype=float,
                )
            )
        graph.add(
            gtsam.BetweenFactorPose2(
                ordinal - 1,
                ordinal,
                gtsam.Pose2(relative_x_m, relative_y_m, relative_theta_rad),
                odometry_noise,
            )
        )

    for closure in closures:
        loop_base_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [loop_translation_sigma, loop_translation_sigma, loop_theta_sigma],
                dtype=float,
            )
        )
        loop_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            loop_base_noise,
        )
        graph.add(
            gtsam.BetweenFactorPose2(
                closure.from_ordinal,
                closure.to_ordinal,
                gtsam.Pose2(
                    closure.relative_x_m,
                    closure.relative_y_m,
                    closure.relative_theta_rad,
                ),
                loop_noise,
            )
        )

    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()

    corrected_poses = []
    for ordinal, pose in enumerate(poses):
        result_pose = result.atPose2(ordinal)
        corrected_poses.append(
            Pose2D(
                scan_index=pose.scan_index,
                x_m=float(result_pose.x()),
                y_m=float(result_pose.y()),
                theta_rad=normalize_angle_rad(float(result_pose.theta())),
                score=pose.score,
                covariance=pose.covariance,
                confidence=pose.confidence,
                surface=pose.surface,
                score_breakdown=pose.score_breakdown,
                degeneracy=pose.degeneracy,
                hypothesis_diagnostics=pose.hypothesis_diagnostics,
                aisle_frame=pose.aisle_frame,
                hypotheses=pose.hypotheses,
                accepted=pose.accepted,
            )
        )

    return corrected_poses


def extract_reference_path_points(reference_path: Path) -> tuple[object, object]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("NumPy and Pillow are required for Stage 11 alignment") from exc

    image = Image.open(reference_path).convert("RGB")
    array = np.asarray(image)
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    height, width = red.shape

    purple_mask = (
        (red > 70)
        & (blue > 120)
        & (green < 120)
        & (blue > green + 40)
        & (red > green + 30)
    )
    legend_mask = (
        (np.indices((height, width))[1] > width * 0.45)
        & (np.indices((height, width))[0] > height * 0.82)
    )
    purple_mask &= ~legend_mask

    y_indices, x_indices = np.nonzero(purple_mask)
    if len(x_indices) == 0:
        raise ValueError(f"could not detect purple reference trajectory in {reference_path}")

    points = np.column_stack((x_indices.astype(float), y_indices.astype(float)))
    return image, points


def pca_angle(points: object) -> float:
    import numpy as np

    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    return math.atan2(float(vector[1]), float(vector[0]))


def sample_array(points: object, max_count: int) -> object:
    import numpy as np

    if len(points) <= max_count:
        return points
    indices = np.linspace(0, len(points) - 1, max_count).astype(int)
    return points[indices]


def mean_nearest_distance(source_points: object, target_points: object) -> float:
    import numpy as np

    if len(source_points) == 0 or len(target_points) == 0:
        return float("inf")

    distances = []
    for start in range(0, len(source_points), 200):
        chunk = source_points[start : start + 200]
        squared = ((chunk[:, None, :] - target_points[None, :, :]) ** 2).sum(axis=2)
        distances.extend(np.sqrt(squared.min(axis=1)).tolist())
    return float(np.mean(distances))


def align_poses_to_reference(
    poses: list[Pose2D],
    reference_points: object,
    image_size: tuple[int, int],
) -> tuple[list[tuple[float, float]], float]:
    import numpy as np

    if not poses:
        return [], 0.0

    source = np.array([[pose.x_m, -pose.y_m] for pose in poses], dtype=float)
    source_center = source.mean(axis=0)
    target_center = reference_points.mean(axis=0)
    target_angle = pca_angle(reference_points)
    target_centered = reference_points - target_center
    target_diag = np.linalg.norm(np.ptp(target_centered, axis=0))
    target_sample = sample_array(reference_points, 3500)

    best_points = None
    best_score = float("inf")
    for mirror_x in (1.0, -1.0):
        for mirror_y in (1.0, -1.0):
            reflected = (source - source_center) * np.array([mirror_x, mirror_y])
            source_angle = pca_angle(reflected)
            for angle_offset in (0.0, math.pi / 2.0, math.pi, math.pi * 3.0 / 2.0):
                theta = target_angle - source_angle + angle_offset
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                rotation = np.array(
                    [
                        [cos_theta, -sin_theta],
                        [sin_theta, cos_theta],
                    ]
                )
                rotated = reflected @ rotation.T
                source_diag = max(np.linalg.norm(np.ptp(rotated, axis=0)), 1e-6)
                scale = target_diag / source_diag
                transformed = rotated * scale + target_center

                width, height = image_size
                outside = (
                    (transformed[:, 0] < 0)
                    | (transformed[:, 0] >= width)
                    | (transformed[:, 1] < 0)
                    | (transformed[:, 1] >= height)
                ).mean()
                source_sample = sample_array(transformed, 1000)
                score = mean_nearest_distance(source_sample, target_sample) + outside * 100.0
                if score < best_score:
                    best_score = score
                    best_points = transformed

    return [tuple(point) for point in best_points], best_score


def write_final_aligned_render(
    poses: list[Pose2D],
    reference_path: Path,
    output_path: Path,
    background_opacity: float,
) -> float:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write the Stage 11 PNG") from exc

    reference_image, reference_points = extract_reference_path_points(reference_path)
    aligned_points, fit_score = align_poses_to_reference(
        poses=poses,
        reference_points=reference_points,
        image_size=reference_image.size,
    )

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
    path_color = (126, 34, 206)
    start_color = (34, 197, 94)
    end_color = (220, 38, 38)

    reference_width, reference_height = reference_image.size
    reference_scale = min(
        plot_width / max(reference_width, 1),
        plot_height / max(reference_height, 1),
    )
    framed_width = max(round(reference_width * reference_scale), 1)
    framed_height = max(round(reference_height * reference_scale), 1)
    framed_left = left + (plot_width - framed_width) / 2.0
    framed_top = top + (plot_height - framed_height) / 2.0

    white_reference = Image.new("RGB", reference_image.size, "white")
    faded_reference = Image.blend(
        white_reference,
        reference_image.convert("RGB"),
        background_opacity,
    )
    faded_reference = faded_reference.resize(
        (framed_width, framed_height),
        Image.Resampling.LANCZOS,
    )
    image.paste(faded_reference, (round(framed_left), round(framed_top)))

    def project_reference_point(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return (
            framed_left + x * reference_scale,
            framed_top + y * reference_scale,
        )

    for step in range(6):
        x = left + round(plot_width * step / 5)
        y = top + round(plot_height * step / 5)
        draw.line([(x, top), (x, top + plot_height)], fill=grid_color)
        draw.line([(left, y), (left + plot_width, y)], fill=grid_color)

    title = "Stage 11 final aligned trajectory"
    subtitle = (
        f"{len(poses)} keyframe poses | aligned to reference | "
        f"fit score {fit_score:.2f}"
    )
    draw.text((left, 18), title, fill=axis_color)
    draw.text((left, 40), subtitle, fill=axis_color)

    path_overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    path_draw = ImageDraw.Draw(path_overlay)
    if len(aligned_points) >= 2:
        framed_path = [project_reference_point(point) for point in aligned_points]
        path_draw.line(framed_path, fill=path_color + (255,), width=4)
        for point in aligned_points[:: max(len(aligned_points) // 30, 1)]:
            x, y = project_reference_point(point)
            path_draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=path_color + (255,))

    if aligned_points:
        start_x, start_y = project_reference_point(aligned_points[0])
        end_x, end_y = project_reference_point(aligned_points[-1])
        path_draw.ellipse(
            (start_x - 6, start_y - 6, start_x + 6, start_y + 6),
            fill=start_color + (255,),
        )
        path_draw.ellipse(
            (end_x - 6, end_y - 6, end_x + 6, end_y + 6),
            fill=end_color + (255,),
        )

    plot_mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(plot_mask)
    mask_draw.rectangle((left, top, left + plot_width, top + plot_height), fill=255)
    clipped_path = Image.composite(
        path_overlay,
        Image.new("RGBA", (width, height), (255, 255, 255, 0)),
        plot_mask,
    )
    image = Image.alpha_composite(image.convert("RGBA"), clipped_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((left, top, left + plot_width, top + plot_height), outline=axis_color)

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
    draw.text((legend_x, legend_y + 96), f"fit score {fit_score:.2f}", fill=axis_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return fit_score


def pose_tuple_to_pose2d(pose: tuple[float, float, float]) -> Pose2D:
    return Pose2D(
        scan_index=-1,
        x_m=pose[0],
        y_m=pose[1],
        theta_rad=pose[2],
        score=0.0,
    )


def average_axes(axes: list[tuple[float, float]]) -> tuple[float, float]:
    if not axes:
        return 1.0, 0.0

    reference = axes[0]
    total_x = 0.0
    total_y = 0.0
    for axis in axes:
        axis_x, axis_y = align_axis_to_reference(axis, reference)
        total_x += axis_x
        total_y += axis_y
    return normalize_vector_2d((total_x, total_y))


def pose_axis(pose: Pose2D) -> tuple[float, float]:
    if pose.aisle_frame is not None:
        return pose.aisle_frame.axis_x, pose.aisle_frame.axis_y
    return math.cos(pose.theta_rad), math.sin(pose.theta_rad)


def segment_length(poses: list[Pose2D], indices: list[int]) -> float:
    if len(indices) < 2:
        return 0.0
    length_m = 0.0
    for previous_index, current_index in zip(indices, indices[1:]):
        previous_pose = poses[previous_index]
        current_pose = poses[current_index]
        length_m += math.hypot(
            current_pose.x_m - previous_pose.x_m,
            current_pose.y_m - previous_pose.y_m,
        )
    return length_m


def compute_corridor_signature(
    poses: list[Pose2D],
    indices: list[int],
    start_reason: str,
) -> dict:
    segment_poses = [poses[index] for index in indices]
    pose_axes = [pose_axis(pose) for pose in segment_poses]
    axis = average_axes(pose_axes)
    axis_angle = math.atan2(axis[1], axis[0]) % math.pi
    axis_bin = min(int(axis_angle / math.pi * 12), 11)
    axis_histogram = [0.0 for _ in range(12)]
    for current_axis in pose_axes:
        current_angle = math.atan2(current_axis[1], current_axis[0]) % math.pi
        current_bin = min(int(current_angle / math.pi * len(axis_histogram)), 11)
        axis_histogram[current_bin] += 1.0
    axis_histogram_total = sum(axis_histogram) or 1.0
    axis_histogram = [value / axis_histogram_total for value in axis_histogram]
    entropies = [
        pose.degeneracy.entropy
        for pose in segment_poses
        if pose.degeneracy is not None
    ]
    confidences = [
        pose.aisle_frame.confidence
        for pose in segment_poses
        if pose.aisle_frame is not None
    ]
    linearities = [
        pose.aisle_frame.local_linearity
        for pose in segment_poses
        if pose.aisle_frame is not None
    ]
    salient_fractions = [
        pose.score_breakdown.salient_score_fraction
        for pose in segment_poses
        if pose.score_breakdown is not None
    ]
    linear_fractions = [
        pose.score_breakdown.linear_score_fraction
        for pose in segment_poses
        if pose.score_breakdown is not None
    ]
    heading_changes = [
        abs(normalize_angle_rad(current.theta_rad - previous.theta_rad))
        for previous, current in zip(segment_poses, segment_poses[1:])
    ]
    turn_count = sum(
        1 for heading_change in heading_changes if heading_change > math.radians(20.0)
    )
    intersection_count = sum(
        1
        for pose in segment_poses
        if pose.degeneracy is not None and pose.degeneracy.mode == "INTERSECTION"
    )
    corridor_reset_count = sum(
        1
        for pose in segment_poses
        if pose.aisle_frame is not None and pose.aisle_frame.corridor_reset
    )
    ambiguity_count = sum(
        1
        for pose in segment_poses
        if pose.hypothesis_diagnostics is not None
        and pose.hypothesis_diagnostics.ambiguity_gate
    )
    drift_values = [
        abs(pose.aisle_frame.longitudinal_drift_m)
        for pose in segment_poses
        if pose.aisle_frame is not None
    ]
    start_pose = segment_poses[0]
    end_pose = segment_poses[-1]
    entry_exit_delta = normalize_angle_rad(end_pose.theta_rad - start_pose.theta_rad)
    net_displacement_m = math.hypot(
        end_pose.x_m - start_pose.x_m,
        end_pose.y_m - start_pose.y_m,
    )
    return {
        "axis_bin": axis_bin,
        "axis_angle_rad": axis_angle,
        "axis_histogram": tuple(axis_histogram),
        "average_entropy": statistics.fmean(entropies) if entropies else 1.0,
        "average_confidence": statistics.fmean(confidences) if confidences else 0.0,
        "average_linearity": statistics.fmean(linearities) if linearities else 1.0,
        "salient_score_fraction": (
            statistics.fmean(salient_fractions) if salient_fractions else 0.0
        ),
        "linear_score_fraction": (
            statistics.fmean(linear_fractions) if linear_fractions else 0.0
        ),
        "salient_to_linear_ratio": (
            (statistics.fmean(salient_fractions) if salient_fractions else 0.0)
            / max(statistics.fmean(linear_fractions) if linear_fractions else 0.0, 1e-6)
        ),
        "turn_count": turn_count,
        "intersection_count": intersection_count,
        "corridor_reset_count": corridor_reset_count,
        "ambiguity_count": ambiguity_count,
        "average_abs_longitudinal_drift_m": (
            statistics.fmean(drift_values) if drift_values else 0.0
        ),
        "entry_exit_heading_delta_rad": entry_exit_delta,
        "net_displacement_m": net_displacement_m,
        "start_scan": start_pose.scan_index,
        "end_scan": end_pose.scan_index,
        "pose_count": len(segment_poses),
        "start_reason": start_reason,
    }


def make_corridor_segment(
    segment_id: int,
    poses: list[Pose2D],
    indices: list[int],
    start_reason: str,
) -> CorridorSegment:
    segment_poses = [poses[index] for index in indices]
    axis = average_axes([pose_axis(pose) for pose in segment_poses])
    signature = compute_corridor_signature(
        poses=poses,
        indices=indices,
        start_reason=start_reason,
    )
    start_pose = segment_poses[0]
    end_pose = segment_poses[-1]
    return CorridorSegment(
        id=segment_id,
        start_pose=(start_pose.x_m, start_pose.y_m, start_pose.theta_rad),
        end_pose=(end_pose.x_m, end_pose.y_m, end_pose.theta_rad),
        axis=axis,
        length_estimate=segment_length(poses, indices),
        signature=signature,
        parent_poses=tuple(indices),
    )


def segment_split_reasons(
    poses: list[Pose2D],
    current_indices: list[int],
    previous_pose: Pose2D,
    current_pose: Pose2D,
    config: MatchQualityConfig,
) -> list[str]:
    reasons = []
    previous_mode = previous_pose.degeneracy.mode if previous_pose.degeneracy else "NORMAL"
    current_mode = current_pose.degeneracy.mode if current_pose.degeneracy else "NORMAL"
    if previous_mode == "AISLE" and current_mode == "INTERSECTION":
        reasons.append("aisle_to_intersection")

    current_axis = pose_axis(current_pose)
    segment_axis = average_axes([pose_axis(poses[index]) for index in current_indices])
    axis_change = axis_angle_delta_rad(current_axis, segment_axis)
    if axis_change > math.radians(config.corridor_segment_axis_threshold_deg):
        reasons.append("axis_change")

    current_entropy = (
        current_pose.hypothesis_diagnostics.weight_entropy
        if current_pose.hypothesis_diagnostics is not None
        else 0.0
    )
    current_reset = (
        current_pose.aisle_frame is not None and current_pose.aisle_frame.corridor_reset
    )
    if (
        current_entropy >= config.corridor_segment_entropy_threshold
        and current_reset
    ):
        reasons.append("entropy_spike_reset")
    elif current_reset:
        reasons.append("corridor_reset")

    previous_confidence = (
        previous_pose.aisle_frame.confidence
        if previous_pose.aisle_frame is not None
        else 0.0
    )
    current_confidence = (
        current_pose.aisle_frame.confidence
        if current_pose.aisle_frame is not None
        else 0.0
    )
    if abs(current_confidence - previous_confidence) > config.corridor_segment_confidence_jump:
        reasons.append("confidence_jump")

    previous_drift = (
        previous_pose.aisle_frame.longitudinal_drift_m
        if previous_pose.aisle_frame is not None
        else 0.0
    )
    current_drift = (
        current_pose.aisle_frame.longitudinal_drift_m
        if current_pose.aisle_frame is not None
        else 0.0
    )
    if abs(current_drift - previous_drift) > config.corridor_segment_drift_reset_m:
        reasons.append("drift_reset")

    previous_corridor_id = (
        previous_pose.aisle_frame.corridor_id
        if previous_pose.aisle_frame is not None
        else None
    )
    current_corridor_id = (
        current_pose.aisle_frame.corridor_id
        if current_pose.aisle_frame is not None
        else None
    )
    if (
        previous_corridor_id is not None
        and current_corridor_id is not None
        and previous_corridor_id != current_corridor_id
    ):
        reasons.append("corridor_id_change")
    return reasons


def build_corridor_segments(
    poses: list[Pose2D],
    config: MatchQualityConfig,
    loop_split_ordinals: set[int] | None = None,
) -> tuple[list[CorridorSegment], dict[int, int]]:
    if not poses:
        return [], {}

    loop_split_ordinals = loop_split_ordinals or set()
    segments = []
    pose_to_segment = {}
    current_indices = [0]
    current_start_reason = "start"
    for ordinal in range(1, len(poses)):
        reasons = segment_split_reasons(
            poses=poses,
            current_indices=current_indices,
            previous_pose=poses[ordinal - 1],
            current_pose=poses[ordinal],
            config=config,
        )
        if ordinal in loop_split_ordinals:
            reasons.append("loop_ambiguity")
        should_split = bool(reasons) and len(current_indices) >= config.corridor_segment_min_poses
        if should_split:
            segment = make_corridor_segment(
                segment_id=len(segments),
                poses=poses,
                indices=current_indices,
                start_reason=current_start_reason,
            )
            segments.append(segment)
            for pose_index in current_indices:
                pose_to_segment[pose_index] = segment.id
            current_indices = [ordinal]
            current_start_reason = ",".join(reasons)
        else:
            current_indices.append(ordinal)

    if current_indices:
        segment = make_corridor_segment(
            segment_id=len(segments),
            poses=poses,
            indices=current_indices,
            start_reason=current_start_reason,
        )
        segments.append(segment)
        for pose_index in current_indices:
            pose_to_segment[pose_index] = segment.id

    return segments, pose_to_segment


def loop_ambiguity_split_ordinals(
    closures: list[LoopConstraint],
    config: MatchQualityConfig,
    pose_count: int,
) -> set[int]:
    split_ordinals: set[int] = set()
    for closure in closures:
        if closure.score_gap >= config.loop_strong_score_gap:
            continue
        if closure.raw_distance_m < config.corridor_graph_conflict_distance_m:
            continue
        for ordinal in (closure.from_ordinal, closure.to_ordinal):
            if 0 < ordinal < pose_count:
                split_ordinals.add(ordinal)
    return split_ordinals


def corridor_segment_similarity(
    first: CorridorSegment,
    second: CorridorSegment,
) -> float:
    axis_score = 1.0 - min(axis_angle_delta_rad(first.axis, second.axis) / (math.pi / 2.0), 1.0)
    entropy_delta = abs(
        first.signature.get("average_entropy", 1.0)
        - second.signature.get("average_entropy", 1.0)
    )
    saliency_delta = abs(
        first.signature.get("salient_to_linear_ratio", 0.0)
        - second.signature.get("salient_to_linear_ratio", 0.0)
    )
    turn_delta = abs(
        first.signature.get("turn_count", 0)
        - second.signature.get("turn_count", 0)
    )
    entropy_score = max(0.0, 1.0 - entropy_delta)
    saliency_score = 1.0 / (1.0 + saliency_delta)
    turn_score = 1.0 / (1.0 + turn_delta)
    return min(max(0.45 * axis_score + 0.25 * entropy_score + 0.2 * saliency_score + 0.1 * turn_score, 0.0), 1.0)


def segment_edge_transform(
    from_pose: tuple[float, float, float],
    to_pose: tuple[float, float, float],
) -> tuple[float, float, float]:
    return relative_pose_delta(
        pose_tuple_to_pose2d(from_pose),
        pose_tuple_to_pose2d(to_pose),
    )


def build_corridor_graph(
    poses: list[Pose2D],
    closures: list[LoopConstraint],
    config: MatchQualityConfig,
) -> CorridorGraph:
    if not config.corridor_graph_enabled:
        return CorridorGraph(
            segments=(),
            edges=(),
            pose_to_segment={},
            ambiguous_edges_count=0,
            conflicting_loop_closures=0,
            topological_inconsistencies=("disabled",),
        )

    segments, pose_to_segment = build_corridor_segments(
        poses,
        config,
        loop_split_ordinals=loop_ambiguity_split_ordinals(
            closures,
            config,
            len(poses),
        ),
    )
    edges: list[CorridorGraphEdge] = []
    inconsistencies: list[str] = []

    for first, second in zip(segments, segments[1:]):
        edge_type = (
            "ambiguity_transition"
            if str(second.signature.get("start_reason", "")).find("entropy") >= 0
            or str(second.signature.get("start_reason", "")).find("reset") >= 0
            else "sequential_transition"
        )
        confidence = min(
            first.signature.get("average_confidence", 0.0),
            second.signature.get("average_confidence", 0.0),
        )
        edges.append(
            CorridorGraphEdge(
                from_segment=first.id,
                to_segment=second.id,
                edge_type=edge_type,
                relative_transform=segment_edge_transform(first.end_pose, second.start_pose),
                confidence_score=confidence,
                supporting_scan_indices=(
                    int(first.signature.get("end_scan", -1)),
                    int(second.signature.get("start_scan", -1)),
                ),
                support_count=1,
                status="LOW_CONFIDENCE" if edge_type == "ambiguity_transition" else "STRONG",
            )
        )

    loop_groups: dict[tuple[int, int], list[LoopConstraint]] = {}
    for closure in closures:
        from_segment = pose_to_segment.get(closure.from_ordinal)
        to_segment = pose_to_segment.get(closure.to_ordinal)
        if from_segment is None or to_segment is None or from_segment == to_segment:
            continue
        key = (min(from_segment, to_segment), max(from_segment, to_segment))
        loop_groups.setdefault(key, []).append(closure)

    segment_by_id = {segment.id: segment for segment in segments}
    conflicting_loop_closures = 0
    for (from_segment, to_segment), group in loop_groups.items():
        first = segment_by_id[from_segment]
        second = segment_by_id[to_segment]
        similarity = corridor_segment_similarity(first, second)
        support_count = len(group)
        representative = max(group, key=lambda closure: closure.match_score)
        base_confidence = representative.match_score * (
            0.5 + 0.5 * similarity
        ) * min(1.0, support_count / max(config.corridor_edge_support_required, 1))
        status = "STRONG"
        if support_count < config.corridor_edge_support_required:
            status = "LOW_CONFIDENCE"
        if similarity < config.corridor_similarity_low_confidence:
            status = "LOW_CONFIDENCE"

        observed_transform = (
            representative.relative_x_m,
            representative.relative_y_m,
            representative.relative_theta_rad,
        )
        pose_transform = segment_edge_transform(first.end_pose, second.start_pose)
        residual_m = math.hypot(
            observed_transform[0] - pose_transform[0],
            observed_transform[1] - pose_transform[1],
        )
        residual_theta = abs(
            normalize_angle_rad(observed_transform[2] - pose_transform[2])
        )
        if (
            residual_m > config.corridor_graph_conflict_distance_m
            or residual_theta > math.radians(config.corridor_graph_conflict_theta_deg)
        ):
            conflicting_loop_closures += 1
            status = "CONFLICT"
            inconsistencies.append(
                f"loop edge {from_segment}->{to_segment} residual "
                f"{residual_m:.2f} m, {math.degrees(residual_theta):.1f} deg"
            )

        supporting_scans = []
        for closure in group:
            supporting_scans.extend(
                (
                    poses[closure.from_ordinal].scan_index,
                    poses[closure.to_ordinal].scan_index,
                )
            )
        edges.append(
            CorridorGraphEdge(
                from_segment=from_segment,
                to_segment=to_segment,
                edge_type="loop_closure_transition",
                relative_transform=observed_transform,
                confidence_score=base_confidence,
                supporting_scan_indices=tuple(supporting_scans),
                support_count=support_count,
                status=status,
            )
        )

    ambiguous_edges_count = sum(1 for edge in edges if edge.status == "LOW_CONFIDENCE")
    if not segments:
        inconsistencies.append("no corridor segments")

    return CorridorGraph(
        segments=tuple(segments),
        edges=tuple(edges),
        pose_to_segment=pose_to_segment,
        ambiguous_edges_count=ambiguous_edges_count,
        conflicting_loop_closures=conflicting_loop_closures,
        topological_inconsistencies=tuple(inconsistencies),
    )


def segment_center(segment: CorridorSegment) -> tuple[float, float]:
    return (
        (segment.start_pose[0] + segment.end_pose[0]) * 0.5,
        (segment.start_pose[1] + segment.end_pose[1]) * 0.5,
    )


def optimize_corridor_graph_layout(
    graph: CorridorGraph,
    config: MatchQualityConfig,
) -> dict[int, tuple[float, float]]:
    positions = {segment.id: segment_center(segment) for segment in graph.segments}
    if not positions:
        return positions

    for _iteration in range(config.corridor_graph_opt_iterations):
        updates = {segment_id: [0.0, 0.0, 0] for segment_id in positions}
        for edge in graph.edges:
            if edge.status == "CONFLICT":
                continue
            if edge.from_segment not in positions or edge.to_segment not in positions:
                continue
            from_position = positions[edge.from_segment]
            to_position = positions[edge.to_segment]
            expected_dx = edge.relative_transform[0]
            expected_dy = edge.relative_transform[1]
            residual_x = (from_position[0] + expected_dx) - to_position[0]
            residual_y = (from_position[1] + expected_dy) - to_position[1]
            weight = edge.confidence_score
            if edge.edge_type == "sequential_transition":
                weight *= 0.4
            elif edge.edge_type == "ambiguity_transition":
                weight *= 0.2
            updates[edge.from_segment][0] -= residual_x * weight
            updates[edge.from_segment][1] -= residual_y * weight
            updates[edge.from_segment][2] += 1
            updates[edge.to_segment][0] += residual_x * weight
            updates[edge.to_segment][1] += residual_y * weight
            updates[edge.to_segment][2] += 1
        for segment_id, (update_x, update_y, count) in updates.items():
            if count <= 0:
                continue
            position = positions[segment_id]
            positions[segment_id] = (
                position[0] + config.corridor_graph_opt_step * update_x / count,
                position[1] + config.corridor_graph_opt_step * update_y / count,
            )
    return positions


def write_corridor_graph_json(
    graph: CorridorGraph,
    optimized_positions: dict[int, tuple[float, float]],
    output_path: Path,
) -> None:
    loop_edges_by_segment: dict[int, list[dict]] = {
        segment.id: [] for segment in graph.segments
    }
    for edge in graph.edges:
        if edge.edge_type != "loop_closure_transition":
            continue
        edge_summary = {
            "other_segment": edge.to_segment,
            "confidence_score": edge.confidence_score,
            "support_count": edge.support_count,
            "status": edge.status,
            "supporting_scan_indices": edge.supporting_scan_indices,
        }
        loop_edges_by_segment.setdefault(edge.from_segment, []).append(edge_summary)
        reverse_summary = dict(edge_summary)
        reverse_summary["other_segment"] = edge.from_segment
        loop_edges_by_segment.setdefault(edge.to_segment, []).append(reverse_summary)

    segment_payloads = []
    for segment in graph.segments:
        payload = asdict(segment)
        payload["optimized_center"] = optimized_positions.get(segment.id)
        payload["loop_closure_edges"] = loop_edges_by_segment.get(segment.id, [])
        segment_payloads.append(payload)

    output_payload = {
        "summary": {
            "segment_count": len(graph.segments),
            "edge_count": len(graph.edges),
            "ambiguous_edges_count": graph.ambiguous_edges_count,
            "conflicting_loop_closures": graph.conflicting_loop_closures,
            "topological_inconsistency_count": len(graph.topological_inconsistencies),
        },
        "segments": segment_payloads,
        "edges": [asdict(edge) for edge in graph.edges],
        "pose_to_segment": graph.pose_to_segment,
        "topological_inconsistencies": graph.topological_inconsistencies,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(output_payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def write_corridor_graph_plot(
    graph: CorridorGraph,
    optimized_positions: dict[int, tuple[float, float]],
    output_path: Path,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write corridor graph plots") from exc

    width = 1200
    height = 900
    left = 70
    right = 260
    top = 64
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    axis_color = (64, 64, 64)
    grid_color = (225, 225, 225)
    node_color = (126, 34, 206)
    loop_color = (37, 99, 235)
    ambiguity_color = (220, 38, 38)
    optimized_color = (34, 197, 94)

    points = []
    for segment in graph.segments:
        points.extend(
            [
                (segment.start_pose[0], segment.start_pose[1]),
                (segment.end_pose[0], segment.end_pose[1]),
                segment_center(segment),
            ]
        )
    points.extend(optimized_positions.values())
    if points:
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
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
    scale = min(plot_width / max(max_x - min_x, 1.0), plot_height / max(max_y - min_y, 1.0))

    def project(point: tuple[float, float]) -> tuple[float, float]:
        return (
            left + (point[0] - min_x) * scale,
            top + plot_height - (point[1] - min_y) * scale,
        )

    for step in range(6):
        x = left + round(plot_width * step / 5)
        y = top + round(plot_height * step / 5)
        draw.line([(x, top), (x, top + plot_height)], fill=grid_color)
        draw.line([(left, y), (left + plot_width, y)], fill=grid_color)
    draw.rectangle((left, top, left + plot_width, top + plot_height), outline=axis_color)
    draw.text((left, 18), "Corridor segment graph", fill=axis_color)
    draw.text(
        (left, 40),
        (
            f"{len(graph.segments)} segments | {len(graph.edges)} edges | "
            f"{graph.ambiguous_edges_count} ambiguous | "
            f"{graph.conflicting_loop_closures} conflicts"
        ),
        fill=axis_color,
    )

    segment_by_id = {segment.id: segment for segment in graph.segments}
    for edge in graph.edges:
        first = segment_by_id.get(edge.from_segment)
        second = segment_by_id.get(edge.to_segment)
        if first is None or second is None:
            continue
        color = loop_color
        width_px = 2
        if edge.edge_type == "sequential_transition":
            color = (148, 163, 184)
            width_px = 1
        elif edge.edge_type == "ambiguity_transition" or edge.status == "LOW_CONFIDENCE":
            color = ambiguity_color
            width_px = 2
        elif edge.status == "CONFLICT":
            color = (127, 29, 29)
            width_px = 3
        draw.line(
            [project(segment_center(first)), project(segment_center(second))],
            fill=color,
            width=width_px,
        )

    for segment in graph.segments:
        start = project((segment.start_pose[0], segment.start_pose[1]))
        end = project((segment.end_pose[0], segment.end_pose[1]))
        center = project(segment_center(segment))
        draw.line([start, end], fill=node_color, width=3)
        draw.ellipse((center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5), fill=node_color)
        draw.text((center[0] + 6, center[1] - 6), str(segment.id), fill=axis_color)
        optimized = optimized_positions.get(segment.id)
        if optimized is not None:
            opt = project(optimized)
            draw.ellipse((opt[0] - 4, opt[1] - 4, opt[0] + 4, opt[1] + 4), fill=optimized_color)
            draw.line([center, opt], fill=optimized_color, width=1)

    legend_x = left + plot_width + 34
    legend_y = top
    draw.line([(legend_x, legend_y), (legend_x + 34, legend_y)], fill=node_color, width=3)
    draw.text((legend_x + 44, legend_y - 7), "segment", fill=axis_color)
    draw.line([(legend_x, legend_y + 28), (legend_x + 34, legend_y + 28)], fill=loop_color, width=2)
    draw.text((legend_x + 44, legend_y + 21), "loop edge", fill=axis_color)
    draw.line([(legend_x, legend_y + 56), (legend_x + 34, legend_y + 56)], fill=ambiguity_color, width=2)
    draw.text((legend_x + 44, legend_y + 49), "ambiguous", fill=axis_color)
    draw.ellipse((legend_x, legend_y + 82, legend_x + 10, legend_y + 92), fill=optimized_color)
    draw.text((legend_x + 44, legend_y + 78), "optimized node", fill=axis_color)
    draw.text((legend_x, legend_y + 124), f"nodes {len(graph.segments)}", fill=axis_color)
    draw.text((legend_x, legend_y + 148), f"edges {len(graph.edges)}", fill=axis_color)
    draw.text((legend_x, legend_y + 172), f"ambiguous {graph.ambiguous_edges_count}", fill=axis_color)
    draw.text((legend_x, legend_y + 196), f"conflicts {graph.conflicting_loop_closures}", fill=axis_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_corridor_consistency_heatmap(
    graph: CorridorGraph,
    output_path: Path,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write corridor graph heatmaps") from exc

    size = max(len(graph.segments), 1)
    cell_px = max(4, min(18, 720 // size))
    left = 72
    top = 72
    right = 260
    bottom = 72
    heatmap_size = size * cell_px
    image = Image.new("RGB", (left + heatmap_size + right, top + heatmap_size + bottom), "white")
    draw = ImageDraw.Draw(image)
    axis_color = (64, 64, 64)
    low_color = (240, 249, 255)
    high_color = (220, 38, 38)
    values = [[0.0 for _ in range(size)] for _ in range(size)]
    segment_by_id = {segment.id: segment for segment in graph.segments}
    for edge in graph.edges:
        if edge.from_segment not in segment_by_id or edge.to_segment not in segment_by_id:
            continue
        first = segment_by_id[edge.from_segment]
        second = segment_by_id[edge.to_segment]
        pose_transform = segment_edge_transform(first.end_pose, second.start_pose)
        residual_m = math.hypot(
            edge.relative_transform[0] - pose_transform[0],
            edge.relative_transform[1] - pose_transform[1],
        )
        residual_theta = abs(normalize_angle_rad(edge.relative_transform[2] - pose_transform[2]))
        value = min(residual_m / 3.0 + residual_theta / math.pi, 1.0)
        values[edge.from_segment][edge.to_segment] = max(values[edge.from_segment][edge.to_segment], value)
        values[edge.to_segment][edge.from_segment] = max(values[edge.to_segment][edge.from_segment], value)

    for row in range(size):
        for column in range(size):
            value = values[row][column]
            color = interpolate_color(low_color, high_color, value)
            x = left + column * cell_px
            y = top + row * cell_px
            draw.rectangle((x, y, x + cell_px, y + cell_px), fill=color)
    draw.rectangle((left, top, left + heatmap_size, top + heatmap_size), outline=axis_color)
    draw.text((left, 22), "Corridor graph consistency heatmap", fill=axis_color)
    draw.text(
        (left, 44),
        "darker cells indicate pose-graph vs corridor-edge transform mismatch",
        fill=axis_color,
    )
    legend_x = left + heatmap_size + 34
    legend_y = top
    for offset in range(max(heatmap_size, 1)):
        ratio = 1.0 - offset / max(heatmap_size - 1, 1)
        color = interpolate_color(low_color, high_color, ratio)
        draw.line([(legend_x, legend_y + offset), (legend_x + 24, legend_y + offset)], fill=color)
    draw.rectangle((legend_x, legend_y, legend_x + 24, legend_y + heatmap_size), outline=axis_color)
    draw.text((legend_x + 34, legend_y - 7), "high", fill=axis_color)
    draw.text((legend_x + 34, legend_y + heatmap_size - 10), "low", fill=axis_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_corridor_graph_outputs(
    poses: list[Pose2D],
    closures: list[LoopConstraint],
    args: argparse.Namespace,
    match_quality: MatchQualityConfig,
    label: str,
) -> CorridorGraph:
    graph = build_corridor_graph(
        poses=poses,
        closures=closures,
        config=match_quality,
    )
    optimized_positions = optimize_corridor_graph_layout(graph, match_quality)
    json_path = args.out_dir / f"corridor_graph_{label}.json"
    graph_path = args.out_dir / f"corridor_graph_{label}.png"
    heatmap_path = args.out_dir / f"corridor_graph_consistency_{label}.png"
    write_corridor_graph_json(graph, optimized_positions, json_path)
    write_corridor_graph_plot(graph, optimized_positions, graph_path)
    write_corridor_consistency_heatmap(graph, heatmap_path)
    print(
        "corridor graph: "
        f"{len(graph.segments)} nodes, {len(graph.edges)} edges, "
        f"{graph.ambiguous_edges_count} ambiguous, "
        f"{graph.conflicting_loop_closures} conflicts"
    )
    print(f"wrote {json_path}")
    print(f"wrote {graph_path}")
    print(f"wrote {heatmap_path}")
    return graph


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


def build_match_quality_config(args: argparse.Namespace) -> MatchQualityConfig:
    if args.confidence_threshold < 0.0:
        raise ValueError("--confidence-threshold must be non-negative")
    if args.score_gap_threshold < 0.0:
        raise ValueError("--score-gap-threshold must be non-negative")
    if args.near_best_score_percent < 0.0:
        raise ValueError("--near-best-score-percent must be non-negative")
    if args.max_near_best_candidates < 1:
        raise ValueError("--max-near-best-candidates must be at least 1")
    if args.covariance_scaling <= 0.0:
        raise ValueError("--covariance-scaling must be greater than 0")
    if args.hessian_conditioning_threshold <= 1.0:
        raise ValueError("--hessian-conditioning-threshold must be greater than 1")
    if args.hessian_min_curvature <= 0.0:
        raise ValueError("--hessian-min-curvature must be greater than 0")
    if args.expanded_search_xy_m < 0.0:
        raise ValueError("--expanded-search-xy-m must be non-negative")
    if args.expanded_search_theta_deg < 0.0:
        raise ValueError("--expanded-search-theta-deg must be non-negative")
    if args.linear_feature_weight < 0.0:
        raise ValueError("--linear-feature-weight must be non-negative")
    if args.salient_feature_weight < 0.0:
        raise ValueError("--salient-feature-weight must be non-negative")
    if args.saliency_neighbor_count < 1:
        raise ValueError("--saliency-neighbor-count must be at least 1")
    if args.saliency_linear_curvature_deg < 0.0:
        raise ValueError("--saliency-linear-curvature-deg must be non-negative")
    if args.saliency_corner_curvature_deg < args.saliency_linear_curvature_deg:
        raise ValueError(
            "--saliency-corner-curvature-deg must be at least "
            "--saliency-linear-curvature-deg"
        )
    if args.saliency_gap_m <= 0.0:
        raise ValueError("--saliency-gap-m must be greater than 0")
    if args.aisle_aligned_feature_weight < 0.0:
        raise ValueError("--aisle-aligned-feature-weight must be non-negative")
    if args.aisle_cross_feature_weight < 0.0:
        raise ValueError("--aisle-cross-feature-weight must be non-negative")
    if args.aisle_yaw_lambda < 0.0:
        raise ValueError("--aisle-yaw-lambda must be non-negative")
    if not 0.0 < args.aisle_theta_window_scale <= 1.0:
        raise ValueError("--aisle-theta-window-scale must be in (0, 1]")
    if args.yaw_bias_correction_alpha < 0.0:
        raise ValueError("--yaw-bias-correction-alpha must be non-negative")
    if args.yaw_bias_top_score_percent < 0.0:
        raise ValueError("--yaw-bias-top-score-percent must be non-negative")
    if args.aisle_persistence_frames < 1:
        raise ValueError("--aisle-persistence-frames must be at least 1")
    if args.aisle_recovery_xy_m < 0.0:
        raise ValueError("--aisle-recovery-xy-m must be non-negative")
    if args.aisle_recovery_theta_deg < 0.0:
        raise ValueError("--aisle-recovery-theta-deg must be non-negative")
    if args.hypothesis_count < 1 or args.hypothesis_count > 5:
        raise ValueError("--hypothesis-count must be between 1 and 5")
    if args.hypothesis_weight_threshold < 0.0:
        raise ValueError("--hypothesis-weight-threshold must be non-negative")
    if args.hypothesis_ambiguous_weight_threshold < 0.0:
        raise ValueError("--hypothesis-ambiguous-weight-threshold must be non-negative")
    if args.hypothesis_score_temperature <= 0.0:
        raise ValueError("--hypothesis-score-temperature must be greater than 0")
    if args.hypothesis_ambiguous_temperature_scale < 1.0:
        raise ValueError("--hypothesis-ambiguous-temperature-scale must be at least 1")
    if args.hypothesis_branch_score_percent < 0.0:
        raise ValueError("--hypothesis-branch-score-percent must be non-negative")
    if args.hypothesis_close_score_gap < 0.0:
        raise ValueError("--hypothesis-close-score-gap must be non-negative")
    if args.hypothesis_near_best_count_threshold < 1:
        raise ValueError("--hypothesis-near-best-count-threshold must be at least 1")
    if args.hypothesis_divergence_threshold_m < 0.0:
        raise ValueError("--hypothesis-divergence-threshold-m must be non-negative")
    if args.hypothesis_theta_distance_weight_m < 0.0:
        raise ValueError("--hypothesis-theta-distance-weight-m must be non-negative")
    if args.hypothesis_ambiguity_covariance_scale < 1.0:
        raise ValueError("--hypothesis-ambiguity-covariance-scale must be at least 1")
    if args.loop_hypothesis_agreement_m < 0.0:
        raise ValueError("--loop-hypothesis-agreement-m must be non-negative")
    if args.loop_hypothesis_agreement_theta_deg < 0.0:
        raise ValueError("--loop-hypothesis-agreement-theta-deg must be non-negative")
    if args.loop_strong_score_gap < 0.0:
        raise ValueError("--loop-strong-score-gap must be non-negative")
    if args.anisotropic_min_confidence < 0.0:
        raise ValueError("--anisotropic-min-confidence must be non-negative")
    if args.anisotropic_aisle_linearity_threshold < 1.0:
        raise ValueError("--anisotropic-aisle-linearity-threshold must be at least 1")
    if args.anisotropic_along_sigma_m <= 0.0:
        raise ValueError("--anisotropic-along-sigma-m must be greater than 0")
    if args.anisotropic_cross_sigma_m <= 0.0:
        raise ValueError("--anisotropic-cross-sigma-m must be greater than 0")
    if args.anisotropic_theta_sigma_deg <= 0.0:
        raise ValueError("--anisotropic-theta-sigma-deg must be greater than 0")
    if args.anisotropic_scan_covariance_floor_m <= 0.0:
        raise ValueError("--anisotropic-scan-covariance-floor-m must be greater than 0")
    if not 0.0 <= args.anisotropic_drift_gain <= 1.0:
        raise ValueError("--anisotropic-drift-gain must be in [0, 1]")
    if not 0.0 <= args.corridor_axis_alpha <= 1.0:
        raise ValueError("--corridor-axis-alpha must be in [0, 1]")
    if not 0.0 <= args.corridor_ema_beta <= 1.0:
        raise ValueError("--corridor-ema-beta must be in [0, 1]")
    if args.corridor_update_min_confidence < 0.0:
        raise ValueError("--corridor-update-min-confidence must be non-negative")
    if args.corridor_update_min_linearity < 1.0:
        raise ValueError("--corridor-update-min-linearity must be at least 1")
    if not 0.0 <= args.corridor_max_hypothesis_entropy <= 1.0:
        raise ValueError("--corridor-max-hypothesis-entropy must be in [0, 1]")
    if args.corridor_reset_axis_deg < 0.0:
        raise ValueError("--corridor-reset-axis-deg must be non-negative")
    if args.corridor_reset_position_gap_m < 0.0:
        raise ValueError("--corridor-reset-position-gap-m must be non-negative")
    if args.corridor_reset_low_confidence_frames < 1:
        raise ValueError("--corridor-reset-low-confidence-frames must be at least 1")
    if args.corridor_segment_axis_threshold_deg < 0.0:
        raise ValueError("--corridor-segment-axis-threshold-deg must be non-negative")
    if not 0.0 <= args.corridor_segment_entropy_threshold <= 1.0:
        raise ValueError("--corridor-segment-entropy-threshold must be in [0, 1]")
    if args.corridor_segment_confidence_jump < 0.0:
        raise ValueError("--corridor-segment-confidence-jump must be non-negative")
    if args.corridor_segment_drift_reset_m < 0.0:
        raise ValueError("--corridor-segment-drift-reset-m must be non-negative")
    if args.corridor_segment_min_poses < 1:
        raise ValueError("--corridor-segment-min-poses must be at least 1")
    if args.corridor_edge_support_required < 1:
        raise ValueError("--corridor-edge-support-required must be at least 1")
    if not 0.0 <= args.corridor_similarity_low_confidence <= 1.0:
        raise ValueError("--corridor-similarity-low-confidence must be in [0, 1]")
    if args.corridor_graph_conflict_distance_m < 0.0:
        raise ValueError("--corridor-graph-conflict-distance-m must be non-negative")
    if args.corridor_graph_conflict_theta_deg < 0.0:
        raise ValueError("--corridor-graph-conflict-theta-deg must be non-negative")
    if args.corridor_graph_opt_iterations < 0:
        raise ValueError("--corridor-graph-opt-iterations must be non-negative")
    if args.corridor_graph_opt_step < 0.0:
        raise ValueError("--corridor-graph-opt-step must be non-negative")

    return MatchQualityConfig(
        confidence_threshold=args.confidence_threshold,
        score_gap_threshold=args.score_gap_threshold,
        near_best_score_percent=args.near_best_score_percent,
        max_near_best_candidates=args.max_near_best_candidates,
        covariance_scaling=args.covariance_scaling,
        hessian_conditioning_threshold=args.hessian_conditioning_threshold,
        hessian_min_curvature=args.hessian_min_curvature,
        expanded_search_xy_m=args.expanded_search_xy_m,
        expanded_search_theta_deg=args.expanded_search_theta_deg,
        feature_weighting_enabled=args.feature_weighting,
        linear_feature_weight=args.linear_feature_weight,
        salient_feature_weight=args.salient_feature_weight,
        saliency_neighbor_count=args.saliency_neighbor_count,
        saliency_linear_curvature_deg=args.saliency_linear_curvature_deg,
        saliency_corner_curvature_deg=args.saliency_corner_curvature_deg,
        saliency_gap_m=args.saliency_gap_m,
        degeneracy_scoring_enabled=args.degeneracy_scoring,
        aisle_aligned_feature_weight=args.aisle_aligned_feature_weight,
        aisle_cross_feature_weight=args.aisle_cross_feature_weight,
        aisle_yaw_lambda=args.aisle_yaw_lambda,
        aisle_theta_window_scale=args.aisle_theta_window_scale,
        yaw_bias_correction_alpha=args.yaw_bias_correction_alpha,
        yaw_bias_top_score_percent=args.yaw_bias_top_score_percent,
        aisle_persistence_frames=args.aisle_persistence_frames,
        aisle_recovery_xy_m=args.aisle_recovery_xy_m,
        aisle_recovery_theta_deg=args.aisle_recovery_theta_deg,
        hypothesis_count=args.hypothesis_count,
        hypothesis_weight_threshold=args.hypothesis_weight_threshold,
        hypothesis_ambiguous_weight_threshold=args.hypothesis_ambiguous_weight_threshold,
        hypothesis_score_temperature=args.hypothesis_score_temperature,
        hypothesis_ambiguous_temperature_scale=args.hypothesis_ambiguous_temperature_scale,
        hypothesis_branch_score_percent=args.hypothesis_branch_score_percent,
        hypothesis_close_score_gap=args.hypothesis_close_score_gap,
        hypothesis_near_best_count_threshold=args.hypothesis_near_best_count_threshold,
        hypothesis_divergence_threshold_m=args.hypothesis_divergence_threshold_m,
        hypothesis_theta_distance_weight_m=args.hypothesis_theta_distance_weight_m,
        hypothesis_ambiguity_covariance_scale=args.hypothesis_ambiguity_covariance_scale,
        loop_hypothesis_agreement_m=args.loop_hypothesis_agreement_m,
        loop_hypothesis_agreement_theta_deg=args.loop_hypothesis_agreement_theta_deg,
        loop_strong_score_gap=args.loop_strong_score_gap,
        anisotropic_constraints_enabled=args.anisotropic_constraints,
        anisotropic_min_confidence=args.anisotropic_min_confidence,
        anisotropic_aisle_linearity_threshold=args.anisotropic_aisle_linearity_threshold,
        anisotropic_along_sigma_m=args.anisotropic_along_sigma_m,
        anisotropic_cross_sigma_m=args.anisotropic_cross_sigma_m,
        anisotropic_theta_sigma_deg=args.anisotropic_theta_sigma_deg,
        anisotropic_scan_covariance_floor_m=args.anisotropic_scan_covariance_floor_m,
        anisotropic_drift_gain=args.anisotropic_drift_gain,
        corridor_frame_enabled=args.corridor_frame,
        corridor_axis_alpha=args.corridor_axis_alpha,
        corridor_ema_beta=args.corridor_ema_beta,
        corridor_update_min_confidence=args.corridor_update_min_confidence,
        corridor_update_min_linearity=args.corridor_update_min_linearity,
        corridor_max_hypothesis_entropy=args.corridor_max_hypothesis_entropy,
        corridor_reset_axis_deg=args.corridor_reset_axis_deg,
        corridor_reset_position_gap_m=args.corridor_reset_position_gap_m,
        corridor_reset_low_confidence_frames=args.corridor_reset_low_confidence_frames,
        corridor_graph_enabled=args.corridor_graph,
        corridor_segment_axis_threshold_deg=args.corridor_segment_axis_threshold_deg,
        corridor_segment_entropy_threshold=args.corridor_segment_entropy_threshold,
        corridor_segment_confidence_jump=args.corridor_segment_confidence_jump,
        corridor_segment_drift_reset_m=args.corridor_segment_drift_reset_m,
        corridor_segment_min_poses=args.corridor_segment_min_poses,
        corridor_edge_support_required=args.corridor_edge_support_required,
        corridor_similarity_low_confidence=args.corridor_similarity_low_confidence,
        corridor_graph_conflict_distance_m=args.corridor_graph_conflict_distance_m,
        corridor_graph_conflict_theta_deg=args.corridor_graph_conflict_theta_deg,
        corridor_graph_opt_iterations=args.corridor_graph_opt_iterations,
        corridor_graph_opt_step=args.corridor_graph_opt_step,
    )


def maybe_write_match_diagnostics(
    result: OdometryResult,
    args: argparse.Namespace,
    label: str,
) -> None:
    if not args.debug_matching:
        return

    output_path = args.out_dir / f"match_diagnostics_{label}.jsonl"
    write_match_diagnostics(result.diagnostics, output_path)
    print_match_diagnostics_summary(result.diagnostics, output_path)
    aisle_plot_path = args.out_dir / f"aisle_frame_diagnostics_{label}.png"
    write_aisle_frame_diagnostics_plot(
        diagnostics=result.diagnostics,
        output_path=aisle_plot_path,
        drift_gain=args.anisotropic_drift_gain,
    )
    print(f"wrote {aisle_plot_path}")


def maybe_write_loop_closure_diagnostics(
    diagnostics: list[LoopClosureDiagnostics],
    args: argparse.Namespace,
    label: str,
    poses: list[Pose2D] | None = None,
) -> None:
    output_path = args.out_dir / f"loop_closure_diagnostics_{label}.jsonl"
    write_loop_closure_diagnostics(diagnostics, output_path)
    print_loop_closure_diagnostics_summary(diagnostics, output_path)
    if poses is not None:
        write_loop_closure_analysis_outputs(diagnostics, poses, args, label)


def run_stage6(args: argparse.Namespace) -> None:
    if args.stride <= 0:
        raise ValueError("--stride must be greater than 0")
    if args.max_points <= 0:
        raise ValueError("--max-points must be greater than 0")
    if args.search_step_m <= 0.0:
        raise ValueError("--search-step-m must be greater than 0")
    if args.search_theta_step_deg <= 0.0:
        raise ValueError("--search-theta-step-deg must be greater than 0")
    if args.heading_weight < 0.0:
        raise ValueError("--heading-weight must be non-negative")
    if args.heading_min_confidence < 0.0:
        raise ValueError("--heading-min-confidence must be non-negative")

    match_quality = build_match_quality_config(args)
    limits = parse_ints(args.limits)
    for limit in limits:
        result = estimate_odometry_detailed(
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
            heading_weight=args.heading_weight,
            heading_min_confidence=args.heading_min_confidence,
            max_step_m=args.max_step_m,
            match_quality=match_quality,
            score_heatmap_dir=(
                args.out_dir / f"score_heatmaps_{limit}"
                if args.debug_score_heatmaps
                else None
            ),
        )
        poses = result.poses
        output_path = args.out_dir / f"05_odom_first_{limit}.png"
        write_odometry_preview(
            poses=poses,
            output_path=output_path,
            limit=limit,
            stride=args.stride,
        )
        maybe_write_match_diagnostics(result, args, f"stage6_{limit}")
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
    if args.heading_weight < 0.0:
        raise ValueError("--heading-weight must be non-negative")
    if args.heading_min_confidence < 0.0:
        raise ValueError("--heading-min-confidence must be non-negative")

    debug_keyframes = parse_ints(args.keyframes)
    if not debug_keyframes:
        raise ValueError("--keyframes must include at least one keyframe ordinal")

    match_quality = build_match_quality_config(args)
    limit = args.limit or ((max(debug_keyframes) + 1) * args.stride + 1)
    result = estimate_odometry_detailed(
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
        heading_weight=args.heading_weight,
        heading_min_confidence=args.heading_min_confidence,
        max_step_m=args.max_step_m,
        debug_keyframe_ordinals=set(debug_keyframes),
        match_quality=match_quality,
        score_heatmap_dir=(
            args.out_dir / "score_heatmaps_debug"
            if args.debug_score_heatmaps
            else None
        ),
    )
    debug_matches = result.debug_matches

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
    maybe_write_match_diagnostics(result, args, "stage7")


def run_stage8(args: argparse.Namespace) -> None:
    if args.stride <= 0:
        raise ValueError("--stride must be greater than 0")
    if args.smoothing_window <= 0:
        raise ValueError("--smoothing-window must be greater than 0")
    if args.bin_count <= 0:
        raise ValueError("--bin-count must be greater than 0")

    samples = compute_heading_samples(
        lidar_path=args.lidar,
        stride=args.stride,
        min_valid_points=args.min_valid_points,
        bin_count=args.bin_count,
        smoothing_window=args.smoothing_window,
        min_segment_m=args.min_segment_m,
        max_segment_m=args.max_segment_m,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
    )
    output_path = args.out_dir / "06_heading_over_time.png"
    write_heading_plot(
        samples=samples,
        output_path=output_path,
        stride=args.stride,
        turn_threshold_deg=args.turn_threshold_deg,
    )
    print(f"computed {len(samples)} heading samples")
    print(f"wrote {output_path}")


def run_stage9(args: argparse.Namespace) -> None:
    if args.stride <= 0:
        raise ValueError("--stride must be greater than 0")
    if args.max_points <= 0:
        raise ValueError("--max-points must be greater than 0")
    if args.search_step_m <= 0.0:
        raise ValueError("--search-step-m must be greater than 0")
    if args.search_theta_step_deg <= 0.0:
        raise ValueError("--search-theta-step-deg must be greater than 0")
    if args.heading_weight < 0.0:
        raise ValueError("--heading-weight must be non-negative")
    if args.heading_min_confidence < 0.0:
        raise ValueError("--heading-min-confidence must be non-negative")

    match_quality = build_match_quality_config(args)
    limit = args.limit or count_scans(args.lidar)
    result = estimate_odometry_detailed(
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
        heading_weight=args.heading_weight,
        heading_min_confidence=args.heading_min_confidence,
        max_step_m=args.max_step_m,
        match_quality=match_quality,
        score_heatmap_dir=(
            args.out_dir / "score_heatmaps_full"
            if args.debug_score_heatmaps
            else None
        ),
    )
    poses = result.poses
    output_path = args.out_dir / "07_full_odometry_raw.png"
    write_odometry_preview(
        poses=poses,
        output_path=output_path,
        limit=limit,
        stride=args.stride,
        title="Stage 9 full raw odometry trajectory",
        subtitle=(
            f"{len(poses)} keyframe poses from first {limit} scans | "
            f"stride {args.stride} | no loop closure"
        ),
    )
    maybe_write_match_diagnostics(result, args, "stage9")
    print(f"estimated {len(poses)} poses for first {limit} scans")
    print(f"wrote {output_path}")


def run_stage10(args: argparse.Namespace) -> None:
    if args.stride <= 0:
        raise ValueError("--stride must be greater than 0")
    if args.max_points <= 0:
        raise ValueError("--max-points must be greater than 0")
    if args.search_step_m <= 0.0:
        raise ValueError("--search-step-m must be greater than 0")
    if args.search_theta_step_deg <= 0.0:
        raise ValueError("--search-theta-step-deg must be greater than 0")
    if args.heading_weight < 0.0:
        raise ValueError("--heading-weight must be non-negative")
    if args.heading_min_confidence < 0.0:
        raise ValueError("--heading-min-confidence must be non-negative")
    if args.loop_weight <= 0.0:
        raise ValueError("--loop-weight must be greater than 0")
    if args.odometry_weight <= 0.0:
        raise ValueError("--odometry-weight must be greater than 0")
    if args.anchor_weight <= 0.0:
        raise ValueError("--anchor-weight must be greater than 0")
    if args.aisle_forward_uncertainty_scale <= 0.0:
        raise ValueError("--aisle-forward-uncertainty-scale must be greater than 0")
    if args.loop_match_search_step_m <= 0.0:
        raise ValueError("--loop-match-search-step-m must be greater than 0")
    if args.loop_match_search_theta_step_deg <= 0.0:
        raise ValueError("--loop-match-search-theta-step-deg must be greater than 0")
    if args.min_loop_match_score < 0.0:
        raise ValueError("--min-loop-match-score must be non-negative")
    if args.min_loop_match_score_gap < 0.0:
        raise ValueError("--min-loop-match-score-gap must be non-negative")
    if args.replay_loop_closure is not None and args.replay_loop_closure < 0:
        raise ValueError("--replay-loop-closure must be non-negative")

    match_quality = build_match_quality_config(args)
    limit = args.limit or count_scans(args.lidar)
    result = estimate_odometry_detailed(
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
        heading_weight=args.heading_weight,
        heading_min_confidence=args.heading_min_confidence,
        max_step_m=args.max_step_m,
        match_quality=match_quality,
        score_heatmap_dir=(
            args.out_dir / "score_heatmaps_loop"
            if args.debug_score_heatmaps
            else None
        ),
    )
    poses = result.poses
    candidate_closures = find_loop_closures(
        lidar_path=args.lidar,
        poses=poses,
        signature_bins=args.signature_bins,
        signature_max_range_m=args.signature_max_range_m,
        min_separation=args.min_loop_separation,
        max_raw_distance_m=args.max_loop_distance_m,
        max_signature_distance=args.max_signature_distance,
        min_overlap_bins=args.min_overlap_bins,
        max_closures=args.max_closures,
    )
    if args.replay_loop_closure is not None:
        replay_loop_closure_candidate(
            args=args,
            poses=poses,
            candidate_closures=candidate_closures,
            match_quality=match_quality,
            label="stage10",
        )
        return
    loop_diagnostics: list[LoopClosureDiagnostics] = []
    closures = build_loop_constraints(
        lidar_path=args.lidar,
        poses=poses,
        closures=candidate_closures,
        max_points=args.max_points,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
        cell_size_m=args.cell_size_m,
        inflation_cells=args.inflation_cells,
        search_xy_m=args.loop_match_search_xy_m,
        search_step_m=args.loop_match_search_step_m,
        search_theta_deg=args.loop_match_search_theta_deg,
        search_theta_step_deg=args.loop_match_search_theta_step_deg,
        min_match_score=args.min_loop_match_score,
        min_score_gap=args.min_loop_match_score_gap,
        match_quality=match_quality,
        diagnostics=loop_diagnostics,
    )
    corrected_poses = solve_loop_closure_graph(
        poses=poses,
        closures=closures,
        loop_weight=args.loop_weight,
        odometry_weight=args.odometry_weight,
        anchor_weight=args.anchor_weight,
        aisle_forward_uncertainty_scale=args.aisle_forward_uncertainty_scale,
        match_quality=match_quality,
    )
    if match_quality.corridor_graph_enabled:
        write_corridor_graph_outputs(
            poses=corrected_poses,
            closures=closures,
            args=args,
            match_quality=match_quality,
            label="stage10",
        )

    before_path = args.out_dir / "08_before_loop_closure.png"
    after_path = args.out_dir / "09_after_loop_closure.png"
    subtitle = (
        f"{len(poses)} keyframe poses from first {limit} scans | "
        f"{len(closures)} accepted scan-matched loop closures"
    )
    write_odometry_preview(
        poses=poses,
        output_path=before_path,
        limit=limit,
        stride=args.stride,
        title="Stage 10 before loop closure",
        subtitle=subtitle,
    )
    write_odometry_preview(
        poses=corrected_poses,
        output_path=after_path,
        limit=limit,
        stride=args.stride,
        title="Stage 10 after GTSAM loop closure",
        subtitle=subtitle,
    )
    print(f"estimated {len(poses)} poses for first {limit} scans")
    maybe_write_match_diagnostics(result, args, "stage10")
    print(
        f"accepted {len(closures)} scan-matched loop closures "
        f"from {len(candidate_closures)} candidates"
    )
    maybe_write_loop_closure_diagnostics(loop_diagnostics, args, "stage10", poses=poses)
    if candidate_closures:
        best = min(closure.signature_distance for closure in candidate_closures)
        worst = max(closure.signature_distance for closure in candidate_closures)
        print(f"candidate signature distance range: {best:.3f} to {worst:.3f}")
    if closures:
        best_score = min(closure.match_score for closure in closures)
        worst_score = max(closure.match_score for closure in closures)
        print(f"accepted loop match score range: {best_score:.3f} to {worst_score:.3f}")
    print(f"wrote {before_path}")
    print(f"wrote {after_path}")


def run_stage11(args: argparse.Namespace) -> None:
    if args.stride <= 0:
        raise ValueError("--stride must be greater than 0")
    if args.max_points <= 0:
        raise ValueError("--max-points must be greater than 0")
    if args.search_step_m <= 0.0:
        raise ValueError("--search-step-m must be greater than 0")
    if args.search_theta_step_deg <= 0.0:
        raise ValueError("--search-theta-step-deg must be greater than 0")
    if args.heading_weight < 0.0:
        raise ValueError("--heading-weight must be non-negative")
    if args.heading_min_confidence < 0.0:
        raise ValueError("--heading-min-confidence must be non-negative")
    if args.loop_weight <= 0.0:
        raise ValueError("--loop-weight must be greater than 0")
    if args.odometry_weight <= 0.0:
        raise ValueError("--odometry-weight must be greater than 0")
    if args.anchor_weight <= 0.0:
        raise ValueError("--anchor-weight must be greater than 0")
    if args.aisle_forward_uncertainty_scale <= 0.0:
        raise ValueError("--aisle-forward-uncertainty-scale must be greater than 0")
    if args.loop_match_search_step_m <= 0.0:
        raise ValueError("--loop-match-search-step-m must be greater than 0")
    if args.loop_match_search_theta_step_deg <= 0.0:
        raise ValueError("--loop-match-search-theta-step-deg must be greater than 0")
    if args.min_loop_match_score < 0.0:
        raise ValueError("--min-loop-match-score must be non-negative")
    if args.min_loop_match_score_gap < 0.0:
        raise ValueError("--min-loop-match-score-gap must be non-negative")
    if args.replay_loop_closure is not None and args.replay_loop_closure < 0:
        raise ValueError("--replay-loop-closure must be non-negative")

    match_quality = build_match_quality_config(args)
    limit = args.limit or count_scans(args.lidar)
    result = estimate_odometry_detailed(
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
        heading_weight=args.heading_weight,
        heading_min_confidence=args.heading_min_confidence,
        max_step_m=args.max_step_m,
        match_quality=match_quality,
        score_heatmap_dir=(
            args.out_dir / "score_heatmaps_final"
            if args.debug_score_heatmaps
            else None
        ),
    )
    poses = result.poses
    candidate_closures = find_loop_closures(
        lidar_path=args.lidar,
        poses=poses,
        signature_bins=args.signature_bins,
        signature_max_range_m=args.signature_max_range_m,
        min_separation=args.min_loop_separation,
        max_raw_distance_m=args.max_loop_distance_m,
        max_signature_distance=args.max_signature_distance,
        min_overlap_bins=args.min_overlap_bins,
        max_closures=args.max_closures,
    )
    if args.replay_loop_closure is not None:
        replay_loop_closure_candidate(
            args=args,
            poses=poses,
            candidate_closures=candidate_closures,
            match_quality=match_quality,
            label="stage11",
        )
        return
    loop_diagnostics: list[LoopClosureDiagnostics] = []
    closures = build_loop_constraints(
        lidar_path=args.lidar,
        poses=poses,
        closures=candidate_closures,
        max_points=args.max_points,
        min_range_m=args.min_range_m,
        max_range_m=args.max_range_m,
        cell_size_m=args.cell_size_m,
        inflation_cells=args.inflation_cells,
        search_xy_m=args.loop_match_search_xy_m,
        search_step_m=args.loop_match_search_step_m,
        search_theta_deg=args.loop_match_search_theta_deg,
        search_theta_step_deg=args.loop_match_search_theta_step_deg,
        min_match_score=args.min_loop_match_score,
        min_score_gap=args.min_loop_match_score_gap,
        match_quality=match_quality,
        diagnostics=loop_diagnostics,
    )
    corrected_poses = solve_loop_closure_graph(
        poses=poses,
        closures=closures,
        loop_weight=args.loop_weight,
        odometry_weight=args.odometry_weight,
        anchor_weight=args.anchor_weight,
        aisle_forward_uncertainty_scale=args.aisle_forward_uncertainty_scale,
        match_quality=match_quality,
    )
    if match_quality.corridor_graph_enabled:
        write_corridor_graph_outputs(
            poses=corrected_poses,
            closures=closures,
            args=args,
            match_quality=match_quality,
            label="stage11",
        )

    output_path = args.out_dir / "10_final_trajectory.png"
    fit_score = write_final_aligned_render(
        poses=corrected_poses,
        reference_path=args.reference,
        output_path=output_path,
        background_opacity=args.background_opacity,
    )
    print(f"estimated {len(poses)} poses for first {limit} scans")
    maybe_write_match_diagnostics(result, args, "stage11")
    print(
        f"accepted {len(closures)} scan-matched loop closures "
        f"from {len(candidate_closures)} candidates"
    )
    maybe_write_loop_closure_diagnostics(loop_diagnostics, args, "stage11", poses=poses)
    print(f"alignment fit score: {fit_score:.2f}")
    print(f"wrote {output_path}")


def add_odometry_constraint_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--heading-weight",
        type=float,
        default=0.15,
        help="penalty weight for scan-structure heading disagreement",
    )
    command.add_argument(
        "--heading-min-confidence",
        type=float,
        default=0.10,
        help="minimum dominant-structure confidence used for heading constraints",
    )
    command.add_argument(
        "--max-step-m",
        type=float,
        default=1.0,
        help="maximum extrapolated keyframe translation before local search",
    )
    command.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
        help="minimum adjusted scan-match score accepted by the frontend",
    )
    command.add_argument(
        "--score-gap-threshold",
        type=float,
        default=0.0001,
        help="minimum best-vs-runner-up adjusted score gap",
    )
    command.add_argument(
        "--near-best-score-percent",
        type=float,
        default=0.5,
        help="score percentage band used to count near-best candidates",
    )
    command.add_argument(
        "--max-near-best-candidates",
        type=int,
        default=60,
        help="maximum candidates allowed inside the near-best score band",
    )
    command.add_argument(
        "--covariance-scaling",
        type=float,
        default=0.02,
        help="scale applied when converting score curvature to covariance",
    )
    command.add_argument(
        "--hessian-conditioning-threshold",
        type=float,
        default=2000.0,
        help="maximum allowed score-surface Hessian condition number",
    )
    command.add_argument(
        "--hessian-min-curvature",
        type=float,
        default=0.001,
        help="minimum accepted score-surface curvature eigenvalue",
    )
    command.add_argument(
        "--expanded-search-xy-m",
        type=float,
        default=1.6,
        help="expanded retry translation search radius in meters",
    )
    command.add_argument(
        "--expanded-search-theta-deg",
        type=float,
        default=16.0,
        help="expanded retry heading search radius in degrees",
    )
    command.add_argument(
        "--debug-matching",
        action="store_true",
        help="write JSONL scan-match confidence and covariance diagnostics",
    )
    command.add_argument(
        "--debug-score-heatmaps",
        action="store_true",
        help="write score-surface heatmaps around accepted and rejected matches",
    )
    command.add_argument(
        "--feature-weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable local saliency weighting for scan-match scoring",
    )
    command.add_argument(
        "--linear-feature-weight",
        type=float,
        default=0.35,
        help="raw score weight for long low-curvature linear scan points",
    )
    command.add_argument(
        "--salient-feature-weight",
        type=float,
        default=2.0,
        help="raw score weight for corners, endpoints, gaps, and high-curvature points",
    )
    command.add_argument(
        "--saliency-neighbor-count",
        type=int,
        default=3,
        help="neighbor span in scan order for local saliency curvature",
    )
    command.add_argument(
        "--saliency-linear-curvature-deg",
        type=float,
        default=8.0,
        help="maximum local turn angle classified as a linear feature",
    )
    command.add_argument(
        "--saliency-corner-curvature-deg",
        type=float,
        default=25.0,
        help="minimum local turn angle classified as a salient feature",
    )
    command.add_argument(
        "--saliency-gap-m",
        type=float,
        default=0.45,
        help="neighbor gap that marks scan endpoints and isolated salient features",
    )
    command.add_argument(
        "--degeneracy-scoring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable AISLE-mode degeneracy scoring adjustments",
    )
    command.add_argument(
        "--aisle-aligned-feature-weight",
        type=float,
        default=0.1,
        help="maximum normalized point weight for features aligned with an aisle axis",
    )
    command.add_argument(
        "--aisle-cross-feature-weight",
        type=float,
        default=2.5,
        help="minimum raw point weight for cross-axis salient features in AISLE mode",
    )
    command.add_argument(
        "--aisle-yaw-lambda",
        type=float,
        default=0.08,
        help="quadratic yaw-prior penalty used only in AISLE mode",
    )
    command.add_argument(
        "--aisle-theta-window-scale",
        type=float,
        default=0.65,
        help="AISLE-mode multiplier applied to the normal heading search radius",
    )
    command.add_argument(
        "--yaw-bias-correction-alpha",
        type=float,
        default=0.18,
        help="fraction of score-surface yaw bias subtracted from the winning pose",
    )
    command.add_argument(
        "--yaw-bias-top-score-percent",
        type=float,
        default=2.0,
        help="near-best score band used to estimate yaw bias in AISLE mode",
    )
    command.add_argument(
        "--aisle-persistence-frames",
        type=int,
        default=8,
        help="consecutive AISLE scans before one recovery retry is attempted",
    )
    command.add_argument(
        "--aisle-recovery-xy-m",
        type=float,
        default=0.35,
        help="extra translation radius for persistent-AISLE recovery retries",
    )
    command.add_argument(
        "--aisle-recovery-theta-deg",
        type=float,
        default=4.0,
        help="extra heading radius for persistent-AISLE recovery retries",
    )
    command.add_argument(
        "--hypothesis-count",
        type=int,
        default=3,
        help="maximum active frontend pose hypotheses; use 1 for legacy single-track behavior",
    )
    command.add_argument(
        "--hypothesis-weight-threshold",
        type=float,
        default=0.01,
        help="normal soft-prune threshold for active pose hypotheses",
    )
    command.add_argument(
        "--hypothesis-ambiguous-weight-threshold",
        type=float,
        default=0.001,
        help="softer prune threshold used while AISLE or score ambiguity is active",
    )
    command.add_argument(
        "--hypothesis-score-temperature",
        type=float,
        default=0.10,
        help="temperature for converting scan-match score to hypothesis likelihood",
    )
    command.add_argument(
        "--hypothesis-ambiguous-temperature-scale",
        type=float,
        default=2.0,
        help="likelihood temperature multiplier used for ambiguous scans",
    )
    command.add_argument(
        "--hypothesis-branch-score-percent",
        type=float,
        default=2.0,
        help="near-best score band used to keep alternate hypothesis branches",
    )
    command.add_argument(
        "--hypothesis-close-score-gap",
        type=float,
        default=0.001,
        help="normalized score gap below which a scan keeps alternate hypotheses",
    )
    command.add_argument(
        "--hypothesis-near-best-count-threshold",
        type=int,
        default=5,
        help="near-best candidate count that triggers alternate hypothesis retention",
    )
    command.add_argument(
        "--hypothesis-divergence-threshold-m",
        type=float,
        default=0.8,
        help="SE(2) hypothesis distance that marks the frontend ambiguous",
    )
    command.add_argument(
        "--hypothesis-theta-distance-weight-m",
        type=float,
        default=1.0,
        help="meters-per-radian scale used for SE(2) hypothesis divergence",
    )
    command.add_argument(
        "--hypothesis-ambiguity-covariance-scale",
        type=float,
        default=4.0,
        help="extra covariance multiplier for GTSAM factors from ambiguous hypotheses",
    )
    command.add_argument(
        "--loop-hypothesis-agreement-m",
        type=float,
        default=0.75,
        help="translation tolerance for loop-closure agreement across hypotheses",
    )
    command.add_argument(
        "--loop-hypothesis-agreement-theta-deg",
        type=float,
        default=12.0,
        help="heading tolerance for loop-closure agreement across hypotheses",
    )
    command.add_argument(
        "--loop-strong-score-gap",
        type=float,
        default=0.04,
        help="loop match score gap allowed to override ambiguous-hypothesis deferral",
    )
    command.add_argument(
        "--anisotropic-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable aisle-frame anisotropic odometry covariance and along-axis regularization",
    )
    command.add_argument(
        "--anisotropic-min-confidence",
        type=float,
        default=0.35,
        help="minimum aisle-frame confidence required for anisotropic odometry factors",
    )
    command.add_argument(
        "--anisotropic-aisle-linearity-threshold",
        type=float,
        default=3.5,
        help="minimum local PCA linearity treated as an aisle-like frame",
    )
    command.add_argument(
        "--anisotropic-along-sigma-m",
        type=float,
        default=0.08,
        help="motion-prior translation sigma along the dominant aisle axis",
    )
    command.add_argument(
        "--anisotropic-cross-sigma-m",
        type=float,
        default=0.18,
        help="maximum scan-match translation sigma across the dominant aisle axis",
    )
    command.add_argument(
        "--anisotropic-theta-sigma-deg",
        type=float,
        default=3.0,
        help="maximum heading sigma for confident aisle-frame odometry factors",
    )
    command.add_argument(
        "--anisotropic-scan-covariance-floor-m",
        type=float,
        default=0.04,
        help="minimum translation sigma when projecting scan covariance into aisle frame",
    )
    command.add_argument(
        "--anisotropic-drift-gain",
        type=float,
        default=0.35,
        help="fraction of along-aisle drift removed from odometry factor measurements",
    )
    command.add_argument(
        "--corridor-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable persistent corridor-level axis smoothing for anisotropic constraints",
    )
    command.add_argument(
        "--corridor-axis-alpha",
        type=float,
        default=0.12,
        help="small correction gain from raw scan PCA axis into corridor axis",
    )
    command.add_argument(
        "--corridor-ema-beta",
        type=float,
        default=0.08,
        help="EMA gain used for the downstream stabilized corridor axis",
    )
    command.add_argument(
        "--corridor-update-min-confidence",
        type=float,
        default=0.35,
        help="minimum raw aisle-frame confidence required to update corridor state",
    )
    command.add_argument(
        "--corridor-update-min-linearity",
        type=float,
        default=3.5,
        help="minimum raw PCA linearity required to update corridor state",
    )
    command.add_argument(
        "--corridor-max-hypothesis-entropy",
        type=float,
        default=0.35,
        help="maximum MHT weight entropy allowed when updating corridor state",
    )
    command.add_argument(
        "--corridor-reset-axis-deg",
        type=float,
        default=35.0,
        help="raw axis jump that starts a new corridor segment",
    )
    command.add_argument(
        "--corridor-reset-position-gap-m",
        type=float,
        default=3.0,
        help="pose gap that starts a new corridor segment",
    )
    command.add_argument(
        "--corridor-reset-low-confidence-frames",
        type=int,
        default=12,
        help="consecutive gated frames before an active corridor is reset",
    )
    command.add_argument(
        "--corridor-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write additive corridor-segment graph diagnostics after loop closure",
    )
    command.add_argument(
        "--corridor-segment-axis-threshold-deg",
        type=float,
        default=25.0,
        help="axis change that starts a new topological corridor segment",
    )
    command.add_argument(
        "--corridor-segment-entropy-threshold",
        type=float,
        default=0.65,
        help="MHT entropy threshold used with corridor resets for segment splitting",
    )
    command.add_argument(
        "--corridor-segment-confidence-jump",
        type=float,
        default=0.35,
        help="aisle-frame confidence jump that starts a new corridor segment",
    )
    command.add_argument(
        "--corridor-segment-drift-reset-m",
        type=float,
        default=0.8,
        help="longitudinal drift jump that starts a new corridor segment",
    )
    command.add_argument(
        "--corridor-segment-min-poses",
        type=int,
        default=2,
        help="minimum pose count before a corridor segment can be closed",
    )
    command.add_argument(
        "--corridor-edge-support-required",
        type=int,
        default=2,
        help="loop-edge observations required before a corridor edge is strong",
    )
    command.add_argument(
        "--corridor-similarity-low-confidence",
        type=float,
        default=0.35,
        help="segment signature similarity below which loop edges stay low confidence",
    )
    command.add_argument(
        "--corridor-graph-conflict-distance-m",
        type=float,
        default=2.0,
        help="corridor-edge translation residual treated as a topological conflict",
    )
    command.add_argument(
        "--corridor-graph-conflict-theta-deg",
        type=float,
        default=35.0,
        help="corridor-edge heading residual treated as a topological conflict",
    )
    command.add_argument(
        "--corridor-graph-opt-iterations",
        type=int,
        default=20,
        help="lightweight corridor graph consistency refinement iterations",
    )
    command.add_argument(
        "--corridor-graph-opt-step",
        type=float,
        default=0.15,
        help="step size for lightweight corridor graph consistency refinement",
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
            default=80,
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
            default=0.15,
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
            default=12,
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
            default=0.2,
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
            default=2.0,
            help="heading search step in degrees",
        )
        add_odometry_constraint_arguments(command)
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
            default=80,
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
            default=0.15,
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
            default=12,
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
            default=0.2,
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
            default=2.0,
            help="heading search step in degrees",
        )
        add_odometry_constraint_arguments(command)
        command.set_defaults(func=run_stage7)

    for command_name in ("stage8", "heading-plot"):
        command = subparsers.add_parser(
            command_name,
            help="generate Stage 8 heading and structure plot",
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
            help="scan stride for heading samples",
        )
        command.add_argument(
            "--min-valid-points",
            type=int,
            default=100,
            help="skip scans with fewer valid returns",
        )
        command.add_argument(
            "--bin-count",
            type=int,
            default=72,
            help="orientation histogram bins over 180 degrees",
        )
        command.add_argument(
            "--smoothing-window",
            type=int,
            default=31,
            help="rolling median window in sampled scans",
        )
        command.add_argument(
            "--min-segment-m",
            type=float,
            default=0.05,
            help="minimum adjacent-point segment length",
        )
        command.add_argument(
            "--max-segment-m",
            type=float,
            default=1.2,
            help="maximum adjacent-point segment length",
        )
        command.add_argument(
            "--min-range-m",
            type=float,
            default=0.7,
            help="minimum point range used for structure extraction",
        )
        command.add_argument(
            "--max-range-m",
            type=float,
            default=12.0,
            help="maximum point range used for structure extraction",
        )
        command.add_argument(
            "--turn-threshold-deg",
            type=float,
            default=45.0,
            help="smoothed angle jump that marks a turn",
        )
        command.set_defaults(func=run_stage8)

    for command_name in ("stage9", "full-odom"):
        command = subparsers.add_parser(
            command_name,
            help="generate Stage 9 full raw odometry trajectory",
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
            "--limit",
            type=int,
            help="scan prefix limit; defaults to the full dataset",
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
            default=80,
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
            default=0.15,
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
            default=12,
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
            default=0.2,
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
            default=2.0,
            help="heading search step in degrees",
        )
        add_odometry_constraint_arguments(command)
        command.set_defaults(func=run_stage9)

    for command_name in ("stage10", "loop-closure"):
        command = subparsers.add_parser(
            command_name,
            help="generate Stage 10 before/after loop-closure plots",
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
            "--limit",
            type=int,
            help="scan prefix limit; defaults to the full dataset",
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
            default=80,
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
            default=0.15,
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
            default=12,
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
            default=0.2,
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
            default=2.0,
            help="heading search step in degrees",
        )
        add_odometry_constraint_arguments(command)
        command.add_argument(
            "--signature-bins",
            type=int,
            default=72,
            help="range-signature bins used for loop candidate rejection",
        )
        command.add_argument(
            "--signature-max-range-m",
            type=float,
            default=12.0,
            help="maximum range encoded in loop signatures",
        )
        command.add_argument(
            "--min-loop-separation",
            type=int,
            default=80,
            help="minimum keyframe ordinal separation for loop candidates",
        )
        command.add_argument(
            "--max-loop-distance-m",
            type=float,
            default=5.0,
            help="maximum raw-pose distance for loop candidates",
        )
        command.add_argument(
            "--max-signature-distance",
            type=float,
            default=0.10,
            help="maximum scan-signature distance for accepted loops",
        )
        command.add_argument(
            "--min-overlap-bins",
            type=int,
            default=10,
            help="minimum overlapping signature bins for loop comparison",
        )
        command.add_argument(
            "--max-closures",
            type=int,
            default=40,
            help="maximum accepted loop closures",
        )
        command.add_argument(
            "--loop-weight",
            type=float,
            default=0.35,
            help="weight for loop-closure constraints",
        )
        command.add_argument(
            "--odometry-weight",
            type=float,
            default=1.0,
            help="weight for adjacent odometry constraints",
        )
        command.add_argument(
            "--anchor-weight",
            type=float,
            default=100.0,
            help="weight anchoring the first pose",
        )
        command.add_argument(
            "--aisle-forward-uncertainty-scale",
            type=float,
            default=4.0,
            help="forward odometry uncertainty multiplier for straight aisle-like motion",
        )
        command.add_argument(
            "--loop-match-search-xy-m",
            type=float,
            default=0.8,
            help="translation search radius for scan-matching loop candidates",
        )
        command.add_argument(
            "--loop-match-search-step-m",
            type=float,
            default=0.2,
            help="translation search step for scan-matching loop candidates",
        )
        command.add_argument(
            "--loop-match-search-theta-deg",
            type=float,
            default=12.0,
            help="heading search radius for scan-matching loop candidates",
        )
        command.add_argument(
            "--loop-match-search-theta-step-deg",
            type=float,
            default=3.0,
            help="heading search step for scan-matching loop candidates",
        )
        command.add_argument(
            "--min-loop-match-score",
            type=float,
            default=0.70,
            help="minimum scan-match score for accepted loop constraints",
        )
        command.add_argument(
            "--min-loop-match-score-gap",
            type=float,
            default=0.02,
            help="minimum best-vs-runner-up score gap for loop constraints",
        )
        command.add_argument(
            "--replay-loop-closure",
            type=int,
            help="rerun one loop closure candidate_id and dump isolated diagnostics",
        )
        command.set_defaults(func=run_stage10)

    for command_name in ("stage11", "final"):
        command = subparsers.add_parser(
            command_name,
            help="generate Stage 11 final aligned trajectory render",
        )
        command.add_argument(
            "--lidar",
            type=Path,
            default=DEFAULT_LIDAR_PATH,
            help=f"input NDJSON LiDAR file (default: {DEFAULT_LIDAR_PATH})",
        )
        command.add_argument(
            "--reference",
            type=Path,
            default=Path("challenge/Output_Trajectory.png"),
            help="reference image used only for final similarity alignment",
        )
        command.add_argument(
            "--out-dir",
            type=Path,
            default=DEFAULT_OUTPUT_DIR,
            help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
        )
        command.add_argument(
            "--background-opacity",
            type=float,
            default=0.35,
            help="opacity of the faded reference background",
        )
        command.add_argument(
            "--limit",
            type=int,
            help="scan prefix limit; defaults to the full dataset",
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
            default=80,
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
            default=0.15,
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
            default=12,
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
            default=0.2,
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
            default=2.0,
            help="heading search step in degrees",
        )
        add_odometry_constraint_arguments(command)
        command.add_argument(
            "--signature-bins",
            type=int,
            default=72,
            help="range-signature bins used for loop candidate rejection",
        )
        command.add_argument(
            "--signature-max-range-m",
            type=float,
            default=12.0,
            help="maximum range encoded in loop signatures",
        )
        command.add_argument(
            "--min-loop-separation",
            type=int,
            default=80,
            help="minimum keyframe ordinal separation for loop candidates",
        )
        command.add_argument(
            "--max-loop-distance-m",
            type=float,
            default=5.0,
            help="maximum raw-pose distance for loop candidates",
        )
        command.add_argument(
            "--max-signature-distance",
            type=float,
            default=0.10,
            help="maximum scan-signature distance for accepted loops",
        )
        command.add_argument(
            "--min-overlap-bins",
            type=int,
            default=10,
            help="minimum overlapping signature bins for loop comparison",
        )
        command.add_argument(
            "--max-closures",
            type=int,
            default=40,
            help="maximum accepted loop closures",
        )
        command.add_argument(
            "--loop-weight",
            type=float,
            default=0.35,
            help="weight for loop-closure constraints",
        )
        command.add_argument(
            "--odometry-weight",
            type=float,
            default=1.0,
            help="weight for adjacent odometry constraints",
        )
        command.add_argument(
            "--anchor-weight",
            type=float,
            default=100.0,
            help="weight anchoring the first pose",
        )
        command.add_argument(
            "--aisle-forward-uncertainty-scale",
            type=float,
            default=4.0,
            help="forward odometry uncertainty multiplier for straight aisle-like motion",
        )
        command.add_argument(
            "--loop-match-search-xy-m",
            type=float,
            default=0.8,
            help="translation search radius for scan-matching loop candidates",
        )
        command.add_argument(
            "--loop-match-search-step-m",
            type=float,
            default=0.2,
            help="translation search step for scan-matching loop candidates",
        )
        command.add_argument(
            "--loop-match-search-theta-deg",
            type=float,
            default=12.0,
            help="heading search radius for scan-matching loop candidates",
        )
        command.add_argument(
            "--loop-match-search-theta-step-deg",
            type=float,
            default=3.0,
            help="heading search step for scan-matching loop candidates",
        )
        command.add_argument(
            "--min-loop-match-score",
            type=float,
            default=0.70,
            help="minimum scan-match score for accepted loop constraints",
        )
        command.add_argument(
            "--min-loop-match-score-gap",
            type=float,
            default=0.02,
            help="minimum best-vs-runner-up score gap for loop constraints",
        )
        command.add_argument(
            "--replay-loop-closure",
            type=int,
            help="rerun one loop closure candidate_id and dump isolated diagnostics",
        )
        command.set_defaults(func=run_stage11)

    return parser


class SlamPipeline:
    """Thin orchestration wrapper around the existing stage entry points.

    The stage functions still contain the behavior-preserving implementation.
    This class gives the package a stable pipeline entry point without changing
    the CLI contract or algorithmic path.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> None:
        self.args.func(self.args)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    SlamPipeline(args).run()


if __name__ == "__main__":
    main()
