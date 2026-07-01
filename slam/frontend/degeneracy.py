from __future__ import annotations

import math

import numpy as np


def _normalized_entropy(histogram: np.ndarray) -> float:
    total = float(histogram.sum())
    if total <= 0.0 or len(histogram) <= 1:
        return 0.0

    probabilities = histogram.astype(float) / total
    probabilities = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return min(max(entropy / math.log(len(histogram)), 0.0), 1.0)


def _line_angle_delta(first: np.ndarray, second: float) -> np.ndarray:
    return (first - second + math.pi / 2.0) % math.pi - math.pi / 2.0


def compute_degeneracy_metrics(points: np.ndarray) -> dict:
    if points.size == 0:
        return {
            "is_aisle": False,
            "mode": "NORMAL",
            "entropy": 1.0,
            "linearity": 1.0,
            "dominant_angle": 0.0,
            "confidence": 0.0,
        }

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape (N, 2)")

    xy = points[:, :2]
    if len(xy) < 3:
        return {
            "is_aisle": False,
            "mode": "NORMAL",
            "entropy": 1.0,
            "linearity": 1.0,
            "dominant_angle": 0.0,
            "confidence": 0.0,
        }

    deltas = xy[1:] - xy[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    valid = lengths > 1e-6
    if not np.any(valid):
        return {
            "is_aisle": False,
            "mode": "NORMAL",
            "entropy": 1.0,
            "linearity": 1.0,
            "dominant_angle": 0.0,
            "confidence": 0.0,
        }

    segment_angles = np.mod(np.arctan2(deltas[valid, 1], deltas[valid, 0]), math.pi)
    histogram, edges = np.histogram(segment_angles, bins=36, range=(0.0, math.pi))
    entropy = _normalized_entropy(histogram)
    peak_index = int(np.argmax(histogram))
    dominant_angle = float((edges[peak_index] + edges[peak_index + 1]) * 0.5)
    aligned_fraction = float(
        np.mean(np.abs(_line_angle_delta(segment_angles, dominant_angle)) <= math.radians(10.0))
    )

    centered = xy - xy.mean(axis=0)
    covariance = centered.T @ centered / max(len(xy) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    small = max(float(eigenvalues[0]), 1e-9)
    large = max(float(eigenvalues[-1]), 1e-9)
    linearity = large / small

    low_entropy = 1.0 - entropy
    linearity_score = min(max((math.log1p(linearity) - math.log1p(6.0)) / 2.0, 0.0), 1.0)
    confidence = min(max((low_entropy + aligned_fraction + linearity_score) / 3.0, 0.0), 1.0)
    is_aisle = linearity >= 8.0 and entropy <= 0.62 and aligned_fraction >= 0.34
    if is_aisle:
        mode = "AISLE"
    elif entropy >= 0.78:
        mode = "INTERSECTION"
    else:
        mode = "NORMAL"

    return {
        "is_aisle": bool(is_aisle),
        "mode": mode,
        "entropy": float(entropy),
        "linearity": float(linearity),
        "dominant_angle": dominant_angle,
        "confidence": float(confidence),
    }
