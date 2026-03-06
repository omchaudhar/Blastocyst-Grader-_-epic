from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .taxonomy import BlastocystGrade

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional runtime fallback
    cv2 = None


@dataclass
class RuleBasedResult:
    grade: BlastocystGrade
    confidence: float
    features: dict[str, float]
    diagnostics: dict[str, float]
    overlay_png_base64: str | None = None
    mode: str = "rule_based_cv2_v2"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "confidence": self.confidence,
            "features": self.features,
            "diagnostics": self.diagnostics,
            **self.grade.to_dict(),
            "explanation": self.grade.explanation(),
        }
        if self.overlay_png_base64:
            payload["overlay_png_base64"] = self.overlay_png_base64
        return payload


def _normalize(values: np.ndarray) -> np.ndarray:
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if v_max - v_min < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - v_min) / (v_max - v_min)).astype(np.float32)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _resize_for_analysis(rgb: np.ndarray, max_dim: int = 640) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    current = max(height, width)
    if current <= max_dim:
        return rgb, 1.0
    scale = max_dim / float(current)
    resized = cv2.resize(rgb, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def _largest_component(mask: np.ndarray, prefer_center: tuple[float, float] | None = None) -> np.ndarray:
    binary = (mask.astype(np.uint8) > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return binary

    best_index = 1
    best_score = -1.0
    for idx in range(1, count):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        score = area
        if prefer_center is not None:
            cx, cy = float(centroids[idx, 0]), float(centroids[idx, 1])
            dx = cx - prefer_center[0]
            dy = cy - prefer_center[1]
            distance = (dx * dx + dy * dy) ** 0.5
            score = area / (1.0 + 0.08 * distance)
        if score > best_score:
            best_index = idx
            best_score = score

    return (labels == best_index).astype(np.uint8)


def _detect_embryo_circle(gray: np.ndarray) -> tuple[float, float, float, float]:
    height, width = gray.shape
    center_x = width / 2.0
    center_y = height / 2.0

    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=min(height, width) / 3.0,
        param1=120,
        param2=28,
        minRadius=int(0.18 * min(height, width)),
        maxRadius=int(0.49 * min(height, width)),
    )

    if circles is not None and circles.shape[1] > 0:
        candidates = circles[0]
        best = None
        best_score = -1.0
        for candidate in candidates:
            cx, cy, radius = float(candidate[0]), float(candidate[1]), float(candidate[2])
            dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
            score = radius - 0.12 * dist
            if score > best_score:
                best = (cx, cy, radius)
                best_score = score
        if best is not None:
            return best[0], best[1], best[2], 0.88

    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    background_level = float(np.median(border))
    diff = np.abs(gray.astype(np.float32) - background_level).astype(np.uint8)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = _largest_component(mask, prefer_center=(center_x, center_y))

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius > 10:
            return float(cx), float(cy), float(radius), 0.72

    fallback_radius = float(0.34 * min(height, width))
    return center_x, center_y, fallback_radius, 0.45


def _component_compactness(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 1e-6:
        return 0.0
    return _clip01((4.0 * np.pi * area) / (perimeter * perimeter))


def _angular_peak(mask: np.ndarray, cx: float, cy: float, bins: int = 16) -> float:
    yy, xx = np.where(mask > 0)
    if yy.size == 0:
        return 0.0
    angles = np.arctan2(yy.astype(np.float32) - cy, xx.astype(np.float32) - cx)
    hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
    total = float(np.sum(hist))
    if total <= 0.0:
        return 0.0
    return float(np.max(hist) / total)


def _ring_uniformity(values: np.ndarray, angles: np.ndarray, bins: int = 16) -> float:
    if values.size == 0:
        return 0.0
    groups = []
    for idx in range(bins):
        low = -np.pi + (2.0 * np.pi * idx / bins)
        high = -np.pi + (2.0 * np.pi * (idx + 1) / bins)
        segment = values[(angles >= low) & (angles < high)]
        groups.append(float(np.mean(segment)) if segment.size else 0.0)
    group_values = np.array(groups, dtype=np.float32)
    mean = float(np.mean(group_values))
    if mean <= 1e-6:
        return 0.0
    cv = float(np.std(group_values) / mean)
    return _clip01(1.0 - cv)


def _quality_metrics(gray: np.ndarray, embryo_mask: np.ndarray, segmentation_conf: float) -> dict[str, float]:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    focus_raw = float(np.var(lap))
    contrast_raw = float(np.std(gray[embryo_mask > 0])) if np.any(embryo_mask > 0) else 0.0

    focus_norm = _clip01((focus_raw - 35.0) / 280.0)
    contrast_norm = _clip01((contrast_raw - 12.0) / 50.0)
    overall = _clip01(0.45 * focus_norm + 0.30 * contrast_norm + 0.25 * segmentation_conf)

    return {
        "focus_raw": focus_raw,
        "contrast_raw": contrast_raw,
        "focus_norm": focus_norm,
        "contrast_norm": contrast_norm,
        "segmentation_conf": segmentation_conf,
        "image_quality": overall,
    }


def _encode_overlay_png(overlay: np.ndarray) -> str | None:
    if cv2 is None:
        return None
    success, encoded = cv2.imencode(".png", overlay)
    if not success:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _grade_without_cv2(image: Image.Image) -> RuleBasedResult:
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    height, width = gray.shape
    yy, xx = np.indices((height, width))
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    radius = np.sqrt(((xx - cx) / max(width, 1)) ** 2 + ((yy - cy) / max(height, 1)) ** 2)

    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    embryo_mask = (np.abs(gray - bg) > np.percentile(np.abs(gray - bg), 65)) & (radius < 0.48)
    embryo_area = float(np.mean(embryo_mask))

    center_region = radius < 0.25
    ring_region = (radius >= 0.25) & (radius < 0.45)
    center_std = float(np.std(gray[center_region])) if np.any(center_region) else 0.0
    ring_std = float(np.std(gray[ring_region])) if np.any(ring_region) else 0.0

    cavity_ratio = float(np.mean((gray > np.percentile(gray, 55)) & center_region))
    if cavity_ratio < 0.18:
        expansion = 2
    elif cavity_ratio < 0.33:
        expansion = 3
    elif cavity_ratio < 0.50:
        expansion = 4
    elif cavity_ratio < 0.65:
        expansion = 5
    else:
        expansion = 6

    icm = "A" if center_std > 0.08 else ("B" if center_std > 0.045 else "C")
    te = "A" if ring_std > 0.09 else ("B" if ring_std > 0.05 else "C")
    confidence = _clip01(0.25 + 0.35 * (center_std + ring_std + embryo_area))

    grade = BlastocystGrade(expansion=expansion, icm=icm, te=te)
    return RuleBasedResult(
        grade=grade,
        confidence=confidence,
        features={
            "cavity_ratio": cavity_ratio,
            "center_std": center_std,
            "ring_std": ring_std,
            "embryo_area_ratio": embryo_area,
        },
        diagnostics={"image_quality": confidence, "segmentation_conf": 0.5},
    )


def grade_image_rule_based(image: Image.Image) -> RuleBasedResult:
    if cv2 is None:
        return _grade_without_cv2(image)

    rgb = np.asarray(image.convert("RGB"))
    rgb, _ = _resize_for_analysis(rgb, max_dim=640)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_blur)

    height, width = gray.shape
    yy, xx = np.indices((height, width))
    cx, cy, radius, segmentation_conf = _detect_embryo_circle(gray_eq)
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    embryo_mask = (distance <= radius).astype(np.uint8)
    embryo_area = float(np.sum(embryo_mask))
    if embryo_area < 20:
        grade = BlastocystGrade(expansion=3, icm="C", te="C")
        return RuleBasedResult(
            grade=grade,
            confidence=0.2,
            features={"error_state": 1.0},
            diagnostics={"image_quality": 0.0, "segmentation_conf": 0.0},
        )

    grad_x = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    grad_norm = _normalize(grad_mag)

    local_delta = np.abs(gray_eq.astype(np.float32) - cv2.GaussianBlur(gray_eq.astype(np.float32), (0, 0), 2.0))
    local_norm = _normalize(local_delta)

    texture = 0.6 * grad_norm + 0.4 * local_norm

    center_prior = np.clip(1.0 - (distance / max(radius * 0.90, 1.0)), 0.0, 1.0)
    inner_region = (distance <= 0.84 * radius) & (embryo_mask > 0)
    cavity_score = 0.72 * (1.0 - texture) + 0.28 * center_prior
    if np.any(inner_region):
        threshold = float(np.percentile(cavity_score[inner_region], 73))
    else:
        threshold = 0.5
    cavity_mask = ((cavity_score >= threshold) & inner_region).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cavity_mask = cv2.morphologyEx(cavity_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cavity_mask = cv2.morphologyEx(cavity_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cavity_mask = _largest_component(cavity_mask, prefer_center=(cx, cy))
    cavity_area = float(np.sum(cavity_mask))
    cavity_ratio = cavity_area / max(embryo_area, 1.0)

    edges = cv2.Canny(gray_eq, threshold1=45, threshold2=130)
    outer_ring = (distance >= 0.88 * radius) & (distance <= 1.03 * radius)
    ring_edge = (edges > 0) & outer_ring
    ring_edge_density = float(np.mean(ring_edge.astype(np.float32))) if np.any(outer_ring) else 0.0

    ring_y, ring_x = np.where(outer_ring)
    ring_angles = np.arctan2(ring_y.astype(np.float32) - cy, ring_x.astype(np.float32) - cx) if ring_y.size else np.array([])
    ring_values = ring_edge[outer_ring].astype(np.float32) if np.any(outer_ring) else np.array([])
    ring_uniformity = _ring_uniformity(ring_values, ring_angles) if ring_values.size else 0.0

    border_level = float(np.median(np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])))
    outside_band = (distance > radius) & (distance <= 1.22 * radius)
    outside_signal = (np.abs(gray.astype(np.float32) - border_level) > 18.0) & outside_band
    outside_ratio = float(np.sum(outside_signal)) / max(embryo_area, 1.0)

    if cavity_ratio < 0.16:
        expansion = 1
    elif cavity_ratio < 0.30:
        expansion = 2
    elif cavity_ratio < 0.46:
        expansion = 3
    elif cavity_ratio < 0.63:
        expansion = 4
    else:
        if cavity_ratio > 0.78 and (outside_ratio > 0.055 or ring_uniformity < 0.22):
            expansion = 6
        elif cavity_ratio > 0.66 and (outside_ratio > 0.020 or ring_uniformity < 0.38):
            expansion = 5
        else:
            expansion = 4

    icm_region = (distance <= 0.72 * radius) & (embryo_mask > 0) & (cavity_mask == 0)
    if np.any(icm_region):
        icm_threshold = float(np.percentile(texture[icm_region], 86))
    else:
        icm_threshold = 0.9
    icm_raw = ((texture >= icm_threshold) & icm_region).astype(np.uint8)
    icm_raw = cv2.morphologyEx(icm_raw, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    icm_mask = _largest_component(icm_raw, prefer_center=(cx, cy))

    icm_area = float(np.sum(icm_mask))
    icm_area_ratio = icm_area / max(embryo_area, 1.0)
    icm_compactness = _component_compactness(icm_mask)
    icm_peak = _angular_peak(icm_mask, cx=cx, cy=cy, bins=14)
    icm_grad_mean = float(np.mean(grad_norm[icm_mask > 0])) if np.any(icm_mask > 0) else 0.0

    icm_score = (
        0.42 * _clip01((icm_area_ratio - 0.007) / 0.030)
        + 0.22 * _clip01((icm_peak - 0.16) / 0.38)
        + 0.20 * icm_compactness
        + 0.16 * _clip01((icm_grad_mean - 0.12) / 0.35)
    )

    if icm_score >= 0.66 and icm_area_ratio >= 0.010:
        icm = "A"
    elif icm_score >= 0.41:
        icm = "B"
    else:
        icm = "C"

    te_region = (distance >= 0.70 * radius) & (distance <= 0.97 * radius) & (embryo_mask > 0)
    te_edge_density = float(np.mean((edges[te_region] > 0).astype(np.float32))) if np.any(te_region) else 0.0
    te_texture_mean = float(np.mean(texture[te_region])) if np.any(te_region) else 0.0

    te_y, te_x = np.where(te_region)
    te_angles = np.arctan2(te_y.astype(np.float32) - cy, te_x.astype(np.float32) - cx) if te_y.size else np.array([])
    te_values = (edges[te_region] > 0).astype(np.float32) if np.any(te_region) else np.array([])
    te_uniformity = _ring_uniformity(te_values, te_angles) if te_values.size else 0.0

    te_blob_mask = ((local_norm > np.percentile(local_norm[te_region], 74)) & te_region).astype(np.uint8) if np.any(te_region) else np.zeros_like(te_region, dtype=np.uint8)
    te_blob_mask = cv2.morphologyEx(te_blob_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(te_blob_mask)
    blob_count = 0
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if 3 <= area <= 180:
            blob_count += 1
    te_blob_density = float(blob_count) / max(2.0 * np.pi * radius, 1.0)

    te_score = (
        0.38 * _clip01((te_edge_density - 0.018) / 0.150)
        + 0.26 * _clip01((te_texture_mean - 0.20) / 0.45)
        + 0.22 * te_uniformity
        + 0.14 * _clip01((te_blob_density - 0.015) / 0.080)
    )

    if te_score >= 0.64 and te_uniformity >= 0.34:
        te = "A"
    elif te_score >= 0.40:
        te = "B"
    else:
        te = "C"

    diagnostics = _quality_metrics(gray=gray, embryo_mask=embryo_mask, segmentation_conf=segmentation_conf)

    icm_margin = min(abs(icm_score - 0.41), abs(icm_score - 0.66)) / 0.25
    te_margin = min(abs(te_score - 0.40), abs(te_score - 0.64)) / 0.24
    stage_boundaries = [0.16, 0.30, 0.46, 0.63, 0.66, 0.78]
    stage_margin = min(abs(cavity_ratio - boundary) for boundary in stage_boundaries) / 0.12
    decision_margin = _clip01(0.4 * _clip01(icm_margin) + 0.4 * _clip01(te_margin) + 0.2 * _clip01(stage_margin))

    confidence = 0.24 + 0.68 * _clip01(0.55 * diagnostics["image_quality"] + 0.45 * decision_margin)

    if diagnostics["image_quality"] < 0.20:
        confidence = min(confidence, 0.38)

    grade = BlastocystGrade(expansion=expansion, icm=icm, te=te)

    overlay = rgb.copy()
    cv2.circle(overlay, (int(round(cx)), int(round(cy))), int(round(radius)), (255, 220, 70), 2)
    cavity_contours, _ = cv2.findContours(cavity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cavity_contours, -1, (80, 170, 255), 2)
    icm_contours, _ = cv2.findContours(icm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, icm_contours, -1, (92, 235, 120), 2)
    cv2.putText(
        overlay,
        f"{grade.code} | conf {confidence:.2f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "Yellow: embryo | Blue: cavity | Green: ICM",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    overlay_b64 = _encode_overlay_png(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    features = {
        "cavity_ratio": cavity_ratio,
        "ring_edge_density": ring_edge_density,
        "ring_uniformity": ring_uniformity,
        "outside_ratio": outside_ratio,
        "icm_area_ratio": icm_area_ratio,
        "icm_compactness": icm_compactness,
        "icm_peak": icm_peak,
        "icm_grad_mean": icm_grad_mean,
        "icm_score": icm_score,
        "te_edge_density": te_edge_density,
        "te_texture_mean": te_texture_mean,
        "te_uniformity": te_uniformity,
        "te_blob_density": te_blob_density,
        "te_score": te_score,
        "embryo_area_ratio": embryo_area / float(height * width),
    }

    return RuleBasedResult(
        grade=grade,
        confidence=float(_clip01(confidence)),
        features=features,
        diagnostics=diagnostics,
        overlay_png_base64=overlay_b64,
    )
