from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blastocyst_grader.rule_based import grade_image_rule_based
from blastocyst_grader.taxonomy import BlastocystGrade


def test_grade_parse_roundtrip() -> None:
    grade = BlastocystGrade.parse("4Ab")
    assert grade.code == "4AB"


def test_invalid_grade_raises() -> None:
    try:
        BlastocystGrade.parse("7ZZ")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid grade.")


def test_quality_band_ordering() -> None:
    high = BlastocystGrade.parse("5AA")
    low = BlastocystGrade.parse("2CD")
    assert high.quality_band in {"high", "moderate"}
    assert low.quality_band in {"low", "very_low"}


def test_rule_based_output_is_valid_grade() -> None:
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[64:192, 64:192] = [180, 180, 180]
    arr[96:160, 96:160] = [120, 120, 120]
    image = Image.fromarray(arr, mode="RGB")

    result = grade_image_rule_based(image)

    assert 1 <= result.grade.expansion <= 6
    assert result.grade.icm in {"A", "B", "C", "D"}
    assert result.grade.te in {"A", "B", "C", "D"}
    assert 0.0 <= result.confidence <= 1.0
