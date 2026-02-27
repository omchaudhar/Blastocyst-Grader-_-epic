from __future__ import annotations

import base64
from io import BytesIO
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blastocyst_grader.research_sources import RESEARCH_SOURCES
from blastocyst_grader.rule_based import grade_image_rule_based
from blastocyst_grader.taxonomy import BlastocystGrade

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_model.pt"
SUPPORTED_COHORTS = {"mixed", "euploid", "frozen_set"}

MIXED_COHORT = {
    "label": "Mixed multicentre single-blastocyst transfer cohort (2023, n=10,018)",
    "category_live_birth": {
        "good": 44.4,
        "moderate": 38.6,
        "low": 30.2,
        "very_low": 13.7,
    },
    "direct_live_birth": {
        "AC": 33.0,
        "CA": 33.0,
        "BC": 33.0,
        "CB": 24.6,
        "CC": 13.7,
    },
    "icm_adjusted_odds": {"A": 1.0, "B": 0.79, "C": 0.40, "D": 0.25},
    "te_adjusted_odds": {"A": 1.0, "B": 0.75, "C": 0.58, "D": 0.35},
    "low_grade_age_benchmarks": {"at25": 40.0, "at35": 21.0},
}

EUPLOID_COHORT = {
    "label": "Euploid NC-FET cohort (2022, n=610)",
    "direct_live_birth": {
        "AA": 64.5,
        "AB": 67.5,
        "BA": 55.6,
        "BB": 56.5,
        "AC": 40.8,
        "CA": 40.8,
        "BC": 40.8,
        "CB": 40.8,
        "CC": 40.8,
        "AD": 35.0,
        "DA": 35.0,
        "BD": 32.0,
        "DB": 32.0,
        "CD": 26.0,
        "DC": 26.0,
        "DD": 20.0,
    },
    "icm_adjusted_odds": {"A": 1.0, "B": 0.70, "C": 0.32, "D": 0.20},
    "te_adjusted_odds": {"A": 1.0, "B": 1.20, "C": 0.84, "D": 0.60},
}

# Additional large FET evidence (2021, n=10,482), used as cross-cohort calibration.
FROZEN_SET_COHORT = {
    "label": "Large FET single blastocyst cohort (2021, n=10,482)",
    "icm_live_birth": {"A": 54.62, "B": 41.29, "C": 28.45, "D": 20.0},
    "te_live_birth": {"A": 52.74, "B": 45.64, "C": 32.57, "D": 24.0},
    "stage_live_birth": {3: 37.07, 4: 44.21, 5: 41.67, 6: 41.67, 1: 34.0, 2: 35.5},
}

META_TOP_PAIR_CODES = {"AA", "AB", "BA", "BB"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _classify_band(icm: str, te: str) -> str:
    if icm == "D" or te == "D":
        return "very_low"
    if icm == "C" and te == "C":
        return "very_low"
    if icm == "C" or te == "C":
        return "low"
    if icm == "B" and te == "B":
        return "moderate"
    return "good"


def _band_label(band: str) -> str:
    return {
        "good": "Higher morphology tier",
        "moderate": "Intermediate morphology tier",
        "low": "Lower morphology tier",
        "very_low": "Lowest morphology tier",
    }[band]


def _transfer_priority(stage: int, band: str) -> str:
    if band == "good" and stage >= 4:
        return "Higher priority in morphology-based ranking."
    if band in {"good", "moderate"}:
        return "Usual priority with patient-specific review."
    if band == "low":
        return "Lower priority but clinically transferable in many programs."
    return "Lowest morphology priority; decision should be individualized."


def _estimate_mixed_outcome(icm: str, te: str, band: str) -> float:
    pair_code = f"{icm}{te}"
    return float(MIXED_COHORT["direct_live_birth"].get(pair_code, MIXED_COHORT["category_live_birth"][band]))


def _estimate_euploid_outcome(icm: str, te: str) -> float:
    pair_code = f"{icm}{te}"
    return float(EUPLOID_COHORT["direct_live_birth"].get(pair_code, EUPLOID_COHORT["direct_live_birth"]["AC"]))


def _estimate_frozen_set_outcome(stage: int, icm: str, te: str) -> float:
    icm_rate = float(FROZEN_SET_COHORT["icm_live_birth"].get(icm, 24.0))
    te_rate = float(FROZEN_SET_COHORT["te_live_birth"].get(te, 24.0))
    stage_rate = float(FROZEN_SET_COHORT["stage_live_birth"].get(stage, 40.0))
    return 0.50 * icm_rate + 0.35 * te_rate + 0.15 * stage_rate


def _infer_low_grade_age_estimate(age: int) -> float:
    slope = (MIXED_COHORT["low_grade_age_benchmarks"]["at35"] - MIXED_COHORT["low_grade_age_benchmarks"]["at25"]) / 10.0
    inferred = MIXED_COHORT["low_grade_age_benchmarks"]["at25"] + (age - 25) * slope
    return _clamp(inferred, 15.0, 45.0)


def _stage_note(stage: int) -> str:
    if stage <= 2:
        return "Stage 1-2 embryos are early in expansion; many labs finalize detailed ICM/TE ranking at stage 3-6."
    return "Stage 3-6 supports standard detailed morphology reporting."


def _meta_rank_note(pair_code: str) -> str:
    if pair_code in META_TOP_PAIR_CODES:
        return "Network meta-analysis trend: this morphology pair is in the higher-ranked group."
    if "C" in pair_code or "D" in pair_code:
        return "Network meta-analysis trend: C/D-containing morphology pairs are generally lower-ranked."
    return "Network meta-analysis trend: mid-ranked morphology pair."


def _cross_study_range(stage: int, icm: str, te: str, central: float) -> dict[str, float]:
    ai_icm = float(FROZEN_SET_COHORT["icm_live_birth"].get(icm, 24.0))
    ai_te = float(FROZEN_SET_COHORT["te_live_birth"].get(te, 24.0))
    ai_stage = float(FROZEN_SET_COHORT["stage_live_birth"].get(stage, 40.0))
    ai_proxy = _estimate_frozen_set_outcome(stage=stage, icm=icm, te=te)

    low = min(central, ai_proxy, ai_icm, ai_te, ai_stage)
    high = max(central, ai_proxy, ai_icm, ai_te, ai_stage)
    return {
        "low": round(_clamp(low, 5.0, 85.0), 1),
        "high": round(_clamp(high, 5.0, 85.0), 1),
        "ai_proxy": round(ai_proxy, 1),
    }


def _report_from_grade(grade: BlastocystGrade, cohort: str, age: int | None, manual_override: bool = False) -> dict[str, Any]:
    icm = grade.icm
    te = grade.te
    stage = grade.expansion
    pair_code = f"{icm}{te}"
    band = _classify_band(icm, te)

    if cohort == "euploid":
        live_birth = _estimate_euploid_outcome(icm=icm, te=te)
        cohort_data = EUPLOID_COHORT
        interpretation = "Euploid cohort data: ICM grade shows the stronger independent association."
        adjusted_odds = {
            "icm": round(float(EUPLOID_COHORT["icm_adjusted_odds"].get(icm, 0.2)), 2),
            "te": round(float(EUPLOID_COHORT["te_adjusted_odds"].get(te, 0.2)), 2),
        }
    elif cohort == "frozen_set":
        live_birth = _estimate_frozen_set_outcome(stage=stage, icm=icm, te=te)
        cohort_data = FROZEN_SET_COHORT
        interpretation = "Large FET cohort estimate from reported stage + ICM + TE live-birth trends."
        adjusted_odds = {
            "icm_proxy": round(float(cohort_data["icm_live_birth"].get(icm, 20.0)) / 41.29, 2),
            "te_proxy": round(float(cohort_data["te_live_birth"].get(te, 20.0)) / 45.64, 2),
        }
    else:
        live_birth = _estimate_mixed_outcome(icm=icm, te=te, band=band)
        cohort_data = MIXED_COHORT
        interpretation = "Mixed multicentre cohort: both ICM and TE grade independently affected live birth odds."
        adjusted_odds = {
            "icm": round(float(MIXED_COHORT["icm_adjusted_odds"].get(icm, 0.2)), 2),
            "te": round(float(MIXED_COHORT["te_adjusted_odds"].get(te, 0.2)), 2),
        }

    age_context = None
    if cohort == "mixed" and age is not None and band in {"low", "very_low"}:
        age_context = {
            "estimated_live_birth": round(_infer_low_grade_age_estimate(age), 1),
            "benchmark_points": "Low-grade subgroup trend benchmark: 40% at age 25 and 21% at age 35.",
        }

    range_data = _cross_study_range(stage=stage, icm=icm, te=te, central=live_birth)

    return {
        "version": "report_v2",
        "manual_override": manual_override,
        "cohort": cohort,
        "cohort_label": str(cohort_data["label"]),
        "grade": grade.code,
        "pair_code": pair_code,
        "quality_band": band,
        "quality_label": _band_label(band),
        "live_birth_estimate_pct": round(float(live_birth), 1),
        "study_range_pct": {"low": range_data["low"], "high": range_data["high"]},
        "ai_proxy_pct": range_data["ai_proxy"],
        "priority_signal": _transfer_priority(stage=stage, band=band),
        "adjusted_odds": adjusted_odds,
        "stage_note": _stage_note(stage=stage),
        "interpretation": interpretation,
        "meta_rank_note": _meta_rank_note(pair_code=pair_code),
        "age_context": age_context,
        "evidence_notes": [
            "Zou et al. 2023 mixed cohort (n=10,018)",
            "Zhang et al. 2022 euploid cohort (n=610)",
            "Ai et al. 2021 large FET cohort (n=10,482)",
            "2025 network meta-analysis morphology ranking trend",
        ],
    }


def _decode_data_url(image_data: str) -> Image.Image:
    if not image_data:
        raise ValueError("image_data is required.")

    token = image_data.strip()
    if token.startswith("data:"):
        try:
            _, token = token.split(",", 1)
        except ValueError as exc:
            raise ValueError("Invalid data URL format.") from exc

    try:
        raw_bytes = base64.b64decode(token, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid base64 image data.") from exc

    try:
        image = Image.open(BytesIO(raw_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError("Decoded payload is not a valid image.") from exc
    return image


def _predict_with_optional_checkpoint(image: Image.Image) -> dict[str, Any]:
    checkpoint_env = os.environ.get("BLASTOCYST_CHECKPOINT")
    checkpoint_path = Path(checkpoint_env) if checkpoint_env else DEFAULT_CHECKPOINT

    if checkpoint_path.exists():
        try:
            import torch  # noqa: F401
            from blastocyst_grader.predict import predict_with_checkpoint

            return predict_with_checkpoint(image=image, checkpoint_path=checkpoint_path)
        except Exception as exc:  # noqa: BLE001
            fallback = grade_image_rule_based(image).to_dict()
            fallback["warning"] = f"Model checkpoint unavailable or failed. Used rule-based fallback: {exc}"
            return fallback

    return grade_image_rule_based(image).to_dict()


def _parse_age(payload: dict[str, Any]) -> int | None:
    age_value = payload.get("age")
    if age_value is None or age_value == "":
        return None
    try:
        age = int(age_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("age must be an integer.") from exc
    if age < 18 or age > 55:
        raise ValueError("age must be between 18 and 55.")
    return age


def _parse_cohort(payload: dict[str, Any]) -> str:
    cohort = str(payload.get("cohort", "mixed")).strip().lower()
    if cohort not in SUPPORTED_COHORTS:
        raise ValueError("cohort must be one of: mixed, euploid, frozen_set.")
    return cohort


class AppHandler(SimpleHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Request body is empty.")
        body = self.rfile.read(content_length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON payload.") from exc

    def do_OPTIONS(self) -> None:  # noqa: N802
        if urlparse(self.path).path.startswith("/api/"):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        if path == "/api/sources":
            self._send_json(200, {"sources": RESEARCH_SOURCES})
            return
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/grade-image":
            self._handle_grade_image()
            return
        if path == "/api/report-from-grade":
            self._handle_report_from_grade()
            return
        self._send_json(404, {"detail": "Endpoint not found."})

    def _handle_grade_image(self) -> None:
        try:
            payload = self._read_json_body()
            cohort = _parse_cohort(payload)
            age = _parse_age(payload)

            image = _decode_data_url(str(payload.get("image_data")))
            prediction = _predict_with_optional_checkpoint(image)
            grade = BlastocystGrade.parse(str(prediction["grade"]))
            report = _report_from_grade(grade=grade, cohort=cohort, age=age, manual_override=False)
            self._send_json(200, {"prediction": prediction, "report": report})
        except ValueError as exc:
            self._send_json(400, {"detail": str(exc)})

    def _handle_report_from_grade(self) -> None:
        try:
            payload = self._read_json_body()
            cohort = _parse_cohort(payload)
            age = _parse_age(payload)

            stage = int(payload.get("stage"))
            icm = str(payload.get("icm", "")).strip().upper()
            te = str(payload.get("te", "")).strip().upper()
            grade = BlastocystGrade(expansion=stage, icm=icm, te=te)
            report = _report_from_grade(grade=grade, cohort=cohort, age=age, manual_override=True)
            self._send_json(200, {"report": report})
        except (TypeError, ValueError) as exc:
            self._send_json(400, {"detail": str(exc)})


def main() -> None:
    os.chdir(PROJECT_ROOT)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Blastocyst AI Grader running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
