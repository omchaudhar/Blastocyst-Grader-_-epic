from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from server import (  # noqa: E402
    RESEARCH_SOURCES,
    _parse_age,
    _parse_cohort,
    _predict_with_optional_checkpoint,
    _report_from_grade,
)
from blastocyst_grader.taxonomy import BlastocystGrade  # noqa: E402


def _allowed_origins_from_env() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _decode_upload_to_image(upload: UploadFile) -> Image.Image:
    payload = upload.file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        image = Image.open(BytesIO(payload))
        image.load()
        return image
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc


class GradeImageRequest(BaseModel):
    image_data: str = Field(..., min_length=16, description="Image as data URL or raw base64 payload.")
    cohort: str = Field(default="mixed")
    age: int | None = Field(default=None, ge=18, le=55)


class ManualReportRequest(BaseModel):
    stage: int = Field(..., ge=1, le=6)
    icm: str = Field(..., min_length=1, max_length=1)
    te: str = Field(..., min_length=1, max_length=1)
    cohort: str = Field(default="mixed")
    age: int | None = Field(default=None, ge=18, le=55)


app = FastAPI(
    title="Blastocyst AI Grader API",
    version="1.0.0",
    description="Production API for blastocyst image grading and report generation.",
)

origins = _allowed_origins_from_env()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.mount("/src", StaticFiles(directory=PROJECT_ROOT / "src"), name="src")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sources")
def sources() -> dict[str, Any]:
    return {"sources": RESEARCH_SOURCES}


@app.post("/api/grade-image")
def grade_image(req: GradeImageRequest) -> dict[str, Any]:
    try:
        cohort = _parse_cohort({"cohort": req.cohort})
        age = _parse_age({"age": req.age})
        from server import _decode_data_url

        image = _decode_data_url(req.image_data)
        prediction = _predict_with_optional_checkpoint(image)
        grade = BlastocystGrade.parse(str(prediction["grade"]))
        report = _report_from_grade(grade=grade, cohort=cohort, age=age, manual_override=False)
        return {"prediction": prediction, "report": report}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/report-from-grade")
def report_from_grade(req: ManualReportRequest) -> dict[str, Any]:
    try:
        cohort = _parse_cohort({"cohort": req.cohort})
        age = _parse_age({"age": req.age})
        grade = BlastocystGrade(
            expansion=req.stage,
            icm=req.icm.strip().upper(),
            te=req.te.strip().upper(),
        )
        report = _report_from_grade(grade=grade, cohort=cohort, age=age, manual_override=True)
        return {"report": report}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/grade-upload")
def grade_upload(
    image: UploadFile = File(...),
    cohort: str = Form("mixed"),
    age: int | None = Form(None),
) -> dict[str, Any]:
    try:
        parsed_cohort = _parse_cohort({"cohort": cohort})
        parsed_age = _parse_age({"age": age})
        pil_image = _decode_upload_to_image(image)
        prediction = _predict_with_optional_checkpoint(pil_image)
        grade = BlastocystGrade.parse(str(prediction["grade"]))
        report = _report_from_grade(grade=grade, cohort=parsed_cohort, age=parsed_age, manual_override=False)
        return {"prediction": prediction, "report": report}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
