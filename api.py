from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blastocyst_grader.predict import predict_with_checkpoint
from blastocyst_grader.rule_based import grade_image_rule_based

app = FastAPI(title="Blastocyst Grading API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    mode: str = Form("rule_based"),
    checkpoint_path: str = Form("models/best_model.pt"),
) -> dict:
    try:
        payload = await image.read()
        pil_img = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    if mode == "rule_based":
        return grade_image_rule_based(pil_img).to_dict()

    if mode == "trained_model":
        try:
            return predict_with_checkpoint(pil_img, checkpoint_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    raise HTTPException(status_code=400, detail="Mode must be 'rule_based' or 'trained_model'.")


import io  # Keep import near usage for explicit visibility.
