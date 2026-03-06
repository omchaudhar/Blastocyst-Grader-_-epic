from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
import torch

from .data import INDEX_TO_EXPANSION, INDEX_TO_LETTER
from .model import MultiHeadBlastocystNet, build_transforms
from .taxonomy import BlastocystGrade


def _device_from_string(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_with_checkpoint(
    image: Image.Image,
    checkpoint_path: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_device = _device_from_string(device)
    checkpoint = torch.load(ckpt_path, map_location=run_device)

    model = MultiHeadBlastocystNet(pretrained=False).to(run_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    image_size = int(checkpoint.get("meta", {}).get("image_size", 224))
    transform = build_transforms(image_size=image_size, train=False)
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(run_device)

    with torch.no_grad():
        outputs = model(image_tensor)

    exp_probs = torch.softmax(outputs["expansion"], dim=1).squeeze(0)
    icm_probs = torch.softmax(outputs["icm"], dim=1).squeeze(0)
    te_probs = torch.softmax(outputs["te"], dim=1).squeeze(0)

    exp_idx = int(torch.argmax(exp_probs).item())
    icm_idx = int(torch.argmax(icm_probs).item())
    te_idx = int(torch.argmax(te_probs).item())

    grade = BlastocystGrade(
        expansion=INDEX_TO_EXPANSION[exp_idx],
        icm=INDEX_TO_LETTER[icm_idx],
        te=INDEX_TO_LETTER[te_idx],
    )

    confidence = float((exp_probs[exp_idx] + icm_probs[icm_idx] + te_probs[te_idx]) / 3.0)

    return {
        "mode": "trained_model",
        "confidence": confidence,
        **grade.to_dict(),
        "probabilities": {
            "expansion": [float(v) for v in exp_probs.tolist()],
            "icm": [float(v) for v in icm_probs.tolist()],
            "te": [float(v) for v in te_probs.tolist()],
        },
        "explanation": grade.explanation(),
        "checkpoint": str(ckpt_path),
    }
