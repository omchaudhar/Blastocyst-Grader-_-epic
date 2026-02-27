from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from .taxonomy import BlastocystGrade

EXPANSION_TO_INDEX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
INDEX_TO_EXPANSION = {v: k for k, v in EXPANSION_TO_INDEX.items()}

LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
INDEX_TO_LETTER = {v: k for k, v in LETTER_TO_INDEX.items()}


class BlastocystDataset(Dataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        image_root: str | Path,
        transform=None,
    ) -> None:
        self.annotations = annotations.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

        if "image_path" not in self.annotations.columns:
            raise ValueError("Annotations must contain an 'image_path' column.")

    def __len__(self) -> int:
        return len(self.annotations)

    def _extract_grade(self, row: pd.Series) -> BlastocystGrade:
        if "grade" in row and isinstance(row["grade"], str) and row["grade"].strip():
            return BlastocystGrade.parse(row["grade"])

        for col in ("expansion", "icm", "te"):
            if col not in row:
                raise ValueError(
                    "Each row must provide either 'grade' or all of 'expansion', 'icm', 'te'."
                )

        return BlastocystGrade(
            expansion=int(row["expansion"]),
            icm=str(row["icm"]).strip().upper(),
            te=str(row["te"]).strip().upper(),
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.annotations.iloc[idx]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = self.image_root / image_path

        image = Image.open(image_path).convert("RGB")
        grade = self._extract_grade(row)

        if self.transform is not None:
            image = self.transform(image)

        targets = {
            "expansion": torch.tensor(EXPANSION_TO_INDEX[grade.expansion], dtype=torch.long),
            "icm": torch.tensor(LETTER_TO_INDEX[grade.icm], dtype=torch.long),
            "te": torch.tensor(LETTER_TO_INDEX[grade.te], dtype=torch.long),
        }

        return {
            "image": image,
            "targets": targets,
            "grade_text": grade.code,
            "image_path": str(image_path),
        }


def load_annotations(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Annotations CSV is empty.")
    return df


def split_train_val(
    annotations: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    shuffled = annotations.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_df = shuffled.iloc[:n_val].reset_index(drop=True)
    train_df = shuffled.iloc[n_val:].reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Training split is empty. Increase dataset size or lower val_fraction.")

    return train_df, val_df
