from __future__ import annotations

from dataclasses import dataclass
import re

EXPANSION_DESCRIPTIONS = {
    1: "Early blastocyst, cavity less than half of embryo volume.",
    2: "Blastocyst, cavity at least half of embryo volume.",
    3: "Full blastocyst, cavity fills embryo.",
    4: "Expanded blastocyst, cavity larger than embryo with thinning zona.",
    5: "Hatching blastocyst.",
    6: "Hatched blastocyst.",
}

ICM_DESCRIPTIONS = {
    "A": "Many tightly packed inner cell mass cells.",
    "B": "Several loosely grouped inner cell mass cells.",
    "C": "Very few inner cell mass cells.",
    "D": "Degenerate or absent inner cell mass cells.",
}

TE_DESCRIPTIONS = {
    "A": "Many trophectoderm cells forming cohesive epithelium.",
    "B": "Few trophectoderm cells with a looser epithelium.",
    "C": "Very few large trophectoderm cells.",
    "D": "Degenerate or absent trophectoderm cells.",
}

GRADE_RE = re.compile(r"^([1-6])([A-D])([A-D])$")

_ICM_SCORE = {"A": 3, "B": 2, "C": 1, "D": 0}
_TE_SCORE = {"A": 3, "B": 2, "C": 1, "D": 0}


@dataclass(frozen=True)
class BlastocystGrade:
    expansion: int
    icm: str
    te: str

    def __post_init__(self) -> None:
        if self.expansion not in EXPANSION_DESCRIPTIONS:
            raise ValueError("Expansion must be between 1 and 6.")
        if self.icm not in ICM_DESCRIPTIONS:
            raise ValueError("ICM must be one of A, B, C, D.")
        if self.te not in TE_DESCRIPTIONS:
            raise ValueError("TE must be one of A, B, C, D.")

    @classmethod
    def parse(cls, grade_text: str) -> "BlastocystGrade":
        token = grade_text.strip().upper()
        match = GRADE_RE.match(token)
        if not match:
            raise ValueError(f"Invalid blastocyst grade: {grade_text!r}")
        return cls(expansion=int(match.group(1)), icm=match.group(2), te=match.group(3))

    @property
    def code(self) -> str:
        return f"{self.expansion}{self.icm}{self.te}"

    @property
    def quality_band(self) -> str:
        # Simple interpretable banding for research dashboards.
        score = self.expansion + (2 * _ICM_SCORE[self.icm]) + (2 * _TE_SCORE[self.te])
        if score >= 14:
            return "high"
        if score >= 10:
            return "moderate"
        if score >= 6:
            return "low"
        return "very_low"

    def to_dict(self) -> dict[str, str | int]:
        return {
            "grade": self.code,
            "expansion": self.expansion,
            "icm": self.icm,
            "te": self.te,
            "quality_band": self.quality_band,
        }

    def explanation(self) -> dict[str, str]:
        return {
            "expansion": EXPANSION_DESCRIPTIONS[self.expansion],
            "icm": ICM_DESCRIPTIONS[self.icm],
            "te": TE_DESCRIPTIONS[self.te],
        }
