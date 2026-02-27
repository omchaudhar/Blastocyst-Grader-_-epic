# Blastocyst Grading Evidence Hub

Static website for educational blastocyst grading support using morphology
criteria (Gardner framework) and outcome context from open research cohorts.

## Run locally

1. Start the local server:

```text
python3 server.py
```

2. Open in browser: `http://127.0.0.1:8000`

## What it includes

- Image upload and automatic blastocyst grading (stage, ICM, TE)
- OpenCV-based morphology pipeline with embryo/cavity/ICM segmentation overlay
- Confidence diagnostics (focus, contrast, segmentation reliability)
- Manual correction controls (override stage/ICM/TE and regenerate report)
- AI mode with trained checkpoint support and CV fallback
- Generated morphology-tier interpretation
- Cohort-specific live birth context:
  - Mixed multicentre single blastocyst transfer cohort (2023, n=10,018)
  - Euploid NC-FET cohort (2022, n=610)
  - Large single-transfer FET cohort (2021, n=10,482)
- Research citations and grading references

## Research sources embedded in the UI

- ASRM grading scales page (Gardner summary)
- Zou et al., 2023 (Human Reproduction, open access)
- Zhang et al., 2022 (JARG, open access)
- Ai et al., 2021 (Reproductive Biology and Endocrinology, open access)
- 2025 network meta-analysis (J Clin Med, open access)
- Kragh et al., 2022 blastocyst AI dataset paper (Sci Data)

## Notes

- Intended for educational and decision-support context.
- Not a replacement for embryologist assessment or clinic-specific protocol.
- To force trained-model inference, set checkpoint path:

```bash
export BLASTOCYST_CHECKPOINT=/absolute/path/to/best_model.pt
```

- Optional model-training dependencies are listed in `requirements.txt`.
