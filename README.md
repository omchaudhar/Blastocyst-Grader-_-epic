# Blastocyst Grading Evidence Hub

Static website for educational blastocyst grading support using morphology
criteria (Gardner framework) and outcome context from open research cohorts.

## Run locally

1. Start the lightweight local server:

```text
python3 server.py
```

2. Open in browser: `http://127.0.0.1:8000`

Shortcut:

```bash
make run-local
```

## Production API (Deployable)

Production entrypoint:

```text
uvicorn api:app --host 0.0.0.0 --port 8000
```

Shortcut:

```bash
make run-api
```

The FastAPI service exposes:

- `GET /api/health`
- `GET /api/sources`
- `POST /api/grade-image` (JSON with base64/data URL image)
- `POST /api/grade-upload` (multipart form upload)
- `POST /api/report-from-grade` (manual override report)

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

## Deploy to Practical Use

### Option A: Docker (recommended)

```bash
docker build -t blastocyst-ai-grader .
docker run -p 8000:8000 \
  -e ALLOWED_ORIGINS="https://yourdomain.com" \
  -e BLASTOCYST_CHECKPOINT="/app/models/best_model.pt" \
  blastocyst-ai-grader
```

Shortcuts:

```bash
make docker-build
make docker-run
```

Open: `http://localhost:8000`

### Option B: Render

1. Push this folder to GitHub.
2. Create a new Render Web Service from the repo.
3. Render will detect `render.yaml` + `Dockerfile`.
4. Set environment variables:
   - `ALLOWED_ORIGINS=https://yourdomain.com`
   - `BLASTOCYST_CHECKPOINT=/app/models/best_model.pt` (optional)
5. Deploy and verify `https://<your-service>/api/health`.

### Option C: Railway

1. Create a new Railway project from your repo.
2. Railway will use `railway.json` start command.
3. Set environment variables:
   - `ALLOWED_ORIGINS=https://yourdomain.com`
   - `BLASTOCYST_CHECKPOINT=/app/models/best_model.pt` (optional)

## Notes

- Intended for educational and decision-support context.
- Not a replacement for embryologist assessment or clinic-specific protocol.
- To force trained-model inference, set checkpoint path:

```bash
export BLASTOCYST_CHECKPOINT=/absolute/path/to/best_model.pt
```

- Optional model-training dependencies are listed in `requirements.txt`.
- `ALLOWED_ORIGINS` should be restricted in production (avoid `*`).
