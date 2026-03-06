.PHONY: run-local run-api docker-build docker-run

run-local:
	python3 server.py

run-api:
	uvicorn api:app --host 0.0.0.0 --port $${PORT:-8000}

docker-build:
	docker build -t blastocyst-ai-grader:latest .

docker-run:
	docker run --rm -p 8000:8000 \
		-e ALLOWED_ORIGINS="*" \
		blastocyst-ai-grader:latest
