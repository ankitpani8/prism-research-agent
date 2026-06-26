# syntax=docker/dockerfile:1
# Multi-stage: deps cached in a builder, slim non-root runtime. Runs the FastAPI
# surface. Ollama runs on the HOST (reached via OLLAMA_HOST) to keep the image
# small and respect the RAM budget — see docker-compose.yml.
FROM python:3.11-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt .
RUN python -m venv /venv && /venv/bin/pip install -r requirements.txt

FROM python:3.11-slim AS runtime
ENV PATH="/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# non-root
RUN useradd --create-home --uid 10001 prism
WORKDIR /app
COPY --from=builder /venv /venv
COPY core/ ./core/
COPY api/ ./api/
COPY data/corpus/ ./data/corpus/
USER prism
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
