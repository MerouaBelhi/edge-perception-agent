# Lightweight inference image: ONNX Runtime + FastAPI, no PyTorch.
# Build after putting a model in models/ (site11hh_v2_int8.onnx from the Releases page,
# or any ONNX file; EPA_MODEL picks one explicitly):
#   docker build -t edge-perception-agent .
#   docker run -p 8000:8000 edge-perception-agent
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY models/*.onnx models/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "epa.api:app", "--host", "0.0.0.0", "--port", "8000"]
