FROM python:3.11-slim AS base

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir . 2>/dev/null || \
    pip install --no-cache-dir \
    onnxruntime>=1.15 \
    fastapi>=0.100 \
    "uvicorn[standard]>=0.23" \
    numpy>=1.24 \
    python-chess>=1.9 \
    pydantic>=2.0

COPY src/ src/
COPY chess_model.onnx .

ENV MODEL_PATH=chess_model.onnx

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
