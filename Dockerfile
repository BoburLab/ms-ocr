# ============================================================
# Stage 1 — Builder: install Python dependencies
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ============================================================
# Stage 2 — Production image
# ============================================================
FROM python:3.11-slim

# Install only essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -s /usr/sbin/nologin appuser

WORKDIR /app

# Create storage outside app root with restrictive permissions
RUN mkdir -p /data/storage/raw /data/storage/preprocessed /data/storage/output \
    && chown -R appuser:appuser /data/storage \
    && chmod -R 700 /data/storage

# Copy application code
COPY --chown=appuser:appuser ./app /app/app

# Environment defaults
ENV RAW_STORAGE_PATH=/data/storage/raw \
    PREPROCESSED_STORAGE_PATH=/data/storage/preprocessed \
    OUTPUT_STORAGE_PATH=/data/storage/output \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production uvicorn: no reload, no access log (JSON logger handles it), single worker
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--no-access-log", \
     "--timeout-keep-alive", "30", \
     "--workers", "1"]
