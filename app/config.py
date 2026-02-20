"""
Centralised, validated application configuration.

All secrets and tunables are loaded from environment variables / ``.env``.
Required variables are validated at import time so the process fails fast.
"""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Storage ──────────────────────────────────────────────
    RAW_STORAGE_PATH: str = "./storage/raw"
    PREPROCESSED_STORAGE_PATH: str = "./storage/preprocessed"
    OUTPUT_STORAGE_PATH: str = "./storage/output"

    # ── vLLM inference server ────────────────────────────────
    VLLM_BASE_URL: str = "http://vllm:8001"
    VLLM_MODEL_NAME: str = "lightonai/LightOnOCR-2-1B"

    # ── File validation ──────────────────────────────────────
    ALLOWED_EXTENSIONS: set[str] = {"pdf", "png", "jpg", "jpeg"}
    ALLOWED_MIME_TYPES: set[str] = {"application/pdf", "image/jpeg", "image/png"}
    MAX_FILE_SIZE_MB: int = 20

    # ── PDF / Image bomb protection ──────────────────────────
    MAX_PDF_PAGES: int = 50
    MAX_IMAGE_DIMENSION: int = 10000  # px (width or height)

    # ── Request timeout (seconds) ────────────────────────────
    REQUEST_TIMEOUT_SECONDS: int = 300

    # ── Trusted proxies (for X-Forwarded-For) ────────────────
    TRUSTED_PROXIES: list[str] = ["127.0.0.1", "172.16.0.0/12", "10.0.0.0/8"]

    # ── File cleanup (hours; 0 = disabled) ───────────────────
    FILE_RETENTION_HOURS: int = 72

    # ── Max request body (bytes, ASGI-level) ─────────────────
    MAX_REQUEST_BODY_BYTES: int = 25 * 1024 * 1024  # 25 MB

    # ── Rate limiting ────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 20

    # ── CORS ─────────────────────────────────────────────────
    CORS_ALLOWED_ORIGINS: list[str] = []

    # ── API key (optional — leave empty to disable) ──────────
    API_KEY: str = ""
    API_KEY_HEADER: str = "X-API-Key"

    # ── Application mode ─────────────────────────────────────
    DEBUG: bool = False

    # ── MLflow (optional) ────────────────────────────────────
    MLFLOW_TRACKING_URI: str = ""

    # ── Server ───────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()


# ── Startup validation ──────────────────────────────────────
def _validate_config() -> None:
    errors: list[str] = []
    if not settings.VLLM_BASE_URL:
        errors.append("VLLM_BASE_URL is required")
    if not settings.VLLM_MODEL_NAME:
        errors.append("VLLM_MODEL_NAME is required")
    if errors:
        raise RuntimeError(f"Configuration errors: {'; '.join(errors)}")


_validate_config()


# ── Create storage dirs with restrictive permissions ────────
for _path in (
    settings.RAW_STORAGE_PATH,
    settings.PREPROCESSED_STORAGE_PATH,
    settings.OUTPUT_STORAGE_PATH,
):
    os.makedirs(_path, mode=0o700, exist_ok=True)
