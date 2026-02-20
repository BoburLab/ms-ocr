"""
FastAPI OCR Microservice — Hardened Entry Point.

Security layers applied (outermost → innermost):
  MaxBodySize → CORS → TrustedProxy → CorrelationID → SecurityHeaders → Timeout → RateLimit → Route
"""

import logging
import os
import time
import asyncio
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.logging_config import setup_logging
from app.middleware import (
    CorrelationIDMiddleware,
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    RequestTimeoutMiddleware,
    MaxBodySizeMiddleware,
    TrustedProxyMiddleware,
)
from app.routers import router

# ── Structured JSON Logging ─────────────────────────────────
setup_logging(debug=settings.DEBUG)
logger = logging.getLogger(__name__)

# ── Application ─────────────────────────────────────────────
app = FastAPI(
    title="OCR Microservice",
    description="Production-ready OCR microservice.",
    version="1.0.0",
    # Disable interactive docs in production
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

# ── Middleware Stack (last added = outermost) ────────────────

# 1. Rate limiter
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
)

# 2. Request timeout
app.add_middleware(
    RequestTimeoutMiddleware,
    timeout_seconds=settings.REQUEST_TIMEOUT_SECONDS,
)

# 3. Secure response headers
app.add_middleware(SecurityHeadersMiddleware)

# 4. Correlation / request ID
app.add_middleware(CorrelationIDMiddleware)

# 5. Trusted proxy validation (strip spoofed X-Forwarded-For)
app.add_middleware(
    TrustedProxyMiddleware,
    trusted_cidrs=settings.TRUSTED_PROXIES,
)

# 6. CORS — only if explicit origins are configured
if settings.CORS_ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "X-Request-ID", "Content-Type"],
    )

# 7. Max body size at ASGI level (outermost — pure ASGI)
app = MaxBodySizeMiddleware(app, max_bytes=settings.MAX_REQUEST_BODY_BYTES)  # type: ignore[assignment]

# ── Global Exception Handlers ───────────────────────────────
# (must be registered on the FastAPI app, not the ASGI wrapper)
_fastapi_app: FastAPI = app.app if isinstance(app, MaxBodySizeMiddleware) else app  # type: ignore[union-attr]


@_fastapi_app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "unknown")
    logger.error("Unhandled exception", exc_info=True, extra={"request_id": rid})
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "request_id": rid},
    )


@_fastapi_app.exception_handler(404)
async def _not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"detail": "Not found."})


@_fastapi_app.exception_handler(405)
async def _method_not_allowed(request: Request, exc):
    return JSONResponse(status_code=405, content={"detail": "Method not allowed."})


# ── Routes ───────────────────────────────────────────────────
_fastapi_app.include_router(router)


# ── Background File Cleanup ─────────────────────────────────

async def _cleanup_old_files() -> None:
    """Periodically delete files older than FILE_RETENTION_HOURS."""
    retention = settings.FILE_RETENTION_HOURS
    if retention <= 0:
        return

    logger.info("File cleanup task started (retention=%dh)", retention)
    max_age = retention * 3600

    while True:
        await asyncio.sleep(3600)  # check every hour
        now = time.time()
        removed = 0
        for base_dir in (
            settings.RAW_STORAGE_PATH,
            settings.PREPROCESSED_STORAGE_PATH,
            settings.OUTPUT_STORAGE_PATH,
        ):
            for fpath in Path(base_dir).rglob("*"):
                if fpath.is_file():
                    try:
                        age = now - fpath.stat().st_mtime
                        if age > max_age:
                            fpath.unlink()
                            removed += 1
                    except OSError:
                        pass
        if removed:
            logger.info("Cleanup: removed %d expired file(s)", removed)


@_fastapi_app.on_event("startup")
async def _startup():
    logger.info(
        "OCR Microservice started",
        extra={
            "debug": settings.DEBUG,
            "rate_limit": settings.RATE_LIMIT_PER_MINUTE,
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
            "max_body_bytes": settings.MAX_REQUEST_BODY_BYTES,
            "request_timeout_s": settings.REQUEST_TIMEOUT_SECONDS,
            "file_retention_h": settings.FILE_RETENTION_HOURS,
        },
    )
    # launch background cleanup
    asyncio.create_task(_cleanup_old_files())
