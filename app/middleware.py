"""
Security middleware stack.

- CorrelationIDMiddleware    — attaches a unique request_id for tracing
- SecurityHeadersMiddleware  — OWASP-recommended response headers
- RateLimitMiddleware        — in-memory per-IP sliding window (429 on breach)
- RequestTimeoutMiddleware   — kills requests exceeding the configured timeout
- MaxBodySizeMiddleware      — rejects oversized request bodies at ASGI level
- TrustedProxyMiddleware     — validates X-Forwarded-For against trusted CIDRs
"""

import asyncio
import ipaddress
import time
import uuid
import logging
from collections import defaultdict
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Correlation ID
# ──────────────────────────────────────────────

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Propagate or generate ``X-Request-ID`` on every request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ──────────────────────────────────────────────
# Secure Response Headers
# ──────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security-related HTTP headers into every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains; preload"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        # suppress server identity
        response.headers.pop("server", None)

        return response


# ──────────────────────────────────────────────
# Per-IP Rate Limiter
# ──────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter.

    Returns ``429 Too Many Requests`` with a ``Retry-After`` header
    when the caller exceeds *requests_per_minute*.
    """

    def __init__(self, app, requests_per_minute: int = 20):
        super().__init__(app)
        self.limit = requests_per_minute
        self.window = 60  # seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    @staticmethod
    def _client_ip(request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_limited(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            self._hits[ip] = [t for t in self._hits[ip] if now - t < self.window]
            if len(self._hits[ip]) >= self.limit:
                return True
            self._hits[ip].append(now)
            return False

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        ip = self._client_ip(request)

        if self._is_limited(ip):
            logger.warning("Rate limit exceeded for %s", ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please try again later."},
                headers={"Retry-After": str(self.window)},
            )

        return await call_next(request)


# ──────────────────────────────────────────────
# Request Timeout
# ──────────────────────────────────────────────

class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """
    Cancel request processing after *timeout_seconds*.
    Returns ``504 Gateway Timeout`` if exceeded.
    """

    def __init__(self, app, timeout_seconds: int = 300):
        super().__init__(app)
        self.timeout = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            rid = getattr(request.state, "request_id", "unknown")
            logger.error(
                "Request timed out after %ds",
                self.timeout,
                extra={"request_id": rid},
            )
            return JSONResponse(
                status_code=504,
                content={"detail": "Request timed out."},
            )


# ──────────────────────────────────────────────
# Max Request Body Size (ASGI-level)
# ──────────────────────────────────────────────

class MaxBodySizeMiddleware:
    """
    Pure ASGI middleware (not BaseHTTPMiddleware) that returns 413
    before the body is fully buffered if ``Content-Length`` exceeds *max_bytes*.
    """

    def __init__(self, app, max_bytes: int = 25 * 1024 * 1024):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")

        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    response = JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"Request body too large. Max: {self.max_bytes // (1024*1024)} MB."
                        },
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass

        await self.app(scope, receive, send)


# ──────────────────────────────────────────────
# Trusted Proxy Validation
# ──────────────────────────────────────────────

class TrustedProxyMiddleware(BaseHTTPMiddleware):
    """
    Only honour ``X-Forwarded-For`` when the direct client IP
    belongs to a trusted CIDR range. Otherwise, strip the header
    to prevent IP spoofing that could bypass rate limiting.
    """

    def __init__(self, app, trusted_cidrs: list[str] | None = None):
        super().__init__(app)
        self._networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for cidr in (trusted_cidrs or []):
            try:
                self._networks.append(ipaddress.ip_network(cidr, strict=False))
            except ValueError:
                logger.warning("Invalid trusted proxy CIDR: %s (skipped)", cidr)

    def _is_trusted(self, ip_str: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip_str)
            return any(addr in net for net in self._networks)
        except ValueError:
            return False

    async def dispatch(self, request: Request, call_next):
        direct_ip = request.client.host if request.client else None

        if direct_ip and not self._is_trusted(direct_ip):
            # Strip X-Forwarded-For — untrusted source
            scope = request.scope
            scope["headers"] = [
                (k, v) for k, v in scope["headers"]
                if k != b"x-forwarded-for"
            ]

        return await call_next(request)
