"""
File security utilities and API key authentication.

Provides:
- MIME type detection via magic bytes
- Malicious content scanning
- Filename sanitization & directory traversal prevention
- UUID-based secure filename generation
- SHA-256 file hash computation
- Optional API key verification
"""

import hashlib
import re
import uuid
import logging
from pathlib import Path

from fastapi import HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader

from app.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Magic-byte signatures for allowed file types
# ──────────────────────────────────────────────

_MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    "application/pdf": [b"%PDF"],
    "image/jpeg": [b"\xff\xd8\xff"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
}

# Byte patterns that should never appear in uploaded content
_MALICIOUS_PATTERNS: list[bytes] = [
    b"<script",
    b"<?php",
    b"<%",
    b"#!/",
    b"eval(",
    b"exec(",
    b"import os",
    b"subprocess",
    b"__import__",
]

# Extensions that can hide behind a valid final extension (double-extension attacks)
_DANGEROUS_EXTENSIONS: set[str] = {
    ".php", ".exe", ".sh", ".bat", ".cmd", ".ps1",
    ".js", ".vbs", ".py", ".rb", ".pl", ".cgi",
    ".asp", ".aspx", ".jsp", ".html", ".htm", ".svg",
}


# ──────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────

def detect_mime_type(header: bytes) -> str | None:
    """Return MIME type based on file magic bytes, or None if unrecognised."""
    for mime, sigs in _MAGIC_SIGNATURES.items():
        for sig in sigs:
            if header.startswith(sig):
                return mime
    return None


def scan_for_malicious_content(sample: bytes) -> bool:
    """Return True if *sample* contains suspicious byte patterns."""
    lower = sample.lower()
    return any(pat.lower() in lower for pat in _MALICIOUS_PATTERNS)


def compute_file_hash(data: bytes) -> str:
    """SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


def _has_dangerous_extension(filename: str) -> bool:
    """Detect double-extension attacks like ``malware.php.pdf``."""
    parts = filename.lower().split(".")
    if len(parts) > 2:
        for part in parts[1:-1]:  # skip base name and final extension
            if f".{part}" in _DANGEROUS_EXTENSIONS:
                return True
    return False


def sanitize_filename(filename: str) -> str:
    """
    Strip directory components, remove dangerous characters,
    and collapse anything that could enable path traversal.
    """
    name = Path(filename).name                       # strip directory parts
    name = name.replace("..", "").replace("/", "").replace("\\", "")
    name = re.sub(r"[^\w.\-]", "_", name)            # keep only safe chars
    return name or "unnamed_file"


def generate_secure_filename(original: str) -> tuple[str, str]:
    """
    Return (uuid_filename, sanitized_original).
    The UUID filename is used for storage; the original is kept for metadata.
    """
    sanitized = sanitize_filename(original)
    ext = Path(sanitized).suffix.lower()
    return f"{uuid.uuid4().hex}{ext}", sanitized


# ──────────────────────────────────────────────
# Comprehensive upload validation
# ──────────────────────────────────────────────

async def validate_file_security(file: UploadFile) -> tuple[str, bytes, str]:
    """
    Full-stack file validation:
      1. Filename presence & sanitization
      2. Extension whitelist
      3. Size limit
      4. Magic-byte MIME verification
      5. Malicious content scan
      6. SHA-256 hash

    Returns
    -------
    (extension, raw_bytes, sha256_hex)

    Raises
    ------
    HTTPException  on any validation failure
    """
    # 1 — filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    safe_name = sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lstrip(".").lower()

    if ext not in settings.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed}.",
        )

    # 1b — double-extension attack
    if _has_dangerous_extension(safe_name):
        logger.warning("Double-extension attack detected: %s", safe_name)
        raise HTTPException(
            status_code=400,
            detail="File rejected: suspicious filename pattern.",
        )

    # 2 — read content (streamed into memory once)
    content = await file.read()
    await file.seek(0)

    # 3 — size
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum: {settings.MAX_FILE_SIZE_MB} MB.",
        )

    # 4 — MIME via magic bytes
    detected = detect_mime_type(content[:16])
    if detected is None or detected not in settings.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="File content does not match an allowed type (PDF, JPEG, PNG).",
        )

    # 5 — malicious patterns (first 8 KB)
    if scan_for_malicious_content(content[:8192]):
        logger.warning("Malicious content detected in upload: %s", safe_name)
        raise HTTPException(
            status_code=400,
            detail="File rejected: suspicious content detected.",
        )

    # 6 — hash
    file_hash = compute_file_hash(content)

    return ext, content, file_hash


# ──────────────────────────────────────────────
# Optional API-key authentication
# ──────────────────────────────────────────────

_api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> None:
    """
    If ``API_KEY`` is configured in settings, every request must supply it.
    If ``API_KEY`` is empty/unset, authentication is skipped.
    """
    if not settings.API_KEY:
        return  # auth disabled

    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
