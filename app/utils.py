import os
import io
import logging
from pathlib import Path

import aiofiles
import numpy as np
from fastapi import UploadFile, HTTPException
from pdf2image import convert_from_path
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# File Validation
# ──────────────────────────────────────────────

def get_file_extension(filename: str) -> str:
    """Extract lowercase file extension without dot."""
    return Path(filename).suffix.lstrip(".").lower()


def validate_file(file: UploadFile) -> str:
    """
    Validate uploaded file: name, extension, and size.
    Returns the validated file extension.
    Raises HTTPException on validation failure.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = get_file_extension(file.filename)
    if ext not in settings.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}.",
        )

    if file.size and file.size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size: {settings.MAX_FILE_SIZE_MB} MB.",
        )

    return ext


# ──────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────

async def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """Async chunked file save."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    async with aiofiles.open(destination, "wb") as out_file:
        while content := await upload_file.read(1024 * 1024):
            await out_file.write(content)
    return destination


# ──────────────────────────────────────────────
# Image Conversion
# ──────────────────────────────────────────────

def pdf_to_images(pdf_path: str, dpi: int = 150) -> list[np.ndarray]:
    """Convert PDF to list of numpy images (one per page)."""
    pages = convert_from_path(pdf_path, dpi=dpi)
    return [np.array(page) for page in pages]


def load_image_as_numpy(file_path: str) -> np.ndarray:
    """Load an image file and return as RGB numpy array."""
    image = Image.open(file_path).convert("RGB")
    return np.array(image)


def file_to_images(file_path: str, ext: str) -> list[np.ndarray]:
    """
    Convert a raw file (PDF or image) to a list of numpy arrays.
    Each element represents one page/image.
    """
    try:
        if ext == "pdf":
            return pdf_to_images(file_path)
        return [load_image_as_numpy(file_path)]
    except Exception as e:
        logger.error(f"Failed to convert file to images: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


# ──────────────────────────────────────────────
# Storage Path Builders
# ──────────────────────────────────────────────

def build_storage_paths(filename: str, engine: str) -> dict[str, str]:
    """
    Build all storage paths for a given file and engine.
    Also ensures directories exist.
    """
    base = Path(filename).stem

    raw_dir = os.path.join(settings.RAW_STORAGE_PATH, engine)
    preprocessed_dir = os.path.join(settings.PREPROCESSED_STORAGE_PATH, engine)
    output_dir = os.path.join(settings.OUTPUT_STORAGE_PATH, engine)

    for d in (raw_dir, preprocessed_dir, output_dir):
        os.makedirs(d, exist_ok=True)

    return {
        "raw_file": os.path.join(raw_dir, filename),
        "preprocessed_dir": preprocessed_dir,
        "output_file": os.path.join(output_dir, f"{base}.md"),
        "base_filename": base,
    }


# ──────────────────────────────────────────────
# Markdown Output Builder
# ──────────────────────────────────────────────

def build_markdown_output(
    filename: str,
    engine: str,
    processing_time: float,
    page_count: int,
    extracted_text: str,
) -> str:
    """Build a formatted Markdown string for OCR results."""
    return (
        f"# OCR Results for {filename}\n\n"
        f"**Engine:** {engine}\n"
        f"**Processing Time:** {processing_time:.2f}s\n"
        f"**Pages:** {page_count}\n\n"
        f"## Extracted Text\n\n"
        f"{extracted_text}"
    )
