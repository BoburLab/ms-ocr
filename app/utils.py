"""
General-purpose utilities: image conversion, storage paths, markdown builder.

NOTE: File validation has moved to ``app.security`` module.
"""

import os
import logging
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from pdf2image import convert_from_path
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Image Conversion
# ──────────────────────────────────────────────

def pdf_to_images(pdf_path: str, dpi: int = 150) -> list[np.ndarray]:
    """Convert PDF to list of numpy images (one per page) with bomb protection."""
    pages = convert_from_path(pdf_path, dpi=dpi)

    if len(pages) > settings.MAX_PDF_PAGES:
        raise HTTPException(
            status_code=400,
            detail=f"PDF has {len(pages)} pages. Maximum allowed: {settings.MAX_PDF_PAGES}.",
        )

    return [np.array(page) for page in pages]


def _validate_image_dimensions(img: np.ndarray) -> None:
    """Reject decompression bombs — images with extreme dimensions."""
    h, w = img.shape[:2]
    limit = settings.MAX_IMAGE_DIMENSION
    if h > limit or w > limit:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions ({w}x{h}) exceed limit ({limit}x{limit}).",
        )


def load_image_as_numpy(file_path: str) -> np.ndarray:
    """Load an image file and return as RGB numpy array."""
    image = Image.open(file_path).convert("RGB")
    arr = np.array(image)
    _validate_image_dimensions(arr)
    return arr


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
        logger.error("Failed to convert file to images: %s", e)
        raise HTTPException(status_code=500, detail="Error processing file.")


# ──────────────────────────────────────────────
# Storage Path Builders
# ──────────────────────────────────────────────

def build_storage_paths(filename: str, engine: str) -> dict[str, str]:
    """
    Build all storage paths for a given *filename* (UUID-based) and engine.
    Creates directories with restrictive permissions.
    """
    base = Path(filename).stem

    raw_dir = os.path.join(settings.RAW_STORAGE_PATH, engine)
    preprocessed_dir = os.path.join(settings.PREPROCESSED_STORAGE_PATH, engine)
    output_dir = os.path.join(settings.OUTPUT_STORAGE_PATH, engine)

    for d in (raw_dir, preprocessed_dir, output_dir):
        os.makedirs(d, mode=0o700, exist_ok=True)

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
    file_hash: str = "",
    request_id: str = "",
) -> str:
    """Build a formatted Markdown string for OCR results."""
    header = (
        f"# OCR Results for {filename}\n\n"
        f"**Engine:** {engine}\n"
        f"**Processing Time:** {processing_time:.2f}s\n"
        f"**Pages:** {page_count}\n"
    )
    if file_hash:
        header += f"**SHA-256:** `{file_hash}`\n"
    if request_id:
        header += f"**Request ID:** `{request_id}`\n"
    header += f"\n## Extracted Text\n\n{extracted_text}"
    return header
