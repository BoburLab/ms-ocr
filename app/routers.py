"""
API endpoints — hardened.

- UUID-based internal filenames (client filename is never trusted)
- Full file-security validation (MIME, size, malicious scan, hash)
- Structured logging with correlation ID
- Optional API-key gate
"""

import os
import time
import logging

import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, Request, Depends
from fastapi.responses import FileResponse
from PIL import Image

from app.security import validate_file_security, generate_secure_filename, verify_api_key
from app.utils import file_to_images, build_storage_paths, build_markdown_output
from app.preprocessing.base import DefaultPreprocessor
from app.ocr_engines.base import get_ocr_engine

router = APIRouter()
logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/ocr", dependencies=[Depends(verify_api_key)])
async def process_ocr(
    request: Request,
    file: UploadFile = File(...),
    engine: str = Form("lighton"),
) -> FileResponse:
    rid = _request_id(request)
    ip = _client_ip(request)
    start = time.time()

    # 1 — comprehensive file validation
    ext, content, file_hash = await validate_file_security(file)

    # 2 — UUID-based secure filename
    secure_name, original_name = generate_secure_filename(file.filename)  # type: ignore[arg-type]
    paths = build_storage_paths(secure_name, engine)

    # 3 — persist raw file
    os.makedirs(os.path.dirname(paths["raw_file"]), exist_ok=True)
    async with aiofiles.open(paths["raw_file"], "wb") as f:
        await f.write(content)

    logger.info(
        "File accepted",
        extra={
            "request_id": rid,
            "client_ip": ip,
            "engine": engine,
            "file_hash": file_hash,
        },
    )

    # 4 — convert to page images
    images = file_to_images(paths["raw_file"], ext)
    page_count = len(images)
    logger.info(
        "%d page(s) detected",
        page_count,
        extra={"request_id": rid},
    )

    # 5 — preprocess + OCR
    preprocessor = DefaultPreprocessor()
    ocr_engine = get_ocr_engine(engine)
    texts: list[str] = []

    for i, img in enumerate(images, 1):
        logger.info("Page %d/%d — preprocessing", i, page_count, extra={"request_id": rid})
        processed = preprocessor.process(img)

        prep_path = os.path.join(
            paths["preprocessed_dir"],
            f"{paths['base_filename']}_page_{i}.png",
        )
        Image.fromarray(processed).save(prep_path)

        logger.info("Page %d/%d — OCR", i, page_count, extra={"request_id": rid})
        text = ocr_engine.infer(processed)
        texts.append(f"========== PAGE {i} ==========\n{text}")

    final_text = "\n\n".join(texts).strip()
    elapsed = time.time() - start

    # 6 — save markdown
    md = build_markdown_output(
        filename=original_name,
        engine=engine,
        processing_time=elapsed,
        page_count=page_count,
        extracted_text=final_text,
        file_hash=file_hash,
        request_id=rid,
    )
    async with aiofiles.open(paths["output_file"], "w") as f:
        await f.write(md)

    logger.info(
        "OCR completed",
        extra={
            "request_id": rid,
            "client_ip": ip,
            "engine": engine,
            "processing_time": f"{elapsed:.2f}",
            "pages": page_count,
            "file_hash": file_hash,
        },
    )

    return FileResponse(
        path=paths["output_file"],
        media_type="text/markdown",
        filename=f"{original_name.rsplit('.', 1)[0]}.md",
    )


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}
