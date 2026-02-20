import os
import time
import logging

import aiofiles
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import FileResponse
from PIL import Image

from app.models import OCRResponse
from app.utils import (
    validate_file,
    save_upload_file,
    file_to_images,
    build_storage_paths,
    build_markdown_output,
)
from app.preprocessing.base import DefaultPreprocessor
from app.ocr_engines.base import get_ocr_engine

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ocr")
async def process_ocr(
    file: UploadFile = File(...),
    engine: str = Form("lighton"),
) -> FileResponse:
    start_time = time.time()

    # 1. Validate input
    file_ext = validate_file(file)
    paths = build_storage_paths(file.filename, engine)  # type: ignore[arg-type]

    # 2. Save raw file
    await save_upload_file(file, paths["raw_file"])

    # 3. Convert to page images
    images = file_to_images(paths["raw_file"], file_ext)
    logger.info(f"{file.filename}: {len(images)} page(s) detected")

    # 4. Preprocess + OCR per page
    preprocessor = DefaultPreprocessor()
    ocr_engine = get_ocr_engine(engine)
    extracted_texts: list[str] = []

    for i, img in enumerate(images, start=1):
        logger.info(f"Page {i}/{len(images)} — preprocessing...")
        processed_img = preprocessor.process(img)

        # Save preprocessed image
        prep_path = os.path.join(
            paths["preprocessed_dir"],
            f"{paths['base_filename']}_page_{i}.png",
        )
        Image.fromarray(processed_img).save(prep_path)

        logger.info(f"Page {i}/{len(images)} — running OCR...")
        text = ocr_engine.infer(processed_img)
        extracted_texts.append(f"========== PAGE {i} ==========\n{text}")
        logger.info(f"Page {i}/{len(images)} — done")

    final_text = "\n\n".join(extracted_texts).strip()
    processing_time = time.time() - start_time

    # 5. Build response
    response = OCRResponse(
        file_name=file.filename,  # type: ignore[arg-type]
        ocr_model=engine,
        extracted_text=final_text,
        metadata={
            "pages": str(len(images)),
            "processing_time_seconds": f"{processing_time:.2f}",
            "preprocessing_type": type(preprocessor).__name__,
            "warnings": "None",
        },
    )

    # 6. Save markdown output
    md_content = build_markdown_output(
        filename=file.filename,  # type: ignore[arg-type]
        engine=engine,
        processing_time=processing_time,
        page_count=len(images),
        extracted_text=final_text,
    )
    async with aiofiles.open(paths["output_file"], "w") as f:
        await f.write(md_content)

    logger.info(f"Completed {file.filename} | engine={engine} | {processing_time:.2f}s")

    return FileResponse(
        path=paths["output_file"],
        media_type="text/markdown",
        filename=f"{paths['base_filename']}.md",
    )


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}
