from pydantic import BaseModel
from typing import Dict

class OCRResponse(BaseModel):
    file_name: str
    ocr_model: str
    extracted_text: str
    metadata: Dict[str, str]
