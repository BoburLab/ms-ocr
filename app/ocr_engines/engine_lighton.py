import base64
import io
import logging

import httpx
import numpy as np
from PIL import Image

from app.ocr_engines.base import OCREngine
from app.config import settings

logger = logging.getLogger(__name__)

VLLM_TIMEOUT = 120.0  # seconds per request


class LightOnOCREngine(OCREngine):
    """
    LightOn OCR 2-1B engine via vLLM OpenAI-compatible API.

    Expects a running vLLM server at VLLM_BASE_URL.
    vLLM handles model loading, GPU memory, batching, and inference.
    """

    _instance = None

    def __new__(cls) -> "LightOnOCREngine":
        if cls._instance is None:
            cls._instance = super(LightOnOCREngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._base_url = settings.VLLM_BASE_URL.rstrip("/")
        self._model_name = settings.VLLM_MODEL_NAME
        self._client = httpx.Client(timeout=VLLM_TIMEOUT)
        self._initialized = True
        logger.info(f"LightOnOCREngine ready (vLLM @ {self._base_url}, model={self._model_name})")

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numpy_to_base64(image: np.ndarray) -> str:
        """Convert numpy image array to base64 PNG string."""
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def infer(self, image: np.ndarray) -> str:
        image_b64 = self._numpy_to_base64(image)

        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 4000,
            "temperature": 0.0,
        }

        try:
            response = self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM returned {e.response.status_code}: {e.response.text}")
            return f"[vLLM error: {e.response.status_code}]"
        except httpx.RequestError as e:
            logger.error(f"vLLM connection failed: {e}")
            return f"[vLLM connection error: {str(e)}]"
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected vLLM response format: {e}")
            return f"[vLLM response parse error: {str(e)}]"
