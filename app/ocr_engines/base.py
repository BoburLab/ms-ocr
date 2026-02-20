import numpy as np
from abc import ABC, abstractmethod
import logging
import importlib

logger = logging.getLogger(__name__)

# Registry of available OCR engines: name -> (module_path, class_name)
ENGINE_REGISTRY: dict[str, tuple[str, str]] = {
    "lighton": ("app.ocr_engines.engine_lighton", "LightOnOCREngine"),
    # Add more engines here, e.g.:
    # "chandra": ("app.ocr_engines.engine_chandra", "ChandraOCREngine"),
    # "marker": ("app.ocr_engines.engine_marker", "MarkerOCREngine"),
}

DEFAULT_ENGINE = "lighton"


class OCREngine(ABC):
    """Abstract base class for all OCR engines."""

    @abstractmethod
    def infer(self, image: np.ndarray) -> str:
        """Run OCR inference on a single image and return extracted text."""
        pass


def get_ocr_engine(engine_name: str) -> OCREngine:
    """
    Factory function that lazily imports and instantiates the requested OCR engine.
    
    This avoids loading heavy ML models (torch, transformers) until they are actually needed,
    and keeps each engine isolated in its own module.
    """
    engine_key = engine_name.lower()

    if engine_key not in ENGINE_REGISTRY:
        available = ", ".join(ENGINE_REGISTRY.keys())
        raise ValueError(f"Unknown engine '{engine_name}'. Available engines: {available}")

    module_path, class_name = ENGINE_REGISTRY[engine_key]

    try:
        module = importlib.import_module(module_path)
        engine_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load engine '{engine_name}': {e}")
        raise RuntimeError(f"Could not load engine '{engine_name}': {e}")

    return engine_class()