import numpy as np
from abc import ABC, abstractmethod
import pytesseract
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class Preprocessor(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        pass

class DefaultPreprocessor(Preprocessor):
    def detect_rotation(self, image: np.ndarray) -> int:
        try:
            pil_img = Image.fromarray(image)
            osd = pytesseract.image_to_osd(pil_img)
            for line in osd.split("\n"):
                if "Rotate" in line:
                    return int(line.split(":")[1].strip())
        except Exception as e:
            logger.warning(f"Rotation detection failed: {e}")
            return 0
        return 0

    def process(self, image: np.ndarray) -> np.ndarray:
        angle = self.detect_rotation(image)
        if angle != 0:
            logger.info(f"Rotating image by {angle} degrees")
            pil_img = Image.fromarray(image)
            # PIL rotate is counter-clockwise. To rotate clockwise by `angle`, we use `-angle`.
            # expand=True ensures the image dimensions are adjusted to fit the rotated image.
            rotated_img = pil_img.rotate(-angle, expand=True)
            return np.array(rotated_img)
        return image
