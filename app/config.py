from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    RAW_STORAGE_PATH: str = os.getenv("RAW_STORAGE_PATH", "./storage/raw")
    PREPROCESSED_STORAGE_PATH: str = os.getenv("PREPROCESSED_STORAGE_PATH", "./storage/preprocessed")
    OUTPUT_STORAGE_PATH: str = os.getenv("OUTPUT_STORAGE_PATH", "./storage/output")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "")

    # vLLM
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://vllm:8001")
    VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "lightonai/LightOnOCR-2-1B")

    # File validation
    ALLOWED_EXTENSIONS: set[str] = {"pdf", "png", "jpg", "jpeg"}
    MAX_FILE_SIZE_MB: int = 50

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure storage directories exist
for path in (settings.RAW_STORAGE_PATH, settings.PREPROCESSED_STORAGE_PATH, settings.OUTPUT_STORAGE_PATH):
    os.makedirs(path, exist_ok=True)
