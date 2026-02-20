FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Python, OpenCV, pdf2image, and Tesseract
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

# Create storage directories
RUN mkdir -p /app/storage/raw /app/storage/preprocessed /app/storage/output

ENV RAW_STORAGE_PATH=/app/storage/raw
ENV PREPROCESSED_STORAGE_PATH=/app/storage/preprocessed
ENV OUTPUT_STORAGE_PATH=/app/storage/output

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
