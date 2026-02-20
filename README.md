# OCR Microservice

Production-ready OCR microservice built with **FastAPI** + **vLLM** + **LightOnOCR-2-1B**.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │            Docker Compose               │
                        │                                         │
  Client ──► :8000      │  ┌──────────────┐    ┌──────────────┐  │
  (PDF/Image upload)    │  │  ocr-service  │───►│  vllm server │  │
                        │  │  (FastAPI)    │    │  (GPU, :8001)│  │
                        │  │  - validation │    │  - model     │  │
                        │  │  - preprocess │    │  - inference │  │
                        │  │  - storage    │    │  - batching  │  │
                        │  └──────────────┘    └──────────────┘  │
                        └─────────────────────────────────────────┘
```

| Service | Vazifasi | GPU kerakmi? |
|---------|----------|:---:|
| **ocr-service** | FastAPI: fayl qabul qilish, preprocessing, natijani saqlash | Yo'q |
| **vllm** | LightOnOCR-2-1B modelni yuklash va inference qilish | Ha |

---

## Loyiha tuzilmasi

```
ms-ocr/
├── app/
│   ├── __init__.py
│   ├── main.py                         # FastAPI entry point
│   ├── routers.py                      # POST /ocr, GET /health endpointlar
│   ├── models.py                       # Pydantic response schema
│   ├── config.py                       # Environment variables (Settings)
│   ├── utils.py                        # Validatsiya, fayl I/O, path builder
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── base.py                     # Preprocessor (burilish aniqlash/to'g'rilash)
│   └── ocr_engines/
│       ├── __init__.py
│       ├── base.py                     # OCREngine abstract class + engine registry
│       └── engine_lighton.py           # LightOnOCR — vLLM orqali inference
├── storage/
│   ├── raw/                            # Yuklangan asl fayllar
│   ├── preprocessed/                   # Preprocessing qilingan rasmlar
│   └── output/                         # OCR natijasi (.md fayllar)
├── .env                                # Environment o'zgaruvchilari
├── .env.example                        # .env namunasi
├── Dockerfile                          # ocr-service uchun Docker image
├── docker-compose.yml                  # Ikki servis: ocr-service + vllm
└── requirements.txt                    # Python kutubxonalari
```

---

## Talablar

- **Docker** va **Docker Compose** o'rnatilgan bo'lishi kerak
- **NVIDIA GPU** (vLLM uchun) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Kamida **8 GB GPU xotirasi** (LightOnOCR-2-1B uchun)

---

## Tez boshlash

### 1. `.env` faylini sozlash

```bash
cp .env.example .env
```

`.env` faylidagi asosiy sozlamalar:

```dotenv
# Storage yo'llari
RAW_STORAGE_PATH=./storage/raw
PREPROCESSED_STORAGE_PATH=./storage/preprocessed
OUTPUT_STORAGE_PATH=./storage/output

# vLLM server
VLLM_BASE_URL=http://vllm:8001
VLLM_MODEL_NAME=lightonai/LightOnOCR-2-1B

# Fayl validatsiyasi
MAX_FILE_SIZE_MB=50

# MLflow (Ixtiyoriy)
MLFLOW_TRACKING_URI=
```

### 2. HuggingFace token (agar kerak bo'lsa)

Agar model gated bo'lsa, tokenni export qiling:

```bash
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxx
```

### 3. Ishga tushirish

```bash
docker-compose up --build
```

> **Eslatma:** Birinchi ishga tushirishda vLLM modelni HuggingFace'dan yuklab oladi (~2-3 GB). Bu 2-5 daqiqa vaqt olishi mumkin. Model `~/.cache/huggingface` papkasiga saqlanadi va keyingi safar qayta yuklanmaydi.

vLLM healthcheck o'tganidan keyin `ocr-service` avtomatik ishga tushadi. Log'larda quyidagini ko'rasiz:

```
vllm-lighton       | INFO: Started server process
ocr-microservice   | INFO: Uvicorn running on http://0.0.0.0:8000
```

### 4. Tekshirish

```bash
# Health check
curl http://localhost:8000/health
# {"status": "healthy"}
```

---

## API

### `POST /ocr`

PDF yoki rasm faylni yuklash va OCR qilish.

**Request:**

| Parametr | Turi | Majburiy | Tavsif |
|----------|------|:---:|--------|
| `file` | File (multipart) | Ha | PDF, PNG, JPG yoki JPEG fayl |
| `engine` | string (form) | Yo'q | OCR engine nomi (standart: `lighton`) |

**cURL misol:**

```bash
# PDF fayl — .md natijani yuklab olish
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@document.pdf" \
  -F "engine=lighton" \
  -o result.md

# Rasm fayl
curl -X POST "http://localhost:8000/ocr" \
  -F "file=@photo.jpg" \
  -o result.md
```

**Python misol:**

```python
import requests

url = "http://localhost:8000/ocr"
files = {"file": open("document.pdf", "rb")}
data = {"engine": "lighton"}

response = requests.post(url, files=files, data=data)

# .md faylni saqlash
with open("result.md", "w") as f:
    f.write(response.text)
```

**Response (200 OK):**

Javob `.md` (Markdown) fayl sifatida qaytadi (`Content-Type: text/markdown`):

```markdown
# OCR Results for document.pdf

**Engine:** lighton
**Processing Time:** 12.34s
**Pages:** 5

## Extracted Text

========== PAGE 1 ==========
Matn...

========== PAGE 2 ==========
Matn...
```

Fayl avtomatik ravishda `storage/output/<engine>/<filename>.md` papkasiga ham saqlanadi.

**Xato javoblari:**

| Status | Sabab |
|--------|-------|
| `400` | Fayl nomi yo'q yoki noto'g'ri format |
| `413` | Fayl hajmi juda katta (standart: 50 MB) |
| `500` | Faylni qayta ishlashda xatolik |

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "healthy"}
```

---

## Pipeline oqimi

Har bir so'rov uchun quyidagi qadamlar bajariladi:

```
1. Fayl validatsiyasi (format, hajm)
         │
2. Raw faylni saqlash → storage/raw/<engine>/<filename>
         │
3. PDF → sahifalarga ajratish (yoki rasm → numpy array)
         │
4. Har bir sahifa uchun:
   ├─ Preprocessing (burilishni aniqlash va to'g'rilash)
   ├─ Preprocessed rasmni saqlash → storage/preprocessed/<engine>/<filename>_page_N.png
   └─ OCR inference (vLLM ga so'rov yuborish)
         │
5. Natijani birlashtirish
         │
6. Markdown faylga saqlash → storage/output/<engine>/<filename>.md
         │
7. JSON response qaytarish
```

---

## Storage tuzilmasi

Faylni `invoice.pdf` nomi bilan `lighton` engine orqali OCR qilganda:

```
storage/
├── raw/
│   └── lighton/
│       └── invoice.pdf                 # Asl yuklangan fayl
├── preprocessed/
│   └── lighton/
│       ├── invoice_page_1.png          # To'g'rilangan 1-sahifa
│       ├── invoice_page_2.png          # To'g'rilangan 2-sahifa
│       └── invoice_page_3.png          # To'g'rilangan 3-sahifa
└── output/
    └── lighton/
        └── invoice.md                  # OCR natijasi (Markdown)
```

---

## Yangi OCR engine qo'shish

Tizim modular — yangi engine qo'shish juda oson.

### 1-qadam: Engine faylini yarating

```python
# app/ocr_engines/engine_chandra.py

import numpy as np
from app.ocr_engines.base import OCREngine


class ChandraOCREngine(OCREngine):
    """Chandra OCR engine."""

    def infer(self, image: np.ndarray) -> str:
        # Sizning inference logikangiz
        ...
        return extracted_text
```

### 2-qadam: Registry'ga qo'shing

```python
# app/ocr_engines/base.py

ENGINE_REGISTRY: dict[str, tuple[str, str]] = {
    "lighton": ("app.ocr_engines.engine_lighton", "LightOnOCREngine"),
    "chandra": ("app.ocr_engines.engine_chandra", "ChandraOCREngine"),  # ← yangi
}
```

Tamom. Endi `engine=chandra` deb so'rov yuborishingiz mumkin.

---

## Environment o'zgaruvchilari

| O'zgaruvchi | Standart qiymati | Tavsif |
|-------------|------------------|--------|
| `RAW_STORAGE_PATH` | `./storage/raw` | Asl fayllar saqlanadigan papka |
| `PREPROCESSED_STORAGE_PATH` | `./storage/preprocessed` | Preprocessed rasmlar papkasi |
| `OUTPUT_STORAGE_PATH` | `./storage/output` | OCR natijalari (.md) papkasi |
| `VLLM_BASE_URL` | `http://vllm:8001` | vLLM server manzili |
| `VLLM_MODEL_NAME` | `lightonai/LightOnOCR-2-1B` | vLLM dagi model nomi |
| `MAX_FILE_SIZE_MB` | `50` | Maksimal fayl hajmi (MB) |
| `MLFLOW_TRACKING_URI` | _(bo'sh)_ | MLflow server manzili (ixtiyoriy) |
| `HUGGING_FACE_HUB_TOKEN` | _(bo'sh)_ | HuggingFace token (gated model uchun) |

---

## Foydali buyruqlar

```bash
# Barcha servislarni ishga tushirish
docker-compose up --build

# Fon rejimida ishga tushirish
docker-compose up --build -d

# Loglarni ko'rish
docker-compose logs -f

# Faqat ocr-service loglarini ko'rish
docker-compose logs -f ocr-service

# Faqat vllm loglarini ko'rish
docker-compose logs -f vllm

# Servislarni to'xtatish
docker-compose down

# Storage'ni tozalash
rm -rf storage/raw/* storage/preprocessed/* storage/output/*
```

---

## Texnologiyalar

| Texnologiya | Maqsad |
|-------------|--------|
| **FastAPI** | Async web framework |
| **vLLM** | Yuqori tezlikdagi LLM inference server |
| **LightOnOCR-2-1B** | OCR model (1B parametr) |
| **Pytesseract** | Preprocessing (burilish aniqlash) |
| **pdf2image** | PDF → rasm konvertatsiya |
| **Pillow** | Rasm qayta ishlash |
| **httpx** | vLLM ga async HTTP so'rovlar |
| **Pydantic** | Ma'lumotlar validatsiyasi va sxema |
| **Docker Compose** | Multi-container orkestrasiya |
