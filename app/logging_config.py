"""
Structured JSON logging configuration.

Every log line is emitted as a single JSON object so it can be ingested
by ELK, Loki, CloudWatch, or any structured-log pipeline.

Extra fields (request_id, client_ip, engine, processing_time …)
are included automatically when passed via ``extra={…}``.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Render each log record as a compact JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # propagate well-known extras
        for attr in (
            "request_id",
            "client_ip",
            "user_id",
            "engine",
            "processing_time",
            "file_hash",
            "pages",
        ):
            val = getattr(record, attr, None)
            if val is not None:
                entry[attr] = val

        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, ensure_ascii=False)


def setup_logging(debug: bool = False) -> None:
    """Replace the root logger's handler with a JSON-emitting one."""
    level = logging.DEBUG if debug else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # reduce noise from chatty libraries
    for name in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
