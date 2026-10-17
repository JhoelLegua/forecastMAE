"""
Logger Service Module for DemandMind.
Sets up unified persistent logging to storage/logs/demandmind_requests.log
and provides helper functions for request lifecycle auditing and failure tracing.
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from app.config import REQUESTS_LOG_PATH, STORAGE_LOGS_DIR

# Asegurar existencia del directorio de logs
STORAGE_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Nombre del logger central
LOGGER_NAME = "demandmind.access"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)

# Evitar duplicación de handlers si se recarga el módulo
if not logger.handlers:
    # 1. File Handler con rotación (10 MB máx por archivo, 5 backups, UTF-8)
    file_handler = RotatingFileHandler(
        filename=str(REQUESTS_LOG_PATH),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 2. Console Handler estándar
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)


def log_request_flow(
    request_id: str,
    method: str,
    path: str,
    client_ip: str,
    status_code: int,
    duration_ms: float,
    error_message: Optional[str] = None,
    extra_details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Registra persistentemente una petición en el archivo de log.
    En casos de falla (status >= 400), emite nivel WARNING o ERROR con detalles.
    """
    status_label = "OK" if status_code < 400 else "ERROR"
    msg = (
        f"[REQ:{request_id}] Client:{client_ip} | {method} {path} | "
        f"Status:{status_code} ({status_label}) | Latency:{duration_ms:.1f}ms"
    )

    if extra_details:
        msg += f" | Info:{extra_details}"

    if error_message or status_code >= 500:
        msg += f" | ErrorDetails:{error_message}"
        logger.error(msg)
    elif status_code >= 400:
        if error_message:
            msg += f" | WarnDetails:{error_message}"
        logger.warning(msg)
    else:
        logger.info(msg)


def get_recent_logs(lines_count: int = 100) -> list[str]:
    """Lee las últimas N líneas del archivo de logs persistente."""
    if not REQUESTS_LOG_PATH.exists():
        return ["No persistent log entries found yet."]

    try:
        with open(REQUESTS_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-lines_count:]]
    except Exception as e:
        return [f"Error reading persistent log file: {str(e)}"]
