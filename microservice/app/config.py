"""
Microservice Configuration Module.
Loads environment variables using Pydantic Settings and resolves absolute paths
for models, data, charts, storage, persistent logs, SQL schemas, and templates.
Supports dynamic switching between AWS S3 / Local Storage and AWS RDS / Local PostgreSQL.
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas del proyecto
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "amazon_sales_cleaned.csv"

# Rutas de almacenamiento persistente
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_CHARTS_DIR = STORAGE_DIR / "charts"
STORAGE_REPORTS_DIR = STORAGE_DIR / "reports"
STORAGE_LOGS_DIR = STORAGE_DIR / "logs"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
CHARTS_DIR = BASE_DIR / "static" / "charts"
SQL_DIR = BASE_DIR / "sql"

# Asegurar existencia de directorios de almacenamiento
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(STORAGE_CHARTS_DIR, exist_ok=True)
os.makedirs(STORAGE_REPORTS_DIR, exist_ok=True)
os.makedirs(STORAGE_LOGS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(SQL_DIR, exist_ok=True)

# Archivo de persistencia de logs de peticiones y SQL
REQUESTS_LOG_PATH = STORAGE_LOGS_DIR / "demandmind_requests.log"
SCHEMA_SQL_PATH = SQL_DIR / "schema.sql"


class Settings(BaseSettings):
    """Configuración centralizada validada con Pydantic Settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    base_url: str = "http://localhost:8000"

    # Control de Análisis de Inteligencia Artificial (LLM)
    llm_analyze: bool = True
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"

    # Notificaciones y Alertas Brevo SMTP
    smtp_enabled: bool = False
    smtp_host: str = "smtp-relay.brevo.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_sender_email: str = "security-alert@aigateway.local"
    alert_recipient_email: str = ""
    alert_min_severity: str = "LOW"  # LOW | MEDIUM | HIGH | CRITICAL

    # ==========================================================================
    # Switches de Infraestructura: AWS S3 vs Almacenamiento Local
    # ==========================================================================
    use_aws_s3: bool = False
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_s3_bucket_name: str = "demandmind-charts"
    aws_s3_custom_endpoint: str = ""

    # ==========================================================================
    # Switches de Infraestructura: PostgreSQL AWS RDS vs Local (DBeaver/Docker)
    # ==========================================================================
    use_aws_rds: bool = False
    local_database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/demandmind_db"
    aws_rds_database_url: str = "postgresql+asyncpg://postgres:password_rds@demandmind-db.xyz.us-east-1.rds.amazonaws.com:5432/demandmind_db"

    @property
    def active_database_url(self) -> str:
        """Determina dinámicamente la URL de conexión activa según use_aws_rds."""
        if self.use_aws_rds and self.aws_rds_database_url:
            return self.aws_rds_database_url
        return self.local_database_url

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Devuelve una instancia cacheada de Settings."""
    return Settings()


# Instancia por defecto y constantes de compatibilidad directa
settings = get_settings()

HOST = settings.host
PORT = settings.port
DEBUG = settings.debug
BASE_URL = settings.base_url

LLM_ANALYZE = settings.llm_analyze
GEMINI_API_KEY = settings.gemini_api_key.strip()
GEMINI_MODEL = settings.gemini_model.strip()
HAS_GEMINI_KEY = bool(GEMINI_API_KEY)

USE_AWS_S3 = settings.use_aws_s3
USE_AWS_RDS = settings.use_aws_rds
ACTIVE_DATABASE_URL = settings.active_database_url
