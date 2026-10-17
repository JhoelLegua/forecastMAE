"""
Database Module for PostgreSQL (AWS RDS & Local).
Configures asynchronous SQLAlchemy engine and session factory using asyncpg.
Defines ORM models for prediction requests, notification recipients, and email audit logs.
Provides connection pooling, health checks, and graceful fallbacks.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional, Dict, Any

from sqlalchemy import (
    Column,
    String,
    Float,
    Numeric,
    Boolean,
    Text,
    DateTime,
    Integer,
    select,
    text
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import declarative_base

from app.config import get_settings

logger = logging.getLogger("demandmind.database")

Base = declarative_base()


# ==============================================================================
# Modelos ORM de SQLAlchemy
# ==============================================================================

class NotificationRecipient(Base):
    """Destinatarios de alertas y notificaciones vía Brevo SMTP."""
    __tablename__ = "notification_recipients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(String(100), nullable=False, default="SUPPLY_CHAIN")
    min_severity = Column(String(20), nullable=False, default="MEDIUM")
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class PredictionRequestRecord(Base):
    """
    Auditoría y registro persistente de cada predicción efectuada.
    IMPORTANTE: No almacena Base64 para evitar saturar el almacenamiento de la base de datos.
    Persiste únicamente el enlace al gráfico (chart_url de S3 o Local).
    """
    __tablename__ = "prediction_requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(String(64), unique=True, nullable=False, index=True)
    request_id = Column(String(64), nullable=True, index=True)
    model_used = Column(String(100), nullable=False, index=True)
    product_category = Column(String(100), nullable=False, index=True)
    predicted_demand = Column(Float, nullable=False)
    demand_tier = Column(String(100), nullable=True)
    category_median = Column(Float, nullable=True)
    relative_vs_median_pct = Column(Float, nullable=True)
    input_parameters = Column(JSONB, nullable=False)
    ai_decision_analysis = Column(Text, nullable=True)
    llm_provider = Column(String(150), nullable=True)
    chart_url = Column(Text, nullable=False)
    storage_backend = Column(String(20), nullable=False, default="LOCAL")  # 'S3' o 'LOCAL'
    execution_time_ms = Column(Float, default=0.0)
    client_ip = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class EmailAuditLog(Base):
    """Historial de auditoría para cada alerta enviada por correo."""
    __tablename__ = "email_audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    recipient_email = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False)
    delivery_status = Column(Boolean, nullable=False, default=True)
    request_id = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    dispatched_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


# ==============================================================================
# Gestión de Motor Asíncrono y Sesiones
# ==============================================================================

_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    """Obtiene o inicializa de forma perezosa el motor SQLAlchemy asíncrono."""
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        db_url = settings.active_database_url
        logger.info("Initializing async database engine with URL: %s (USE_AWS_RDS=%s)", 
                    db_url.split("@")[-1] if "@" in db_url else "configured", 
                    settings.use_aws_rds)
        _engine = create_async_engine(
            db_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )
        _session_factory = async_sessionmaker(
            bind=_engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Retorna la fábrica de sesiones asíncronas."""
    global _session_factory
    if _session_factory is None:
        get_engine()
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Generador de sesión asíncrona inyectable en rutas de FastAPI."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def check_database_connectivity() -> Dict[str, Any]:
    """Verifica si la base de datos PostgreSQL (Local o AWS RDS) está accesible."""
    settings = get_settings()
    target_backend = "AWS RDS (PostgreSQL)" if settings.use_aws_rds else "Local PostgreSQL"
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1;"))
            row = result.scalar()
            return {
                "status": "connected",
                "backend": target_backend,
                "use_aws_rds": settings.use_aws_rds,
                "ping": row == 1,
                "message": f"Conexión exitosa a {target_backend}"
            }
    except Exception as exc:
        return {
            "status": "disconnected",
            "backend": target_backend,
            "use_aws_rds": settings.use_aws_rds,
            "error": str(exc),
            "message": f"No se pudo conectar a {target_backend}. El microservicio operará con persistencia JSON local."
        }


async def init_db_tables() -> bool:
    """Crea las tablas en la base de datos si aún no existen."""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            # Crear extensión UUID si no existe
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables verified/created successfully.")
        return True
    except Exception as exc:
        logger.warning("Could not auto-create tables in database: %s", exc)
        return False
