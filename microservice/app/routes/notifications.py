"""
Notification, SMTP Alerts, and Dynamic Recipients Routes.
Exposes endpoints to test Brevo SMTP connectivity, inspect AWS S3 / RDS status,
manage alert notification recipients in PostgreSQL, dispatch alerts, and inspect persistent request logs.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, BackgroundTasks, Depends, Query, HTTPException, status
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import select, update, delete

from app.config import Settings, get_settings
from app.services.email_notifier import send_alert_email
from app.services.logger_service import get_recent_logs
from app.services.s3_service import check_s3_connectivity
from app.database import (
    get_db_session,
    check_database_connectivity,
    NotificationRecipient
)

router = APIRouter(prefix="/notifications", tags=["Notifications & Cloud Status"])


# ==============================================================================
# Modelos Pydantic para Destinatarios y Alertas
# ==============================================================================

class CreateRecipientRequest(BaseModel):
    """Payload para dar de alta a un destinatario de alertas en PostgreSQL."""
    first_name: str = Field(description="Nombre del destinatario", examples=["Carlos"])
    last_name: str = Field(description="Apellido del destinatario", examples=["Pérez"])
    email: str = Field(description="Correo corporativo o personal", examples=["carlos.perez@empresa.com"])
    role: str = Field(default="SUPPLY_CHAIN", description="Rol (SUPPLY_CHAIN, ADMIN, SOC)", examples=["SUPPLY_CHAIN"])
    min_severity: str = Field(default="MEDIUM", description="Severidad mínima a recibir (LOW, MEDIUM, HIGH, CRITICAL)", examples=["MEDIUM"])
    is_active: bool = Field(default=True, description="Estado activo para recibir alertas")


class RecipientResponse(BaseModel):
    """Respuesta con información del destinatario registrado."""
    id: int
    first_name: str
    last_name: str
    email: str
    role: str
    min_severity: str
    is_active: bool
    created_at: Any


class TestEmailRequest(BaseModel):
    """Payload opcional para probar el envío SMTP."""
    recipient_email: Optional[str] = Field(
        default=None,
        description="Dirección de correo destino opcional. Si se omite, se usa la BD o ALERT_RECIPIENT_EMAIL de .env"
    )
    custom_message: Optional[str] = Field(
        default=None,
        description="Mensaje personalizado opcional para la prueba"
    )


class CustomAlertRequest(BaseModel):
    """Payload para emitir una alerta personalizada de sistema."""
    event_type: str = Field(
        default="extreme_demand_spike",
        description="Identificador del evento o capa",
        examples=["extreme_demand_spike", "system_failure", "prediction_error", "test_alert"]
    )
    severity: str = Field(
        default="MEDIUM",
        description="Nivel de severidad (LOW, MEDIUM, HIGH, CRITICAL)",
        examples=["LOW", "MEDIUM", "HIGH"]
    )
    title: str = Field(
        default="Alerta de Demanda Excepcional Detectada",
        description="Título descriptivo de la alerta"
    )
    reason: str = Field(
        default="La demanda proyectada supera el 400% respecto a la mediana del sector.",
        description="Explicación o diagnóstico principal"
    )
    details: Optional[str] = Field(
        default="Categoría: Laptops | Modelo: LightGBM | Proyección: 499 unidades/mes vs Mediana: 100",
        description="Detalles técnicos o traza de auditoría"
    )
    extra_data: Optional[Dict[str, Any]] = Field(
        default={"SKU Category": "Laptops", "Model": "LightGBM", "Predicted Units": 499.0},
        description="Diccionario con datos contextuales adicionales"
    )
    override_recipient: Optional[str] = Field(
        default=None,
        description="Destinatario específico opcional"
    )


# ==============================================================================
# Endpoints de Monitoreo de Infraestructura y SMTP
# ==============================================================================

@router.get("/status", summary="Get SMTP, S3, and Database Infrastructure Status")
async def get_infrastructure_status(settings: Settings = Depends(get_settings)):
    """
    Retorna el estado integral de la infraestructura:
    - Estado del Relay Brevo SMTP y credenciales ofuscadas.
    - Estado del servicio AWS S3 (Bucket, Región, Modo Cloud vs Local).
    - Estado de la conexión a PostgreSQL (AWS RDS vs Local DBeaver).
    """
    masked_user = ""
    if settings.smtp_user:
        parts = settings.smtp_user.split("@")
        if len(parts) == 2:
            masked_user = f"{parts[0][:3]}***@{parts[1]}"
        else:
            masked_user = f"{settings.smtp_user[:3]}***"

    # Verificación de conectividad S3 y Base de Datos
    s3_status = await check_s3_connectivity(settings)
    db_status = await check_database_connectivity()

    return {
        "smtp": {
            "smtp_enabled": settings.smtp_enabled,
            "smtp_host": settings.smtp_host,
            "smtp_port": settings.smtp_port,
            "smtp_user_masked": masked_user,
            "alert_sender_email": settings.alert_sender_email,
            "alert_recipient_email": settings.alert_recipient_email or "Not configured",
            "alert_min_severity": settings.alert_min_severity,
            "has_password": bool(settings.smtp_password)
        },
        "storage_backend": {
            "use_aws_s3": settings.use_aws_s3,
            "active_mode": "AWS S3 Bucket" if settings.use_aws_s3 else "Local Storage (storage/charts/)",
            "details": s3_status
        },
        "database_backend": {
            "use_aws_rds": settings.use_aws_rds,
            "active_mode": "AWS RDS PostgreSQL" if settings.use_aws_rds else "Local PostgreSQL (DBeaver/Docker)",
            "details": db_status
        }
    }


@router.post("/test", summary="Send Test Alert Email via Brevo SMTP")
async def test_smtp_notification(
    payload: Optional[TestEmailRequest] = None,
    settings: Settings = Depends(get_settings)
):
    """
    Envía inmediatamente un correo de prueba a través de Brevo SMTP Relay.
    Resuelve destinatarios dinámicamente desde PostgreSQL (o fallback del .env).
    """
    target = payload.recipient_email if payload else None
    user_msg = payload.custom_message if (payload and payload.custom_message) else "Conexión de relay validada con éxito."

    success = await send_alert_email(
        settings=settings,
        event_type="test_alert",
        severity="LOW",
        title="Validación de Conectividad Brevo SMTP Relay",
        reason=user_msg,
        details="Este mensaje confirma que el servicio de alertas vía Brevo SMTP en DemandMind está activo y operando con STARTTLS.",
        extra_data={"Relay Host": settings.smtp_host, "Relay Port": str(settings.smtp_port), "Status": "Connected"},
        override_recipient=target
    )

    if not success:
        recipient = target or settings.alert_recipient_email
        raise HTTPException(
            status_code=502,
            detail=f"Fallo al enviar correo vía Brevo SMTP a '{recipient}'. Revisa si SMTP_ENABLED=True, credenciales y logs."
        )

    return {
        "status": "success",
        "message": f"Alerta de prueba enviada exitosamente vía Brevo SMTP Relay.",
        "smtp_host": settings.smtp_host,
        "severity": "LOW"
    }


@router.post("/alert", summary="Dispatch Operational Alert (Background)")
async def dispatch_custom_alert(
    payload: CustomAlertRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings)
):
    """
    Programa el envío no bloqueante de una alerta operativa
    utilizando FastAPI BackgroundTasks, garantizando latencia cero en la respuesta.
    """
    background_tasks.add_task(
        send_alert_email,
        settings=settings,
        event_type=payload.event_type,
        severity=payload.severity,
        title=payload.title,
        reason=payload.reason,
        details=payload.details,
        extra_data=payload.extra_data,
        override_recipient=payload.override_recipient
    )

    return {
        "status": "queued",
        "message": f"Alerta '{payload.title}' programada para despacho en segundo plano.",
        "severity": payload.severity,
        "event_type": payload.event_type
    }


# ==============================================================================
# Endpoints de Gestión Dinámica de Destinatarios (PostgreSQL RDS / Local)
# ==============================================================================

@router.get("/recipients", response_model=List[RecipientResponse], summary="List Notification Recipients (DB)")
async def list_recipients(active_only: bool = Query(default=False)):
    """
    Consulta los destinatarios de correo configurados en la tabla PostgreSQL notification_recipients.
    Si la base de datos no está disponible, retorna error informativo.
    """
    try:
        from app.database import get_session_factory
        factory = get_session_factory()
        async with factory() as session:
            stmt = select(NotificationRecipient)
            if active_only:
                stmt = stmt.where(NotificationRecipient.is_active.is_(True))
            result = await session.execute(stmt)
            recipients = result.scalars().all()
            return [
                RecipientResponse(
                    id=r.id,
                    first_name=r.first_name,
                    last_name=r.last_name,
                    email=r.email,
                    role=r.role,
                    min_severity=r.min_severity,
                    is_active=r.is_active,
                    created_at=r.created_at
                )
                for r in recipients
            ]
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo conectar a la base de datos PostgreSQL: {str(exc)}. Verifica tu conexión local o en AWS RDS."
        )


@router.post("/recipients", response_model=RecipientResponse, status_code=status.HTTP_201_CREATED, summary="Add Notification Recipient (DB)")
async def create_recipient(payload: CreateRecipientRequest):
    """Agrega un nuevo destinatario a la tabla notification_recipients en PostgreSQL."""
    try:
        from app.database import get_session_factory
        factory = get_session_factory()
        async with factory() as session:
            # Verificar duplicados
            existing = await session.execute(
                select(NotificationRecipient).where(NotificationRecipient.email == payload.email)
            )
            if existing.scalar_one_or_none():
                raise HTTPException(status_code=400, detail=f"El correo '{payload.email}' ya está registrado.")

            recipient = NotificationRecipient(
                first_name=payload.first_name,
                last_name=payload.last_name,
                email=payload.email,
                role=payload.role,
                min_severity=payload.min_severity.upper(),
                is_active=payload.is_active
            )
            session.add(recipient)
            await session.commit()
            await session.refresh(recipient)

            return RecipientResponse(
                id=recipient.id,
                first_name=recipient.first_name,
                last_name=recipient.last_name,
                email=recipient.email,
                role=recipient.role,
                min_severity=recipient.min_severity,
                is_active=recipient.is_active,
                created_at=recipient.created_at
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al registrar destinatario en BD: {str(exc)}")


@router.patch("/recipients/{recipient_id}/toggle", summary="Toggle Recipient Active Status (DB)")
async def toggle_recipient(recipient_id: int):
    """Activa o desactiva un destinatario para pausar o reanudar el envío de alertas sin borrarlo."""
    try:
        from app.database import get_session_factory
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(NotificationRecipient).where(NotificationRecipient.id == recipient_id)
            )
            recipient = result.scalar_one_or_none()
            if not recipient:
                raise HTTPException(status_code=404, detail="Destinatario no encontrado")

            recipient.is_active = not recipient.is_active
            await session.commit()
            return {
                "id": recipient.id,
                "email": recipient.email,
                "is_active": recipient.is_active,
                "message": f"Estado de {recipient.email} actualizado a {'Activo' if recipient.is_active else 'Inactivo'}"
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al alternar estado en BD: {str(exc)}")


@router.delete("/recipients/{recipient_id}", summary="Delete Recipient (DB)")
async def delete_recipient(recipient_id: int):
    """Elimina definitivamente un destinatario de la base de datos."""
    try:
        from app.database import get_session_factory
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                select(NotificationRecipient).where(NotificationRecipient.id == recipient_id)
            )
            recipient = result.scalar_one_or_none()
            if not recipient:
                raise HTTPException(status_code=404, detail="Destinatario no encontrado")

            await session.delete(recipient)
            await session.commit()
            return {"status": "deleted", "id": recipient_id, "email": recipient.email}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al eliminar destinatario en BD: {str(exc)}")


# ==============================================================================
# Endpoint de Inspección de Logs Persistentes
# ==============================================================================

@router.get("/logs", summary="Inspect Persistent Request Logs")
async def view_persistent_logs(
    limit: int = Query(default=50, ge=1, le=500, description="Número de líneas a consultar")
):
    """Devuelve las líneas más recientes del log persistente de peticiones (storage/logs/demandmind_requests.log)."""
    lines = get_recent_logs(lines_count=limit)
    return {
        "lines_count": len(lines),
        "log_file": "storage/logs/demandmind_requests.log",
        "logs": lines
    }
