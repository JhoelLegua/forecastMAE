"""
Notification and System Alert Routes.
Exposes endpoints to test Brevo SMTP connectivity, inspect email relay status,
dispatch custom or simulated operational alerts, and inspect persistent request logs.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, BackgroundTasks, Depends, Query, HTTPException
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.services.email_notifier import send_alert_email
from app.services.logger_service import get_recent_logs

router = APIRouter(prefix="/notifications", tags=["Notifications & SMTP Alerts"])


class TestEmailRequest(BaseModel):
    """Payload opcional para probar el envío SMTP."""
    recipient_email: Optional[str] = Field(
        default=None,
        description="Dirección de correo destino opcional. Si se omite, se usa ALERT_RECIPIENT_EMAIL de .env"
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


@router.get("/status", summary="Get SMTP Relay Configuration Status")
async def get_smtp_status(settings: Settings = Depends(get_settings)):
    """Retorna el estado de configuración del servicio de correo Brevo SMTP (credenciales ofuscadas)."""
    masked_user = ""
    if settings.smtp_user:
        parts = settings.smtp_user.split("@")
        if len(parts) == 2:
            masked_user = f"{parts[0][:3]}***@{parts[1]}"
        else:
            masked_user = f"{settings.smtp_user[:3]}***"

    return {
        "smtp_enabled": settings.smtp_enabled,
        "smtp_host": settings.smtp_host,
        "smtp_port": settings.smtp_port,
        "smtp_user_masked": masked_user,
        "alert_sender_email": settings.alert_sender_email,
        "alert_recipient_email": settings.alert_recipient_email or "Not configured",
        "alert_min_severity": settings.alert_min_severity,
        "has_password": bool(settings.smtp_password)
    }


@router.post("/test", summary="Send Test Alert Email via Brevo SMTP")
async def test_smtp_notification(
    payload: Optional[TestEmailRequest] = None,
    settings: Settings = Depends(get_settings)
):
    """
    Envía inmediatamente un correo de prueba a través de Brevo SMTP Relay.
    Permite validar las credenciales de Brevo y verificar la entrega en bandeja de entrada.
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
            detail=f"Fallo al enviar correo vía Brevo SMTP a '{recipient}'. Revisa si SMTP_ENABLED=True, tus credenciales y logs."
        )

    return {
        "status": "success",
        "message": f"Alerta de prueba enviada exitosamente vía Brevo SMTP Relay a '{target or settings.alert_recipient_email}'.",
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
    Programa el envío no bloqueante de una alerta operativa o de seguridad
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
