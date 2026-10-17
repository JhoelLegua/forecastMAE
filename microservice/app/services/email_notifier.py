"""
Email Notifier Service via Brevo SMTP Relay.
Implements non-blocking, asynchronous email dispatch for system alerts,
critical forecast anomalies, service failures, and administrative notifications.
Follows fault-isolation principles to never block or interrupt the microservice flow.
"""

import ssl
import smtplib
import asyncio
import logging
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List, Dict, Any

from app.config import Settings, get_settings

logger = logging.getLogger("demandmind.smtp")

# Orden jerárquico de severidades
_SEVERITY_ORDER = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3
}

# Mapeo de severidad por tipo de evento / capa
_EVENT_SEVERITY = {
    "system_failure": "HIGH",
    "prediction_error": "HIGH",
    "extreme_demand_spike": "MEDIUM",
    "llm_fallback_warning": "LOW",
    "service_startup": "LOW",
    "test_alert": "LOW"
}


def _get_active_recipients(settings: Settings, override_recipient: Optional[str] = None) -> List[str]:
    """
    Resuelve los destinatarios activos para la alerta.
    Prioriza override_recipient si se proporciona, luego settings.alert_recipient_email.
    """
    recipients = []
    if override_recipient and override_recipient.strip():
        recipients.append(override_recipient.strip())
    elif settings.alert_recipient_email and settings.alert_recipient_email.strip():
        recipients.append(settings.alert_recipient_email.strip())

    return recipients


def _build_html_email(
    event_type: str,
    severity: str,
    title: str,
    reason: str,
    details: Optional[str] = None,
    request_id: Optional[str] = None,
    client_ip: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None
) -> str:
    """Construye un correo HTML profesional, responsivo y visualmente estructurado."""
    severity_colors = {
        "LOW": ("#3b82f6", "#eff6ff"),
        "MEDIUM": ("#f59e0b", "#fffbeb"),
        "HIGH": ("#ef4444", "#fef2f2"),
        "CRITICAL": ("#b91c1c", "#fef2f2")
    }
    badge_border, badge_bg = severity_colors.get(severity.upper(), ("#64748b", "#f8fafc"))
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    extra_rows = ""
    if extra_data:
        for k, v in extra_data.items():
            extra_rows += f"""
            <tr>
              <td style="padding: 6px 12px; font-weight: 600; color: #475569; border-bottom: 1px solid #e2e8f0;">{k}</td>
              <td style="padding: 6px 12px; font-family: monospace; color: #0f172a; border-bottom: 1px solid #e2e8f0;">{v}</td>
            </tr>
            """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>{title}</title>
    </head>
    <body style="margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f1f5f9; color: #0f172a;">
      <div style="max-width: 640px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;">
        
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #0f172a, #1e293b); padding: 20px 28px; color: #ffffff;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 18px; font-weight: 800; letter-spacing: -0.02em; color: #38bdf8;">DemandMind</span>
            <span style="font-size: 12px; background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; padding: 3px 8px; border-radius: 12px; font-weight: 600;">
              Alerta de Sistema
            </span>
          </div>
          <h2 style="margin: 12px 0 4px 0; font-size: 20px; font-weight: 700; color: #ffffff;">{title}</h2>
          <p style="margin: 0; font-size: 13px; color: #94a3b8;">Timestamp: {timestamp_str}</p>
        </div>

        <!-- Body Content -->
        <div style="padding: 24px 28px;">
          <!-- Severity and Event Banner -->
          <div style="display: flex; gap: 12px; margin-bottom: 20px;">
            <span style="display: inline-block; padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 700; border: 1px solid {badge_border}; background-color: {badge_bg}; color: {badge_border};">
              Severidad: {severity.upper()}
            </span>
            <span style="display: inline-block; padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 600; background: #f8fafc; border: 1px solid #cbd5e1; color: #475569;">
              Evento: {event_type}
            </span>
          </div>

          <!-- Motivo Principal -->
          <div style="background-color: #f8fafc; border-left: 4px solid {badge_border}; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #1e293b; font-weight: 500;">
              <strong>Diagnóstico:</strong> {reason}
            </p>
          </div>

          {f'<div style="margin-bottom: 20px;"><strong style="font-size: 13px; color: #64748b; text-transform: uppercase;">Detalles Técnicos:</strong><pre style="background: #0f172a; color: #e2e8f0; padding: 12px 16px; border-radius: 8px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; font-family: monospace; margin-top: 6px;">{details}</pre></div>' if details else ''}

          <!-- Context Data Table -->
          <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 24px;">
            <tbody>
              <tr>
                <td style="padding: 6px 12px; font-weight: 600; color: #475569; border-bottom: 1px solid #e2e8f0; width: 35%;">Request ID</td>
                <td style="padding: 6px 12px; font-family: monospace; color: #0f172a; border-bottom: 1px solid #e2e8f0;">{request_id or 'N/A'}</td>
              </tr>
              <tr>
                <td style="padding: 6px 12px; font-weight: 600; color: #475569; border-bottom: 1px solid #e2e8f0;">Client IP</td>
                <td style="padding: 6px 12px; font-family: monospace; color: #0f172a; border-bottom: 1px solid #e2e8f0;">{client_ip or 'Local / Internal'}</td>
              </tr>
              {extra_rows}
            </tbody>
          </table>

          <!-- Footer Note -->
          <p style="font-size: 12px; color: #94a3b8; margin: 0; border-top: 1px solid #e2e8f0; padding-top: 14px;">
            Mensaje generado automáticamente por el microservicio DemandMind a través del Relay SMTP de Brevo.
          </p>
        </div>

      </div>
    </body>
    </html>
    """


def _send_smtp_blocking(
    settings: Settings,
    recipients: List[str],
    message_str: str,
    event_type: str,
) -> bool:
    """
    Ejecuta el despacho SMTP de forma síncrona/bloqueante en un hilo dedicado.
    Maneja el aislamiento total de excepciones para proteger la disponibilidad del servicio.
    """
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(
                settings.alert_sender_email,
                recipients,
                message_str.encode("utf-8")
            )
        logger.info(
            "Security/System alert email sent via Brevo to %d recipient(s) for event '%s'.",
            len(recipients),
            event_type
        )
        return True
    except Exception as exc:
        # Aislamiento de fallos: nunca propaga error ni interrumpe el microservicio
        logger.error("Failed to send alert email via Brevo SMTP: %s", str(exc))
        return False


async def send_alert_email(
    settings: Optional[Settings] = None,
    event_type: str = "system_failure",
    severity: str = "MEDIUM",
    title: str = "Alerta del Sistema DemandMind",
    reason: str = "Evento registrado",
    details: Optional[str] = None,
    request_id: Optional[str] = None,
    client_ip: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    override_recipient: Optional[str] = None
) -> bool:
    """
    Punto de entrada principal asíncrono y no bloqueante para el envío de alertas.
    Valida habilitación, filtra por severidad mínima y despacha a través de asyncio.to_thread.
    """
    cfg = settings or get_settings()

    # 1. Switch maestro
    if not cfg.smtp_enabled:
        logger.debug("SMTP is disabled in settings. Skipping alert '%s'.", event_type)
        return False

    # 2. Filtro por severidad mínima
    event_sev = severity.upper()
    min_sev = cfg.alert_min_severity.upper()

    event_level = _SEVERITY_ORDER.get(event_sev, 1)
    min_level = _SEVERITY_ORDER.get(min_sev, 1)

    if event_level < min_level:
        logger.debug(
            "Alert '%s' severity (%s) is lower than configured minimum threshold (%s). Skipped.",
            event_type, event_sev, min_sev
        )
        return False

    # 3. Resolución de destinatarios
    recipients = _get_active_recipients(cfg, override_recipient)
    if not recipients:
        logger.warning("No active recipient configured for SMTP alert. Notification aborted.")
        return False

    # 4. Construcción del mensaje MIME multipart
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[{event_sev}] {title}"
    msg["From"] = cfg.alert_sender_email
    msg["To"] = ", ".join(recipients)

    text_plain = f"""DemandMind System Alert
Severidad: {event_sev}
Evento: {event_type}
Título: {title}
Motivo: {reason}
Request ID: {request_id or 'N/A'}
Timestamp: {datetime.now(timezone.utc).isoformat()}
Detalles: {details or 'N/A'}
"""
    html_content = _build_html_email(
        event_type=event_type,
        severity=event_sev,
        title=title,
        reason=reason,
        details=details,
        request_id=request_id,
        client_ip=client_ip,
        extra_data=extra_data
    )

    msg.attach(MIMEText(text_plain, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    # 5. Despacho no bloqueante delegado al worker pool de hilos
    return await asyncio.to_thread(
        _send_smtp_blocking,
        settings=cfg,
        recipients=recipients,
        message_str=msg.as_string(),
        event_type=event_type
    )


async def send_security_alert(
    settings: Settings,
    user_id: str = "system",
    session_id: str = "default",
    layer: str = "system_layer",
    reason: str = "Alerta detectada",
    prompt: Optional[str] = None,
    score: Optional[float] = None,
) -> None:
    """
    Firma compatible con la guía de arquitectura compartida por el usuario.
    Permite invocar alertas con parámetros equivalentes a los del AI Gateway.
    """
    severity = _EVENT_SEVERITY.get(layer, "HIGH")
    extra = {}
    if user_id: extra["User ID"] = user_id
    if session_id: extra["Session ID"] = session_id
    if score is not None: extra["Confidence Score"] = f"{score:.4f}"

    await send_alert_email(
        settings=settings,
        event_type=layer,
        severity=severity,
        title=f"Alerta de Seguridad/Integridad en {layer}",
        reason=reason,
        details=prompt,
        extra_data=extra
    )
