"""
DemandMind Microservice Main Application.
Configures FastAPI, mounts modern interactive Scalar API Documentation at /docs,
serves static & persistent storage charts, persistent request logging middleware,
Brevo SMTP alert notifications, and registers health, prediction, notifications, and interactive report routes.
"""

import os
import time
import uuid
import asyncio
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from scalar_fastapi import get_scalar_api_reference

from app.config import (
    CHARTS_DIR,
    STORAGE_CHARTS_DIR,
    STORAGE_LOGS_DIR,
    HOST,
    PORT,
    get_settings
)
from app.routes.health import router as health_router
from app.routes.prediction import router as prediction_router
from app.routes.reports import router as reports_router
from app.routes.notifications import router as notifications_router
from app.services.logger_service import log_request_flow
from app.services.email_notifier import send_alert_email


app = FastAPI(
    title="DemandMind API - Retail Demand Forecasting & AI Intelligence",
    description="""
    Microservicio empresarial de Machine Learning e Inteligencia Artificial Multimodal para estimación 
    de demanda mensual de productos en Amazon según categoría.
    
    ### Características Principales:
    * **Modelos Supervisados:** LightGBM, Random Forest, XGBoost, Decision Tree y Ridge Regression.
    * **Control de Análisis LLM:** Flag configurable `LLM_ANALYZE = true/false` para activar/desactivar la interpretación del LLM.
    * **Trazabilidad Persistente:** Flujo completo de peticiones registrado en `storage/logs/demandmind_requests.log`.
    * **Servicio de Alertas Brevo SMTP:** Despacho asíncrono no bloqueante de alertas ante fallas, anomalías de demanda y eventos críticos.
    * **Persistencia Local:** Activos gráficos y reportes almacenados en `storage/charts/` y `storage/reports/`.
    * **Vistas Interactivas:** Reporte web interactivo con selector de tema (Oscuro/Claro) y descarga de PDF respetando el estilo.
    * **Documentación Interactiva:** Renderizada con **Scalar API Reference** en `/docs`.
    """,
    version="1.2.0",
    docs_url=None,   # Desactiva Swagger UI por defecto para dar paso a Scalar
    redoc_url=None  # Desactiva ReDoc por defecto
)

# Configuración CORS para permitir consumo desde aplicaciones web y dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Middleware de Trazabilidad y Persistencia de Logs del Flujo de Petición
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """
    Registra persistentemente cada petición con su ID, método, ruta, IP, código HTTP y latencia.
    En caso de excepción no controlada (500), almacena la traza completa y dispara una alerta Brevo SMTP.
    """
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    request.state.request_id = request_id
    client_ip = request.client.host if request.client else "unknown"
    start_time = time.time()

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000.0
        response.headers["X-Request-ID"] = request_id

        log_request_flow(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
            status_code=response.status_code,
            duration_ms=duration_ms
        )
        return response

    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000.0
        tb_str = traceback.format_exc()

        log_request_flow(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
            status_code=500,
            duration_ms=duration_ms,
            error_message=f"{type(exc).__name__}: {str(exc)}",
            extra_details={"traceback_tail": tb_str[-300:]}
        )

        # Disparar alerta de falla crítica vía Brevo SMTP de manera no bloqueante
        settings = get_settings()
        if settings.smtp_enabled:
            asyncio.create_task(
                send_alert_email(
                    settings=settings,
                    event_type="system_failure",
                    severity="HIGH",
                    title=f"Error 500 Crítico en {request.method} {request.url.path}",
                    reason=f"Excepción en petición: {type(exc).__name__}: {str(exc)}",
                    details=tb_str,
                    request_id=request_id,
                    client_ip=client_ip,
                    extra_data={"HTTP Method": request.method, "URL Path": request.url.path}
                )
            )

        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal Server Error",
                "request_id": request_id,
                "error": str(exc)
            },
            headers={"X-Request-ID": request_id}
        )


# Servir archivos de almacenamiento persistente y estáticos
os.makedirs(STORAGE_CHARTS_DIR, exist_ok=True)
os.makedirs(STORAGE_LOGS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
app.mount("/storage/charts", StaticFiles(directory=str(STORAGE_CHARTS_DIR)), name="storage_charts")
app.mount("/static/charts", StaticFiles(directory=str(CHARTS_DIR)), name="static_charts")

# Registrar Routers
app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(reports_router)
app.include_router(notifications_router)


# Endpoint de documentación interactiva moderna con Scalar
@app.get("/docs", include_in_schema=False)
async def scalar_html():
    """Sirve la interfaz interactiva moderna de documentación API de Scalar."""
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="DemandMind - Scalar API Reference",
        theme="moon",
        hide_download_button=False
    )


# Redirección de la raíz / hacia /docs
@app.get("/", include_in_schema=False)
async def root():
    """Redirige automáticamente la raíz a la documentación interactiva en Scalar."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    print(f"[DEMANDMIND] Starting microservice on http://{HOST}:{PORT}")
    print(f"[DEMANDMIND] Scalar API Documentation available at: http://{HOST}:{PORT}/docs")
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
