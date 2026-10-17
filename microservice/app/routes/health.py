"""
Health and Status Routes.
Provides service liveness, model inventory, and operational metadata.
"""

from fastapi import APIRouter
from app.config import HAS_GEMINI_KEY, GEMINI_MODEL, LLM_ANALYZE, settings
from app.schemas import HealthResponse
from app.services.model_service import model_service


router = APIRouter(tags=["Health & Status"])


@router.get("/health", response_model=HealthResponse, summary="Service Health Check")
@router.get("/status", response_model=HealthResponse, summary="Operational Status")
async def get_health_status():
    """Retorna el estado operativo del microservicio, modelos cargados y configuración de IA y SMTP."""
    llm_desc = (
        (GEMINI_MODEL if HAS_GEMINI_KEY else "Heuristic Fallback Engine")
        if LLM_ANALYZE
        else "Disabled via LLM_ANALYZE=False"
    )

    return HealthResponse(
        status="operational",
        service_name="DemandMind Retail Demand Forecasting Microservice",
        version="1.3.0",
        models_loaded=model_service.get_loaded_models(),
        llm_analyze_enabled=LLM_ANALYZE,
        llm_multimodal_active=HAS_GEMINI_KEY and LLM_ANALYZE,
        llm_model=llm_desc,
        smtp_enabled=settings.smtp_enabled,
        use_aws_s3=settings.use_aws_s3,
        use_aws_rds=settings.use_aws_rds,
        scalar_docs_url="/docs"
    )
