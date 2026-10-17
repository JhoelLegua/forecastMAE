"""
Storage Service Module with Dual Cloud / Local Architecture.
- Storage Backend: AWS S3 (when USE_AWS_S3=true) or Local storage/charts/ (when USE_AWS_S3=false).
- Database Backend: PostgreSQL (AWS RDS or Local DBeaver) for structured queryable persistence
  without base64 bloat, accompanied by local JSON document mirroring.
"""

import os
import json
import uuid
import shutil
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from app.config import (
    STORAGE_DIR,
    STORAGE_CHARTS_DIR,
    STORAGE_REPORTS_DIR,
    BASE_URL,
    get_settings
)
from app.services.s3_service import _upload_file_blocking

logger = logging.getLogger("demandmind.storage")


def _persist_to_db_blocking(report_document: Dict[str, Any]) -> None:
    """Inserta síncronamente el registro de predicción en PostgreSQL sin almacenar Base64."""
    try:
        from app.database import get_session_factory, PredictionRequestRecord

        async def _async_insert():
            try:
                factory = get_session_factory()
                async with factory() as session:
                    record = PredictionRequestRecord(
                        report_id=report_document["report_id"],
                        request_id=report_document.get("request_id"),
                        model_used=report_document["model_used"],
                        product_category=report_document["output_data"].get("category", "General"),
                        predicted_demand=float(report_document["output_data"].get("predicted_monthly_demand") or 
                                               report_document["output_data"].get("ensemble_median_demand") or 0.0),
                        demand_tier=report_document["output_data"].get("demand_tier", "Standard"),
                        category_median=float(report_document["output_data"].get("category_median_demand") or 0.0),
                        relative_vs_median_pct=float(report_document["output_data"].get("relative_vs_category_pct") or 0.0),
                        input_parameters=report_document["input_data"],
                        ai_decision_analysis=report_document.get("ai_decision_analysis"),
                        llm_provider=report_document.get("llm_provider"),
                        chart_url=report_document["chart_url"],
                        storage_backend=report_document.get("storage_backend", "LOCAL"),
                        execution_time_ms=report_document.get("metadata", {}).get("execution_time_ms", 0.0),
                        client_ip=report_document.get("client_ip")
                    )
                    session.add(record)
                    await session.commit()
                    logger.info("Prediction successfully saved to PostgreSQL (%s)", record.report_id)
            except Exception as e:
                logger.warning("Could not persist prediction to PostgreSQL (operating with local JSON): %s", str(e).splitlines()[0])

        # Ejecutar en loop existente o nuevo
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_async_insert())
            else:
                loop.run_until_complete(_async_insert())
        except RuntimeError:
            asyncio.run(_async_insert())

    except Exception as exc:
        logger.warning("Could not persist prediction to PostgreSQL (operating with local JSON): %s", exc)


def save_prediction_report(
    report_type: str,
    model_used: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    ai_decision_analysis: str,
    llm_provider: str,
    chart_source_path: str,
    chart_base64: str,
    execution_time_ms: float = 0.0,
    request_id: Optional[str] = None,
    client_ip: Optional[str] = None
) -> Dict[str, Any]:
    """
    Persiste el reporte y su activo gráfico conmutando entre AWS S3 y almacenamiento Local.
    Inserta el registro en PostgreSQL (AWS RDS o Local) sin base64 y guarda copia JSON local.
    """
    settings = get_settings()
    report_id = f"rep_{uuid.uuid4().hex[:10]}"
    chart_filename = f"{report_id}.png"
    persistent_chart_path = STORAGE_CHARTS_DIR / chart_filename

    # 1. Copiar siempre al almacenamiento local como respaldo seguro
    if os.path.exists(chart_source_path):
        shutil.copy2(chart_source_path, persistent_chart_path)
    else:
        persistent_chart_path = Path(chart_source_path)

    # 2. Conmutación de Storage: AWS S3 vs Local
    storage_backend = "LOCAL"
    web_chart_url = f"/storage/charts/{chart_filename}"

    if settings.use_aws_s3:
        logger.info("USE_AWS_S3 is enabled. Attempting upload to S3 bucket '%s'...", settings.aws_s3_bucket_name)
        s3_url = _upload_file_blocking(
            settings=settings,
            file_path=str(persistent_chart_path),
            object_key=f"charts/{chart_filename}",
            content_type="image/png"
        )
        if s3_url:
            web_chart_url = s3_url
            storage_backend = "S3"
            logger.info("Chart URL resolved to S3: %s", web_chart_url)
        else:
            logger.warning("S3 upload failed. Falling back to local chart URL: %s", web_chart_url)

    report_web_url = f"{BASE_URL}/reports/{report_id}"
    now_iso = datetime.now(timezone.utc).isoformat()
    now_formatted = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # 3. Documento de Reporte consolidado
    report_document = {
        "report_id": report_id,
        "request_id": request_id,
        "created_at_iso": now_iso,
        "created_at_formatted": now_formatted,
        "report_type": report_type,
        "model_used": model_used,
        "storage_backend": storage_backend,
        "input_data": input_data,
        "output_data": output_data,
        "ai_decision_analysis": ai_decision_analysis,
        "llm_provider": llm_provider,
        "chart_url": web_chart_url,
        "chart_base64": chart_base64,
        "storage_chart_path": str(persistent_chart_path),
        "report_url": report_web_url,
        "client_ip": client_ip,
        "metadata": {
            "execution_time_ms": round(execution_time_ms, 1),
            "microservice_version": "1.3.0",
            "framework": "FastAPI + Scalar",
            "use_aws_s3": settings.use_aws_s3,
            "use_aws_rds": settings.use_aws_rds,
            "storage_root": str(STORAGE_DIR)
        }
    }

    # 4. Guardar copia JSON local
    report_json_path = STORAGE_REPORTS_DIR / f"{report_id}.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_document, f, indent=2, ensure_ascii=False)

    # 5. Persistir en PostgreSQL (AWS RDS o Local) de manera no bloqueante y sin Base64
    _persist_to_db_blocking(report_document)

    return report_document


def load_prediction_report(report_id: str) -> Optional[Dict[str, Any]]:
    """Carga un reporte persistido desde storage/reports/{report_id}.json."""
    clean_id = report_id.strip()
    report_json_path = STORAGE_REPORTS_DIR / f"{clean_id}.json"
    if not report_json_path.exists():
        return None

    with open(report_json_path, "r", encoding="utf-8") as f:
        return json.load(f)
