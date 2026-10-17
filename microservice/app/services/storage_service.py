"""
Storage Service Module.
Persists generated prediction charts in storage/charts/ and full report
metadata documents in storage/reports/ for web viewing, PDF generation, and auditing.
"""

import os
import json
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from app.config import (
    STORAGE_DIR,
    STORAGE_CHARTS_DIR,
    STORAGE_REPORTS_DIR,
    BASE_URL
)


def save_prediction_report(
    report_type: str,
    model_used: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    ai_decision_analysis: str,
    llm_provider: str,
    chart_source_path: str,
    chart_base64: str,
    execution_time_ms: float = 0.0
) -> Dict[str, Any]:
    """
    Persiste el reporte y su activo gráfico en el storage local.
    Retorna la metadata consolidada con URLs de acceso web.
    """
    report_id = f"rep_{uuid.uuid4().hex[:10]}"
    chart_filename = f"{report_id}.png"
    persistent_chart_path = STORAGE_CHARTS_DIR / chart_filename

    # Copiar o mover el gráfico generado al storage permanente
    if os.path.exists(chart_source_path):
        shutil.copy2(chart_source_path, persistent_chart_path)
    else:
        # Si el archivo original no está en disco, persistir el archivo destino
        persistent_chart_path = Path(chart_source_path)

    # Rutas web
    web_chart_url = f"/storage/charts/{chart_filename}"
    report_web_url = f"{BASE_URL}/reports/{report_id}"

    now_iso = datetime.now(timezone.utc).isoformat()
    now_formatted = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    report_document = {
        "report_id": report_id,
        "created_at_iso": now_iso,
        "created_at_formatted": now_formatted,
        "report_type": report_type,
        "model_used": model_used,
        "input_data": input_data,
        "output_data": output_data,
        "ai_decision_analysis": ai_decision_analysis,
        "llm_provider": llm_provider,
        "chart_url": web_chart_url,
        "chart_base64": chart_base64,
        "storage_chart_path": str(persistent_chart_path),
        "report_url": report_web_url,
        "metadata": {
            "execution_time_ms": round(execution_time_ms, 1),
            "microservice_version": "1.0.0",
            "framework": "FastAPI + Scalar",
            "storage_root": str(STORAGE_DIR)
        }
    }

    report_json_path = STORAGE_REPORTS_DIR / f"{report_id}.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_document, f, indent=2, ensure_ascii=False)

    print(f"[STORAGE] Report persisted: {report_json_path} & Chart: {persistent_chart_path}")
    return report_document


def load_prediction_report(report_id: str) -> Optional[Dict[str, Any]]:
    """Carga un reporte persistido desde storage/reports/{report_id}.json."""
    clean_id = report_id.strip()
    report_json_path = STORAGE_REPORTS_DIR / f"{clean_id}.json"
    if not report_json_path.exists():
        return None

    with open(report_json_path, "r", encoding="utf-8") as f:
        return json.load(f)
