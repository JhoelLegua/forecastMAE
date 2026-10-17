"""
Interactive Report View Routes.
Serves full-page interactive HTML reports with theme switching (dark/light),
high-resolution charts, LLM decision analysis, and instant PDF download.
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.config import TEMPLATES_DIR
from app.services.storage_service import load_prediction_report


router = APIRouter(prefix="/reports", tags=["Executive Reports & Views"])
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/{report_id}", response_class=HTMLResponse, summary="View Interactive HTML Report")
async def view_report(request: Request, report_id: str):
    """
    Despliega la vista ejecutiva interactiva del reporte.
    Incluye: Datos de Entrada, KPIs de Salida, Gráfica de Benchmarking,
    Reporte del LLM formateado, Modo Oscuro/Claro y Descarga en PDF.
    """
    report_data = load_prediction_report(report_id)
    if not report_data:
        raise HTTPException(
            status_code=404,
            detail=f"Report with ID '{report_id}' was not found in storage."
        )

    return templates.TemplateResponse(
        request=request,
        name="report.html",
        context={"report": report_data}
    )


@router.get("/{report_id}/json", response_class=JSONResponse, summary="Get Raw Report JSON")
async def get_report_json(report_id: str):
    """Devuelve la información íntegra persistida en formato JSON para integración API."""
    report_data = load_prediction_report(report_id)
    if not report_data:
        raise HTTPException(
            status_code=404,
            detail=f"Report with ID '{report_id}' was not found."
        )
    return report_data
