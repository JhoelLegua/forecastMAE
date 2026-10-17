"""
Demand Prediction Routes.
Exposes endpoints for individual machine learning models, the best model (LightGBM),
and multi-model comparative predictions with automated chart rendering,
storage persistence, optional LLM decision analysis (controlled via LLM_ANALYZE),
and automated Brevo SMTP alerts for demand spikes or operational incidents.
"""

import time
import traceback
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
import numpy as np

from app.schemas import (
    ProductPredictionInput,
    PredictionResponse,
    MultiModelPredictionResponse
)
from app.services.model_service import model_service
from app.services.chart_service import (
    generate_single_prediction_chart,
    generate_multi_model_chart
)
from app.services.llm_service import analyze_prediction_with_llm
from app.services.storage_service import save_prediction_report
from app.services.email_notifier import send_alert_email

router = APIRouter(prefix="/predict", tags=["Demand Forecasting"])


def process_single_prediction(
    model_name: str,
    payload: ProductPredictionInput,
    background_tasks: Optional[BackgroundTasks] = None
) -> PredictionResponse:
    """Función unificada para procesar predicción individual, generar gráfico, persistir y consultar LLM."""
    start_time = time.time()
    try:
        # 1. Inferencia del modelo
        pred_units = model_service.predict(model_name, payload)

        # 2. Métricas de referencia de la categoría
        category = payload.product_category
        benchmarks = model_service.get_category_benchmark(category)
        p50 = benchmarks.get("median", 100.0)
        rel_diff_pct = round(((pred_units - p50) / max(1.0, p50)) * 100.0, 1)
        tier = model_service.classify_demand_tier(pred_units, p50)

        # 3. Preparación de data de salida
        output_data = {
            "predicted_monthly_demand": pred_units,
            "demand_tier": tier,
            "category": category,
            "category_median_demand": p50,
            "relative_vs_category_pct": rel_diff_pct
        }

        # 4. Generación del gráfico ejecutivo
        file_path, b64_str, img_bytes = generate_single_prediction_chart(
            model_name=model_name,
            predicted_demand=pred_units,
            category=category,
            benchmark_data=benchmarks,
            input_data=payload
        )

        # 5. Análisis del gráfico, datos de entrada y predicción con LLM (si LLM_ANALYZE=true)
        ai_analysis, llm_provider = analyze_prediction_with_llm(
            model_name=model_name,
            predicted_demand=pred_units,
            category=category,
            benchmark_data=benchmarks,
            input_data=payload,
            image_bytes=img_bytes,
            prediction_details=output_data
        )

        elapsed_ms = (time.time() - start_time) * 1000.0

        # 6. Persistencia obligatoria en storage local (storage/reports/ y storage/charts/)
        persisted_report = save_prediction_report(
            report_type="single",
            model_used=model_name.replace("_", " ").title(),
            input_data=payload.model_dump(),
            output_data=output_data,
            ai_decision_analysis=ai_analysis,
            llm_provider=llm_provider,
            chart_source_path=file_path,
            chart_base64=b64_str,
            execution_time_ms=elapsed_ms
        )

        # 7. Disparo automático de alerta en segundo plano si hay demanda excepcional
        if background_tasks and (pred_units >= 1000 or rel_diff_pct >= 300):
            background_tasks.add_task(
                send_alert_email,
                event_type="extreme_demand_spike",
                severity="MEDIUM",
                title=f"Demanda Excepcional Detectada ({category})",
                reason=f"El modelo {model_name.replace('_', ' ').title()} proyectó {pred_units:.1f} u/mes ({rel_diff_pct:+}% vs mediana de {p50} u).",
                details=f"Acceso a reporte interactivo: {persisted_report['report_url']}",
                extra_data={
                    "SKU Category": category,
                    "Model": model_name,
                    "Proyección": f"{pred_units:.1f} u/mes",
                    "Mediana Sector": f"{p50} u/mes"
                }
            )

        return PredictionResponse(
            report_id=persisted_report["report_id"],
            report_url=persisted_report["report_url"],
            model_used=model_name.replace("_", " ").title(),
            predicted_monthly_demand=pred_units,
            demand_tier=tier,
            category=category,
            category_median_demand=p50,
            relative_vs_category_pct=rel_diff_pct,
            chart_url=persisted_report["chart_url"],
            chart_base64=b64_str,
            storage_chart_path=persisted_report["storage_chart_path"],
            ai_decision_analysis=ai_analysis,
            llm_provider=llm_provider
        )

    except Exception as e:
        tb_str = traceback.format_exc()
        if background_tasks:
            background_tasks.add_task(
                send_alert_email,
                event_type="prediction_error",
                severity="HIGH",
                title=f"Error en Inferencia con Modelo '{model_name}'",
                reason=f"Error al procesar la predicción: {str(e)}",
                details=tb_str,
                extra_data={"Model": model_name, "Category": payload.product_category}
            )
        raise HTTPException(status_code=500, detail=f"Prediction error for model '{model_name}': {str(e)}")


@router.post("/lightgbm", response_model=PredictionResponse, summary="Predict via LightGBM (Recommended Model)")
async def predict_lightgbm(payload: ProductPredictionInput, background_tasks: BackgroundTasks):
    """Estima la demanda mensual empleando LightGBM (R² = 0.9311 en pruebas)."""
    return process_single_prediction("lightgbm", payload, background_tasks)


@router.post("/best", response_model=PredictionResponse, summary="Predict via Best Model (LightGBM)")
async def predict_best(payload: ProductPredictionInput, background_tasks: BackgroundTasks):
    """Alias automático que ejecuta el modelo con mejor precisión y confiabilidad validada."""
    return process_single_prediction("lightgbm", payload, background_tasks)


@router.post("/random-forest", response_model=PredictionResponse, summary="Predict via Random Forest")
async def predict_random_forest(payload: ProductPredictionInput, background_tasks: BackgroundTasks):
    """Estima la demanda mensual empleando ensamble Random Forest de 150 árboles."""
    return process_single_prediction("random_forest", payload, background_tasks)


@router.post("/xgboost", response_model=PredictionResponse, summary="Predict via XGBoost")
async def predict_xgboost(payload: ProductPredictionInput, background_tasks: BackgroundTasks):
    """Estima la demanda mensual empleando Gradient Boosting optimizado con XGBoost."""
    return process_single_prediction("xgboost", payload, background_tasks)


@router.post("/decision-tree", response_model=PredictionResponse, summary="Predict via Decision Tree")
async def predict_decision_tree(payload: ProductPredictionInput, background_tasks: BackgroundTasks):
    """Estima la demanda mensual empleando Árbol de Decisión no paramétrico."""
    return process_single_prediction("decision_tree", payload, background_tasks)


@router.post("/ridge", response_model=PredictionResponse, summary="Predict via Ridge Regression")
async def predict_ridge(payload: ProductPredictionInput, background_tasks: BackgroundTasks):
    """Estima la demanda mensual empleando Regresión Lineal Regularizada L2 (Ridge)."""
    return process_single_prediction("ridge", payload, background_tasks)


@router.post("/all", response_model=MultiModelPredictionResponse, summary="Predict with all 5 models in consensus")
async def predict_all_models(payload: ProductPredictionInput, background_tasks: BackgroundTasks):
    """
    Ejecuta la predicción en los 5 modelos simultáneamente, genera un gráfico comparativo de consenso,
    persiste el reporte en storage y evalúa la dispersión de pronósticos mediante el LLM (si LLM_ANALYZE=true).
    """
    start_time = time.time()
    try:
        preds = model_service.predict_all(payload)
        category = payload.product_category
        benchmarks = model_service.get_category_benchmark(category)

        pred_values = list(preds.values())
        mean_val = float(round(np.mean(pred_values), 1))
        median_val = float(round(np.median(pred_values), 1))

        # Preparar data de salida
        output_data = {
            "predictions_by_model": preds,
            "best_model": "lightgbm",
            "ensemble_mean_demand": mean_val,
            "ensemble_median_demand": median_val,
            "category": category,
            "category_median_demand": benchmarks.get("median", 100.0)
        }

        # Generar gráfico comparativo
        file_path, b64_str, img_bytes = generate_multi_model_chart(
            predictions_dict=preds,
            category=category,
            benchmark_data=benchmarks
        )

        # Análisis con LLM del mejor modelo (LightGBM) o consenso
        best_pred = preds.get("lightgbm", median_val)
        ai_analysis, llm_provider = analyze_prediction_with_llm(
            model_name="Multi-Model Ensemble Consensus (Leader: LightGBM)",
            predicted_demand=best_pred,
            category=category,
            benchmark_data=benchmarks,
            input_data=payload,
            image_bytes=img_bytes,
            prediction_details=output_data
        )

        elapsed_ms = (time.time() - start_time) * 1000.0

        # Persistir en storage local
        persisted_report = save_prediction_report(
            report_type="multi",
            model_used="Ensemble Consensus (5 Models)",
            input_data=payload.model_dump(),
            output_data=output_data,
            ai_decision_analysis=ai_analysis,
            llm_provider=llm_provider,
            chart_source_path=file_path,
            chart_base64=b64_str,
            execution_time_ms=elapsed_ms
        )

        return MultiModelPredictionResponse(
            report_id=persisted_report["report_id"],
            report_url=persisted_report["report_url"],
            predictions_by_model=preds,
            best_model="lightgbm",
            ensemble_mean_demand=mean_val,
            ensemble_median_demand=median_val,
            category=category,
            category_median_demand=benchmarks.get("median", 100.0),
            chart_url=persisted_report["chart_url"],
            chart_base64=b64_str,
            storage_chart_path=persisted_report["storage_chart_path"],
            ai_decision_analysis=ai_analysis,
            llm_provider=llm_provider
        )

    except Exception as e:
        tb_str = traceback.format_exc()
        background_tasks.add_task(
            send_alert_email,
            event_type="prediction_error",
            severity="HIGH",
            title="Error en Predicción Multi-Modelo",
            reason=f"Excepción en predicción de ensamble: {str(e)}",
            details=tb_str,
            extra_data={"Category": payload.product_category}
        )
        raise HTTPException(status_code=500, detail=f"Multi-model prediction error: {str(e)}")
