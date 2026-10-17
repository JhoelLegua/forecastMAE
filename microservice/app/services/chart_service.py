"""
Chart Generation Service.
Renders high-resolution business decision charts for demand predictions,
exports PNG files, and returns base64 Data URIs for LLM inspection and UI rendering.
"""

import os
import io
import uuid
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
from app.config import CHARTS_DIR
from app.schemas import ProductPredictionInput


# Configuración estética
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "figure.titlesize": 13,
    "figure.dpi": 160
})


def generate_single_prediction_chart(
    model_name: str,
    predicted_demand: float,
    category: str,
    benchmark_data: Dict[str, float],
    input_data: ProductPredictionInput
) -> Tuple[str, str, bytes]:
    """
    Genera un gráfico ejecutivo de 2 paneles para toma de decisiones:
    1. Posicionamiento del producto vs Percentiles de Mercado (P25, Mediana, P75).
    2. Factores de impulso comercial (Precio, Calificación, Descuento, Insignias).
    """
    chart_id = f"pred_{uuid.uuid4().hex[:8]}"
    file_path = os.path.join(CHARTS_DIR, f"{chart_id}.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: Posicionamiento competitivo
    ax1 = axes[0]
    p25 = benchmark_data.get("p25", 50.0)
    p50 = benchmark_data.get("median", 100.0)
    p75 = benchmark_data.get("p75", 250.0)

    categories = ["Cat. P25 (Bajo)", "Cat. Mediana (Promedio)", "Cat. P75 (Líderes)", f"Predicción ({model_name.title()})"]
    values = [p25, p50, p75, predicted_demand]
    colors = ["#95a5a6", "#34495e", "#7f8c8d", "#27ae60" if predicted_demand >= p50 else "#e67e22"]

    bars = ax1.bar(categories, values, color=colors, width=0.55, alpha=0.9, edgecolor="black", linewidth=0.8)
    ax1.set_title(f"Posicionamiento de Demanda en '{category}'", fontweight="bold", pad=12)
    ax1.set_ylabel("Demanda Mensual Estimada (Unidades)")
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=15, ha="right")
    ax1.axhline(p50, color="#2980b9", linestyle="--", linewidth=1.2, label=f"Mediana Sector ({p50:.0f} u)")

    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + (max(values)*0.02), f"{val:,.0f} u", ha="center", fontsize=9, fontweight="bold")

    # Ratio relativo respecto a la mediana
    diff_pct = ((predicted_demand - p50) / max(1.0, p50)) * 100.0
    sign = "+" if diff_pct >= 0 else ""
    ax1.text(0.95, 0.90, f"vs Mediana: {sign}{diff_pct:.1f}%\nModelo: {model_name.upper()}",
             transform=ax1.transAxes, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#eafaf1" if diff_pct >= 0 else "#fef5e7", edgecolor="#27ae60" if diff_pct >= 0 else "#e67e22", alpha=0.95),
             fontsize=9, fontweight="bold")
    ax1.legend(loc="upper left")

    # Panel 2: Resumen de Atributos Clave
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Palancas Comerciales y Perfil de Entrada", fontweight="bold", pad=12)

    table_data = [
        ["Categoría", category],
        ["Precio Descontado", f"${input_data.discounted_price:,.2f}"],
        ["Precio Original", f"${input_data.original_price:,.2f}"],
        ["Descuento Efectivo", f"{input_data.discount_percentage:.1f}%"],
        ["Calificación Cliente", f"{input_data.rating_num:.1f} / 5.0"],
        ["Volumen Reseñas", f"{input_data.reviews_num:,.0f}"],
        ["Días Entrega Prometida", f"{input_data.delivery_lead_days:.0f} días"],
        ["Insignia Best Seller", "Sí (Activa)" if input_data.is_best_seller else "No"],
        ["Oferta Limitada", "Sí" if input_data.is_limited_deal else "No"],
        ["Cupón de Descuento", f"Sí (${input_data.coupon_discount_value:,.1f})" if input_data.has_coupon else "No"],
        ["Buy Box / Add to Cart", "Activo" if input_data.buy_box_availability else "No"]
    ]

    table = ax2.table(
        cellText=table_data,
        colLabels=["Parámetro", "Valor"],
        cellLoc="left",
        loc="center",
        colWidths=[0.48, 0.48]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fa")

    fig.suptitle(f"DEMANDMIND PREDICTIVE INTELLIGENCE - {category.upper()}", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()

    # Guardar archivo PNG
    fig.savefig(file_path, bbox_inches="tight")

    # Generar Bytes y Base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image_bytes = buf.getvalue()
    b64_str = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    return file_path, b64_str, image_bytes


def generate_multi_model_chart(
    predictions_dict: Dict[str, float],
    category: str,
    benchmark_data: Dict[str, float]
) -> Tuple[str, str, bytes]:
    """Genera gráfico comparativo entre los 5 modelos versus la referencia del sector."""
    chart_id = f"multi_{uuid.uuid4().hex[:8]}"
    file_path = os.path.join(CHARTS_DIR, f"{chart_id}.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Comparativa entre modelos
    ax1 = axes[0]
    model_names = [m.replace("_", " ").title() for m in predictions_dict.keys()]
    preds = list(predictions_dict.values())
    p50 = benchmark_data.get("median", 100.0)

    palette = ["#3498db", "#2ecc71", "#1abc9c", "#e67e22", "#9b59b6"]
    bars = ax1.bar(model_names, preds, color=palette, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax1.set_title("Consenso de Pronóstico por Algoritmo", fontweight="bold")
    ax1.set_ylabel("Demanda Mensual Estimada (Unidades)")
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=20, ha="right")
    ax1.axhline(p50, color="red", linestyle="--", linewidth=1.5, label=f"Mediana Sector ({p50:.0f} u)")

    for bar, val in zip(bars, preds):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + (max(preds)*0.02), f"{val:,.0f}", ha="center", fontsize=8.5, fontweight="bold")
    ax1.legend(loc="upper left")

    # Panel 2: Rango de Incertidumbre y Resumen
    ax2 = axes[1]
    mean_val = float(np.mean(preds))
    median_val = float(np.median(preds))
    min_val = float(np.min(preds))
    max_val = float(np.max(preds))
    best_pred = predictions_dict.get("lightgbm", median_val)

    ax2.axis("off")
    ax2.set_title("Métricas de Consenso para Decisiones", fontweight="bold")

    summary_text = (
        f"CATEGORÍA EVALUADA: {category.upper()}\n"
        f"--------------------------------------------------\n"
        f"• Modelo Recomendado (LightGBM):   {best_pred:,.1f} unidades/mes\n"
        f"• Promedio de Ensamble:             {mean_val:,.1f} unidades/mes\n"
        f"• Mediana de Ensamble:              {median_val:,.1f} unidades/mes\n"
        f"• Rango Estimado de Dispersión:    [{min_val:,.0f} - {max_val:,.0f}] u\n"
        f"• Mediana Histórica de Categoría:   {p50:,.0f} unidades/mes\n\n"
        f"EVALUACIÓN DE SENSIBILIDAD:\n"
        f"Los modelos basados en árboles y boosting (LightGBM, XGBoost, RF)\n"
        f"capturan interacciones no lineales entre ratings y precios,\n"
        f"ofreciendo pronósticos robustos frente a modelos lineales."
    )
    ax2.text(0.05, 0.85, summary_text, fontsize=9.5, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f4f6f7", edgecolor="#bdc3c7", alpha=0.9))

    fig.suptitle(f"COMPARATIVA MULTI-MODELO DE DEMANDA - {category.upper()}", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()

    fig.savefig(file_path, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    image_bytes = buf.getvalue()
    b64_str = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    return file_path, b64_str, image_bytes
