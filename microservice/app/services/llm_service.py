"""
LLM Decision Analysis Service.
Connects to Google Gemini Multimodal Vision API (gemini-3.8-flash)
to inspect the generated prediction chart and prediction data, generating executive decision-making insights.
Enforces an unvarying, standardized 4-section structure including plain natural language and technical S&OP analysis.
Includes an intelligent Heuristic Decision Engine fallback.
"""

from typing import Dict, Any, Optional, Tuple
from app.config import GEMINI_API_KEY, GEMINI_MODEL, HAS_GEMINI_KEY, LLM_ANALYZE
from app.schemas import ProductPredictionInput


def format_prediction_details_text(prediction_details: Optional[Dict[str, Any]]) -> str:
    """Convierte los detalles de la predicción realizada en texto estructurado para el prompt."""
    if not prediction_details:
        return "Demanda calculada sobre parámetros del modelo base."

    lines = []
    if "predicted_monthly_demand" in prediction_details:
        lines.append(f"- Demanda Mensual Estimada: {prediction_details['predicted_monthly_demand']} unidades/mes")
    if "demand_tier" in prediction_details:
        lines.append(f"- Nivel / Tier de Rendimiento: {prediction_details['demand_tier']}")
    if "relative_vs_category_pct" in prediction_details:
        lines.append(f"- Diferencial vs Mediana del Sector: {prediction_details['relative_vs_category_pct']:+}%")
    if "predictions_by_model" in prediction_details:
        lines.append("- Desglose de predicciones por modelo:")
        for m_name, val in prediction_details["predictions_by_model"].items():
            lines.append(f"  * {m_name.replace('_', ' ').title()}: {val} unidades")
        if "ensemble_mean_demand" in prediction_details:
            lines.append(f"  * Promedio de Ensamble: {prediction_details['ensemble_mean_demand']} unidades")
        if "ensemble_median_demand" in prediction_details:
            lines.append(f"  * Mediana de Ensamble: {prediction_details['ensemble_median_demand']} unidades")
    return "\n".join(lines)


def generate_heuristic_decision_analysis(
    model_name: str,
    predicted_demand: float,
    category: str,
    benchmark_data: Dict[str, float],
    input_data: ProductPredictionInput,
    prediction_details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Motor Heurístico de Análisis con estructura fija obligatoria en 4 secciones.
    Combina explicación en lenguaje natural cotidiano con diagnóstico cuantitativo S&OP.
    """
    p50 = benchmark_data.get("median", 100.0)
    p25 = benchmark_data.get("p25", 50.0)
    p75 = benchmark_data.get("p75", 250.0)
    diff_pct = ((predicted_demand - p50) / max(1.0, p50)) * 100.0

    # Lógica de clasificación
    if predicted_demand >= p75:
        simple_resumen = (
            f"Este producto es un 'éxito de ventas' rotundo. Estimamos que se venderán cerca de {predicted_demand:,.0f} unidades al mes, "
            f"lo cual es muchísimo más que una tienda promedio en este rubro (que solo vende {p50:.0f} al mes)."
        )
        simple_porque = (
            f"A la gente le encanta porque tiene una nota muy alta ({input_data.rating_num:.1f} de 5 estrellas) con miles de personas recomendándolo "
            f"({input_data.reviews_num:,.0f} opiniones). Además, tiene la insignia de producto más vendido y llega rápido en {input_data.delivery_lead_days:.0f} días."
        )
        simple_accion = (
            f"Hay que asegurar que la fábrica o el depósito tenga suficiente mercadería (al menos {int(predicted_demand * 1.35):,} unidades listas). "
            f"Si nos quedamos sin producto, perderemos clientes inmediatamente."
        )
        tier = "ALTO RENDIMIENTO (Top Tier / Líder de Categoría)"
        stock_advice = f"Planificar compras sobre tasa de ~{int(predicted_demand/4):,} u/semana. Fijar stock de seguridad de {int(predicted_demand*0.35):,} u (cobertura 12-15 días)."
        price_advice = "Existe margen de elasticidad. Se sugiere probar subida gradual de $15 a $30 o retirar cupones para capturar mayor margen neto sin resentir volumen."
    elif predicted_demand >= p50:
        simple_resumen = (
            f"El producto tendrá una venta saludable y constante de aproximadamente {predicted_demand:,.0f} unidades al mes, "
            f"situándose por encima del promedio normal de la categoría ({p50:.0f} unidades)."
        )
        simple_porque = (
            f"El precio de ${input_data.discounted_price:,.2f} resulta razonable para lo que ofrece y cuenta con buen respaldo de reseñas ({input_data.reviews_num:,.0f} comentarios), "
            f"lo que genera confianza en el comprador."
        )
        simple_accion = (
            f"Mantener pedidos regulares cada mes para unas {int(predicted_demand * 1.15):,} unidades y conservar activa cualquier oferta o cupón para no bajar el ritmo."
        )
        tier = "DESEMPEÑO MEDIO-ALTO (Sobre la Mediana de Mercado)"
        stock_advice = f"Planificar reposición mensual de ~{int(predicted_demand):,} u. Stock de seguridad de {int(predicted_demand*0.25):,} u para absorber variaciones."
        price_advice = "Mantener el precio y los cupones activos; el punto actual equilibra volumen de ventas con margen competitivo."
    else:
        simple_resumen = (
            f"El producto se moverá a un ritmo pausado o de nicho, con unas {predicted_demand:,.0f} unidades al mes, "
            f"lo cual está por debajo del promedio del sector ({p50:.0f} unidades)."
        )
        simple_porque = (
            f"Puede que el precio sea percibido como elevado para las características ofrecidas o que aún le falte mayor cantidad de opiniones de compradores para generar plena confianza."
        )
        simple_accion = (
            f"No sobrecargarse de inventario; pedir lotes pequeños de prueba (unas {int(predicted_demand * 1.1):,} unidades) e impulsar promociones o conseguir más comentarios positivos."
        )
        tier = "DEMANDA MODERADA / NICHO (Bajo la Mediana de Mercado)"
        stock_advice = f"Adoptar estrategia Just-In-Time con pedidos reducidos de {int(predicted_demand):,} u para evitar costos de almacenamiento prolongado."
        price_advice = "Evaluar descuentos temporales más agresivos o cupones de bienvenida para activar la velocidad de conversión."

    badges_list = []
    if input_data.is_best_seller: badges_list.append("Best Seller")
    if input_data.is_amazons_choice: badges_list.append("Amazon's Choice")
    if input_data.is_limited_deal: badges_list.append("Oferta Limitada")
    if input_data.has_coupon: badges_list.append(f"Cupón (${input_data.coupon_discount_value:.0f})")
    if input_data.buy_box_availability: badges_list.append("Buy Box Ganada")
    badges_str = ", ".join(badges_list) if badges_list else "Sin insignias destacadas (Tráfico orgánico estándar)"

    # Estructura uniforme fija de 4 secciones
    return f"""# INFORME EJECUTIVO DE DEMANDA Y TOMA DE DECISIONES

## 1. EXPLICACIÓN EN LENGUAJE SIMPLE (NO TÉCNICO)
- **¿Qué significa este pronóstico?:** {simple_resumen}
- **¿Por qué se venderá esta cantidad?:** {simple_porque}
- **¿Qué debemos hacer ahora mismo?:** {simple_accion}

## 2. DIAGNÓSTICO TÉCNICO Y POSICIONAMIENTO DE MERCADO
- **Demanda Proyectada:** {predicted_demand:,.1f} unidades mensuales pronosticadas mediante el modelo **{model_name}**.
- **Benchmark Sectorial:** En la categoría **{category}**, el percentil 25 se ubica en {p25:.0f} u, la mediana en {p50:.0f} u y el percentil 75 en {p75:.0f} u. La proyección se sitúa en un diferencial de **{diff_pct:+.1f}%** respecto a la mediana del sector.
- **Clasificación de Rendimiento:** {tier}.
- **Lectura de la Gráfica:** La barra de predicción refleja el posicionamiento relativo frente a los tres cuartiles históricos, confirmando la posición competitiva del SKU.

## 3. ANÁLISIS DE PALANCAS COMERCIALES (DRIVERS)
- **Precio y Descuento:** Precio descontado de **${input_data.discounted_price:,.2f}** frente al precio original de **${input_data.original_price:,.2f}** ({input_data.discount_percentage:.1f}% de descuento), estimulando la elasticidad-precio.
- **Confianza y Reseñas:** Calificación de **{input_data.rating_num:.1f} / 5.0** con **{input_data.reviews_num:,.0f} reseñas**, consolidando una alta tasa de conversión por prueba social.
- **Insignias y Visibilidad:** Presencia de: {badges_str}.
- **Eficiencia en la Cadena de Suministro:** Promesa de entrega en **{input_data.delivery_lead_days:.0f} días**, alineada con las expectativas de compras de alta rotación.

## 4. PLAN DE ACCIÓN OPERATIVO Y ESTRATÉGICO
- **Cadena de Suministro (Inventario):** {stock_advice}
- **Fijación de Precios y Margen:** {price_advice}
- **Marketing y Ventas:** Proteger la titularidad del Buy Box y reasignar pauta publicitaria hacia retargeting defensivo de SKU."""


def analyze_prediction_with_llm(
    model_name: str,
    predicted_demand: float,
    category: str,
    benchmark_data: Dict[str, float],
    input_data: ProductPredictionInput,
    image_bytes: bytes,
    prediction_details: Optional[Dict[str, Any]] = None
) -> Tuple[str, str]:
    """
    Envía la imagen del gráfico y la data de predicción a Google Gemini Vision
    exigiendo una estructura estricta e invariable de 4 secciones, incluyendo lenguaje natural.
    Si LLM_ANALYZE es False, se omite el análisis y el informe no contendrá la explicación.
    """
    if not LLM_ANALYZE:
        return "", "Desactivado (LLM_ANALYZE=False)"

    if not HAS_GEMINI_KEY:
        analysis = generate_heuristic_decision_analysis(
            model_name=model_name,
            predicted_demand=predicted_demand,
            category=category,
            benchmark_data=benchmark_data,
            input_data=input_data,
            prediction_details=prediction_details
        )
        return analysis, "Heuristic Decision Engine (Fallback)"

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        prediction_data_text = format_prediction_details_text(prediction_details)
        p50 = benchmark_data.get("median", 100.0)
        p25 = benchmark_data.get("p25", 50.0)
        p75 = benchmark_data.get("p75", 250.0)

        prompt = f"""
Eres el Director de Estrategia Comercial y Cadena de Suministro (Chief Retail & Supply Chain Officer).
Se te proporciona la imagen del gráfico de predicción y benchmarking generado por el sistema de Machine Learning DemandMind, junto con los datos técnicos y la data exacta de la predicción realizada:

DATA DE LA PREDICCIÓN REALIZADA:
- Modelo de Machine Learning: {model_name}
- Demanda Mensual Estimada: {predicted_demand} unidades/mes
{prediction_data_text}

BENCHMARKS DE LA CATEGORÍA ({category}):
- Percentil 25 (Bajo): {p25} u/mes
- Mediana del Sector: {p50} u/mes
- Percentil 75 (Líderes): {p75} u/mes

ATRIBUTOS DE ENTRADA DEL PRODUCTO:
- Categoría: {category}
- Precio de Venta: ${input_data.discounted_price} (Precio Original: ${input_data.original_price}, Descuento: {input_data.discount_percentage}%)
- Calificación del Cliente: {input_data.rating_num} / 5.0
- Total de Reseñas: {input_data.reviews_num}
- Días de Entrega Prometida: {input_data.delivery_lead_days} días
- Insignia Best Seller: {'Sí' if input_data.is_best_seller else 'No'}
- Oferta por Tiempo Limitado: {'Sí' if input_data.is_limited_deal else 'No'}
- Cupón Activo: {'Sí ($' + str(input_data.coupon_discount_value) + ')' if input_data.has_coupon else 'No'}
- Buy Box Ganada: {'Sí' if input_data.buy_box_availability else 'No'}
- Insignia Sustentabilidad: {'Sí' if input_data.has_sustainability else 'No'}

================================================================================
REGLA MANDATORIA DE FORMATO:
DEBES GENERAR TU REPORTE SIGUIENDO EXACTA Y LITERALMENTE LA SIGUIENTE ESTRUCTURA DE 4 SECCIONES.
NO agregues títulos nuevos, NO omitas ninguna sección y NO alteres los nombres de los encabezados `##`:

# INFORME EJECUTIVO DE DEMANDA Y TOMA DE DECISIONES

## 1. EXPLICACIÓN EN LENGUAJE SIMPLE (NO TÉCNICO)
- **¿Qué significa este pronóstico?:** [Explica en lenguaje 100% cotidiano, claro y amigable qué significa esta cifra. PROHIBIDO usar palabras como "algoritmo", "percentil", "regresión", "heterocedasticidad", "desviación estándar" o "R²". Debe entenderlo cualquier persona sin formación técnica]
- **¿Por qué se venderá esta cantidad?:** [Explica la razón humana por la que los compradores eligen o no el producto: si el precio es atractivo, si confían por las estrellas y reseñas, o si la entrega rápida convence]
- **¿Qué debemos hacer ahora mismo?:** [Instrucciones prácticas y sencillas: cuántas unidades encargar a los proveedores o qué hacer con el precio hoy mismo]

## 2. DIAGNÓSTICO TÉCNICO Y POSICIONAMIENTO DE MERCADO
- **Demanda Proyectada:** [Cifra calculada, modelo de machine learning utilizado y nivel de rotación]
- **Benchmark Sectorial:** [Comparativa cuantitativa exacta frente al P25 ({p25} u), la Mediana ({p50} u) y el P75 ({p75} u), indicando el porcentaje de diferencia]
- **Clasificación de Rendimiento:** [Nivel competitivo: Bajo / Moderado / Alto / Líder de Categoría]
- **Lectura de la Gráfica:** [Análisis visual técnico de la gráfica adjunta y de la dispersión de estimaciones]

## 3. ANÁLISIS DE PALANCAS COMERCIALES (DRIVERS)
- **Precio y Descuento:** [Análisis cuantitativo de elasticidad precio-demanda y efecto del cupón]
- **Confianza y Reseñas:** [Impacto del volumen de reseñas y calificación en la tasa de conversión]
- **Insignias y Logística:** [Efecto comercial de las insignias activas y de los tiempos de entrega]

## 4. PLAN DE ACCIÓN OPERATIVO Y ESTRATÉGICO
- **Cadena de Suministro (Inventario):** [Ritmo de salida semanal estimado, stock de seguridad en unidades y días de cobertura para evitar roturas]
- **Fijación de Precios y Margen:** [Recomendación concreta sobre mantener, subir o ajustar el precio/descuento para optimizar rentabilidad]
- **Marketing y Ventas:** [Estrategia de defensa del Buy Box y enfoque de pauta publicitaria]
================================================================================
"""

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[image_part, prompt]
        )
        return response.text, f"Google Gemini Vision ({GEMINI_MODEL})"

    except Exception as e:
        print(f"[LLM SERVICE WARNING] Error calling Gemini API: {e}. Falling back to heuristic engine.")
        fallback = generate_heuristic_decision_analysis(
            model_name=model_name,
            predicted_demand=predicted_demand,
            category=category,
            benchmark_data=benchmark_data,
            input_data=input_data,
            prediction_details=prediction_details
        )
        return fallback, f"Heuristic Decision Engine (Gemini fallback due to: {type(e).__name__})"
