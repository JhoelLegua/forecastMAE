"""
Pydantic Request and Response Schemas.
Includes pre-populated default values and examples for seamless interactive testing in Scalar.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ProductPredictionInput(BaseModel):
    """Payload de entrada para predicción de demanda de producto."""
    product_category: str = Field(
        default="Laptops",
        description="Categoría del producto en Amazon",
        examples=["Laptops", "Phones", "Cameras", "Storage", "Speakers", "TV & Display"]
    )
    discounted_price: float = Field(
        default=699.99,
        ge=0.0,
        description="Precio de venta actual con descuento en USD",
        examples=[699.99]
    )
    original_price: float = Field(
        default=899.99,
        ge=0.0,
        description="Precio de lista original en USD",
        examples=[899.99]
    )
    discount_percentage: float = Field(
        default=22.2,
        ge=0.0,
        le=100.0,
        description="Porcentaje de descuento aplicado",
        examples=[22.2]
    )
    rating_num: float = Field(
        default=4.6,
        ge=1.0,
        le=5.0,
        description="Calificación promedio del producto de 1 a 5 estrellas",
        examples=[4.6]
    )
    reviews_num: float = Field(
        default=1850.0,
        ge=0.0,
        description="Cantidad total de reseñas recibidas",
        examples=[1850.0]
    )
    delivery_lead_days: float = Field(
        default=3.0,
        ge=0.0,
        description="Días estimados para la entrega del producto",
        examples=[3.0]
    )
    is_best_seller: int = Field(
        default=1,
        ge=0,
        le=1,
        description="1 si posee la insignia Best Seller, 0 en caso contrario",
        examples=[1]
    )
    is_amazons_choice: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 si posee insignia Amazon's Choice, 0 en caso contrario",
        examples=[0]
    )
    is_limited_deal: int = Field(
        default=1,
        ge=0,
        le=1,
        description="1 si es oferta por tiempo limitado, 0 en caso contrario",
        examples=[1]
    )
    is_sponsored: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 si es listado patrocinado (Sponsored), 0 si es orgánico",
        examples=[0]
    )
    has_coupon: int = Field(
        default=1,
        ge=0,
        le=1,
        description="1 si tiene cupón de descuento adicional, 0 en caso contrario",
        examples=[1]
    )
    coupon_discount_value: float = Field(
        default=50.0,
        ge=0.0,
        description="Valor del cupón en dólares o porcentaje",
        examples=[50.0]
    )
    buy_box_availability: int = Field(
        default=1,
        ge=0,
        le=1,
        description="1 si el botón 'Add to Cart' (Buy Box) está activo, 0 en caso contrario",
        examples=[1]
    )
    has_sustainability: int = Field(
        default=1,
        ge=0,
        le=1,
        description="1 si cuenta con insignias de sustentabilidad ecológica, 0 en caso contrario",
        examples=[1]
    )
    title_length: int = Field(
        default=115,
        ge=1,
        description="Longitud en caracteres del título del producto",
        examples=[115]
    )
    title_word_count: int = Field(
        default=18,
        ge=1,
        description="Número de palabras en el título del producto",
        examples=[18]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "product_category": "Laptops",
                "discounted_price": 699.99,
                "original_price": 899.99,
                "discount_percentage": 22.2,
                "rating_num": 4.6,
                "reviews_num": 1850.0,
                "delivery_lead_days": 3.0,
                "is_best_seller": 1,
                "is_amazons_choice": 0,
                "is_limited_deal": 1,
                "is_sponsored": 0,
                "has_coupon": 1,
                "coupon_discount_value": 50.0,
                "buy_box_availability": 1,
                "has_sustainability": 1,
                "title_length": 115,
                "title_word_count": 18
            }
        }
    }


class PredictionResponse(BaseModel):
    """Respuesta estructurada para predicción de demanda individual."""
    report_id: str = Field(description="Identificador único persistente del reporte")
    report_url: str = Field(description="Enlace directo a la vista web interactiva del reporte")
    model_used: str = Field(description="Nombre del algoritmo utilizado")
    predicted_monthly_demand: float = Field(description="Demanda estimada en unidades mensuales")
    demand_tier: str = Field(description="Nivel de demanda (Low / Moderate / High / Very High)")
    category: str = Field(description="Categoría evaluada")
    category_median_demand: float = Field(description="Mediana histórica de la categoría en unidades")
    relative_vs_category_pct: float = Field(description="Diferencia porcentual respecto a la mediana de categoría")
    chart_url: str = Field(description="Ruta de acceso web al gráfico generado")
    chart_base64: str = Field(description="Data URI en base64 de la imagen generada")
    storage_chart_path: str = Field(description="Ruta en storage local del archivo PNG persistido")
    ai_decision_analysis: str = Field(description="Interpretación ejecutiva y recomendaciones generadas por LLM")
    llm_provider: str = Field(description="Motor de análisis LLM (Gemini Vision o Heurística)")


class MultiModelPredictionResponse(BaseModel):
    """Respuesta para predicción comparativa entre los 5 algoritmos."""
    report_id: str = Field(description="Identificador único persistente del reporte")
    report_url: str = Field(description="Enlace directo a la vista web interactiva del reporte")
    predictions_by_model: Dict[str, float] = Field(description="Predicciones individuales por algoritmo")
    best_model: str = Field(description="Modelo recomendado con mayor R² (LightGBM)")
    ensemble_mean_demand: float = Field(description="Promedio de unidades estimadas por todos los modelos")
    ensemble_median_demand: float = Field(description="Mediana de unidades estimadas por todos los modelos")
    category: str = Field(description="Categoría del producto")
    category_median_demand: float = Field(description="Mediana histórica de la categoría")
    chart_url: str = Field(description="Ruta de acceso web al gráfico comparativo")
    chart_base64: str = Field(description="Data URI en base64 de la imagen generada")
    storage_chart_path: str = Field(description="Ruta en storage local del archivo PNG persistido")
    ai_decision_analysis: str = Field(description="Análisis comparativo de toma de decisiones por LLM")
    llm_provider: str = Field(description="Motor de análisis utilizado")


class HealthResponse(BaseModel):
    """Estado y diagnóstico del microservicio."""
    status: str
    service_name: str
    version: str
    models_loaded: list[str]
    llm_analyze_enabled: bool = Field(default=True, description="Estado de habilitación de análisis LLM (LLM_ANALYZE)")
    llm_multimodal_active: bool = Field(description="Indica si existe API key activa de Gemini Vision")
    llm_model: str = Field(description="Identificador del modelo LLM o motor heurístico activo")
    smtp_enabled: bool = Field(default=False, description="Estado de notificaciones Brevo SMTP")
    use_aws_s3: bool = Field(default=False, description="Indica si el almacenamiento de gráficos está en AWS S3")
    use_aws_rds: bool = Field(default=False, description="Indica si la base de datos apunta a PostgreSQL en AWS RDS")
    scalar_docs_url: str = Field(description="Ruta de acceso a la documentación interactiva en Scalar")
