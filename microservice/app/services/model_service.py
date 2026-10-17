"""
Model Management Service.
Loads, caches, and provides prediction utilities for all 5 machine learning models.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from app.config import MODELS_DIR
from app.schemas import ProductPredictionInput


MODEL_KEYS = ["ridge", "decision_tree", "random_forest", "xgboost", "lightgbm"]

# Medias y percentiles históricos de demanda por categoría calculados del dataset limpio
CATEGORY_BENCHMARKS = {
    "Cameras": {"median": 100.0, "p25": 100.0, "p75": 300.0, "catalog_items": 2599},
    "Chargers & Cables": {"median": 200.0, "p25": 100.0, "p75": 400.0, "catalog_items": 1546},
    "Gaming": {"median": 100.0, "p25": 50.0, "p75": 100.0, "catalog_items": 666},
    "Headphones": {"median": 100.0, "p25": 50.0, "p75": 100.0, "catalog_items": 993},
    "Laptops": {"median": 100.0, "p25": 50.0, "p75": 200.0, "catalog_items": 6634},
    "Networking": {"median": 200.0, "p25": 100.0, "p75": 200.0, "catalog_items": 846},
    "Other Electronics": {"median": 200.0, "p25": 100.0, "p75": 300.0, "catalog_items": 5425},
    "Phones": {"median": 300.0, "p25": 100.0, "p75": 500.0, "catalog_items": 5804},
    "Power & Batteries": {"median": 1000.0, "p25": 200.0, "p75": 20000.0, "catalog_items": 2783},
    "Printers & Scanners": {"median": 100.0, "p25": 100.0, "p75": 200.0, "catalog_items": 876},
    "Smart Home": {"median": 100.0, "p25": 50.0, "p75": 100.0, "catalog_items": 253},
    "Speakers": {"median": 200.0, "p25": 100.0, "p75": 300.0, "catalog_items": 959},
    "Storage": {"median": 200.0, "p25": 100.0, "p75": 300.0, "catalog_items": 958},
    "TV & Display": {"median": 200.0, "p25": 100.0, "p75": 300.0, "catalog_items": 1373},
    "Wearables": {"median": 800.0, "p25": 50.0, "p75": 800.0, "catalog_items": 447}
}


class ModelService:
    """Administra la carga en memoria y ejecución de inferencias sobre los modelos."""

    def __init__(self):
        self.pipelines: Dict[str, Any] = {}
        self._load_all_models()

    def _load_all_models(self):
        """Carga en caché todos los pipelines .joblib al inicializar el servicio."""
        print("[MODEL SERVICE] Loading machine learning pipelines into memory...")
        for key in MODEL_KEYS:
            pipeline_path = MODELS_DIR / f"{key}_pipeline.joblib"
            if pipeline_path.exists():
                self.pipelines[key] = joblib.load(pipeline_path)
                print(f"[MODEL SERVICE] Loaded: {key}_pipeline.joblib")
            else:
                print(f"[MODEL SERVICE WARNING] File not found: {pipeline_path}")

    def get_loaded_models(self) -> List[str]:
        return list(self.pipelines.keys())

    def format_input_dataframe(self, input_data: ProductPredictionInput) -> pd.DataFrame:
        """Convierte y enriquece el payload Pydantic al DataFrame esperado por el modelo."""
        data_dict = input_data.model_dump()
        
        # Ingeniería de variables en tiempo real
        data_dict["log_reviews"] = float(np.log1p(data_dict["reviews_num"]))
        data_dict["log_price"] = float(np.log1p(data_dict["discounted_price"]))
        data_dict["rating_review_score"] = float(data_dict["rating_num"] * data_dict["log_reviews"])

        return pd.DataFrame([data_dict])

    def predict(self, model_key: str, input_data: ProductPredictionInput) -> float:
        """Ejecuta la inferencia con el modelo indicado devolviendo unidades estimadas."""
        norm_key = model_key.lower().replace("-", "_")
        if norm_key not in self.pipelines:
            raise ValueError(f"Model '{norm_key}' is not available. Available: {list(self.pipelines.keys())}")

        df = self.format_input_dataframe(input_data)
        raw_pred = self.pipelines[norm_key].predict(df)
        pred_units = float(np.clip(raw_pred[0], 0.0, None))
        return round(pred_units, 1)

    def predict_all(self, input_data: ProductPredictionInput) -> Dict[str, float]:
        """Ejecuta la predicción en paralelo conceptual sobre los 5 modelos."""
        results = {}
        df = self.format_input_dataframe(input_data)
        for key, pipe in self.pipelines.items():
            pred = float(np.clip(pipe.predict(df)[0], 0.0, None))
            results[key] = round(pred, 1)
        return results

    def get_category_benchmark(self, category: str) -> Dict[str, float]:
        """Devuelve las métricas de referencia históricas de la categoría."""
        # Búsqueda insensible a mayúsculas
        for cat_name, stats in CATEGORY_BENCHMARKS.items():
            if cat_name.lower() == category.lower():
                return stats
        # Categoría por defecto
        return {"median": 150.0, "p25": 100.0, "p75": 300.0, "catalog_items": 1000}

    @staticmethod
    def classify_demand_tier(predicted: float, median: float) -> str:
        """Clasifica el nivel de demanda en relación al percentil de mercado."""
        ratio = predicted / max(1.0, median)
        if ratio < 0.5:
            return "Low (Below Market Average)"
        elif ratio <= 1.5:
            return "Moderate (Average Market Demand)"
        elif ratio <= 4.0:
            return "High (Strong Performer)"
        else:
            return "Very High (Market Leader / Outlier)"


# Instancia singleton del servicio de modelos
model_service = ModelService()
