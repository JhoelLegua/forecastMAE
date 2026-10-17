"""
Inference and Consumption Module.
Loads serialized models and pipelines to perform single-item,
batch, or category-aggregated monthly demand predictions.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List


MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CLEANED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "amazon_sales_cleaned.csv")

AVAILABLE_MODELS = ["ridge", "decision_tree", "random_forest", "xgboost", "lightgbm"]


def load_model(model_name: str = "lightgbm"):
    """
    Carga el pipeline completo entrenado para inferencia directa.
    
    :param model_name: Nombre del modelo ('ridge', 'decision_tree', 'random_forest', 'xgboost', 'lightgbm').
    :return: Pipeline serializado listo para predict().
    """
    clean_name = model_name.lower().replace(" ", "_")
    pipeline_path = os.path.join(MODELS_DIR, f"{clean_name}_pipeline.joblib")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Model pipeline not found at: {pipeline_path}")
    return joblib.load(pipeline_path)


def predict_demand(
    input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    model_name: str = "lightgbm"
) -> np.ndarray:
    """
    Predice la demanda mensual (unidades estimadas) para uno o varios productos.
    """
    pipeline = load_model(model_name)

    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df_input = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise ValueError("Input data must be a dict, list of dicts, or pandas DataFrame.")

    # Asegurar columnas necesarias
    from src.train import FEATURE_COLUMNS
    for col in FEATURE_COLUMNS:
        if col not in df_input.columns:
            if col == "product_category":
                df_input[col] = "Other Electronics"
            else:
                df_input[col] = 0.0

    raw_preds = pipeline.predict(df_input[FEATURE_COLUMNS])
    # Demanda física no puede ser negativa
    clean_preds = np.clip(raw_preds, 0.0, None)
    return np.round(clean_preds, 1)


def compare_all_models_on_sample(sample_dict: Dict[str, Any]) -> pd.DataFrame:
    """Ejecuta inferencia comparativa en los 5 modelos sobre una muestra de producto."""
    records = []
    for model_name in AVAILABLE_MODELS:
        pred = predict_demand(sample_dict, model_name=model_name)[0]
        records.append({
            "Model": model_name.replace("_", " ").title(),
            "Predicted Monthly Demand (Units)": pred
        })
    return pd.DataFrame(records)


def predict_category_monthly_demand(
    category: str = "Laptops",
    model_name: str = "lightgbm",
    dataset_path: str = CLEANED_CSV_PATH
) -> Dict[str, Any]:
    """
    Calcula la demanda total y promedio estimada por mes para una categoría dada
    analizando el catálogo de productos disponible.
    """
    df = pd.read_csv(dataset_path)
    cat_df = df[df["product_category"].str.lower() == category.lower()]
    if cat_df.empty:
        raise ValueError(f"Category '{category}' not found in dataset.")

    preds = predict_demand(cat_df, model_name=model_name)
    total_demand = float(np.sum(preds))
    mean_demand = float(np.mean(preds))
    median_demand = float(np.median(preds))

    return {
        "category": category,
        "model_used": model_name,
        "catalog_items_analyzed": len(cat_df),
        "total_estimated_monthly_demand": round(total_demand, 1),
        "mean_monthly_demand_per_item": round(mean_demand, 1),
        "median_monthly_demand_per_item": round(median_demand, 1)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run demand predictions using trained models.")
    parser.add_argument("--model", type=str, default="lightgbm", choices=AVAILABLE_MODELS)
    parser.add_argument("--category", type=str, default="Laptops")
    args = parser.parse_args()

    print(f"\n[DEMAND INFERENCE ENGINE] Model: {args.model.upper()}")

    # Muestra de producto representativo
    sample_product = {
        "product_category": "Laptops",
        "discounted_price": 699.99,
        "original_price": 899.99,
        "discount_percentage": 22.2,
        "rating_num": 4.6,
        "reviews_num": 1850.0,
        "log_reviews": np.log1p(1850.0),
        "log_price": np.log1p(699.99),
        "rating_review_score": 4.6 * np.log1p(1850.0),
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

    print("\n[SAMPLE PRODUCT SPECIFICATION]:")
    for k, v in sample_product.items():
        print(f"  {k:25}: {v}")

    print("\n--- Model Comparison on Sample Product ---")
    comp_df = compare_all_models_on_sample(sample_product)
    print(comp_df.to_string(index=False))

    print(f"\n--- Category Demand Forecast: '{args.category}' ---")
    cat_summary = predict_category_monthly_demand(category=args.category, model_name=args.model)
    for k, v in cat_summary.items():
        print(f"  {k:35}: {v}")
