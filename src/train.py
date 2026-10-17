"""
Model Training Module.
Trains Ridge Regression, Decision Tree, Random Forest, XGBoost, and LightGBM
using TransformedTargetRegressor with logarithmic demand scaling and robust pipelines.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
CLEANED_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "amazon_sales_cleaned.csv")

FEATURE_COLUMNS = [
    "product_category",
    "discounted_price",
    "original_price",
    "discount_percentage",
    "rating_num",
    "reviews_num",
    "log_reviews",
    "log_price",
    "rating_review_score",
    "delivery_lead_days",
    "is_best_seller",
    "is_amazons_choice",
    "is_limited_deal",
    "is_sponsored",
    "has_coupon",
    "coupon_discount_value",
    "buy_box_availability",
    "has_sustainability",
    "title_length",
    "title_word_count"
]

CATEGORICAL_FEATURES = ["product_category"]
NUMERICAL_FEATURES = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Construye el transformador de columnas para normalización y codificación one-hot."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )


def initialize_candidate_models() -> Dict[str, Any]:
    """Define los estimadores base con hiperparámetros optimizados."""
    return {
        "Ridge": Ridge(
            alpha=25.0,
            solver="auto",
            random_state=42
        ),
        "Decision_Tree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=15,
            random_state=42
        ),
        "Random_Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_split=12,
            min_samples_leaf=8,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=200,
            max_depth=7,
            num_leaves=35,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
    }


def train_models(
    csv_path: str = CLEANED_CSV_PATH,
    models_dir: str = MODELS_DIR,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Entrena los 5 modelos de machine learning, guarda los artefactos serializados
    y retorna los pipelines junto a las particiones de entrenamiento y prueba.
    """
    os.makedirs(models_dir, exist_ok=True)
    print(f"[INFO] Loading processed data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filtrar únicamente registros con demanda observada para supervisión
    df_trainable = df[df["has_demand_data"] == 1].copy()
    print(f"[INFO] Total supervised records available for training: {len(df_trainable)}")

    X = df_trainable[FEATURE_COLUMNS]
    y = df_trainable["monthly_demand"]

    # División entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Train split: {len(X_train)} samples | Test split: {len(X_test)} samples")

    # Ajustar preprocesador independiente
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    print("[INFO] Saved standalone preprocessor.joblib")

    candidate_models = initialize_candidate_models()
    trained_pipelines: Dict[str, Pipeline] = {}

    for model_key, base_estimator in candidate_models.items():
        print(f"[TRAINING] Training model: {model_key}...")
        
        # Pipeline unificado con TransformedTargetRegressor (log1p <-> expm1)
        full_pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", TransformedTargetRegressor(
                regressor=base_estimator,
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ])

        # Ajuste de pipeline
        full_pipeline.fit(X_train, y_train)
        trained_pipelines[model_key] = full_pipeline

        # Serialización de artefactos listos para consumo
        pipeline_filename = f"{model_key.lower()}_pipeline.joblib"
        raw_model_filename = f"{model_key.lower()}.joblib"

        joblib.dump(full_pipeline, os.path.join(models_dir, pipeline_filename))
        joblib.dump(full_pipeline.named_steps["regressor"].regressor_, os.path.join(models_dir, raw_model_filename))
        print(f"[SUCCESS] Model {model_key} saved -> {pipeline_filename} & {raw_model_filename}")

    return trained_pipelines, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_models()
