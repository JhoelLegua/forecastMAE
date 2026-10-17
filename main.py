"""
Master Orchestration Pipeline.
Executes the end-to-end Machine Learning Workflow:
1. Dataset acquisition (kagglehub)
2. Comprehensive preprocessing & clean CSV generation
3. Training of 5 candidate regression models (Ridge, DT, RF, XGBoost, LightGBM)
4. Evaluation with statistical reliability metrics (R², MAE, RMSE, MedAE, MAPE, CV, CI 95%)
5. Generation of all visualizations (preprocessing flow, metrics, residuals, internal algorithmic flows)
6. Model artifact persistence & demonstration of inference
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
from src.download_data import download_dataset
from src.preprocess import preprocess_data
from src.train import train_models
from src.evaluate import evaluate_all_models
from src.visualize import generate_all_visualizations
from src.predict import compare_all_models_on_sample, predict_category_monthly_demand


def run_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("DEMANDMIND: END-TO-END DEMAND FORECASTING & MACHINE LEARNING PIPELINE")
    print("=" * 80)

    # 1. Descarga de datos
    print("\n>>> STEP 1: DATASET ACQUISITION VIA KAGGLEHUB")
    raw_dir = download_dataset()

    # 2. Preprocesamiento y exportación de CSV limpio
    print("\n>>> STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
    df_clean, meta = preprocess_data()
    print(f"[STATUS] Clean CSV produced with {meta['final_rows']:,} rows and {meta['final_cols']} columns.")

    # 3. Entrenamiento de los 5 modelos
    print("\n>>> STEP 3: MODEL TRAINING (RIDGE, DECISION TREE, RANDOM FOREST, XGBOOST, LIGHTGBM)")
    pipelines, X_train, X_test, y_train, y_test = train_models()

    # 4. Evaluación de métricas de confiabilidad
    print("\n>>> STEP 4: RELIABILITY METRICS & BOOTSTRAP CONFIDENCE INTERVALS")
    evaluation_results = evaluate_all_models(pipelines, X_train, X_test, y_train, y_test)

    # 5. Generación de todos los gráficos visuales
    print("\n>>> STEP 5: VISUAL ASSET GENERATION (PREPROCESSING, METRICS & ALGORITHMIC FLOWS)")
    generate_all_visualizations()

    # 6. Demostración de inferencia
    print("\n>>> STEP 6: INFERENCE DEMONSTRATION & CATEGORY DEMAND AGGREGATION")
    sample_product = {
        "product_category": "Laptops",
        "discounted_price": 549.99,
        "original_price": 699.99,
        "discount_percentage": 21.4,
        "rating_num": 4.5,
        "reviews_num": 1200.0,
        "log_reviews": 7.09,
        "log_price": 6.31,
        "rating_review_score": 31.9,
        "delivery_lead_days": 2.0,
        "is_best_seller": 1,
        "is_amazons_choice": 0,
        "is_limited_deal": 1,
        "is_sponsored": 0,
        "has_coupon": 1,
        "coupon_discount_value": 30.0,
        "buy_box_availability": 1,
        "has_sustainability": 1,
        "title_length": 95,
        "title_word_count": 15
    }
    comp_df = compare_all_models_on_sample(sample_product)
    print("\n[INFERENCE BENCHMARK ON SAMPLE ITEM]:")
    print(comp_df.to_string(index=False))

    cat_demand = predict_category_monthly_demand("Laptops", model_name="lightgbm")
    print("\n[CATEGORY DEMAND AGGREGATION: LAPTOPS]:")
    for k, v in cat_demand.items():
        print(f"  {k:35}: {v}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
