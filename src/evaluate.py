"""
Model Evaluation and Reliability Analysis Module.
Computes statistical reliability metrics (R², RMSE, MAE, MedAE, MAPE),
5-Fold Cross-Validation, Bootstrap Confidence Intervals, and Residual Diagnostics.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import KFold


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")


def compute_bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """Calcula intervalos de confianza al 95% mediante bootstrapping no paramétrico."""
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    r2_samples: List[float] = []
    mae_samples: List[float] = []
    rmse_samples: List[float] = []

    for _ in range(n_iterations):
        indices = rng.integers(0, n, size=n)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]

        r2_samples.append(r2_score(sample_true, sample_pred))
        mae_samples.append(mean_absolute_error(sample_true, sample_pred))
        rmse_samples.append(np.sqrt(mean_squared_error(sample_true, sample_pred)))

    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    return {
        "r2_ci95": {
            "lower": float(np.percentile(r2_samples, lower_pct)),
            "upper": float(np.percentile(r2_samples, upper_pct))
        },
        "mae_ci95": {
            "lower": float(np.percentile(mae_samples, lower_pct)),
            "upper": float(np.percentile(mae_samples, upper_pct))
        },
        "rmse_ci95": {
            "lower": float(np.percentile(rmse_samples, lower_pct)),
            "upper": float(np.percentile(rmse_samples, upper_pct))
        }
    }


def compute_cross_validation_metrics(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """Calcula métricas de validación cruzada k-fold para evaluar generalización."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    r2_scores = []
    mae_scores = []
    rmse_scores = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        val_pred = np.clip(pipeline.predict(X_val), 0, None)

        r2_scores.append(r2_score(y_val, val_pred))
        mae_scores.append(mean_absolute_error(y_val, val_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, val_pred)))

    return {
        "cv_r2_mean": float(np.mean(r2_scores)),
        "cv_r2_std": float(np.std(r2_scores)),
        "cv_mae_mean": float(np.mean(mae_scores)),
        "cv_mae_std": float(np.std(mae_scores)),
        "cv_rmse_mean": float(np.mean(rmse_scores)),
        "cv_rmse_std": float(np.std(rmse_scores))
    }


def evaluate_all_models(
    trained_pipelines: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_path: str = METRICS_JSON_PATH
) -> Dict[str, Any]:
    """
    Evalúa exhaustivamente todos los modelos con métricas de desempeño y confiabilidad.
    Genera reporte JSON y resumen en consola.
    """
    results: Dict[str, Any] = {}
    y_test_arr = y_test.values
    y_log_test = np.log1p(y_test_arr)

    print("\n================================ MODEL EVALUATION SUMMARY ================================")
    for model_name, pipe in trained_pipelines.items():
        print(f"\n[EVALUATING] {model_name}...")

        # Predicciones
        y_train_pred = np.clip(pipe.predict(X_train), 0, None)
        y_test_pred = np.clip(pipe.predict(X_test), 0, None)

        y_log_test_pred = np.log1p(y_test_pred)

        # Métricas de escala real
        r2_train = float(r2_score(y_train, y_train_pred))
        r2_test = float(r2_score(y_test, y_test_pred))
        mae_test = float(mean_absolute_error(y_test, y_test_pred))
        rmse_test = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        medae_test = float(median_absolute_error(y_test, y_test_pred))
        mape_test = float(mean_absolute_percentage_error(y_test, y_test_pred))

        # Métricas en escala logarítmica
        r2_log = float(r2_score(y_log_test, y_log_test_pred))
        mae_log = float(mean_absolute_error(y_log_test, y_log_test_pred))
        rmse_log = float(np.sqrt(mean_squared_error(y_log_test, y_log_test_pred)))

        # Residuos y sesgo
        residuals = y_test_arr - y_test_pred
        mean_residual = float(np.mean(residuals))
        std_residual = float(np.std(residuals))
        pct_within_30 = float(np.mean(np.abs(residuals / np.maximum(y_test_arr, 1)) <= 0.30) * 100.0)

        # Intervalos de confianza Bootstrap (95%)
        ci_95 = compute_bootstrap_confidence_interval(y_test_arr, y_test_pred)

        # Validación cruzada 5-fold sobre submuestra de validación rápida
        cv_sample_n = min(6000, len(X_train))
        cv_metrics = compute_cross_validation_metrics(
            pipe, X_train.iloc[:cv_sample_n], y_train.iloc[:cv_sample_n], n_splits=5
        )

        results[model_name] = {
            "r2_train": r2_train,
            "r2_test": r2_test,
            "mae_test": mae_test,
            "rmse_test": rmse_test,
            "medae_test": medae_test,
            "mape_test": mape_test,
            "r2_log": r2_log,
            "mae_log": mae_log,
            "rmse_log": rmse_log,
            "mean_residual": mean_residual,
            "std_residual": std_residual,
            "pct_within_30_pct_error": pct_within_30,
            "ci_95": ci_95,
            "cross_validation": cv_metrics
        }

        print(f"  Test R²:        {r2_test:.4f} (95% CI: [{ci_95['r2_ci95']['lower']:.4f}, {ci_95['r2_ci95']['upper']:.4f}])")
        print(f"  Test MAE:       {mae_test:.2f} units (95% CI: [{ci_95['mae_ci95']['lower']:.2f}, {ci_95['mae_ci95']['upper']:.2f}])")
        print(f"  Test RMSE:      {rmse_test:.2f} units (95% CI: [{ci_95['rmse_ci95']['lower']:.2f}, {ci_95['rmse_ci95']['upper']:.2f}])")
        print(f"  Median AE:      {medae_test:.2f} units | MAPE: {mape_test:.2%}")
        print(f"  Log R²:         {r2_log:.4f} | Log MAE: {mae_log:.4f}")
        print(f"  5-Fold CV R²:   {cv_metrics['cv_r2_mean']:.4f} ± {cv_metrics['cv_r2_std']:.4f}")
        print(f"  Mean Bias:      {mean_residual:.2f} (Residual Std: {std_residual:.2f})")

    # Guardar en archivo JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n[SUCCESS] All evaluation metrics exported to {output_path}")

    return results


if __name__ == "__main__":
    from src.train import train_models
    pipelines, X_train, X_test, y_train, y_test = train_models()
    evaluate_all_models(pipelines, X_train, X_test, y_train, y_test)
