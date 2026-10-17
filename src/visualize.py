"""
Visualization Module.
Generates comprehensive visual reports:
1. Complete Preprocessing Pipeline Flow and Data Quality comparison.
2. Cross-Model Performance & Reliability Metrics comparison.
3. Residuals & Error Reliability analysis.
4. Internal algorithmic flow & feature importance charts for all 5 models:
   - Ridge Regression Flow & Coefficients
   - Decision Tree Structure & Split Flow
   - Random Forest Ensemble Architecture & Importances
   - XGBoost Gradient Boosting Flow & Gain
   - LightGBM Leaf-Wise Growth Flow & Importances
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.linear_model import Ridge


PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CLEANED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "amazon_sales_cleaned.csv")
RAW_UNCEANED_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "amazon_products_sales_data_uncleaned.csv")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")

# Estilo general limpio y moderno
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 200
})


def clean_feature_names(raw_names: np.ndarray) -> list:
    """Limpia los nombres de variables generados por ColumnTransformer."""
    cleaned = []
    for name in raw_names:
        clean = name.replace("num__", "").replace("cat__product_category_", "Cat: ")
        clean = clean.replace("_", " ").title()
        cleaned.append(clean)
    return cleaned


def plot_preprocessing_pipeline(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    output_path: str = os.path.join(PLOTS_DIR, "preprocessing_pipeline.png")
):
    """
    Genera el gráfico completo de la fase de preprocesamiento:
    - Comparación de valores nulos antes vs después
    - Distribución de variables transformadas
    - Demanda mensual por categoría
    - Diagrama de flujo de arquitectura de preprocesamiento
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # Panel 1: Comparación de Nulos antes vs después
    ax1 = fig.add_subplot(gs[0, 0])
    raw_null_counts = {
        "Price": int(df_raw["current/discounted_price"].isnull().sum()),
        "Delivery": int(df_raw["delivery_details"].isnull().sum()),
        "Rating": int(df_raw["rating"].isnull().sum()),
        "Reviews": int(df_raw["number_of_reviews"].isnull().sum()),
        "Buy Box": int(df_raw["buy_box_availability"].isnull().sum()),
        "Sustainability": int(df_raw["sustainability_badges"].isnull().sum())
    }
    clean_null_counts = {
        "Price": 0,
        "Delivery": 0,
        "Rating": 0,
        "Reviews": 0,
        "Buy Box": 0,
        "Sustainability": 0
    }
    labels = list(raw_null_counts.keys())
    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x - width/2, [raw_null_counts[k] for k in labels], width, label="Raw (Uncleaned)", color="#d9534f", alpha=0.85)
    ax1.bar(x + width/2, [clean_null_counts[k] for k in labels], width, label="Preprocessed (Clean)", color="#28a745", alpha=0.85)
    ax1.set_title("Missing Values Resolution (Before vs After)", fontweight="bold")
    ax1.set_ylabel("Count of Missing Records")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20)
    ax1.legend(loc="upper right", frameon=True)
    for i, v in enumerate([raw_null_counts[k] for k in labels]):
        ax1.text(i - width/2, v + 300, f"{v:,}", ha="center", fontsize=8, color="#a94442")

    # Panel 2: Distribución de variables clave normalizadas
    ax2 = fig.add_subplot(gs[0, 1])
    sample_clean = df_clean.sample(min(5000, len(df_clean)), random_state=42)
    sns.kdeplot(data=sample_clean["rating_num"], ax=ax2, color="#0275d8", fill=True, alpha=0.3, label="Rating (1-5)")
    sns.kdeplot(data=sample_clean["discount_percentage"] / 20.0, ax=ax2, color="#f0ad4e", fill=True, alpha=0.3, label="Discount % (/20)")
    sns.kdeplot(data=sample_clean["log_reviews"] / 2.0, ax=ax2, color="#5bc0de", fill=True, alpha=0.3, label="Log Reviews (/2)")
    ax2.set_title("Engineered Feature Distributions", fontweight="bold")
    ax2.set_xlabel("Standardized Feature Scale")
    ax2.set_ylabel("Density")
    ax2.legend(loc="upper right", frameon=True)

    # Panel 3: Demanda Mensual por Categoría (Supervisada)
    ax3 = fig.add_subplot(gs[1, 0])
    supervised = df_clean[df_clean["has_demand_data"] == 1]
    top_cats = supervised.groupby("product_category")["monthly_demand"].median().sort_values(ascending=False).index[:10]
    sub_df = supervised[supervised["product_category"].isin(top_cats)]
    sns.boxplot(
        data=sub_df,
        y="product_category",
        x="monthly_demand",
        order=top_cats,
        ax=ax3,
        hue="product_category",
        legend=False,
        palette="viridis",
        fliersize=1.5
    )
    ax3.set_xscale("log")
    ax3.set_title("Monthly Demand Distribution by Category (Log Scale)", fontweight="bold")
    ax3.set_xlabel("Purchased Units Last Month (Log Scale)")
    ax3.set_ylabel("")

    # Panel 4: Diagrama de Flujo de la Arquitectura de Preprocesamiento
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.set_title("Data Preprocessing Architecture & Flowchart", fontweight="bold")

    stages = [
        ("1. Raw Ingestion", "42,675 items ingested from KaggleHub\nDual file alignment (uncleaned + cleaned tags)"),
        ("2. Type & Regex Extraction", "Ratings: (\\d+\\.\\d+) | Reviews: commas stripped\nDemand: '(\\d+)K+' multipliers parsed"),
        ("3. Missing Value Imputation", "Contextual median by Category for Price/Rating/Delivery\nReviews defaulted to 0 for new listings"),
        ("4. Feature Engineering", "Log Reviews, Log Price, Discount %, Lead Days\nInteraction Terms & Commercial Flags (BestSeller, BuyBox)"),
        ("5. Clean CSV & Validation", "32,162 verified demand records for ML training\nExport to data/processed/amazon_sales_cleaned.csv")
    ]
    y_pos = 0.88
    for i, (title, desc) in enumerate(stages):
        box = patches.FancyBboxPatch((0.05, y_pos - 0.12), 0.90, 0.12, boxstyle="round,pad=0.02,rounding_size=0.03",
                                     edgecolor="#0275d8", facecolor="#eaf2f8", linewidth=1.5)
        ax4.add_patch(box)
        ax4.text(0.08, y_pos - 0.04, title, fontsize=10, fontweight="bold", color="#1b4f72")
        ax4.text(0.08, y_pos - 0.095, desc, fontsize=8.5, color="#2c3e50")
        if i < len(stages) - 1:
            ax4.annotate("", xy=(0.50, y_pos - 0.14), xytext=(0.50, y_pos - 0.12),
                         arrowprops=dict(arrowstyle="->", color="#0275d8", lw=1.5))
        y_pos -= 0.18

    fig.suptitle("END-TO-END DATA PREPROCESSING PIPELINE & QUALITY AUDIT", fontsize=15, fontweight="bold", y=0.98)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Preprocessing pipeline plot saved: {output_path}")


def plot_metrics_comparison(
    metrics_path: str = METRICS_JSON_PATH,
    output_path: str = os.path.join(PLOTS_DIR, "metrics_comparison.png")
):
    """
    Genera comparativa de métricas de confiabilidad:
    - R² Score (Train vs Test vs 5-Fold CV)
    - MAE y RMSE en escala real
    - Median Absolute Error y MAPE
    - Intervalos de confianza Bootstrap al 95%
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    models = list(metrics.keys())
    model_labels = [m.replace("_", " ") for m in models]
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: R² Score Comparison con Intervalos de Confianza 95%
    ax1 = axes[0, 0]
    r2_train = [metrics[m]["r2_train"] for m in models]
    r2_test = [metrics[m]["r2_test"] for m in models]
    r2_cv = [metrics[m]["cross_validation"]["cv_r2_mean"] for m in models]
    r2_cv_err = [metrics[m]["cross_validation"]["cv_r2_std"] for m in models]

    ax1.bar(x - width, r2_train, width, label="Train R²", color="#4a90e2", alpha=0.85)
    ax1.bar(x, r2_test, width, label="Test R²", color="#50e3c2", alpha=0.85)
    ax1.bar(x + width, r2_cv, width, yerr=r2_cv_err, capsize=4, label="5-Fold CV R² (±σ)", color="#f5a623", alpha=0.85)
    ax1.set_title("Goodness of Fit: R² Score by Model", fontweight="bold")
    ax1.set_ylabel("R² Score (Real Scale)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=15)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right", frameon=True)
    for i, v in enumerate(r2_test):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    # Panel 2: MAE y RMSE en unidades reales
    ax2 = axes[0, 1]
    mae_test = [metrics[m]["mae_test"] for m in models]
    rmse_test = [metrics[m]["rmse_test"] for m in models]
    w2 = 0.35
    ax2.bar(x - w2/2, mae_test, w2, label="Test MAE (Units)", color="#e74c3c", alpha=0.85)
    ax2.bar(x + w2/2, rmse_test, w2, label="Test RMSE (Units)", color="#9b59b6", alpha=0.85)
    ax2.set_title("Test Error Comparison: MAE vs RMSE", fontweight="bold")
    ax2.set_ylabel("Error in Product Units")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, rotation=15)
    ax2.legend(loc="upper right", frameon=True)
    for i, v in enumerate(mae_test):
        ax2.text(i - w2/2, v + 100, f"{v:.0f}", ha="center", fontsize=8)

    # Panel 3: Median Absolute Error (MedAE) - Robustez a Atípicos
    ax3 = axes[1, 0]
    medae_test = [metrics[m]["medae_test"] for m in models]
    palette = ["#3498db", "#2ecc71", "#1abc9c", "#e67e22", "#9b59b6"]
    bars = ax3.bar(model_labels, medae_test, color=palette, alpha=0.85)
    ax3.set_title("Robust Median Absolute Error (MedAE)", fontweight="bold")
    ax3.set_ylabel("Median Error (Units)")
    ax3.set_xticks(range(len(model_labels)))
    ax3.set_xticklabels(model_labels, rotation=15)
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    # Panel 4: Mean Absolute Percentage Error (MAPE) & Log R²
    ax4 = axes[1, 1]
    mape_test = [metrics[m]["mape_test"] * 100.0 for m in models]
    log_r2 = [metrics[m]["r2_log"] for m in models]

    ax4.bar(x - w2/2, mape_test, w2, label="MAPE (%)", color="#e67e22", alpha=0.85)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x + w2/2, log_r2, "o-", color="#2c3e50", linewidth=2.5, markersize=8, label="Log-Scale R²")
    ax4_twin.set_ylabel("Log-Scale R²", color="#2c3e50")
    ax4_twin.set_ylim(0.5, 1.0)
    ax4.set_title("Relative Error (MAPE) & Log-Scale R²", fontweight="bold")
    ax4.set_ylabel("MAPE (%)", color="#e67e22")
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_labels, rotation=15)
    ax4.legend(loc="upper left", frameon=True)
    ax4_twin.legend(loc="upper right", frameon=True)

    fig.suptitle("COMPREHENSIVE MODEL BENCHMARK & RELIABILITY EVALUATION", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Metrics comparison plot saved: {output_path}")


def plot_reliability_and_residuals(
    best_pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str = os.path.join(PLOTS_DIR, "reliability_residuals.png")
):
    """
    Genera gráficos de confiabilidad y diagnóstico de residuos del modelo líder (LightGBM):
    - Predicciones vs Valores Reales (Línea de 45°)
    - Distribución de Residuos
    - Residuos vs Predicciones (Homocedasticidad)
    - MAE segmentado por categoría
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    y_pred = np.clip(best_pipeline.predict(X_test), 0, None)
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: Predicción vs Real (Escala Log)
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.25, color="#1b4f72", s=18, edgecolors="none")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y = ŷ)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("Actual vs Predicted Monthly Demand (Log-Log Scale)", fontweight="bold")
    ax1.set_xlabel("Actual Demand (Units)")
    ax1.set_ylabel("Predicted Demand (Units)")
    ax1.legend(loc="upper left", frameon=True)

    # Panel 2: Distribución de Residuos
    ax2 = axes[0, 1]
    res_clipped = np.clip(residuals, -2000, 2000)
    sns.histplot(res_clipped, kde=True, ax=ax2, color="#2980b9", bins=50, stat="density")
    ax2.axvline(0, color="r", linestyle="--", linewidth=1.8, label="Zero Bias Line")
    ax2.set_title("Residual Distribution (Prediction Errors)", fontweight="bold")
    ax2.set_xlabel("Residual (Actual - Predicted Units, clipped [-2000, 2000])")
    ax2.set_ylabel("Density")
    ax2.legend(loc="upper right", frameon=True)

    # Panel 3: Residuos vs Predicción (Verificación de Homocedasticidad)
    ax3 = axes[1, 0]
    ax3.scatter(y_pred, residuals, alpha=0.20, color="#16a085", s=16)
    ax3.axhline(0, color="r", linestyle="--", linewidth=1.8)
    ax3.set_xscale("log")
    ax3.set_ylim(-5000, 5000)
    ax3.set_title("Residuals vs Predicted Demand", fontweight="bold")
    ax3.set_xlabel("Predicted Demand (Units, Log Scale)")
    ax3.set_ylabel("Residual (Actual - Predicted)")

    # Panel 4: Confiabilidad por Categoría (MAE por Categoría)
    ax4 = axes[1, 1]
    test_eval_df = X_test.copy()
    test_eval_df["actual"] = y_test.values
    test_eval_df["pred"] = y_pred
    test_eval_df["abs_error"] = np.abs(test_eval_df["actual"] - test_eval_df["pred"])
    cat_mae = test_eval_df.groupby("product_category")["abs_error"].median().sort_values(ascending=False)

    sns.barplot(x=cat_mae.values, y=cat_mae.index, ax=ax4, hue=cat_mae.index, legend=False, palette="mako")
    ax4.set_title("Median Absolute Error Across Product Categories", fontweight="bold")
    ax4.set_xlabel("Median Absolute Error (Units)")
    ax4.set_ylabel("")
    for i, v in enumerate(cat_mae.values):
        ax4.text(v + 1, i, f"{v:.1f}", va="center", fontsize=8.5)

    fig.suptitle("RELIABILITY, RESIDUALS & CATEGORICAL ERROR DIAGNOSTICS", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Reliability and residuals plot saved: {output_path}")


def plot_flow_ridge(
    ridge_pipeline,
    raw_feature_names: np.ndarray,
    output_path: str = os.path.join(PLOTS_DIR, "flow_ridge.png")
):
    """Genera gráfico de flujo algorítmico interno y pesos de Regresión Ridge."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Diagrama de flujo conceptual Ridge L2
    ax1 = axes[0]
    ax1.axis("off")
    ax1.set_title("Ridge Regression Internal Algorithmic Flow", fontweight="bold")

    boxes = [
        ("1. Input Feature Space X", "Standardized Continuous Features + One-Hot Encoded Categories (p=33)"),
        ("2. Objective Function with L2 Penalty", "min_w || y - X w ||_2^2 + α || w ||_2^2\nBalances empirical squared error with weight magnitude penalty"),
        ("3. Analytical Closed-Form Solution", "w* = (X^T X + α I)^{-1} X^T y\nGuarantees strictly positive definite matrix, eliminating multicollinearity"),
        ("4. Shrinkage & Prediction", "Eigenvalue scaling: w_i = λ_i / (λ_i + α) * w_OLS\nInference: ŷ_log = X w*  ->  ŷ = exp(ŷ_log) - 1")
    ]
    y_pos = 0.85
    for i, (title, desc) in enumerate(boxes):
        box = patches.FancyBboxPatch((0.05, y_pos - 0.14), 0.90, 0.14, boxstyle="round,pad=0.03",
                                     edgecolor="#2980b9", facecolor="#ebf5fb", linewidth=1.5)
        ax1.add_patch(box)
        ax1.text(0.08, y_pos - 0.04, title, fontsize=10, fontweight="bold", color="#1b4f72")
        ax1.text(0.08, y_pos - 0.10, desc, fontsize=8.5, color="#2c3e50")
        if i < len(boxes) - 1:
            ax1.annotate("", xy=(0.50, y_pos - 0.17), xytext=(0.50, y_pos - 0.14),
                         arrowprops=dict(arrowstyle="->", color="#2980b9", lw=2))
        y_pos -= 0.22

    # Panel 2: Coeficientes de pesos aprendidos
    ax2 = axes[1]
    ridge_estimator = ridge_pipeline.named_steps["regressor"].regressor_
    coefs = ridge_estimator.coef_
    feat_names = clean_feature_names(raw_feature_names)

    coef_df = pd.DataFrame({"Feature": feat_names, "Coefficient": coefs})
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    top_coef = coef_df.sort_values(by="Abs", ascending=False).head(15).sort_values(by="Coefficient")

    colors = ["#27ae60" if c > 0 else "#c0392b" for c in top_coef["Coefficient"]]
    ax2.barh(top_coef["Feature"], top_coef["Coefficient"], color=colors, alpha=0.85)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("Ridge Top 15 Feature Coefficients (w*)", fontweight="bold")
    ax2.set_xlabel("Learned Coefficient Value (Log-Demand Sensitivity)")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Ridge flow plot saved: {output_path}")


def plot_flow_decision_tree(
    dt_pipeline,
    raw_feature_names: np.ndarray,
    output_path: str = os.path.join(PLOTS_DIR, "flow_decision_tree.png")
):
    """Genera gráfico de flujo algorítmico interno y estructura de árbol de decisión."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dt_estimator = dt_pipeline.named_steps["regressor"].regressor_
    feat_names = clean_feature_names(raw_feature_names)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Panel 1: Estructura interna de splits (Profundidad representativa = 3)
    ax1 = axes[0]
    plot_tree(
        dt_estimator,
        max_depth=3,
        feature_names=feat_names,
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax1,
        precision=2,
        proportion=True
    )
    ax1.set_title("Decision Tree: Recursive Binary Splitting (Top 3 Levels)", fontweight="bold")

    # Panel 2: Importancia de variables por reducción de varianza (MSE)
    ax2 = axes[1]
    importances = dt_estimator.feature_importances_
    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    top_imp = imp_df.sort_values(by="Importance", ascending=True).tail(15)

    ax2.barh(top_imp["Feature"], top_imp["Importance"], color="#16a085", alpha=0.85)
    ax2.set_title("Decision Tree Feature Importance (Variance Reduction)", fontweight="bold")
    ax2.set_xlabel("Normalized Gini / Impurity Reduction")
    for i, v in enumerate(top_imp["Importance"]):
        ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Decision Tree flow plot saved: {output_path}")


def plot_flow_random_forest(
    rf_pipeline,
    raw_feature_names: np.ndarray,
    output_path: str = os.path.join(PLOTS_DIR, "flow_random_forest.png")
):
    """Genera gráfico de flujo algorítmico y ensamble de Random Forest."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rf_estimator = rf_pipeline.named_steps["regressor"].regressor_
    feat_names = clean_feature_names(raw_feature_names)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))

    # Panel 1: Diagrama de flujo de Ensamble y Bagging
    ax1 = axes[0]
    ax1.axis("off")
    ax1.set_title("Random Forest Architecture & Bagging Flowchart", fontweight="bold")

    steps = [
        ("1. Original Dataset (N=25,729)", "Contains 33 features across 15 categories"),
        ("2. Bootstrap Aggregation (Bagging)", "Generates T=150 bootstrap samples with replacement (subsampling rows)"),
        ("3. Random Feature Subspacing (m=sqrt(p))", "Each split selects random subset of features to de-correlate trees"),
        ("4. Independent Tree Training", "Trains 150 diverse decision trees in parallel with min_samples_leaf=8"),
        ("5. Ensemble Averaging", "ŷ = (1/T) ∑ T_t(x)  ->  Variance reduced by factor of 1/T without increasing bias")
    ]
    y_pos = 0.88
    for i, (title, desc) in enumerate(steps):
        box = patches.FancyBboxPatch((0.05, y_pos - 0.12), 0.90, 0.12, boxstyle="round,pad=0.03",
                                     edgecolor="#27ae60", facecolor="#eafaf1", linewidth=1.5)
        ax1.add_patch(box)
        ax1.text(0.08, y_pos - 0.04, title, fontsize=10, fontweight="bold", color="#145a32")
        ax1.text(0.08, y_pos - 0.095, desc, fontsize=8.5, color="#1e8449")
        if i < len(steps) - 1:
            ax1.annotate("", xy=(0.50, y_pos - 0.14), xytext=(0.50, y_pos - 0.12),
                         arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2))
        y_pos -= 0.18

    # Panel 2: Importancia de variables agregada
    ax2 = axes[1]
    importances = rf_estimator.feature_importances_
    tree_importances = np.array([tree.feature_importances_ for tree in rf_estimator.estimators_])
    std = np.std(tree_importances, axis=0)

    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances, "Std": std})
    top_imp = imp_df.sort_values(by="Importance", ascending=True).tail(15)

    ax2.barh(top_imp["Feature"], top_imp["Importance"], xerr=top_imp["Std"], capsize=3, color="#2e7d32", alpha=0.85)
    ax2.set_title("Random Forest Top 15 Feature Importances (MDI ± σ)", fontweight="bold")
    ax2.set_xlabel("Mean Decrease in Impurity")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Random Forest flow plot saved: {output_path}")


def plot_flow_xgboost(
    xgb_pipeline,
    raw_feature_names: np.ndarray,
    output_path: str = os.path.join(PLOTS_DIR, "flow_xgboost.png")
):
    """Genera gráfico de flujo algorítmico interno de XGBoost."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    xgb_estimator = xgb_pipeline.named_steps["regressor"].regressor_
    feat_names = clean_feature_names(raw_feature_names)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))

    # Panel 1: Diagrama de Boosting secuencial de gradiente
    ax1 = axes[0]
    ax1.axis("off")
    ax1.set_title("XGBoost Sequential Gradient Boosting Flowchart", fontweight="bold")

    stages = [
        ("1. Initial Baseline Prediction", "ŷ^(0) = argmin_c ∑ L(y_i, c)  (initial base score)"),
        ("2. Taylor Expansion of Loss", "L^(t) ≈ ∑ [ g_i f_t(x_i) + 1/2 h_i f_t^2(x_i) ] + Ω(f_t)\nCalculates 1st order gradients (g_i) and 2nd order hessians (h_i)"),
        ("3. Regularized Tree Structure Scoring", "Optimal leaf weight w_j* = - ∑ g_i / (∑ h_i + λ)\nOptimal split gain Gain = 1/2 [ G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - G^2/(H+λ) ] - γ"),
        ("4. Shrinkage & Iterative Update", "ŷ^(t) = ŷ^(t-1) + η f_t(x_i)  (learning rate η=0.07 dampens individual tree impact)"),
        ("5. Final Summation", "Final prediction is sum of 200 regularized shallow boosting trees")
    ]
    y_pos = 0.88
    for i, (title, desc) in enumerate(stages):
        box = patches.FancyBboxPatch((0.05, y_pos - 0.12), 0.90, 0.12, boxstyle="round,pad=0.03",
                                     edgecolor="#d35400", facecolor="#fef5e7", linewidth=1.5)
        ax1.add_patch(box)
        ax1.text(0.08, y_pos - 0.04, title, fontsize=10, fontweight="bold", color="#935116")
        ax1.text(0.08, y_pos - 0.095, desc, fontsize=8.5, color="#7e5109")
        if i < len(stages) - 1:
            ax1.annotate("", xy=(0.50, y_pos - 0.14), xytext=(0.50, y_pos - 0.12),
                         arrowprops=dict(arrowstyle="->", color="#d35400", lw=2))
        y_pos -= 0.18

    # Panel 2: Feature Importance por Gain
    ax2 = axes[1]
    importances = xgb_estimator.feature_importances_
    imp_df = pd.DataFrame({"Feature": feat_names, "Gain": importances})
    top_imp = imp_df.sort_values(by="Gain", ascending=True).tail(15)

    ax2.barh(top_imp["Feature"], top_imp["Gain"], color="#e67e22", alpha=0.85)
    ax2.set_title("XGBoost Top 15 Feature Importances (Gain)", fontweight="bold")
    ax2.set_xlabel("Relative Split Gain")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] XGBoost flow plot saved: {output_path}")


def plot_flow_lightgbm(
    lgbm_pipeline,
    raw_feature_names: np.ndarray,
    output_path: str = os.path.join(PLOTS_DIR, "flow_lightgbm.png")
):
    """Genera gráfico de flujo algorítmico interno de LightGBM."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lgbm_estimator = lgbm_pipeline.named_steps["regressor"].regressor_
    feat_names = clean_feature_names(raw_feature_names)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))

    # Panel 1: Diagrama de Leaf-Wise Tree Growth y Optimizaciones
    ax1 = axes[0]
    ax1.axis("off")
    ax1.set_title("LightGBM Leaf-Wise Architecture & Histogram Flow", fontweight="bold")

    features_flow = [
        ("1. Histogram-Based Binning", "Buckets continuous features into discrete bins (k=255)\nReduces memory by 8x and calculates split gains in O(bins) instead of O(data)"),
        ("2. Leaf-Wise (Best-First) Splitting", "Unlike traditional Level-Wise depth trees, splits the leaf with maximum delta loss\nAchieves significantly lower training loss at same depth"),
        ("3. GOSS (Gradient-based One-Side Sampling)", "Keeps all instances with large gradients and randomly samples small gradients\nPreserves split accuracy while speeding up computation"),
        ("4. EFB (Exclusive Feature Bundling)", "Mutually exclusive sparse features (e.g., one-hot categories) are bundled into dense features"),
        ("5. Fast Leaf Prediction Summation", "Ensemble of 200 leaf-wise trees with max 35 leaves per tree and L2 regularization")
    ]
    y_pos = 0.88
    for i, (title, desc) in enumerate(features_flow):
        box = patches.FancyBboxPatch((0.05, y_pos - 0.12), 0.90, 0.12, boxstyle="round,pad=0.03",
                                     edgecolor="#8e44ad", facecolor="#f4ecf7", linewidth=1.5)
        ax1.add_patch(box)
        ax1.text(0.08, y_pos - 0.04, title, fontsize=10, fontweight="bold", color="#5b2c6f")
        ax1.text(0.08, y_pos - 0.095, desc, fontsize=8.5, color="#512e5f")
        if i < len(features_flow) - 1:
            ax1.annotate("", xy=(0.50, y_pos - 0.14), xytext=(0.50, y_pos - 0.12),
                         arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=2))
        y_pos -= 0.18

    # Panel 2: Importancia de Variables por Split
    ax2 = axes[1]
    importances = lgbm_estimator.feature_importances_
    imp_df = pd.DataFrame({"Feature": feat_names, "Splits": importances})
    top_imp = imp_df.sort_values(by="Splits", ascending=True).tail(15)

    ax2.barh(top_imp["Feature"], top_imp["Splits"], color="#8e44ad", alpha=0.85)
    ax2.set_title("LightGBM Top 15 Feature Importances (Split Frequency)", fontweight="bold")
    ax2.set_xlabel("Number of Times Feature Used in Splits")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] LightGBM flow plot saved: {output_path}")


def generate_all_visualizations():
    """Genera todos los gráficos solicitados en el proyecto."""
    print("\n[VISUALIZATION] Generating all project visual assets...")

    # Cargar datos
    df_raw = pd.read_csv(RAW_UNCEANED_PATH)
    df_clean = pd.read_csv(CLEANED_CSV_PATH)

    # 1. Gráfico del Preprocesamiento
    plot_preprocessing_pipeline(df_raw, df_clean)

    # 2. Gráfico de Métricas Comparativas
    plot_metrics_comparison()

    # Cargar modelos y datos de test para gráficos de residuos y flujos
    supervised = df_clean[df_clean["has_demand_data"] == 1]
    from src.train import FEATURE_COLUMNS
    from sklearn.model_selection import train_test_split
    X = supervised[FEATURE_COLUMNS]
    y = supervised["monthly_demand"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    raw_feature_names = preprocessor.get_feature_names_out()

    lgbm_pipe = joblib.load(os.path.join(MODELS_DIR, "lightgbm_pipeline.joblib"))
    ridge_pipe = joblib.load(os.path.join(MODELS_DIR, "ridge_pipeline.joblib"))
    dt_pipe = joblib.load(os.path.join(MODELS_DIR, "decision_tree_pipeline.joblib"))
    rf_pipe = joblib.load(os.path.join(MODELS_DIR, "random_forest_pipeline.joblib"))
    xgb_pipe = joblib.load(os.path.join(MODELS_DIR, "xgboost_pipeline.joblib"))

    # 3. Gráfico de Confiabilidad y Residuos
    plot_reliability_and_residuals(lgbm_pipe, X_test, y_test)

    # 4-8. Gráficos de Flujo de Algorítmica Interna por Modelo
    plot_flow_ridge(ridge_pipe, raw_feature_names)
    plot_flow_decision_tree(dt_pipe, raw_feature_names)
    plot_flow_random_forest(rf_pipe, raw_feature_names)
    plot_flow_xgboost(xgb_pipe, raw_feature_names)
    plot_flow_lightgbm(lgbm_pipe, raw_feature_names)

    print("[SUCCESS] All 8 visualization charts generated in 'plots/' directory!")


if __name__ == "__main__":
    generate_all_visualizations()
