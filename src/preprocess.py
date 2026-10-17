"""
Data Preprocessing and Feature Engineering Module.
Performs data cleaning, regex extraction, type conversion,
missing value imputation, feature engineering, and exports clean CSV.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


RAW_UNCEANED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "amazon_products_sales_data_uncleaned.csv")
RAW_CLEANED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "amazon_products_sales_data_cleaned.csv")
PROCESSED_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "amazon_sales_cleaned.csv")


def extract_rating(val: Any) -> float:
    """Extrae el valor numérico de la calificación (rating)."""
    if pd.isna(val):
        return np.nan
    match = re.search(r"(\d+\.?\d*)", str(val))
    if match:
        val_float = float(match.group(1))
        return val_float if 1.0 <= val_float <= 5.0 else np.nan
    return np.nan


def extract_reviews(val: Any) -> float:
    """Extrae el número de reseñas eliminando comas."""
    if pd.isna(val):
        return np.nan
    match = re.search(r"([\d,]+)", str(val))
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return np.nan
    return np.nan


def extract_demand(val: Any) -> float:
    """Extrae la cantidad de unidades compradas en el último mes."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    match_k = re.search(r"(\d+(?:\.\d+)?)\s*K\+\s*bought", val_str, re.IGNORECASE)
    if match_k:
        return float(match_k.group(1)) * 1000.0
    match_m = re.search(r"(\d+(?:\.\d+)?)\s*M\+\s*bought", val_str, re.IGNORECASE)
    if match_m:
        return float(match_m.group(1)) * 1000000.0
    match_num = re.search(r"(\d+)\+\s*bought", val_str, re.IGNORECASE)
    if match_num:
        return float(match_num.group(1))
    return np.nan


def extract_price(val: Any) -> float:
    """Extrae el valor monetario en float eliminando caracteres especiales."""
    if pd.isna(val):
        return np.nan
    clean_str = str(val).replace("$", "").replace(",", "").strip()
    match = re.search(r"(\d+\.?\d*)", clean_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return np.nan
    return np.nan


def extract_delivery_lead_days(delivery_str: Any, collected_at_str: Any) -> float:
    """Calcula los días estimados de entrega a partir de la fecha de recolección."""
    if pd.isna(delivery_str) or pd.isna(collected_at_str):
        return np.nan
    match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d+)", str(delivery_str), re.IGNORECASE)
    if match:
        month_str, day_str = match.group(1).capitalize(), match.group(2)
        try:
            # El dataset corresponde a recolectas en agosto 2025
            deliv_dt = pd.to_datetime(f"2025 {month_str} {day_str}", format="%Y %b %d")
            col_dt = pd.to_datetime(collected_at_str)
            days = (deliv_dt.date() - col_dt.date()).days
            return float(days) if 0 <= days <= 45 else np.nan
        except Exception:
            return np.nan
    return np.nan


def extract_coupon_discount(coupon_str: Any) -> float:
    """Extrae el porcentaje o monto del cupón si existe."""
    if pd.isna(coupon_str):
        return 0.0
    s = str(coupon_str)
    # Porcentaje
    match_pct = re.search(r"Save\s+(\d+)%", s, re.IGNORECASE)
    if match_pct:
        return float(match_pct.group(1))
    # Dólares
    match_dlr = re.search(r"Save\s+\$(\d+\.?\d*)", s, re.IGNORECASE)
    if match_dlr:
        return float(match_dlr.group(1))
    return 0.0


def preprocess_data(
    raw_uncleaned_path: str = RAW_UNCEANED_PATH,
    raw_cleaned_path: str = RAW_CLEANED_PATH,
    output_csv_path: str = PROCESSED_CSV_PATH
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta el pipeline completo de preprocesamiento y limpieza de datos.
    Exporta el CSV limpio y devuelve metadata del proceso.
    """
    print("[INFO] Loading raw datasets...")
    df_raw = pd.read_csv(raw_uncleaned_path)
    df_ref = pd.read_csv(raw_cleaned_path)

    initial_shape = df_raw.shape
    initial_nulls = df_raw.isnull().sum().to_dict()

    # Consolidar categoría verificada
    df = df_raw.copy()
    if "product_category" in df_ref.columns and len(df_ref) == len(df):
        df["product_category"] = df_ref["product_category"].fillna("Other Electronics").str.strip()
    else:
        df["product_category"] = "Other Electronics"

    # 1. Limpieza de texto de título y métricas textuales
    df["title_clean"] = df["title"].fillna("").astype(str).str.strip()
    df["title_length"] = df["title_clean"].apply(len)
    df["title_word_count"] = df["title_clean"].apply(lambda t: len(t.split()))

    # 2. Extracción de rating y número de reseñas
    df["rating_num"] = df["rating"].apply(extract_rating)
    df["reviews_num"] = df["number_of_reviews"].apply(extract_reviews)

    # 3. Extracción de variable objetivo: demanda mensual (bought_in_last_month)
    df["monthly_demand"] = df["bought_in_last_month"].apply(extract_demand)
    df["has_demand_data"] = df["monthly_demand"].notna().astype(int)

    # 4. Precios: actual/descuento, variante y precio de lista
    curr_p = df["current/discounted_price"].apply(extract_price)
    var_p = df["price_on_variant"].apply(extract_price)
    list_p = df["listed_price"].apply(extract_price)

    # Reconstrucción de precio descontado y precio original
    df["discounted_price"] = curr_p.combine_first(var_p).combine_first(list_p)
    df["original_price"] = list_p.combine_first(df["discounted_price"])

    # 5. Fechas y tiempo estimado de entrega (lead days)
    lead_days = [
        extract_delivery_lead_days(d, c) for d, c in zip(df["delivery_details"], df["collected_at"])
    ]
    df["delivery_lead_days"] = lead_days

    # 6. Indicadores comerciales binarios
    df["is_best_seller"] = df["is_best_seller"].astype(str).str.contains("Best Seller", case=False).astype(int)
    df["is_amazons_choice"] = df["is_best_seller"].astype(str).str.contains("Amazon's", case=False).astype(int)
    df["is_limited_deal"] = df["is_best_seller"].astype(str).str.contains("Limited time deal", case=False).astype(int)
    df["is_sponsored"] = df["is_sponsored"].astype(str).str.contains("Sponsored", case=False).astype(int)
    
    # Cupones
    df["has_coupon"] = df["is_couponed"].astype(str).str.contains("Save", case=False).astype(int)
    df["coupon_discount_value"] = df["is_couponed"].apply(extract_coupon_discount)

    # Buy box y sustentabilidad
    df["buy_box_availability"] = df["buy_box_availability"].astype(str).str.contains("Add to cart", case=False).astype(int)
    df["has_sustainability"] = df["sustainability_badges"].notna().astype(int)

    # 7. Imputación contextual basada en categorías
    # Ratings: mediana por categoría
    category_rating_median = df.groupby("product_category")["rating_num"].transform("median")
    df["rating_num"] = df["rating_num"].fillna(category_rating_median).fillna(df["rating_num"].median()).fillna(4.3)

    # Reviews: nuevos listados = 0 o mediana de categoría
    df["reviews_num"] = df["reviews_num"].fillna(0.0)

    # Precios: mediana por categoría si persiste nulo
    cat_price_median = df.groupby("product_category")["discounted_price"].transform("median")
    df["discounted_price"] = df["discounted_price"].fillna(cat_price_median).fillna(df["discounted_price"].median()).fillna(25.0)
    df["original_price"] = df["original_price"].fillna(df["discounted_price"])
    # Corrección de consistencia: original_price >= discounted_price
    df["original_price"] = np.maximum(df["original_price"], df["discounted_price"])

    # Lead days: mediana por categoría
    cat_lead_median = df.groupby("product_category")["delivery_lead_days"].transform("median")
    df["delivery_lead_days"] = df["delivery_lead_days"].fillna(cat_lead_median).fillna(df["delivery_lead_days"].median()).fillna(7.0)

    # 8. Ingeniería de variables derivadas
    df["discount_amount"] = np.maximum(0.0, df["original_price"] - df["discounted_price"])
    df["discount_percentage"] = np.clip(
        (df["discount_amount"] / np.maximum(1e-4, df["original_price"])) * 100.0, 0.0, 100.0
    )
    df["log_reviews"] = np.log1p(df["reviews_num"])
    df["log_price"] = np.log1p(df["discounted_price"])
    df["rating_review_score"] = df["rating_num"] * df["log_reviews"]

    # Deduplicación considerando título, categoría y fecha de recolección
    df = df.drop_duplicates(subset=["title_clean", "product_category", "collected_at"]).reset_index(drop=True)

    # Selección final de columnas limpias y ordenadas
    cleaned_columns = [
        "title_clean",
        "product_category",
        "monthly_demand",
        "has_demand_data",
        "discounted_price",
        "original_price",
        "discount_amount",
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
        "title_word_count",
        "collected_at"
    ]
    df_clean = df[cleaned_columns].copy()

    # Exportar CSV limpio
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_clean.to_csv(output_csv_path, index=False)
    print(f"[SUCCESS] Clean dataset saved to: {output_csv_path} with {len(df_clean)} records.")

    final_nulls = df_clean.isnull().sum().to_dict()
    metadata = {
        "initial_rows": initial_shape[0],
        "initial_cols": initial_shape[1],
        "final_rows": len(df_clean),
        "final_cols": len(df_clean.columns),
        "initial_nulls": initial_nulls,
        "final_nulls": final_nulls,
        "demand_records_count": int(df_clean["has_demand_data"].sum()),
        "categories_count": int(df_clean["product_category"].nunique())
    }

    return df_clean, metadata


if __name__ == "__main__":
    df_clean, meta = preprocess_data()
    print("[METADATA SUMMARY]:", meta)
