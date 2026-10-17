# DemandMind: Amazon Product Sales Demand Forecasting

DemandMind es una solucion integral de Machine Learning disenada para estimar la demanda mensual de productos (`purchased_last_month`) segun su categoria e indicadores comerciales, a partir del dataset de Amazon de 42,000 articulos recolectado en 2025 (`ikramshah512/amazon-products-sales-dataset-42k-items-2025`).

---

## 1. Arquitectura del Proyecto

```
DemandMind/
├── .venv/                         # Entorno virtual de Python
├── data/
│   ├── raw/                       # Datasets crudos descargados desde Kaggle
│   │   ├── amazon_products_sales_data_uncleaned.csv
│   │   └── amazon_products_sales_data_cleaned.csv
│   └── processed/
│       └── amazon_sales_cleaned.csv  # CSV preprocesado y estandarizado
├── models/                        # Artefactos entrenados y serializados
│   ├── preprocessor.joblib        # Pipeline de transformacion y escalado
│   ├── ridge.joblib / ridge_pipeline.joblib
│   ├── decision_tree.joblib / decision_tree_pipeline.joblib
│   ├── random_forest.joblib / random_forest_pipeline.joblib
│   ├── xgboost.joblib / xgboost_pipeline.joblib
│   ├── lightgbm.joblib / lightgbm_pipeline.joblib
│   └── evaluation_metrics.json   # Metricas de desempeno y confiabilidad
├── plots/                         # Graficos generados
│   ├── preprocessing_pipeline.png # Resumen visual del preprocesamiento
│   ├── metrics_comparison.png     # Comparativa de metricas entre modelos
│   ├── reliability_residuals.png  # Diagnostico de residuos y confiabilidad
│   ├── flow_ridge.png             # Algoritmica interna y pesos de Ridge
│   ├── flow_decision_tree.png     # Algoritmica y splits de Arbol de Decision
│   ├── flow_random_forest.png     # Flujo de ensamble y feature importance de Random Forest
│   ├── flow_xgboost.png           # Flujo de boosting secuencial y ganancia de XGBoost
│   └── flow_lightgbm.png          # Flujo leaf-wise e importancia de splits de LightGBM
├── microservice/                  # Microservicio FastAPI con documentacion en Scalar
│   ├── app/                       # Logica modular (config, schemas, services, routes)
│   ├── static/charts/             # Almacenamiento de graficos generados para inferencias
│   ├── .env / .env.example        # Configuracion de entorno y Google Gemini API Key
│   ├── requirements.txt           # Dependencias del microservicio
│   └── run.py                     # Lanzador del servidor Uvicorn
├── src/
│   ├── download_data.py           # Descarga automatica via kagglehub
│   ├── preprocess.py              # Limpieza, imputacion e ingenieria de variables
│   ├── train.py                   # Entrenamiento y persistencia de modelos
│   ├── evaluate.py                # Calculo de metricas de confiabilidad y bootstrapping
│   ├── visualize.py               # Renderizado de graficos tecnicos
│   └── predict.py                 # Modulo de inferencia y consumo de modelos
├── main.py                        # Orquestador del pipeline completo
└── requirements.txt               # Dependencias del proyecto
```

---

## 2. Instalacion y Preparacion del Entorno

### Requisitos Previos
- Python 3.10 o superior (validado en Python 3.12).

### Creacion y Activacion del Entorno Virtual (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Instalacion de Dependencias
```powershell
pip install -r requirements.txt
```

---

## 3. Fase de Preprocesamiento y Limpieza de Datos

El archivo crudo presenta cadenas con simbolos monetarios, calificaciones en formato textual, conteos de resenas con comas, texto de insignias en lugar de demandas numericas y multiples valores nulos.

El modulo `src/preprocess.py` implementa tecnicas de preparacion de datos:
1. **Extraccion de Demanda Mensual (Target)**: Procesamiento mediante expresiones regulares de cadenas como `10K+ bought in past month`, `300+ bought in past month`, convirtiendolas a unidades numericas escaladas (`10,000`, `300`). Se identifican 32,162 registros con demanda mensual verificada.
2. **Normalizacion de Calificaciones**: Extraccion numerica en rango [1.0, 5.0] e imputacion contextual basada en la mediana de cada categoria.
3. **Saneamiento de Precios**: Reconstruccion del precio descontado combinando `current/discounted_price`, `price_on_variant` y `listed_price`. Imputacion por mediana de categoria para listados residuales y garantia de consistencia (`original_price >= discounted_price`).
4. **Estimacion de Tiempos Logisticos (`delivery_lead_days`)**: Calculo de dias transcurridos entre la fecha de recoleccion y la fecha de entrega prometida.
5. **Ingenieria de Caracteristicas**:
   - `discount_amount` y `discount_percentage`.
   - `log_reviews` = $\log(1 + \text{reviews\_num})$.
   - `log_price` = $\log(1 + \text{discounted\_price})$.
   - `rating_review_score` = $\text{rating\_num} \times \text{log\_reviews}$.
   - Variables binarias para insignias: `is_best_seller`, `is_amazons_choice`, `is_limited_deal`, `is_sponsored`, `has_coupon`, `buy_box_availability`, `has_sustainability`.
6. **Generacion del CSV Limpio**: Exportado en `data/processed/amazon_sales_cleaned.csv` con 42,673 filas y 25 columnas sin nulos en las variables de entrada.

---

## 4. Modelos de Machine Learning Implementados

Se entrenaron y evaluaron 5 familias de algoritmos supervisados para estimar la demanda mensual segun categoria y atributos de producto:

1. **Regresion Ridge**: Modelo lineal con penalizacion $L_2$ ($\alpha = 25.0$) que mitiga la multicolinealidad y provee una linea base analitica.
2. **Arboles de Decision (`DecisionTreeRegressor`)**: Modelo no parametrico con particiones binarias recursivas basadas en reduccion de varianza (MSE), acotado a `max_depth=12` y `min_samples_leaf=15`.
3. **Random Forest (`RandomForestRegressor`)**: Ensamble de 150 arboles independientes mediante Bagging y subespacio aleatorio de variables para reducir drasticamente la varianza.
4. **XGBoost (`XGBRegressor`)**: Algoritmo de gradient boosting secuencial de 200 arboles optimizado con expansion de Taylor de segundo orden y regularizacion $L_1$ y $L_2$.
5. **LightGBM (`LGBMRegressor`)**: Algoritmo de gradient boosting con crecimiento por hojas (Leaf-Wise / Best-First), agrupamiento por histogramas y muestreo GOSS.

### Escalado Logaritmico de la Variable Objetivo
Debido al sesgo de la demanda (valores desde 50 hasta 100,000 unidades), los modelos se entrenan sobre $\log(1 + y)$ utilizando `TransformedTargetRegressor`. Durante la inferencia se aplica automaticamente $\exp(\hat{y}_{\log}) - 1$, garantizando predicciones estrictamente no negativas y minimizando errores porcentuales.

---

## 5. Tabla de Metricas y Evaluacion de Confiabilidad

Las metricas se evaluaron sobre un conjunto de prueba independiente ($N = 6,433$) y mediante validacion cruzada de 5 particiones (5-Fold CV):

| Algoritmo | Test $R^2$ | 95% CI $R^2$ (Bootstrap) | Test MAE (Unidades) | Test RMSE (Unidades) | Median AE (Unidades) | MAPE (%) | 5-Fold CV $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.9311** | [0.8801, 0.9650] | **256.20** | **1,531.51** | 9.87 | 27.29% | **0.8506 ± 0.087** |
| **Decision Tree** | 0.9068 | [0.8453, 0.9497] | 273.68 | 1,781.10 | **6.28** | 31.72% | 0.7447 ± 0.145 |
| **Random Forest** | 0.8814 | [0.8015, 0.9384] | 289.60 | 2,009.75 | 6.58 | **23.88%** | 0.6278 ± 0.118 |
| **XGBoost** | 0.8471 | [0.6313, 0.9547] | 263.36 | 2,281.25 | 11.88 | 26.59% | 0.8459 ± 0.091 |
| **Ridge** | 0.1858 | [0.1709, 0.2080] | 980.22 | 5,264.78 | 76.40 | 80.25% | 0.1931 ± 0.040 |

### Conclusiones de Rendimiento:
- **LightGBM** obtiene el mejor balance general con $R^2 = 0.9311$, el menor RMSE (1,531.51) y la mayor consistencia en validacion cruzada ($0.8506$).
- **Random Forest** presenta el menor error porcentual absoluto medio (MAPE = 23.88%).
- **Decision Tree** exhibe una mediana de error absoluto de solo 6.28 unidades para la gran mayoria de articulos tipicos.

---

## 6. Galeria de Graficos Generados

Todos los graficos tecnicos se encuentran en la carpeta `plots/`:

1. `plots/preprocessing_pipeline.png`: Comparacion de valores nulos antes y despues de la limpieza, distribucion de caracteristicas y flujo del pipeline.
2. `plots/metrics_comparison.png`: Comparativa integral de $R^2$, MAE, RMSE, MedAE y MAPE para los 5 algoritmos.
3. `plots/reliability_residuals.png`: Grafico de dispersion valor real vs predicho en escala logaritmica, distribucion de residuos y error desagregado por categoria.
4. `plots/flow_ridge.png`: Diagrama de flujo de contraccion $L_2$ y ranking de coeficientes aprendidos.
5. `plots/flow_decision_tree.png`: Representacion del arbol con los nodos de decision principales y reduccion de impureza.
6. `plots/flow_random_forest.png`: Arquitectura de agregacion por bootstrap (Bagging) y error estandar de importancia de variables.
7. `plots/flow_xgboost.png`: Flujo de optimizacion de segundo orden de Taylor y ganancia relativa de splits.
8. `plots/flow_lightgbm.png`: Crecimiento por hojas (Leaf-Wise), cuantizacion por histogramas e importancia por conteo de divisiones.

---

## 7. Ejecucion y Consumo de Modelos

### Ejecutar el Pipeline Completo
Para ejecutar la adquisicion, preprocesamiento, reentrenamiento, evaluacion y renderizado de graficos:
```powershell
python main.py
```

### Inferencia sobre Nuevos Datos / Consumo Programatico
Los pipelines serializados en `models/*_pipeline.joblib` incluyen el preprocesador y la transformacion inversa:

```python
import joblib
import pandas as pd

# Cargar el pipeline listo para produccion
pipeline = joblib.load("models/lightgbm_pipeline.joblib")

# Definir un nuevo articulo
item = pd.DataFrame([{
    "product_category": "Laptops",
    "discounted_price": 799.99,
    "original_price": 999.99,
    "discount_percentage": 20.0,
    "rating_num": 4.7,
    "reviews_num": 2500.0,
    "log_reviews": 7.82,
    "log_price": 6.68,
    "rating_review_score": 36.75,
    "delivery_lead_days": 2.0,
    "is_best_seller": 1,
    "is_amazons_choice": 0,
    "is_limited_deal": 1,
    "is_sponsored": 0,
    "has_coupon": 1,
    "coupon_discount_value": 50.0,
    "buy_box_availability": 1,
    "has_sustainability": 1,
    "title_length": 110,
    "title_word_count": 16
}])

predicted_units = pipeline.predict(item)
print(f"Demanda mensual estimada: {predicted_units[0]:.1f} unidades")
```

### Ejecutar Inferencia via CLI
```powershell
python src/predict.py --model lightgbm --category Laptops
```
