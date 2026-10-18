# 🚀 Guía de Operación y Despliegue: DemandMind Microservice

Esta guía detalla los pasos para activar el entorno, iniciar el microservicio en local, verificar la conectividad de los servicios Cloud (AWS S3 + AWS RDS) mediante `curl` y exponer el servicio de forma segura a Internet usando **Cloudflare Tunnel**.

---

## 📋 Índice
1. [Activación del Entorno Virtual (.venv)](#1-activación-del-entorno-virtual-venv)
2. [Levantamiento del Microservicio](#2-levantamiento-del-microservicio)
3. [Verificación de Servicios Cloud con cURL](#3-verificación-de-servicios-cloud-con-curl)
   - [3.1. Estado General de Salud (Health Check)](#31-estado-general-de-salud-health-check)
   - [3.2. Conectividad AWS S3 y AWS RDS](#32-conectividad-aws-s3-y-aws-rds)
   - [3.3. Prueba de Inferencia con Almacenamiento en S3 y RDS](#33-prueba-de-inferencia-con-almacenamiento-en-s3-y-rds)
   - [3.4. Consulta de Destinatarios de Alertas en RDS](#34-consulta-de-destinatarios-de-alertas-en-rds)
4. [Exposición a Internet mediante Cloudflare Tunnel](#4-exposición-a-internet-mediante-cloudflare-tunnel)
   - [4.1. Instalación de cloudflared](#41-instalación-de-cloudflared)
   - [4.2. Iniciar el Túnel Gratuito](#42-iniciar-el-túnel-gratuito)
   - [4.3. Validación End-to-End por el Túnel](#43-validación-end-to-end-por-el-túnel)

---

## 1. Activación del Entorno Virtual (.venv)

Abre tu terminal en la raíz del proyecto (`DemandMind`) y activa el entorno:

### Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```
*(Si recibes un error de políticas de ejecución en PowerShell: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y vuelve a ejecutar).*

### Windows (Git Bash / WSL / Linux / macOS):
```bash
source .venv/Scripts/activate      # En Windows Git Bash
# o
source .venv/bin/activate          # En Linux / macOS
```

Al activarse, tu terminal mostrará el prefijo `(.venv)`.

---

## 2. Levantamiento del Microservicio

Con el entorno virtual activo y ubicado en la carpeta raíz del proyecto:

```bash
python microservice/run.py
```

### Salida esperada en terminal:
```text
[MODEL SERVICE] Loading machine learning pipelines into memory...
[MODEL SERVICE] Loaded: ridge_pipeline.joblib
[MODEL SERVICE] Loaded: decision_tree_pipeline.joblib
[MODEL SERVICE] Loaded: random_forest_pipeline.joblib
[MODEL SERVICE] Loaded: xgboost_pipeline.joblib
[MODEL SERVICE] Loaded: lightgbm_pipeline.joblib
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### URLs Locales Disponibles:
* **API Docs (Scalar UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **API Docs (Swagger UI):** [http://localhost:8000/swagger](http://localhost:8000/swagger)
* **Diagnóstico de Salud:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 3. Verificación de Servicios Cloud con cURL

Abre una **segunda terminal** para ejecutar los siguientes comandos de diagnóstico:

### 3.1. Estado General de Salud (Health Check)
Verifica que los modelos de ML estén cargados en memoria y los switches de AWS estén encendidos:

```bash
curl -s http://localhost:8000/health
```

**Respuesta esperada (JSON):**
```json
{
  "status": "healthy",
  "service_name": "DemandMind Intelligent Forecasting Microservice",
  "version": "1.3.0",
  "models_loaded": ["ridge", "decision_tree", "random_forest", "xgboost", "lightgbm"],
  "llm_analyze_enabled": true,
  "use_aws_s3": true,
  "use_aws_rds": true,
  "scalar_docs_url": "http://localhost:8000/docs"
}
```

---

### 3.2. Conectividad AWS S3 y AWS RDS
Comprueba la conexión en vivo con el bucket S3 (`demandmind-charts`) y la base de datos PostgreSQL en AWS RDS:

```bash
curl -s http://localhost:8000/notifications/status
```

**Respuesta esperada:**
* **S3:** `"status": "connected"`, `"bucket": "demandmind-charts"`
* **RDS:** `"status": "connected"`, `"backend": "AWS RDS (PostgreSQL)"`, `"ping": true`

---

### 3.3. Prueba de Inferencia con Almacenamiento en S3 y RDS
Envía un SKU de prueba al modelo recomendado (`LightGBM`). El microservicio generará el gráfico, lo subirá a S3, guardará la auditoría en RDS y devolverá un payload ligero (sin Base64):

```bash
curl -X POST http://localhost:8000/predict/best \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Respuesta esperada:**
```json
{
  "report_id": "rep_xxxxxxxxxx",
  "report_url": "http://localhost:8000/reports/rep_xxxxxxxxxx",
  "model_used": "Lightgbm",
  "predicted_monthly_demand": 826.2,
  "demand_tier": "High Demand",
  "category": "Laptops",
  "category_median_demand": 710.0,
  "relative_vs_category_pct": 16.37,
  "chart_url": "https://demandmind-charts.s3.amazonaws.com/charts/rep_xxxxxxxxxx.png",
  "storage_chart_path": "d:\\MAE\\MAE\\MOD VI\\DemandMind\\microservice\\storage\\charts\\rep_xxxxxxxxxx.png",
  "ai_decision_analysis": "...",
  "llm_provider": "Gemini 2.0 Flash / Heuristic Engine"
}
```

> **Verificación visual del gráfico en S3:**
> Copia el enlace `chart_url` que te devuelva y pégalo en tu navegador. Deberá abrir la imagen PNG pública alojada en tu bucket S3.

---

### 3.4. Consulta de Destinatarios de Alertas en RDS
Verifica que la tabla `notification_recipients` en AWS RDS esté respondiendo:

```bash
curl -s http://localhost:8000/notifications/recipients
```

**Respuesta esperada:**
```json
[
  {
    "id": 1,
    "first_name": "Jhoel",
    "last_name": "Legua",
    "email": "leguajhoel@gmail.com",
    "role": "ADMIN",
    "min_severity": "LOW",
    "is_active": true
  }
]
```

---

## 4. Exposición a Internet mediante Cloudflare Tunnel

Cloudflare Tunnel te permite exponer tu microservicio que corre en `localhost:8000` hacia una URL pública con HTTPS gratuito sin necesidad de abrir puertos en tu router ni configurar DDNS.

### 4.1. Instalación de cloudflared

Si aún no tienes la utilidad `cloudflared`:

* **Windows (con winget):**
  ```powershell
  winget install --id Cloudflare.cloudflared
  ```
* **O descarga directa:** Descarga el binario `.exe` desde [GitHub Releases de Cloudflare](https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe) y colócalo en tu `PATH`.

---

### 4.2. Iniciar el Túnel Gratuito

En una terminal independiente, ejecuta:

```bash
cloudflared tunnel --url http://localhost:8000
```

Cloudflare generará una URL pública temporal segura, por ejemplo:
```text
2026-10-17T23:55:00Z INF +--------------------------------------------------------------------------------------------+
2026-10-17T23:55:00Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2026-10-17T23:55:00Z INF |  https://alpha-bravo-charlie.trycloudflare.com                                             |
2026-10-17T23:55:00Z INF +--------------------------------------------------------------------------------------------+
```

---

### 4.3. Validación End-to-End por el Túnel

Prueba invocar la API desde cualquier lugar usando la URL de Cloudflare Tunnel:

```bash
curl -X POST https://TU-SUBDOMINIO.trycloudflare.com/predict/best \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### 🎉 ¡Magia de la Auto-vinculación!
Gracias al resolvedor dinámico de cabeceras implementado en el microservicio:
1. El campo `report_url` devuelto apuntará directamente a tu túnel:
   `https://TU-SUBDOMINIO.trycloudflare.com/reports/rep_xxxxxxxxxx`
2. El campo `chart_url` apuntará a la imagen alojada en tu **AWS S3**:
   `https://demandmind-charts.s3.amazonaws.com/charts/rep_xxxxxxxxxx.png`
3. Todos los datos quedarán auditados y guardados en tu base de datos **AWS RDS**.
