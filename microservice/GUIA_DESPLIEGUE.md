# Guia de Operacion y Despliegue: DemandMind Microservice

Esta guia detalla los pasos para activar el entorno, iniciar el microservicio en local, verificar la conectividad de los servicios Cloud (AWS S3 + AWS RDS) mediante `curl` y exponer el servicio de forma segura a Internet usando **Cloudflare Tunnel**.

---

## Indice
0. [Modo Todo-en-Uno (Recomendado)](#0-modo-todo-en-uno-recomendado)
1. [Activacion del Entorno Virtual (.venv)](#1-activacion-del-entorno-virtual-venv)
2. [Levantamiento Manual del Microservicio](#2-levantamiento-manual-del-microservicio)
3. [Verificacion de Servicios Cloud con cURL](#3-verificacion-de-servicios-cloud-con-curl)
   - [3.1. Estado General de Salud (Health Check)](#31-estado-general-de-salud-health-check)
   - [3.2. Conectividad AWS S3 y AWS RDS](#32-conectividad-aws-s3-y-aws-rds)
   - [3.3. Prueba de Inferencia con Almacenamiento en S3 y RDS](#33-prueba-de-inferencia-con-almacenamiento-en-s3-y-rds)
   - [3.4. Consulta de Destinatarios de Alertas en RDS](#34-consulta-de-destinatarios-de-alertas-en-rds)
4. [Exposicion a Internet mediante Cloudflare Tunnel](#4-exposicion-a-internet-mediante-cloudflare-tunnel)
   - [4.1. Instalacion de cloudflared](#41-instalacion-de-cloudflared)
   - [4.2. Iniciar el Tunel Gratuito](#42-iniciar-el-tunel-gratuito)
   - [4.3. Validacion End-to-End por el Tunel](#43-validacion-end-to-end-por-el-tunel)

---

## 0. Modo Todo-en-Uno (Recomendado)

Si deseas hacer todo en **un solo comando** (verificar AWS S3 + RDS, generar diagnosticos en `status/`, arrancar FastAPI y levantar el tunel de Cloudflare):

```bash
python microservice/start.py
```

### Funcionalidad automatica de este script:
1. Detecta la ruta de `cloudflared` y los modelos ML en memoria.
2. Levanta el servidor Uvicorn en `http://localhost:8000`.
3. Ejecuta las verificaciones y **actualiza los archivos JSON en la carpeta `microservice/status/`**:
   - `status/health.json`: Salud de la aplicacion y modelos cargados.
   - `status/aws_connectivity.json`: Conectividad en vivo con S3 y RDS.
   - `status/recipients.json`: Destinatarios configurados para las alertas.
   - `status/sample_prediction.json`: Inferencia de prueba en S3 y RDS.
   - `status/tunnel.json`: URL publica generada por Cloudflare.
4. Inicia el tunel de Cloudflare y muestra la URL publica en terminal.
5. Al presionar `Ctrl + C`, apaga ambos procesos de forma limpia.

---

## 1. Activacion del Entorno Virtual (.venv)

Abre la terminal en la raiz del proyecto (`DemandMind`) y activa el entorno:

### Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```
*(Si recibes un error de politicas de ejecucion en PowerShell: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y vuelve a ejecutar).*

### Windows (Git Bash / WSL / Linux / macOS):
```bash
source .venv/Scripts/activate      # En Windows Git Bash
# o
source .venv/bin/activate          # En Linux / macOS
```

Al activarse, la terminal mostrara el prefijo `(.venv)`.

---

## 2. Levantamiento Manual del Microservicio

Con el entorno virtual activo y ubicado en la carpeta raiz del proyecto:

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
* **Diagnostico de Salud:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 3. Verificacion de Servicios Cloud con cURL

En una segunda terminal, ejecuta los siguientes comandos de diagnostico:

### 3.1. Estado General de Salud (Health Check)
Verifica que los modelos de ML esten cargados en memoria y los switches de AWS esten encendidos:

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
Comprueba la conexion en vivo con el bucket S3 (`demandmind-charts`) y la base de datos PostgreSQL en AWS RDS:

```bash
curl -s http://localhost:8000/notifications/status
```

**Respuesta esperada:**
* **S3:** `"status": "connected"`, `"bucket": "demandmind-charts"`
* **RDS:** `"status": "connected"`, `"backend": "AWS RDS (PostgreSQL)"`, `"ping": true`

---

### 3.3. Prueba de Inferencia con Almacenamiento en S3 y RDS
Envia un SKU de prueba al modelo recomendado (`LightGBM`). El microservicio generara el grafico, lo subira a S3, guardara la auditoria en RDS y devolvera un payload ligero sin Base64:

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

> **Verificacion visual del grafico en S3:**
> Copia el enlace `chart_url` que te devuelva y pegalo en el navegador. Abrira directamente la imagen PNG publica alojada en el bucket S3.

---

### 3.4. Consulta de Destinatarios de Alertas en RDS
Verifica que la tabla `notification_recipients` en AWS RDS responda:

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

## 4. Exposicion a Internet mediante Cloudflare Tunnel

Cloudflare Tunnel permite exponer el microservicio local hacia una URL publica con HTTPS sin necesidad de abrir puertos en el router ni configurar DDNS.

### 4.1. Instalacion de cloudflared

Si aun no tienes la utilidad `cloudflared`:

* **Windows (con winget):**
  ```powershell
  winget install --id Cloudflare.cloudflared
  ```
* **O descarga directa:** Descarga el binario `.exe` desde [GitHub Releases de Cloudflare](https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe) y colocalo en tu `PATH`.

---

### 4.2. Iniciar el Tunel Gratuito

En una terminal independiente, puedes ejecutarlo de cualquiera de estas dos formas:

#### Opcion Directa (Git Bash tras instalar con winget):
```bash
"/c/Program Files (x86)/cloudflared/cloudflared.exe" tunnel --url http://localhost:8000
```

#### Opcion Estandar (si cloudflared ya esta en el PATH):
```bash
cloudflared tunnel --url http://localhost:8000
```

Cloudflare generara una URL publica temporal segura, por ejemplo:
```text
2026-10-17T23:55:00Z INF +--------------------------------------------------------------------------------------------+
2026-10-17T23:55:00Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2026-10-17T23:55:00Z INF |  https://alpha-bravo-charlie.trycloudflare.com                                             |
2026-10-17T23:55:00Z INF +--------------------------------------------------------------------------------------------+
```

---

### 4.3. Validacion End-to-End por el Tunel

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

### Auto-vinculacion del Reporte:
Gracias al resolvedor dinamico de cabeceras implementado en el microservicio:
1. El campo `report_url` devuelto apuntara directamente al tunel:
   `https://TU-SUBDOMINIO.trycloudflare.com/reports/rep_xxxxxxxxxx`
2. El campo `chart_url` apuntara a la imagen alojada en **AWS S3**:
   `https://demandmind-charts.s3.amazonaws.com/charts/rep_xxxxxxxxxx.png`
3. Todos los datos quedaran auditados y guardados en la base de datos **AWS RDS**.
