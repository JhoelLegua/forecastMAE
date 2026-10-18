# DemandMind Microservice: Demand Forecasting & Multimodal AI Intelligence

Microservicio empresarial desarrollado con **FastAPI** y documentado con la interfaz interactiva moderna **Scalar API Reference**, diseñado para estimar demanda mensual de productos en Amazon, generar gráficos de benchmarking persistentes, interpretar escenarios mediante **LLM Multimodal (Google Gemini Vision)**, persistir el flujo de peticiones en disco, emitir alertas operativas vía **Brevo SMTP Relay**, y operar con **infraestructura dual conmutable entre la Nube (AWS S3 y AWS RDS) y Entorno Local (Storage y PostgreSQL local con DBeaver)**.

---

## 1. Arquitectura del Microservicio

```
microservice/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI, lifespan, middlewares de logs, Scalar docs y routers
│   ├── config.py                # Pydantic Settings, switches S3/RDS y variables (.env)
│   ├── database.py              # Conexión SQLAlchemy Async (asyncpg), pooling y modelos ORM
│   ├── schemas.py               # Modelos Pydantic con payloads y schemas de salida
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py     # Carga y ejecución de modelos .joblib y benchmarks
│   │   ├── chart_service.py     # Generación de gráficos ejecutivos de benchmarking
│   │   ├── s3_service.py        # Subida asíncrona de gráficos a AWS S3 y generación de URLs
│   │   ├── storage_service.py   # Persistencia dual (S3/Local) y registro en PostgreSQL/JSON
│   │   ├── logger_service.py    # Persistencia de logs de peticiones y fallas en disco
│   │   ├── email_notifier.py    # Servicio de alertas y notificaciones vía Brevo SMTP Relay
│   │   └── llm_service.py       # Análisis de decisiones con Google Gemini Vision (o desactivado)
│   ├── templates/
│   │   └── report.html          # Vista HTML ejecutiva con tema oscuro/claro y exportación PDF
│   └── routes/
│       ├── __init__.py
│       ├── health.py            # Endpoints /health y /status
│       ├── prediction.py        # Endpoints /predict/* con BackgroundTasks
│       ├── reports.py           # Endpoints /reports/{id} y /reports/{id}/json
│       └── notifications.py    # Endpoints /notifications/* (alertas, status, logs y CRUD destinatarios)
├── sql/
│   └── schema.sql               # Script DDL oficial para crear las tablas en PostgreSQL (DBeaver / RDS)
├── storage/
│   ├── charts/                  # Archivos PNG persistidos de cada predicción
│   ├── reports/                 # Archivos JSON con metadata y datos de cada predicción
│   └── logs/                    # Archivo persistente demandmind_requests.log
├── static/
│   └── charts/                  # Almacenamiento web legacy
├── .env                         # Variables de entorno locales
├── .env.example                 # Plantilla con guía de variables
├── requirements.txt             # Dependencias del microservicio
├── run.py                       # Script lanzador del servidor Uvicorn
└── README.md                    # Documentación del microservicio
```

---

## 2. Infraestructura Dual: Cloud (AWS) vs Local

El microservicio desacopla el almacenamiento y la base de datos a través de dos variables de entorno booleanas:

```ini
# ==============================================================================
# 1. Almacenamiento de Gráficos (AWS S3 vs Local)
# ==============================================================================
USE_AWS_S3=false
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_S3_BUCKET_NAME="demandmind-charts"
AWS_S3_CUSTOM_ENDPOINT=""

# ==============================================================================
# 2. Base de Datos Relacional (AWS RDS vs Local PostgreSQL)
# ==============================================================================
USE_AWS_RDS=false
LOCAL_DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/demandmind_db"
AWS_RDS_DATABASE_URL="postgresql+asyncpg://postgres:tu_password_rds@demandmind-db.xyz.us-east-1.rds.amazonaws.com:5432/demandmind_db"
```

### Comportamiento del Almacenamiento (`USE_AWS_S3`):
* **`USE_AWS_S3=true`**: El gráfico generado se sube directamente a tu Bucket de **AWS S3** usando `boto3`. El reporte y la API retornan el enlace público HTTPS directo (ej. `https://demandmind-charts.s3.amazonaws.com/charts/rep_123.png`).
* **`USE_AWS_S3=false`**: El gráfico se almacena localmente en `storage/charts/rep_123.png` y se expone a través de la ruta `/storage/charts/rep_123.png`.

### Comportamiento de la Base de Datos (`USE_AWS_RDS`):
* **`USE_AWS_RDS=true`**: Conecta a la instancia administrada de PostgreSQL en **AWS RDS**.
* **`USE_AWS_RDS=false`**: Conecta a tu servidor PostgreSQL local (Docker o servicio de Windows que puedes administrar desde **DBeaver**).
* **Aislamiento y Resiliencia**:
  - **No se persiste Base64 en la base de datos**: Para evitar saturar las tablas y degradar el buffer pool, únicamente se guarda la URL del gráfico (`chart_url`).
  - Si la base de datos no está levantada o falla la conexión, el microservicio **nunca se cae**; emite una advertencia en el log y continúa operando de forma transparente con persistencia en documentos JSON locales (`storage/reports/`).

---

## 3. Despliegue de la Base de Datos con DBeaver (`sql/schema.sql`)

En [`microservice/sql/schema.sql`](file:///d:/MAE/MAE/MOD%20VI/DemandMind/microservice/sql/schema.sql) se encuentra el script DDL completo listo para ejecutar en **DBeaver**:

1. Abre **DBeaver** y conéctate a tu base de datos PostgreSQL local (`localhost:5432`) o a tu instancia de **AWS RDS**.
2. Abre un editor SQL (`Ctrl + ]`).
3. Pega y ejecuta el contenido de [`schema.sql`](file:///d:/MAE/MAE/MOD%20VI/DemandMind/microservice/sql/schema.sql).
4. El script creará automáticamente:
   * **`prediction_requests`**: Registro histórico de cada inferencia con parámetros en JSONB, demanda estimada, KPIs y URL del gráfico.
   * **`notification_recipients`**: Destinatarios dinámicos para las alertas de correo, con roles y niveles de severidad configurables.
   * **`email_audit_logs`**: Trazabilidad y auditoría de alertas despachadas.

---

## 4. Control de Análisis LLM (`LLM_ANALYZE`)

En el archivo `.env` se define el interruptor maestro del LLM:

```ini
# true: envía imagen y métricas al LLM para generar explicación de decisiones
# false: desactiva el análisis LLM (el reporte muestra métricas y gráficos sin explicación)
LLM_ANALYZE=true
```

* **`LLM_ANALYZE=true`**: Se envía el gráfico a **Google Gemini Vision** para generar la explicación ejecutiva de 4 secciones.
* **`LLM_ANALYZE=false`**: Se omite el envío de la imagen. La inferencia es ultra rápida (~300 ms) y en el reporte web / PDF la sección explicativa se oculta limpiamente.

---

## 5. Trazabilidad de Peticiones y Persistencia de Logs

- **Archivo de log persistente**: `microservice/storage/logs/demandmind_requests.log`
- **Rotación automática**: Archivos de hasta 10 MB con 5 respaldos rotativos.
- **Formato**:
  ```text
  [2026-10-17 18:08:58] [INFO] [REQ:req_437cf3b6] Client:127.0.0.1 | POST /predict/best | Status:200 (OK) | Latency:3042.1ms
  ```
- **Auditoría ante errores 500**: Almacena el `traceback` completo y dispara una alerta `system_failure` vía Brevo SMTP.

---

## 6. Endpoints Disponibles

### Diagnóstico, Salud e Infraestructura
* `GET /health`: Estado del microservicio, modelos cargados, estado de `LLM_ANALYZE`, `USE_AWS_S3` y `USE_AWS_RDS`.
* `GET /notifications/status`: Diagnóstico detallado con estado del relay Brevo SMTP, prueba de bucket S3 y conexión a PostgreSQL (RDS / Local).
* `GET /docs`: Documentación interactiva en **Scalar**.

### Pronósticos de Demanda
* `POST /predict/lightgbm`: Inferencia con **LightGBM** (Modelo líder, $R^2 = 0.9311$).
* `POST /predict/best`: Alias directo al mejor modelo evaluado.
* `POST /predict/random-forest`: Inferencia con **Random Forest** (Menor MAPE: 23.88%).
* `POST /predict/xgboost`: Inferencia con **XGBoost** (Gradient Boosting optimizado).
* `POST /predict/decision-tree`: Inferencia con **Árbol de Decisión**.
* `POST /predict/ridge`: Inferencia con **Regresión Ridge** ($L_2$).
* `POST /predict/all`: Ejecuta los 5 modelos simultáneamente y grafica consenso.

### Gestión Dinámica de Destinatarios de Alertas (PostgreSQL)
* `GET /notifications/recipients`: Lista los destinatarios activos/inactivos configurados en la base de datos.
* `POST /notifications/recipients`: Registra un nuevo destinatario (nombre, apellido, email, rol, severidad mínima).
* `PATCH /notifications/recipients/{id}/toggle`: Activa o pausa la recepción de alertas sin borrar al usuario.
* `DELETE /notifications/recipients/{id}`: Elimina un destinatario de la base de datos.
* `POST /notifications/test`: Dispara un correo de prueba vía Brevo SMTP Relay.
* `GET /notifications/logs`: Muestra las últimas líneas del log persistente.

### Vistas Interactivas y Persistencia
* `GET /reports/{report_id}`: Vista web ejecutiva interactiva (Modo Oscuro / Claro y descarga PDF).
* `GET /reports/{report_id}/json`: Metadata completa del reporte persistido.

---

## 7. Ejecución Local

1. Activar el entorno virtual:
   ```bash
   source .venv/Scripts/activate
   ```

2. Iniciar el servidor:
   ```bash
   python microservice/run.py
   ```

3. Abrir la documentación interactiva:
   - **Scalar API Reference**: `http://localhost:8000/docs`
   - **Estado de Infraestructura**: `http://localhost:8000/notifications/status`

> **Guia Rapida de Despliegue y Pruebas**: Consulta [`GUIA_DESPLIEGUE.md`](GUIA_DESPLIEGUE.md) para ver los comandos detallados de activacion, pruebas con `curl` para AWS S3/RDS y exposicion a Internet mediante Cloudflare Tunnel.


