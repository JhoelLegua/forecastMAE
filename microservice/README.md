# DemandMind Microservice: Demand Forecasting & Multimodal AI Intelligence

Microservicio empresarial desarrollado con **FastAPI** y documentado con la interfaz interactiva moderna **Scalar API Reference**, diseñado para estimar demanda mensual de productos en Amazon, generar gráficos de benchmarking persistentes, interpretar escenarios mediante **LLM Multimodal (Google Gemini Vision)**, persistir el flujo de peticiones en disco y emitir alertas operativas vía **Brevo SMTP Relay**.

---

## 1. Arquitectura del Microservicio

```
microservice/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI, middlewares de trazabilidad de logs, Scalar docs y routers
│   ├── config.py                # Pydantic Settings, variables (.env), control LLM y Brevo SMTP
│   ├── schemas.py               # Modelos Pydantic con payloads y schemas de salida
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py     # Carga y ejecución de modelos .joblib y benchmarks
│   │   ├── chart_service.py     # Generación de gráficos ejecutivos de benchmarking
│   │   ├── storage_service.py   # Persistencia local de reportes JSON y gráficos PNG
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
│       └── notifications.py    # Endpoints /notifications/test, /status, /alert y /logs
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

## 2. Framework y Documentación Interactiva (Scalar)

El microservicio utiliza **Scalar API Reference** (`scalar-fastapi`) como sustituto interactivo moderno de Swagger UI y ReDoc.

- **URL de Documentación Scalar**: `http://localhost:8000/docs`
- **Características de Scalar**:
  - Interfaz moderna oscura/clara con tema `moon`.
  - Prueba de endpoints interactiva ("Test Request") con **payloads precargados por defecto**.
  - Generador automático de fragmentos de código cliente en JavaScript, Python, cURL, Go, PHP, etc.
  - Búsqueda instantánea y visualización completa de esquemas.

---

## 3. Control de Análisis LLM (`LLM_ANALYZE`)

En el archivo `.env` se define el interruptor maestro del LLM:

```ini
# true: envía imagen y métricas al LLM para generar explicación de decisiones
# false: desactiva el análisis LLM (el reporte muestra métricas y gráficos sin explicación)
LLM_ANALYZE=true
```

### Comportamiento según el estado:
* **`LLM_ANALYZE=true`**:
  - El gráfico generado y las métricas se envían a **Google Gemini Multimodal Vision** (o al motor heurístico si no hay API key).
  - La respuesta incluye la sección ejecutiva obligatoria de 4 puntos (explicación en lenguaje no técnico, diagnóstico cuantitativo, análisis de drivers y plan de acción operativo).
  - La vista web `/reports/{report_id}` y el PDF desplegarán el informe explicativo completo.
* **`LLM_ANALYZE=false`**:
  - No se realiza ninguna llamada de red al LLM ni se genera texto explicativo (`ai_decision_analysis` queda vacío y `llm_provider` indica `"Desactivado (LLM_ANALYZE=False)"`).
  - La respuesta y la generación del reporte es ultra rápida (~300 ms).
  - En la vista HTML y en el PDF, la sección de explicación de IA se omite automáticamente, mostrando de manera limpia los KPIs de salida, gráfico de benchmarking, datos de entrada y auditoría técnica.

---

## 4. Persistencia de Logs del Flujo de Peticiones

El microservicio incluye un middleware HTTP global que registra persistentemente cada petición recibida:

- **Archivo de log persistente**: `microservice/storage/logs/demandmind_requests.log`
- **Rotación automática**: Archivos de hasta 10 MB con histórico rotativo de 5 respaldos.
- **Formato del registro**:
  ```text
  [2026-10-17 18:08:39] [INFO] [REQ:req_e329f9d6] Client:127.0.0.1 | GET /health | Status:200 (OK) | Latency:1.0ms
  [2026-10-17 18:08:58] [INFO] [REQ:req_437cf3b6] Client:127.0.0.1 | POST /predict/best | Status:200 (OK) | Latency:19052.4ms
  [2026-10-17 18:09:21] [INFO] [REQ:req_ec910c87] Client:127.0.0.1 | POST /predict/best | Status:200 (OK) | Latency:394.0ms
  ```
- **Auditoría ante fallas (Status 500 / Excepciones)**:
  - En caso de error no controlado, se captura la traza completa (traceback), se persiste en el log y se asocia al `Request-ID` para diagnóstico inmediato.
  - Además, se dispara automáticamente una alerta de alta severidad vía Brevo SMTP al equipo administrador.
- **Consulta vía API**:
  - `GET /notifications/logs?limit=50`: Permite inspeccionar las últimas N líneas del archivo persistente directamente desde Scalar o navegador.

---

## 5. Servicio de Notificaciones y Alertas Brevo SMTP

El microservicio incorpora un cliente SMTP no bloqueante optimizado para **Brevo SMTP Relay** con soporte de `STARTTLS` sobre el puerto 587.

### 5.1. Variables en `.env`
```ini
SMTP_ENABLED=True
SMTP_HOST="smtp-relay.brevo.com"
SMTP_PORT=587
SMTP_USER="8061d7001@smtp-brevo.com"
SMTP_PASSWORD="xsmtpsib-tu-clave-aqui"
ALERT_SENDER_EMAIL="leguajhoel@gmail.com"
ALERT_RECIPIENT_EMAIL="leguajhoel@gmail.com"
ALERT_MIN_SEVERITY="LOW" # LOW | MEDIUM | HIGH | CRITICAL
```

### 5.2. Arquitectura de Despacho (`email_notifier.py`)
- **No bloqueante**: Utiliza `asyncio.to_thread` y `FastAPI BackgroundTasks` para asegurar que el despacho del correo nunca agregue latencia a la respuesta HTTP del cliente.
- **Aislamiento de fallos**: Los errores de red SMTP son capturados y registrados; nunca provocan la caída del servicio.
- **Filtrado por severidad**: Compara la severidad del evento contra `ALERT_MIN_SEVERITY` (`LOW` = 0, `MEDIUM` = 1, `HIGH` = 2, `CRITICAL` = 3).
- **Plantilla HTML ejecutiva**: Genera correos formateados con encabezado corporativo DemandMind, distintivo de severidad en colores, diagnóstico, tabla contextual y Request ID.

### 5.3. Casos de Activación Automática:
1. **Fallas críticas (500 Internal Server Error)**: `system_failure` (Severidad HIGH).
2. **Errores de inferencia en modelos**: `prediction_error` (Severidad HIGH).
3. **Picos de demanda atípicos**: `extreme_demand_spike` (Severidad MEDIUM) cuando la demanda estimada supera 1,000 unidades o +300% de la mediana del sector.
4. **Prueba de relay manual**: `test_alert` (Severidad LOW) para comprobar entrega.

---

## 6. Endpoints Disponibles

### Diagnóstico y Salud
- `GET /health`: Estado del servicio, modelos en memoria, estado de `LLM_ANALYZE` y SMTP.
- `GET /status`: Alias de monitoreo operacional.
- `GET /docs`: Documentación interactiva en **Scalar**.

### Pronósticos de Demanda
- `POST /predict/lightgbm`: Inferencia con **LightGBM** (Modelo líder, $R^2 = 0.9311$).
- `POST /predict/best`: Alias directo al mejor modelo evaluado.
- `POST /predict/random-forest`: Inferencia con **Random Forest** (Menor MAPE: 23.88%).
- `POST /predict/xgboost`: Inferencia con **XGBoost** (Gradient Boosting optimizado).
- `POST /predict/decision-tree`: Inferencia con **Árbol de Decisión**.
- `POST /predict/ridge`: Inferencia con **Regresión Ridge** ($L_2$).
- `POST /predict/all`: Ejecuta los 5 modelos en simultáneo, grafica consenso y reporta en storage.

### Alertas y Notificaciones Brevo SMTP
- `GET /notifications/status`: Estado de la configuración SMTP y ofuscación de credenciales.
- `POST /notifications/test`: Disparo inmediato de correo de prueba para verificar relay.
- `POST /notifications/alert`: Emisión de alerta en segundo plano con severidad y mensaje personalizado.
- `GET /notifications/logs`: Lectura de los registros de peticiones persistidos.

### Vistas Interactivas y Persistencia en Storage
- `GET /reports/{report_id}`: Vista web ejecutiva interactiva que despliega:
  - Reporte descriptivo de IA renderizado en Markdown (si `LLM_ANALYZE=true`).
  - Gráfico de benchmarking de alta resolución.
  - Data de entrada detallada (precios, ratings, insignias, logística).
  - Data de salida (demanda proyectada, tier, benchmarks de sector).
  - Metadatos de ejecución y auditoría técnica.
  - Selector de Tema: **Modo Oscuro / Modo Claro** con transiciones suaves.
  - **Exportación a PDF**: Descarga el reporte completo en formato A4 respetando el tema activo.
- `GET /reports/{report_id}/json`: Recupera la estructura cruda persistida en `storage/reports/{report_id}.json`.

---

## 7. Ejecución Local

1. Activar el entorno virtual:
   ```bash
   source .venv/Scripts/activate   # En Windows Git Bash
   # o bien: .\.venv\Scripts\Activate.ps1 en PowerShell
   ```

2. Iniciar el servidor Uvicorn:
   ```bash
   python microservice/run.py
   ```

3. Abrir en el navegador:
   - **Scalar API Reference**: `http://localhost:8000/docs`
   - **Verificación de Salud**: `http://localhost:8000/health`
