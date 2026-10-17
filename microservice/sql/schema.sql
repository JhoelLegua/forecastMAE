-- ==============================================================================
-- DemandMind Database Schema for PostgreSQL (Local / AWS RDS)
-- Compatible with PostgreSQL 13+ (tested with asyncpg & SQLAlchemy)
-- ==============================================================================

-- 1. Habilitar extensión UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Tabla: notification_recipients (Destinatarios de Alertas Brevo SMTP)
CREATE TABLE IF NOT EXISTS notification_recipients (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    role VARCHAR(100) NOT NULL DEFAULT 'SUPPLY_CHAIN',
    min_severity VARCHAR(20) NOT NULL DEFAULT 'MEDIUM', -- LOW, MEDIUM, HIGH, CRITICAL
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Índices para búsqueda rápida de destinatarios activos por severidad
CREATE INDEX IF NOT EXISTS idx_recipients_active_severity 
ON notification_recipients(is_active, min_severity);

-- Insertar destinatario inicial por defecto (leguajhoel@gmail.com)
INSERT INTO notification_recipients (first_name, last_name, email, role, min_severity, is_active)
VALUES ('Jhoel', 'Legua', 'leguajhoel@gmail.com', 'ADMIN', 'LOW', TRUE)
ON CONFLICT (email) DO NOTHING;


-- 3. Tabla: prediction_requests (Auditoría y Persistencia de Inferencia)
-- IMPORTANTE: No se almacena base64 para evitar saturar el buffer pool de la BD.
-- Se persiste únicamente el enlace público/local a la imagen (chart_url).
CREATE TABLE IF NOT EXISTS prediction_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_id VARCHAR(64) NOT NULL UNIQUE,
    request_id VARCHAR(64),
    model_used VARCHAR(100) NOT NULL,
    product_category VARCHAR(100) NOT NULL,
    predicted_demand NUMERIC(10, 2) NOT NULL,
    demand_tier VARCHAR(100),
    category_median NUMERIC(10, 2),
    relative_vs_median_pct NUMERIC(10, 2),
    input_parameters JSONB NOT NULL,
    ai_decision_analysis TEXT,
    llm_provider VARCHAR(150),
    chart_url TEXT NOT NULL,
    storage_backend VARCHAR(20) NOT NULL DEFAULT 'LOCAL', -- 'S3' o 'LOCAL'
    execution_time_ms NUMERIC(10, 2) DEFAULT 0.0,
    client_ip VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Índices para consultas analíticas de demanda, reportería y trazabilidad
CREATE INDEX IF NOT EXISTS idx_predictions_report_id ON prediction_requests(report_id);
CREATE INDEX IF NOT EXISTS idx_predictions_category ON prediction_requests(product_category);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON prediction_requests(model_used);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON prediction_requests(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_input_params ON prediction_requests USING gin (input_parameters);


-- 4. Tabla: email_audit_logs (Historial de Alertas Despachadas)
CREATE TABLE IF NOT EXISTS email_audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    recipient_email VARCHAR(255) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    delivery_status BOOLEAN NOT NULL DEFAULT TRUE,
    request_id VARCHAR(64),
    error_message TEXT,
    dispatched_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_email_logs_dispatched ON email_audit_logs(dispatched_at DESC);
