"""
Database Initializer Script for DemandMind.
Connects to PostgreSQL (Local or AWS RDS), creates demandmind_db if it does not exist,
and executes microservice/sql/schema.sql to set up tables, indices, and default recipients.
"""

import asyncio
import asyncpg
import sys
from pathlib import Path
from urllib.parse import urlparse

# Asegurar que microservice está en el path
MICROSERVICE_DIR = Path(__file__).resolve().parent.parent
if str(MICROSERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(MICROSERVICE_DIR))

from app.config import get_settings


async def setup_database():
    settings = get_settings()
    db_url = settings.active_database_url

    # asyncpg expects postgresql:// instead of postgresql+asyncpg://
    clean_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    parsed = urlparse(clean_url)
    target_db = parsed.path.lstrip("/")

    # URL administrativa conectando a la base por defecto 'postgres'
    admin_url = clean_url.replace(f"/{target_db}", "/postgres")

    print(f"[DB SETUP] Conectando a PostgreSQL ({parsed.hostname}:{parsed.port or 5432})...")
    print(f"[DB SETUP] Modo activo: {'AWS RDS' if settings.use_aws_rds else 'Local PostgreSQL'}")
    print(f"[DB SETUP] Base de datos destino: '{target_db}'")

    # 1. Verificar / Crear base de datos
    try:
        conn_admin = await asyncpg.connect(admin_url)
        exists = await conn_admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1;", target_db)
        if not exists:
            print(f"[DB SETUP] Base de datos '{target_db}' no existe. Creándola...")
            await conn_admin.execute(f'CREATE DATABASE "{target_db}";')
            print(f"[DB SETUP] Base de datos '{target_db}' creada con éxito.")
        else:
            print(f"[DB SETUP] Base de datos '{target_db}' ya existe.")
        await conn_admin.close()
    except Exception as exc:
        print(f"[DB SETUP WARNING] No se pudo verificar en base 'postgres' ({exc}). Intentando conectar directo a '{target_db}'...")

    # 2. Conectar a demandmind_db y ejecutar schema.sql
    print(f"[DB SETUP] Conectando a '{target_db}' para ejecutar schema.sql...")
    sql_file = Path(__file__).resolve().parent / "schema.sql"
    if not sql_file.exists():
        print(f"[DB SETUP ERROR] No se encontró el archivo: {sql_file}")
        return

    sql_content = sql_file.read_text(encoding="utf-8")
    conn_target = await asyncpg.connect(clean_url)

    await conn_target.execute(sql_content)
    print("[DB SETUP] Esquema DDL ejecutado exitosamente (Tablas, Índices y Extensión UUID).")

    # 3. Listar tablas creadas
    tables = await conn_target.fetch("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
    """)
    print("\n[TABLAS EN SCHEMA PUBLIC]:")
    for t in tables:
        print(f"  * {t['table_name']}")

    # 4. Verificar destinatario inicial
    recipients = await conn_target.fetch(
        "SELECT id, first_name, last_name, email, role, min_severity, is_active FROM notification_recipients;"
    )
    print("\n[DESTINATARIOS CONFIGURADOS EN BD]:")
    for r in recipients:
        print(f"  * ID {r['id']}: {r['first_name']} {r['last_name']} <{r['email']}> [{r['role']}] (Severidad Minima: {r['min_severity']}, Activo: {r['is_active']})")

    await conn_target.close()
    print("\n[DB SETUP] ¡Inicialización completada con éxito!")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    asyncio.run(setup_database())
