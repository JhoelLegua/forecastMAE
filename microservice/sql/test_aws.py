"""
AWS RDS & S3 Connectivity and Initialization Script.
Tests connectivity to S3 bucket, uploads a test image, connects to AWS RDS PostgreSQL,
creates demandmind_db (if needed), and executes schema.sql to set up all tables.
"""

import sys
import asyncio
import asyncpg
import boto3
from pathlib import Path
from urllib.parse import quote_plus

# Asegurar path
MICROSERVICE_DIR = Path(__file__).resolve().parent.parent
if str(MICROSERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(MICROSERVICE_DIR))

from app.config import get_settings


async def test_and_init_aws():
    settings = get_settings()

    print("=" * 70)
    print("1. PROBANDO CONECTIVIDAD Y SUBIDA A AWS S3")
    print("=" * 70)
    s3_client = boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key
    )

    bucket = settings.aws_s3_bucket_name
    print(f"[S3] Verificando bucket '{bucket}' en región '{settings.aws_region}'...")

        s3_client.head_bucket(Bucket=bucket)
        print(f"[S3 OK] El bucket '{bucket}' existe y es accesible con tus credenciales.")

        # Asegurar política de lectura pública para las imágenes de los gráficos
        try:
            import json
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "PublicReadForCharts",
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": "s3:GetObject",
                        "Resource": f"arn:aws:s3:::{bucket}/*"
                    }
                ]
            }
            s3_client.put_bucket_policy(Bucket=bucket, Policy=json.dumps(policy))
            print(f"[S3 OK] Bucket Policy de lectura pública configurada para '{bucket}'.")
        except Exception as pol_err:
            print(f"[S3 NOTICE] No se pudo actualizar bucket policy: {pol_err}")

        # Subir un archivo de prueba (ej. un gráfico existente en plots/)
        test_file = Path("plots/flow_lightgbm.png")
        if not test_file.exists():
            test_file = Path("microservice/README.md")

        test_key = f"charts/test_connection_{test_file.name}"
        print(f"[S3] Subiendo archivo de prueba '{test_file}' como '{test_key}'...")
        with open(test_file, "rb") as f:
            s3_client.upload_fileobj(
                Fileobj=f,
                Bucket=bucket,
                Key=test_key,
                ExtraArgs={"ContentType": "image/png"}
            )

        public_url = f"https://{bucket}.s3.amazonaws.com/{test_key}"
        print(f"[S3 OK] ¡Archivo subido con éxito!")
        print(f"[S3 URL PÚBLICA]: {public_url}")

    except Exception as exc:
        print(f"[S3 ERROR]: {exc}")

    print("\n" + "=" * 70)
    print("2. PROBANDO CONEXIÓN A AWS RDS POSTGRESQL")
    print("=" * 70)
    from urllib.parse import urlparse, unquote
    parsed_url = urlparse(settings.aws_rds_database_url)
    rds_host = parsed_url.hostname or "demandmind-db.c0lio2884xnk.us-east-1.rds.amazonaws.com"
    rds_user = parsed_url.username or "postgres"
    rds_pass = unquote(parsed_url.password or "")

    print(f"[RDS] Conectando a {rds_host}:{parsed_url.port or 5432}...")

    # Probar conexión a la base inicial 'postgres'
    conn_admin = None
    connected_db = None
    for db_candidate in ["demandmind_db", "postgres"]:
        try:
            print(f"[RDS] Intentando conectar a base '{db_candidate}'...")
            conn = await asyncpg.connect(
                host=rds_host,
                port=5432,
                user=rds_user,
                password=rds_pass,
                database=db_candidate,
                timeout=15
            )
            ver = await conn.fetchval("SELECT version();")
            print(f"[RDS OK] Conexión establecida a '{db_candidate}'!")
            print(f"[RDS INFO] {ver.split(',')[0]}")
            connected_db = db_candidate
            conn_admin = conn
            break
        except Exception as exc:
            print(f"[RDS NOTICE] No se pudo conectar directo a '{db_candidate}': {exc}")

    if not conn_admin:
        print("[RDS ERROR] No se pudo conectar a la instancia RDS. Revisa el Security Group (debe permitir entrada en puerto 5432 desde tu IP o 0.0.0.0/0 si es pública).")
        return

    # Si estamos en 'postgres', verificar si 'demandmind_db' existe, si no, crearla
    target_db = "demandmind_db"
    if connected_db == "postgres":
        exists = await conn_admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1;", target_db)
        if not exists:
            print(f"[RDS] Base '{target_db}' no existe en RDS. Creándola...")
            await conn_admin.execute(f'CREATE DATABASE "{target_db}";')
            print(f"[RDS OK] Base '{target_db}' creada en RDS.")
        await conn_admin.close()

        # Conectar a la base destino
        conn_target = await asyncpg.connect(
            host=rds_host,
            port=5432,
            user=rds_user,
            password=rds_pass,
            database=target_db,
            timeout=15
        )
    else:
        conn_target = conn_admin

    print(f"\n[RDS] Ejecutando schema.sql en '{target_db}' de AWS RDS...")
    sql_file = Path("microservice/sql/schema.sql")
    sql_content = sql_file.read_text(encoding="utf-8")
    await conn_target.execute(sql_content)
    print("[RDS OK] Tablas, extensiones e índices creados exitosamente en AWS RDS.")

    # Listar tablas en RDS
    tables = await conn_target.fetch("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
    """)
    print("\n[TABLAS CREADAS EN AWS RDS]:")
    for t in tables:
        print(f"  * {t['table_name']}")

    # Destinatarios
    recipients = await conn_target.fetch(
        "SELECT id, first_name, last_name, email, role, min_severity, is_active FROM notification_recipients;"
    )
    print("\n[DESTINATARIOS EN AWS RDS]:")
    for r in recipients:
        print(f"  * ID {r['id']}: {r['first_name']} {r['last_name']} <{r['email']}> [{r['role']}]")

    await conn_target.close()
    print("\n[PROCESO AWS COMPLETADO CON ÉXITO]")


if __name__ == "__main__":
    asyncio.run(test_and_init_aws())
