"""
AWS S3 Storage Service Module.
Handles asynchronous and non-blocking image uploads to Amazon S3 buckets
and generates public/presigned URLs.
Includes fallback mechanisms if AWS S3 is disabled or unreachable.
"""

import os
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

from app.config import Settings, get_settings

logger = logging.getLogger("demandmind.s3")


def _get_s3_client(settings: Settings):
    """Instancia el cliente de boto3 S3 con las credenciales configuradas."""
    import boto3
    from botocore.config import Config

    kwargs = {
        "region_name": settings.aws_region,
        "config": Config(signature_version="s3v4", connect_timeout=5, read_timeout=10)
    }

    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

    if settings.aws_s3_custom_endpoint:
        kwargs["endpoint_url"] = settings.aws_s3_custom_endpoint

    return boto3.client("s3", **kwargs)


def _upload_file_blocking(
    settings: Settings,
    file_path: str,
    object_key: str,
    content_type: str = "image/png"
) -> Optional[str]:
    """Operación síncrona de subida de archivo a S3 delegada a un worker thread."""
    try:
        s3 = _get_s3_client(settings)
        bucket = settings.aws_s3_bucket_name

        extra_args = {"ContentType": content_type}
        
        # Subir archivo al bucket
        with open(file_path, "rb") as f:
            s3.upload_fileobj(
                Fileobj=f,
                Bucket=bucket,
                Key=object_key,
                ExtraArgs=extra_args
            )

        # Construir URL pública directa
        if settings.aws_s3_custom_endpoint:
            url = f"{settings.aws_s3_custom_endpoint.rstrip('/')}/{bucket}/{object_key}"
        elif settings.aws_region == "us-east-1":
            url = f"https://{bucket}.s3.amazonaws.com/{object_key}"
        else:
            url = f"https://{bucket}.s3.{settings.aws_region}.amazonaws.com/{object_key}"

        logger.info("Successfully uploaded chart to AWS S3: %s", url)
        return url

    except Exception as exc:
        logger.error("Failed to upload chart to AWS S3 bucket '%s': %s", settings.aws_s3_bucket_name, exc)
        return None


async def upload_chart_to_s3(
    file_path: str | Path,
    report_id: str,
    settings: Optional[Settings] = None
) -> Optional[str]:
    """
    Sube un archivo de gráfico a AWS S3 de forma asíncrona no bloqueante.
    Retorna la URL pública si tiene éxito, o None si S3 está desactivado o falla.
    """
    cfg = settings or get_settings()

    if not cfg.use_aws_s3:
        logger.debug("AWS S3 is disabled in settings. Skipping cloud upload.")
        return None

    path_str = str(file_path)
    if not os.path.exists(path_str):
        logger.warning("Chart file does not exist at path: %s", path_str)
        return None

    object_key = f"charts/{report_id}.png"

    return await asyncio.to_thread(
        _upload_file_blocking,
        settings=cfg,
        file_path=path_str,
        object_key=object_key,
        content_type="image/png"
    )


async def check_s3_connectivity(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Verifica si el bucket S3 existe y las credenciales son válidas."""
    cfg = settings or get_settings()

    if not cfg.use_aws_s3:
        return {
            "s3_enabled": False,
            "status": "disabled",
            "message": "AWS S3 está desactivado (USE_AWS_S3=false). Modo local activo."
        }

    def _check():
        try:
            s3 = _get_s3_client(cfg)
            s3.head_bucket(Bucket=cfg.aws_s3_bucket_name)
            return {
                "s3_enabled": True,
                "status": "connected",
                "bucket": cfg.aws_s3_bucket_name,
                "region": cfg.aws_region,
                "message": f"Conectado exitosamente al bucket '{cfg.aws_s3_bucket_name}'"
            }
        except Exception as exc:
            return {
                "s3_enabled": True,
                "status": "error",
                "bucket": cfg.aws_s3_bucket_name,
                "region": cfg.aws_region,
                "error": str(exc)
            }

    return await asyncio.to_thread(_check)
