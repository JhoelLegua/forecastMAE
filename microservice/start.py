"""
DemandMind All-In-One Launcher & Verifier (start.py)
-----------------------------------------------------
1. Verifica entorno, credenciales y modelos.
2. Inicia el microservicio FastAPI (Uvicorn).
3. Ejecuta diagnósticos HTTP automáticos y persiste los JSONs en microservice/status/:
   - status/health.json
   - status/aws_connectivity.json
   - status/recipients.json
   - status/sample_prediction.json
   - status/tunnel.json
4. Inicia el túnel seguro de Cloudflare (cloudflared).
5. Extrae la URL pública y presenta un banner ejecutivo con accesos directos.
6. Maneja CTRL+C para detener ambos procesos de forma limpia y ordenada.
"""

import os
import sys
import time
import json
import signal
import shutil
import urllib.request
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Ubicar raíz del microservicio
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATUS_DIR = BASE_DIR / "status"
STATUS_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str, tag: str = "INFO"):
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "RESET": "\033[0m"
    }
    prefix = f"[{tag}]"
    if sys.stdout.isatty():
        prefix = f"{colors.get(tag, '')}{prefix}{colors['RESET']}"
    print(f" {prefix} {msg}")


def find_cloudflared_path() -> str:
    """Busca el ejecutable de cloudflared en PATH y rutas comunes de Windows."""
    in_path = shutil.which("cloudflared")
    if in_path:
        return in_path

    candidates = [
        r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
        r"C:\Program Files\cloudflared\cloudflared.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\cloudflared\cloudflared.exe"),
        os.path.expandvars(r"%USERPROFILE%\cloudflared.exe")
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return "cloudflared"


def save_status_json(filename: str, new_data: any):
    """
    Guarda o actualiza un archivo JSON dentro de microservice/status/.
    Agrega timestamp de actualización sin perder la estructura.
    """
    filepath = STATUS_DIR / filename
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data": new_data
    } if isinstance(new_data, (dict, list)) else new_data

    # Si es lista directa (como status.json del usuario), guardamos respetando formato
    data_to_write = new_data
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_to_write, f, indent=2, ensure_ascii=False)
        log(f"Diagnóstico actualizado: status/{filename}", "SUCCESS")
    except Exception as e:
        log(f"No se pudo guardar status/{filename}: {e}", "WARNING")


def check_local_endpoint(url: str, method: str = "GET", payload: dict = None, timeout: int = 15):
    """Realiza una petición HTTP local de diagnóstico."""
    data_bytes = json.dumps(payload).encode("utf-8") if payload else None
    headers = {"Content-Type": "application/json"} if payload else {}
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print("\n" + "=" * 76)
    print("      DEMANDMIND FORECASTING: LANZADOR Y VERIFICADOR AUTOMATICO")
    print("=" * 76)

    cloudflared_bin = find_cloudflared_path()
    log(f"Ruta de Cloudflared detectada: {cloudflared_bin}", "INFO")

    # Mover status.json legacy a status/recipients.json si existía
    legacy_status = BASE_DIR / "status.json"
    if legacy_status.exists() and not (STATUS_DIR / "recipients.json").exists():
        try:
            shutil.copy2(legacy_status, STATUS_DIR / "recipients.json")
        except Exception:
            pass

    # 1. Comprobar si el microservicio ya está corriendo en el puerto 8000
    server_process = None
    already_running = False
    try:
        health = check_local_endpoint("http://localhost:8000/health", timeout=2)
        log("El microservicio ya se encuentra en ejecución en http://localhost:8000", "SUCCESS")
        already_running = True
    except Exception:
        log("Iniciando microservicio FastAPI con Uvicorn en http://localhost:8000...", "INFO")
        env = os.environ.copy()
        python_exe = sys.executable

        run_script = BASE_DIR / "run.py"
        server_process = subprocess.Popen(
            [python_exe, str(run_script)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Esperar hasta que /health responda
        log("Esperando que el microservicio esté listo...", "INFO")
        ready = False
        for _ in range(30):
            time.sleep(1)
            try:
                check_local_endpoint("http://localhost:8000/health", timeout=2)
                ready = True
                break
            except Exception:
                continue

        if not ready:
            log("El servidor tardó demasiado en responder o falló al iniciar.", "ERROR")
            if server_process:
                server_process.terminate()
            return
        log("Microservicio listo y respondiendo en http://localhost:8000", "SUCCESS")

    # 2. Ejecutar diagnósticos y persistir en microservice/status/
    print("-" * 76)
    log("Ejecutando verificaciones de salud y servicios Cloud (AWS S3 + RDS)...", "INFO")

    try:
        health_data = check_local_endpoint("http://localhost:8000/health")
        save_status_json("health.json", health_data)
    except Exception as e:
        log(f"Error verificando health: {e}", "WARNING")

    try:
        aws_data = check_local_endpoint("http://localhost:8000/notifications/status")
        save_status_json("aws_connectivity.json", aws_data)
        s3_ok = aws_data.get("s3", {}).get("active_mode") or "Desconocido"
        rds_ok = aws_data.get("database", {}).get("active_mode") or "Desconocido"
        log(f"Estado S3: {s3_ok} | Estado BD: {rds_ok}", "SUCCESS")
    except Exception as e:
        log(f"Error verificando status AWS: {e}", "WARNING")

    try:
        recipients_data = check_local_endpoint("http://localhost:8000/notifications/recipients")
        save_status_json("recipients.json", recipients_data)
        log(f"Destinatarios de alerta registrados en RDS: {len(recipients_data)} activos", "SUCCESS")
    except Exception as e:
        log(f"Error consultando destinatarios en RDS: {e}", "WARNING")

    try:
        log("Ejecutando predicción de prueba para verificar S3 y RDS...", "INFO")
        sample_payload = {
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
        }
        sample_pred = check_local_endpoint(
            "http://localhost:8000/predict/best",
            method="POST",
            payload=sample_payload,
            timeout=20
        )
        save_status_json("sample_prediction.json", sample_pred)
        log(f"Predicción OK (Reporte: {sample_pred.get('report_id')})", "SUCCESS")
        log(f"Gráfico subido a S3: {sample_pred.get('chart_url')}", "SUCCESS")
    except Exception as e:
        log(f"Error en predicción de prueba: {e}", "WARNING")

    # 3. Arrancar el Túnel de Cloudflare
    print("-" * 76)
    log("Iniciando Cloudflare Tunnel para exponer el microservicio...", "INFO")

    tunnel_cmd = [cloudflared_bin, "tunnel", "--url", "http://localhost:8000"]
    try:
        tunnel_process = subprocess.Popen(
            tunnel_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
    except Exception as e:
        log(f"No se pudo iniciar cloudflared: {e}", "ERROR")
        log(f"Verifica la instalación o ejecuta manualmente: {cloudflared_bin} tunnel --url http://localhost:8000", "WARNING")
        return

    # Buscar la URL en la salida de cloudflared
    tunnel_url = None
    start_wait = time.time()
    for line in tunnel_process.stdout:
        if "trycloudflare.com" in line:
            import re
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                tunnel_url = match.group(0)
                break
        if time.time() - start_wait > 30:
            break

    if tunnel_url:
        save_status_json("tunnel.json", {
            "tunnel_url": tunnel_url,
            "docs_url": f"{tunnel_url}/docs",
            "health_url": f"{tunnel_url}/health",
            "local_target": "http://localhost:8000",
            "connected_at": datetime.now(timezone.utc).isoformat()
        })

        print("\n" + "=" * 76)
        print("  DEMANDMIND MICROSERVICE Y TUNEL CLOUDFLARE EN LINEA")
        print("=" * 76)
        print(f"  URL Publica (Tunel Cloudflare): {tunnel_url}")
        print(f"  Scalar API Docs Interactiva:    {tunnel_url}/docs")
        print(f"  Swagger UI Alternativo:         {tunnel_url}/swagger")
        print(f"  Diagnostico de Salud:           {tunnel_url}/health")
        print(f"  Servidor Local:                 http://localhost:8000")
        print(f"  Archivos JSON de diagnostico:   microservice/status/")
        print("=" * 76)
        print("  Presiona CTRL+C para detener el servicio y cerrar el tunel limpiamente.\n")
    else:
        log("No se pudo detectar la URL del túnel en los primeros 30s. Revisa la consola.", "WARNING")

    # 4. Mantener la ejecución viva y capturar CTRL+C
    def cleanup(sig=None, frame=None):
        print("\n[INFO] Apagando túnel y microservicio de forma segura...")
        if tunnel_process:
            tunnel_process.terminate()
        if server_process and not already_running:
            server_process.terminate()
        print("[OK] Procesos detenidos con éxito.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        while True:
            time.sleep(1)
            if tunnel_process.poll() is not None:
                log("El proceso de Cloudflare Tunnel se ha detenido inesperadamente.", "WARNING")
                break
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
