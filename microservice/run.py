"""
Microservice Runner Script.
Starts the DemandMind FastAPI server with Uvicorn and Scalar documentation.
Usage: python run.py (or from root: python microservice/run.py)
"""

import os
import sys
from pathlib import Path

# Asegurar que microservice está en sys.path
MICROSERVICE_DIR = Path(__file__).resolve().parent
if str(MICROSERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(MICROSERVICE_DIR))

import uvicorn
from app.config import HOST, PORT, DEBUG


if __name__ == "__main__":
    print("=" * 75)
    print("DEMANDMIND: RETAIL DEMAND FORECASTING & MULTIMODAL AI MICROSERVICE")
    print(f"Scalar Documentation UI : http://localhost:{PORT}/docs")
    print(f"Health Status Endpoint  : http://localhost:{PORT}/health")
    print("=" * 75)

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG
    )
