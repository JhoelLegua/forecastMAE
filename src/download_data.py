"""
Data Acquisition Module.
Downloads the Amazon Sales Dataset from Kaggle via kagglehub
and places raw files into data/raw directory.
"""

import os
import shutil
import kagglehub


DATASET_IDENTIFIER = "ikramshah512/amazon-products-sales-dataset-42k-items-2025"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def download_dataset(target_dir: str = RAW_DATA_DIR) -> str:
    """
    Descarga el dataset desde KaggleHub y copia los archivos a data/raw.
    
    :param target_dir: Carpeta destino local para almacenar los archivos crudos.
    :return: Ruta a la carpeta con los datos descargados.
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"[INFO] Downloading dataset '{DATASET_IDENTIFIER}' via kagglehub...")
    downloaded_path = kagglehub.dataset_download(DATASET_IDENTIFIER)
    print(f"[INFO] Downloaded to cache: {downloaded_path}")

    # Copiar archivos a data/raw
    for filename in os.listdir(downloaded_path):
        source_file = os.path.join(downloaded_path, filename)
        dest_file = os.path.join(target_dir, filename)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, dest_file)
            print(f"[INFO] Copied {filename} -> {dest_file}")

    print(f"[SUCCESS] Raw files ready in: {target_dir}")
    return target_dir


if __name__ == "__main__":
    download_dataset()
