"""Descarga los datos versionados con DVC desde el remoto S3 del proyecto.

Uso desde cualquier directorio:
    python scripts/01_download_data.py

Requiere DVC con soporte S3 y credenciales AWS disponibles para boto3/DVC
(por ejemplo, el perfil configurado en AWS CLI o variables de entorno).
"""

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_TARGETS = (
    "data/sleep-edfx.dvc",
    "data/SC-subjects-split.xlsx.dvc",
)


def main() -> int:
    dvc = shutil.which("dvc")
    if dvc is None:
        print(
            """No se encontró el comando 'dvc'. 
            Instala las dependencias del proyecto con 'pip install -r requirements.txt'.""",
            file=sys.stderr,
        )
        return 1

    print("Descargando los datos versionados desde el remoto S3 configurado en DVC...")
    print(f"Repositorio: {PROJECT_ROOT}")
    try:
        subprocess.run(
            [dvc, "pull", *DATA_TARGETS],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            "Falló 'dvc pull'. Verifica que tengas acceso al remoto S3 y que "
            "las credenciales AWS estén disponibles en este entorno.",
            file=sys.stderr,
        )
        return exc.returncode or 1

    print("Descarga completada. Los datos están disponibles bajo data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
