"""
Genera los .npz con 3 canales de 100 Hz: EEG Fpz-Cz, EEG Pz-Oz y EOG horizontal.
Reutiliza src/data/generar_npz.py (cambiando solo la carpeta de salida y los
canales), asi que el resultado sale con el mismo formato de los datos de la
version base. Es reanudable: si se corta, vuelve a correrlo.

Uso (desde la raiz del repo):
    python scripts/generar_3canales.py
Luego el recorte de vigilia:
    python src\\data\\aplicar_recorte_vigilia.py data\\interim\\epochs_por_sujeto_3canales
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "data"))

import generar_npz as g  # noqa: E402

g.OUT_DIR = Path("data/interim/epochs_por_sujeto_3canales")
g.CHANNEL_PICKS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]

if __name__ == "__main__":
    g.main()
