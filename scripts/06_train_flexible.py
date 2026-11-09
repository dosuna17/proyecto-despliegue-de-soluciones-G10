"""Validacion cruzada de TinySleepNet con canales flexibles.

Uso (desde la raiz del repositorio, con SLEEP_NPZ_DIR apuntando a la carpeta de 3 canales):
    python scripts/06_train_flexible.py                         # modelo flexible, 6 combinaciones
    python scripts/06_train_flexible.py --control eeg_fpz_cz    # un modelo de control
    python scripts/06_train_flexible.py --pliegues 0 1          # solo algunos pliegues
    python scripts/06_train_flexible.py --final                 # modelo final (todos los sujetos de train, sin pliegues)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="TinySleepNet con canales flexibles (validacion cruzada).")
    parser.add_argument("--control", default=None,
                        help="Nombre de una combinacion (p. ej. eeg_fpz_cz) para entrenar un modelo de control.")
    parser.add_argument("--salida", default="resultados_flexible", help="Carpeta donde se guardan los JSON.")
    parser.add_argument("--max-epocas", type=int, default=None, help="Por defecto, el valor del yaml.")
    parser.add_argument("--paciencia", type=int, default=None,
                        help="Paciencia del early stopping. Por defecto, el valor del yaml.")
    parser.add_argument("--pliegues", type=int, nargs="+", default=None, help="Pliegues a correr (por defecto todos).")
    parser.add_argument("--final", action="store_true",
                        help="Entrena el modelo final con todos los sujetos de train y guarda los pesos (.pth).")
    parser.add_argument("--epocas-final", type=int, default=None,
                        help="Epocas del modelo final. Por defecto, la media de las mejores epocas de la CV en --salida.")
    args = parser.parse_args()

    if args.final:
        from sleep_staging.training.train_flexible import run_final_flexible
        run_final_flexible(control=args.control, salida=args.salida, n_epocas=args.epocas_final)
        return

    from sleep_staging.training.train_flexible import run_cv_flexible

    run_cv_flexible(control=args.control, salida=args.salida,
                    max_epochs=args.max_epocas, pliegues=args.pliegues,
                    paciencia=args.paciencia)


if __name__ == "__main__":
    main()
