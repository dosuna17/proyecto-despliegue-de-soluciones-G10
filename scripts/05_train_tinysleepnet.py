"""Comando para entrenar TinySleepNet.

Uso (desde la raiz del repositorio):
    python scripts/05_train_tinysleepnet.py --modo cv
    python scripts/05_train_tinysleepnet.py --modo final
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena TinySleepNet.")
    parser.add_argument(
        "--modo",
        choices=["cv", "final"],
        required=True,
        help="cv: validacion cruzada por sujeto. final: entrenamiento final con validacion interna.",
    )
    args = parser.parse_args()

    from sleep_staging.training import train_tinysleepnet as tr

    if args.modo == "cv":
        tr.run_cv()
    else:
        tr.run_final()


if __name__ == "__main__":
    main()
