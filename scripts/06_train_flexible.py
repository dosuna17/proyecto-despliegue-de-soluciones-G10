"""Validacion cruzada de TinySleepNet con canales flexibles.

Uso (desde la raiz del repositorio, con SLEEP_NPZ_DIR apuntando a la carpeta de 3 canales):
    python scripts/06_train_flexible.py                         # modelo flexible, 6 combinaciones
    python scripts/06_train_flexible.py --control eeg_fpz_cz    # un modelo de control
    python scripts/06_train_flexible.py --pliegues 0 1          # solo algunos pliegues
    python scripts/06_train_flexible.py --emg mlp               # con EMG en una rama aparte (12 combinaciones)
    python scripts/06_train_flexible.py --emg conv              # igual, con la rama convolucional
    python scripts/06_train_flexible.py --final                 # modelo final (todos los sujetos de train, sin pliegues)
    python scripts/06_train_flexible.py --sin-lstm              # ablacion: el flexible de 3 senales sin la LSTM
    python scripts/06_train_flexible.py --emg mlp --control eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental
                                                                # control con las 4 senales (para comparar con el EMG)
    python scripts/06_train_flexible.py --final --noche-completa   # final con las noches completas (la CV usa la ventana)

La CV guarda ademas, por pliegue y combinacion, las probabilidades de validacion de la mejor epoca
(carpeta <salida>/probs_validacion/); sirven para fijar el umbral de confianza sin mirar el test.
Con --sin-probs no se guardan.
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
    parser.add_argument("--emg", choices=["mlp", "conv"], default=None,
                        help="Usa el EMG submental en una rama aparte (mlp o conv). Sin esta opcion, 3 senales.")
    parser.add_argument("--sin-lstm", action="store_true",
                        help="Ablacion: TinySleepNet de 3 senales sin la LSTM (cada epoca por separado).")
    parser.add_argument("--sin-probs", action="store_true",
                        help="No guardar las probabilidades de validacion (por defecto se guardan).")
    parser.add_argument("--noche-completa", action="store_true",
                        help="Solo con --final: usa cada grabacion completa en vez de la ventana de epocas de la CV.")
    parser.add_argument("--escalar-epocas", action="store_true",
                        help="Solo con --final --noche-completa: divide las epocas de la CV por el cociente "
                             "(epocas con ventana / epocas con noche completa). Por defecto, mismas epocas que la CV.")
    parser.add_argument("--pliegues", type=int, nargs="+", default=None, help="Pliegues a correr (por defecto todos).")
    parser.add_argument("--final", action="store_true",
                        help="Entrena el modelo final con todos los sujetos de train y guarda los pesos (.pth).")
    parser.add_argument("--epocas-final", type=int, default=None,
                        help="Epocas del modelo final. Por defecto, la media de las mejores epocas de la CV en --salida.")
    args = parser.parse_args()

    if args.final:
        from sleep_staging.training.train_flexible import run_final_flexible
        run_final_flexible(control=args.control, salida=args.salida, n_epocas=args.epocas_final,
                           emg=args.emg, sin_lstm=args.sin_lstm,
                           noche_completa=args.noche_completa, escalar_epocas=args.escalar_epocas)
        return
    if args.noche_completa or args.escalar_epocas:
        parser.error("--noche-completa y --escalar-epocas solo se usan con --final.")

    from sleep_staging.training.train_flexible import run_cv_flexible

    run_cv_flexible(control=args.control, salida=args.salida,
                    max_epochs=args.max_epocas, pliegues=args.pliegues,
                    paciencia=args.paciencia, emg=args.emg,
                    sin_lstm=args.sin_lstm, guardar_probs=not args.sin_probs)


if __name__ == "__main__":
    main()
