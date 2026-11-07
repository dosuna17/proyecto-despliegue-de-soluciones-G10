"""Sube a MLflow los resultados de la validacion cruzada flexible (o de un control).

Lee los archivos que deja scripts/06_train_flexible.py; no entrena nada.
Dos modos, cada uno crea su propio run:
  - validacion cruzada (por defecto): metricas por combinacion e historial por pliegue.
  - --final: el modelo final (pesos .pth + JSON), con la etiqueta stage=final_model, como la
    version base. El .pth queda en la raiz del run, que es donde la API de Daniel lo busca.

Uso:
    python scripts/07_log_mlflow_flexible.py --carpeta resultados --run flexible
    python scripts/07_log_mlflow_flexible.py --carpeta resultados --run control_eeg_fpz_cz
    python scripts/07_log_mlflow_flexible.py --carpeta resultados --run flexible --final
    python scripts/07_log_mlflow_flexible.py --carpeta resultados --run flexible --final --cv-run-id <id>

Variables de entorno: MLFLOW_TRACKING_URI (y MLFLOW_TRACKING_USERNAME / _PASSWORD si el
servidor las pide). Las escribes tu al ejecutarlo; no se guardan en el codigo.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import mlflow

EXPERIMENTO = "sleep-staging-advanced"  # el mismo que usa la version base


def subir_modelo_final(carpeta: Path, args) -> None:
    """Run propio del modelo final: pesos + JSON en la raiz del run (la API busca asi el .pth)."""
    info = json.load(open(carpeta / f"{args.run}_final.json", encoding="utf-8"))
    pesos = carpeta / info["pesos"]
    if not pesos.exists():
        sys.exit(f"No existe {pesos}")
    es_flexible = args.run == "flexible"

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENTO)
    mlflow.start_run(run_name=f"tinysleepnet_final_{args.run}")
    try:
        mlflow.set_tag("model_type", "advanced")
        mlflow.set_tag("stage", "final_model")
        mlflow.set_tag("signals_used", ("flexible: " + ", ".join(info["combinaciones"])) if es_flexible
                       else info["combinaciones"][0])
        if args.cv_run_id:
            mlflow.set_tag("cv_run_id", args.cv_run_id)
        mlflow.set_tag("rendimiento", "ver el run de validacion cruzada; aqui no hay metrica de validacion ni de test")
        mlflow.log_params({
            "architecture": "TinySleepNet (CNN1D + LSTM), entrada 2 x n_posiciones",
            "flexible": es_flexible,
            "n_canales_entrada": info["n_canales_entrada"],
            "combinaciones": ", ".join(info["combinaciones"]),
            "n_epocas": info["n_epocas"],
            "n_grabaciones_train": info["n_grabaciones"],
            "seq_length": info["seq_length"],
            "clases": "5 (W, N1, N2, N3, REM), sin M",
            "vigilia": "recortada en los extremos (30 min)",
            "canales_ausentes": "ceros + marca de presencia",
            "n_epocas_origen": "media de las mejores epocas de la validacion cruzada",
        })
        for h in info["historial"]:  # solo entrenamiento: sirve para ver que converge, no como rendimiento
            mlflow.log_metrics({"train_loss": h["train_loss"], "train_macro_f1": h["train_macro_f1"]},
                               step=h["epoca"])
        mlflow.log_metric("segundos_totales", info["segundos_totales"])
        mlflow.log_artifact(str(pesos))                                  # raiz del run
        mlflow.log_artifact(str(carpeta / f"{args.run}_final.json"))
        print("Subido. Run:", mlflow.active_run().info.run_id)
    finally:
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"[AVISO] No se pudo cerrar el run (lo subido ya esta): {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--carpeta", required=True)
    p.add_argument("--run", required=True, help="'flexible' o 'control_<combinacion>'")
    p.add_argument("--adjuntos", nargs="*", default=[], help="Archivos extra (figuras, etc.).")
    p.add_argument("--final", action="store_true", help="Sube el modelo final en vez de la validacion cruzada.")
    p.add_argument("--cv-run-id", default=None, help="Con --final: id del run de validacion cruzada (queda como etiqueta).")
    args = p.parse_args()

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        sys.exit("Falta MLFLOW_TRACKING_URI.")
    carpeta = Path(args.carpeta)
    if args.final:
        subir_modelo_final(carpeta, args)
        return
    resumen = json.load(open(carpeta / f"{args.run}_resumen.json", encoding="utf-8"))
    pliegues = [json.load(open(a, encoding="utf-8"))
                for a in sorted(carpeta.glob(f"{args.run}_pliegue*.json"))]
    es_flexible = args.run == "flexible"
    combos = list(resumen["combinaciones"])

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENTO)

    # Run manual (no "with"): igual que la version base, para poder proteger el cierre.
    mlflow.start_run(run_name=f"tinysleepnet_{args.run}_cv")
    try:
        mlflow.set_tag("model_type", "advanced")
        mlflow.set_tag("stage", "cv")
        mlflow.set_tag("signals_used", ("flexible: " + ", ".join(combos)) if es_flexible else combos[0])
        mlflow.log_params({
            "architecture": "TinySleepNet (CNN1D + LSTM), entrada 2 x n_posiciones",
            "flexible": es_flexible,
            "n_combinaciones": len(combos),
            "n_folds": resumen["n_pliegues"],
            "clases": "5 (W, N1, N2, N3, REM), sin M",
            "vigilia": "recortada en los extremos (30 min)",
            "canales_ausentes": "ceros + marca de presencia",
            "repeticion_por_combinacion": es_flexible,
        })
        for nombre, m in resumen["combinaciones"].items():
            mlflow.log_metrics({
                f"cv_macro_f1_media__{nombre}": m["macro_f1_media"],
                f"cv_macro_f1_desv__{nombre}": m["macro_f1_desv"],
                f"cv_kappa_media__{nombre}": m["kappa_media"],
                f"cv_kappa_desv__{nombre}": m["kappa_desv"],
            })
        g = resumen["macro_f1_medio_global"]
        mlflow.log_metrics({"cv_mean_macro_f1": g["media"], "cv_std_macro_f1": g["desv"],
                            "segundos_totales": resumen["segundos_totales"]})
        for pl in pliegues:  # historial por epoca de cada pliegue, como la version base
            k = pl["pliegue"]
            for h in pl["historial"]:
                mlflow.log_metrics({
                    f"fold{k}_train_loss": h["train_loss"],
                    f"fold{k}_train_macro_f1": h["train_macro_f1"],
                    f"fold{k}_val_loss": h["val_loss"],
                    f"fold{k}_val_macro_f1": h["val_macro_f1_medio"],
                }, step=h["epoca"])

        for a in sorted(carpeta.glob(f"{args.run}_*.json")):
            mlflow.log_artifact(str(a), artifact_path="resultados")
        for a in args.adjuntos:
            mlflow.log_artifact(a)
        print("Subido. Run:", mlflow.active_run().info.run_id)
    finally:
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"[AVISO] No se pudo cerrar el run (lo subido ya esta): {e}")


if __name__ == "__main__":
    main()
