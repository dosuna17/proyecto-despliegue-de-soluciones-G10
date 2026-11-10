"""Sube a MLflow los resultados de la validacion cruzada flexible (o de un control).

Lee los archivos que deja scripts/06_train_flexible.py (y scripts/08_evaluar_test.py); no entrena nada.
Pasos, cada uno por separado:
  1. Validacion cruzada (por defecto): crea el run `tinysleepnet_<run>_cv` con las metricas por
     combinacion y el historial por pliegue. Al terminar imprime su run_id.
  2. --final --cv-run-id <id>: agrega el modelo final AL MISMO run de la validacion cruzada, en la
     carpeta `modelo_final/` (pesos .pth + JSON). Asi un solo run tiene el rendimiento y los pesos,
     y es donde la API (configs/models/model_registry.yaml) los busca.
  3. --test-carpeta <carpeta> --cv-run-id <id>: agrega al mismo run los resultados del test
     (scripts/08_evaluar_test.py). Solo funciona si el test ya se evaluo (existe TEST_YA_EJECUTADO.json).
  4. --campeon --cv-run-id <id>: marca ese run con champion=true y quita la etiqueta a los runs
     anteriores con el mismo nombre. Es lo que decide que modelo carga la API.

Sin --cv-run-id, --final conserva el comportamiento anterior: crea un run aparte
(`tinysleepnet_final_<run>`) con los pesos en la raiz, que la API NO usa.

Uso:
    python scripts/07_log_mlflow_flexible.py --carpeta resultados --run flexible
    python scripts/07_log_mlflow_flexible.py --carpeta resultados --run flexible --final --cv-run-id <id>
    python scripts/07_log_mlflow_flexible.py --carpeta resultados --run flexible --cv-run-id <id> --test-carpeta resultados_test
    python scripts/07_log_mlflow_flexible.py --run flexible --cv-run-id <id> --campeon
    python scripts/07_log_mlflow_flexible.py --carpeta resultados --run control_eeg_fpz_cz

Los pasos 2, 3 y 4 se pueden pedir juntos en un solo comando. El --run es el nombre del modelo:
'flexible', 'flexible_emg_mlp' o 'control_<combinacion>'.

Variables de entorno: MLFLOW_TRACKING_URI (y MLFLOW_TRACKING_USERNAME / _PASSWORD si el
servidor las pide). Las escribes tu al ejecutarlo; no se guardan en el codigo.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
from mlflow.entities import ViewType

EXPERIMENTO = "sleep-staging-advanced"  # el mismo que usa la version base
RAIZ = Path(__file__).resolve().parents[1]
CARPETA_MODELO = "modelo_final"          # carpeta dentro del run donde la API busca el .pth


def es_flexible(run: str) -> bool:
    return run.startswith("flexible")


def rama_emg(run: str) -> str:
    """'flexible_emg_mlp' -> 'mlp'; cualquier otro modelo -> 'no'."""
    return run.removeprefix("flexible_emg_") if run.startswith("flexible_emg_") else "no"


def descripcion_arquitectura(run: str) -> str:
    if rama_emg(run) != "no":
        return (f"TinySleepNetEMG (CNN1D + rama EMG {rama_emg(run)} + LSTM), entrada 2 x n_posiciones; "
                "el EMG (1 Hz) viaja repetido 100 veces")
    return "TinySleepNet (CNN1D + LSTM), entrada 2 x n_posiciones"


def commit_actual() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=RAIZ, timeout=10).stdout.strip()
    except Exception:
        return ""


def cerrar_run() -> None:
    try:
        mlflow.end_run()
    except Exception as e:
        print(f"[AVISO] No se pudo cerrar el run (lo subido ya esta): {e}")


# ---------------------------------------------------------------------
# Paso 2: modelo final
# ---------------------------------------------------------------------
def subir_modelo_final(carpeta: Path, args) -> None:
    info = json.load(open(carpeta / f"{args.run}_final.json", encoding="utf-8"))
    pesos = carpeta / info["pesos"]
    if not pesos.exists():
        sys.exit(f"No existe {pesos}")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENTO)
    en_run_cv = bool(args.cv_run_id)
    if en_run_cv:
        mlflow.start_run(run_id=args.cv_run_id)          # el mismo run de la validacion cruzada
    else:
        mlflow.start_run(run_name=f"tinysleepnet_final_{args.run}")   # comportamiento anterior
    try:
        prefijo = "final_" if en_run_cv else ""          # para no mezclar con las curvas de la CV
        mlflow.set_tag("model_type", "advanced")
        if not en_run_cv:
            mlflow.set_tag("stage", "final_model")
            mlflow.set_tag("signals_used", ("flexible: " + ", ".join(info["combinaciones"]))
                           if es_flexible(args.run) else info["combinaciones"][0])
            mlflow.set_tag("rendimiento", "ver el run de validacion cruzada; aqui no hay metrica de validacion ni de test")
        mlflow.set_tag("modelo_final_subido", "true")
        commit = commit_actual()
        if commit:
            mlflow.set_tag("git_commit_modelo_final", commit)
        mlflow.log_params({
            f"{prefijo}architecture": descripcion_arquitectura(args.run),
            f"{prefijo}flexible": es_flexible(args.run),
            f"{prefijo}emg": rama_emg(args.run),
            f"{prefijo}n_canales_entrada": info["n_canales_entrada"],
            f"{prefijo}combinaciones": ", ".join(info["combinaciones"]),
            f"{prefijo}n_epocas": info["n_epocas"],
            f"{prefijo}n_grabaciones_train": info["n_grabaciones"],
            f"{prefijo}seq_length": info["seq_length"],
            f"{prefijo}n_epocas_origen": "media de las mejores epocas de la validacion cruzada",
        })
        for h in info["historial"]:  # solo entrenamiento: sirve para ver que converge, no como rendimiento
            mlflow.log_metrics({f"{prefijo}train_loss": h["train_loss"],
                                f"{prefijo}train_macro_f1": h["train_macro_f1"]}, step=h["epoca"])
        mlflow.log_metric(f"{prefijo}segundos_totales", info["segundos_totales"])

        destino = CARPETA_MODELO if en_run_cv else None   # sin run de CV, raiz del run (como antes)
        mlflow.log_artifact(str(pesos), artifact_path=destino)
        mlflow.log_artifact(str(carpeta / f"{args.run}_final.json"), artifact_path=destino)
        configuracion = RAIZ / "configs" / "models" / "tinysleepnet.yaml"
        if configuracion.exists():                         # hiperparametros exactos con los que se entreno
            mlflow.log_artifact(str(configuracion), artifact_path="config")
        print("Modelo final subido. Run:", mlflow.active_run().info.run_id,
              f"| carpeta: {destino or '(raiz)'}")
    finally:
        cerrar_run()


# ---------------------------------------------------------------------
# Paso 3: resultados del test
# ---------------------------------------------------------------------
def subir_test(carpeta_test: Path, args) -> None:
    if not (carpeta_test / "TEST_YA_EJECUTADO.json").exists():
        sys.exit(f"No hay {carpeta_test / 'TEST_YA_EJECUTADO.json'}: el test todavia no se evaluo.")
    resultados = json.load(open(carpeta_test / "resultados_test.json", encoding="utf-8"))
    if resultados.get("modo") != "test":
        sys.exit("resultados_test.json no es del modo test (es un ensayo).")
    if args.run not in resultados["modelos"]:
        sys.exit(f"'{args.run}' no esta en los resultados del test: {list(resultados['modelos'])}")
    combos = resultados["modelos"][args.run]["combinaciones"]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENTO)
    mlflow.start_run(run_id=args.cv_run_id)
    try:
        mlflow.set_tag("test_evaluado", "true")
        mlflow.set_tag("test_nota", "evaluado una sola vez; el umbral de confianza se fijo sobre el propio test (optimista)")
        mlflow.log_param("test_n_sujetos", len({s for s in resultados["sujetos"]}))
        for nombre, r in combos.items():
            mlflow.log_metrics({
                f"test_macro_f1__{nombre}": r["macro_f1"],
                f"test_macro_f1_ic95_bajo__{nombre}": r["macro_f1_ic95"][0],
                f"test_macro_f1_ic95_alto__{nombre}": r["macro_f1_ic95"][1],
                f"test_accuracy__{nombre}": r["accuracy"],
                f"test_balanced_accuracy__{nombre}": r["balanced_accuracy"],
                f"test_kappa__{nombre}": r["cohens_kappa"],
            })
        mlflow.log_metric("test_macro_f1_medio_combinaciones",
                          sum(r["macro_f1"] for r in combos.values()) / len(combos))
        for nombre_archivo in ["resultados_test.json", "tabla_test.csv", "TEST_YA_EJECUTADO.json"]:
            if (carpeta_test / nombre_archivo).exists():
                mlflow.log_artifact(str(carpeta_test / nombre_archivo), artifact_path="test")
        figuras = [carpeta_test / "figuras", carpeta_test.parent / "figures", Path("figures")]
        for carpeta_fig in figuras:
            for pdf in sorted(carpeta_fig.glob(f"test_*_{args.run}.pdf")) if carpeta_fig.exists() else []:
                mlflow.log_artifact(str(pdf), artifact_path="test/figuras")
        print("Resultados del test subidos. Run:", mlflow.active_run().info.run_id)
    finally:
        cerrar_run()


# ---------------------------------------------------------------------
# Paso 4: campeon
# ---------------------------------------------------------------------
def marcar_campeon(args) -> None:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = mlflow.MlflowClient()
    run = client.get_run(args.cv_run_id)
    nombre = run.data.tags.get("mlflow.runName", "")
    pesos = f"{CARPETA_MODELO}/{args.run}_final.pth"
    presentes = [a.path for a in client.list_artifacts(args.cv_run_id, CARPETA_MODELO)]
    if pesos not in presentes:
        sys.exit(f"El run {args.cv_run_id} no tiene {pesos}; sube primero el modelo final con --final.")
    if args.run != "flexible":
        print(f"[AVISO] La API busca '{CARPETA_MODELO}/flexible_final.pth' (configs/models/model_registry.yaml). "
              f"Con '{args.run}' hay que cambiar el registro y avisar a Daniel antes de desplegar.")
    escapado = nombre.replace("'", "\\'")
    anteriores = client.search_runs(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"tags.champion = 'true' and tags.mlflow.runName = '{escapado}'",
        run_view_type=ViewType.ALL)
    for r in anteriores:
        if r.info.run_id != args.cv_run_id:
            client.set_tag(r.info.run_id, "champion", "false")
            print(f"Quitada la etiqueta champion al run {r.info.run_id}")
    client.set_tag(args.cv_run_id, "champion", "true")
    print(f"Campeon: run {args.cv_run_id} ({nombre})")


# ---------------------------------------------------------------------
# Paso 1: validacion cruzada
# ---------------------------------------------------------------------
def subir_cv(carpeta: Path, args) -> None:
    resumen = json.load(open(carpeta / f"{args.run}_resumen.json", encoding="utf-8"))
    pliegues = [json.load(open(a, encoding="utf-8"))
                for a in sorted(carpeta.glob(f"{args.run}_pliegue*.json"))]
    combos = list(resumen["combinaciones"])

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENTO)

    # Run manual (no "with"): igual que la version base, para poder proteger el cierre.
    mlflow.start_run(run_name=f"tinysleepnet_{args.run}_cv")
    try:
        mlflow.set_tag("model_type", "advanced")
        mlflow.set_tag("stage", "cv")
        mlflow.set_tag("signals_used", ("flexible: " + ", ".join(combos)) if es_flexible(args.run) else combos[0])
        mlflow.log_params({
            "architecture": descripcion_arquitectura(args.run),
            "flexible": es_flexible(args.run),
            "emg": rama_emg(args.run),
            "n_combinaciones": len(combos),
            "n_folds": resumen["n_pliegues"],
            "clases": "5 (W, N1, N2, N3, REM), sin M",
            "vigilia": "recortada en los extremos (30 min)",
            "canales_ausentes": "ceros + marca de presencia",
            "repeticion_por_combinacion": es_flexible(args.run),
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
            if a.name.endswith("_final.json"):   # el final va en modelo_final/ con --final
                continue
            mlflow.log_artifact(str(a), artifact_path="resultados")
        for a in args.adjuntos:
            mlflow.log_artifact(a)
        print("Subido. Run:", mlflow.active_run().info.run_id)
    finally:
        cerrar_run()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--carpeta", default=None, help="Carpeta con los resultados de 06 (CV y final). No hace falta con --campeon.")
    p.add_argument("--run", required=True, help="'flexible', 'flexible_emg_mlp' o 'control_<combinacion>'")
    p.add_argument("--adjuntos", nargs="*", default=[], help="Archivos extra (figuras, etc.).")
    p.add_argument("--final", action="store_true", help="Sube el modelo final (con --cv-run-id, dentro del run de la CV).")
    p.add_argument("--cv-run-id", default=None, help="Id del run de validacion cruzada al que se agrega el modelo final, el test o la etiqueta de campeon.")
    p.add_argument("--test-carpeta", default=None, help="Carpeta con los resultados de scripts/08_evaluar_test.py (requiere --cv-run-id).")
    p.add_argument("--campeon", action="store_true", help="Marca el run --cv-run-id como campeon (champion=true).")
    args = p.parse_args()

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        sys.exit("Falta MLFLOW_TRACKING_URI.")
    if (args.test_carpeta or args.campeon) and not args.cv_run_id:
        sys.exit("--test-carpeta y --campeon necesitan --cv-run-id.")
    if (args.final or not (args.test_carpeta or args.campeon)) and not args.carpeta:
        sys.exit("Falta --carpeta.")

    if args.final:
        subir_modelo_final(Path(args.carpeta), args)
    if args.test_carpeta:
        subir_test(Path(args.test_carpeta), args)
    if args.campeon:
        marcar_campeon(args)
    if not (args.final or args.test_carpeta or args.campeon):
        subir_cv(Path(args.carpeta), args)


if __name__ == "__main__":
    main()
