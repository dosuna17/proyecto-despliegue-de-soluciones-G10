"""Datos de entrenamiento de los 8 modelos, desde MLflow.

Por modelo: épocas máximas y paciencia de la validación cruzada, épocas corridas y mejores por pliegue,
épocas y datos del modelo final, número de parámetros (contado en los pesos guardados), hardware y tiempo por pliegue.
Las épocas máximas y la paciencia son las de los comandos de entrenamiento (README, sección 10); el script
comprueba con los historiales que la paciencia fue esa (cada pliegue paró esas épocas después de su mejor época,
salvo los que llegaron al tope). Hardware: GPU NVIDIA T4 en Google Colab para todas las corridas.

Uso, desde la raíz del repositorio:
    python scripts/verificacion/tabla_entrenamiento.py --salida reports/tables
    (pide el URI, el usuario y la clave de MLflow si no están en el entorno)
"""
import argparse
import getpass
import json
import os
import tempfile
from pathlib import Path

import numpy as np

RUNS = {"flexible_emg_mlp": "95951e2a2cfd4368970636cc6bdea9b5", "flexible": "029cc677c3734b10959735a6e9246cb8",
        "flexible_sin_lstm": "c424e04708c34f04a5bfec810b705982",
        "control_eeg_fpz_cz": "2bd024695ad44d19b175490e5abbdedb", "control_eeg_pz_oz": "30877a344b304c5a8a2b5f2e3163805f",
        "control_eeg_fpz_cz__eeg_pz_oz": "076282f5e5534691b441514b50e90441",
        "control_eeg_fpz_cz__eog_horizontal": "038a654fc695444ab650f4e2aac9056f",
        "control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": "74b16a9d1a854bb5a5589353fec14f91"}
NOMBRE = {"flexible_emg_mlp": "EMG-MLP (seleccionado)", "flexible": "Flexible sin EMG", "flexible_sin_lstm": "Flexible sin LSTM",
          "control_eeg_fpz_cz": "Dedicado Fpz-Cz", "control_eeg_pz_oz": "Dedicado Pz-Oz",
          "control_eeg_fpz_cz__eeg_pz_oz": "Dedicado Fpz-Cz + Pz-Oz", "control_eeg_fpz_cz__eog_horizontal": "Dedicado Fpz-Cz + EOG",
          "control_eeg_fpz_cz__eeg_pz_oz__eog_horizontal__emg_submental": "Dedicado de 4 señales"}
TOPE = lambda m: (30, 5) if m.startswith("flexible") else (240, 60)     # --max-epocas y paciencia de cada corrida
HARDWARE = "GPU NVIDIA T4 (16 GB), Google Colab"

ap = argparse.ArgumentParser()
ap.add_argument("--salida", default="reports/tables")
ap.add_argument("--local", default=None, help="(pruebas) carpeta con los JSON y .pth ya bajados")
args = ap.parse_args()
sal = Path(args.salida)
sal.mkdir(parents=True, exist_ok=True)

if args.local:
    carp = fin = {m: Path(args.local) for m in RUNS}
else:
    import mlflow
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = input("MLFLOW_TRACKING_URI: ").strip()
        u = input("MLFLOW_TRACKING_USERNAME (vacío si no hace falta): ").strip()
        if u:
            os.environ["MLFLOW_TRACKING_USERNAME"] = u
            os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass.getpass("MLFLOW_TRACKING_PASSWORD: ")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    tmp = Path(tempfile.mkdtemp(prefix="tabla_entrenamiento_"))
    carp, fin = {}, {}
    for m, rid in RUNS.items():
        carp[m] = Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="resultados", dst_path=str(tmp / m)))
        fin[m] = Path(mlflow.artifacts.download_artifacts(run_id=rid, artifact_path="modelo_final", dst_path=str(tmp / m)))


def n_parametros(pth):
    import torch
    sd = torch.load(pth, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd.get("model_state_dict", sd)) if isinstance(sd, dict) else sd
    return sum(v.numel() for k, v in sd.items() if hasattr(v, "numel") and "running_" not in k and "num_batches_tracked" not in k)


T = {}
for m in RUNS:
    pl = sorted((json.load(open(a, encoding="utf-8")) for a in carp[m].glob(f"{m}_pliegue[0-9].json")), key=lambda x: x["pliegue"])
    jf_ruta = next(fin[m].glob(f"{m}_final.json"), None)
    if len(pl) != 5 or jf_ruta is None:
        print(f"   (omitido {m}: faltan archivos)")
        continue
    jf = json.load(open(jf_ruta, encoding="utf-8"))
    pth = next(fin[m].glob(f"{m}_final.pth"), None)
    tope, pac = TOPE(m)
    corr = [len(x["historial"]) for x in pl]
    mej = [x["mejor_epoca"] for x in pl]
    pac_obs = sorted({c - b for c, b in zip(corr, mej) if c < tope})
    al_tope = sum(c == tope for c in corr)
    T[m] = dict(tope=tope, pac=pac, pac_obs=pac_obs, al_tope=al_tope, corr=corr, mej=mej,
                min_pl=[x["segundos_pliegue"] / 60 for x in pl],
                ep_s=np.mean([h["segundos_entrenar"] + h["segundos_validar"] for x in pl for h in x["historial"]]),
                fin_ep=jf["n_epocas"], fin_cv=jf.get("epocas_cv"), noche=jf.get("noche_completa"),
                fin_grab=jf.get("n_grabaciones"), fin_n=jf.get("n_epocas_totales_entrenamiento"),
                fin_min=(jf.get("segundos_totales") or 0) / 60, canales=jf.get("n_canales_entrada"),
                emg=jf.get("emg"), sin_lstm=jf.get("sin_lstm"), params=n_parametros(pth) if pth else None)

md = [f"**Entrenamiento por modelo** (hardware: {HARDWARE}; validación cruzada de 5 pliegues por persona sobre 62 personas, "
      "ventana de 500 épocas continuas por noche; modelo final con las 121 noches completas de esas personas, sin validación)", "",
      "| Modelo | Run de MLflow | Parámetros | Canales de entrada | Épocas máx. / paciencia (VC) | Épocas corridas por pliegue | Mejor época por pliegue | Épocas del final | Min por pliegue (VC) | Min del final |",
      "|---|---|---|---|---|---|---|---|---|---|"]
csv = ["modelo;run_id;parametros;canales_entrada;epocas_max;paciencia;corridas;mejores;epocas_final;min_por_pliegue;min_final;grabaciones_final;epocas_entrenamiento_final"]
for m, t in T.items():
    md.append(f"| {NOMBRE[m]} | `{RUNS[m]}` | {t['params']:,} | {t['canales']} | {t['tope']} / {t['pac']} | {', '.join(map(str, t['corr']))} | "
              f"{', '.join(map(str, t['mej']))} | {t['fin_ep']} | {', '.join(f'{x:.0f}' for x in t['min_pl'])} | {t['fin_min']:.0f} |")
    csv.append(";".join([m, RUNS[m], str(t["params"]), str(t["canales"]), str(t["tope"]), str(t["pac"]), str(t["corr"]), str(t["mej"]),
                         str(t["fin_ep"]), str([round(x, 1) for x in t["min_pl"]]), f"{t['fin_min']:.1f}", str(t["fin_grab"]), str(t["fin_n"])]))
md += ["", "Comunes a todos: Adam (tasa 1e-4), lote de 20 secuencias de 15 épocas, recorte del gradiente a norma 5, L2 1e-3 en las "
       "convoluciones, abandono 0.5, pesos de clase 1/√frecuencia, semilla 42. Las épocas del modelo final son la media de las mejores "
       "épocas de la validación cruzada. Los canales de entrada cuentan señales y marcas de presencia (6: Fpz-Cz, Pz-Oz, EOG y sus marcas; "
       "8: además el EMG y su marca, que van a la rama MLP).", ""]
(sal / "tabla_entrenamiento.md").write_text("\n".join(md) + "\n", encoding="utf-8")
(sal / "tabla_entrenamiento.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
print("\n".join(md))

fallos = 0
def check(a, o, ok):
    global fallos
    fallos += not ok
    print(f"[{'OK ' if ok else 'NO '}] {a}\n       observado: {o}")
print()
for m, t in T.items():
    check(f"{NOMBRE[m]}: paciencia {t['pac']} (cada pliegue que no llegó al tope paró {t['pac']} épocas después de su mejor época); "
          f"épocas del final = media de las mejores redondeada",
          f"diferencias observadas {t['pac_obs']}; pliegues en el tope {t['al_tope']}; final {t['fin_ep']} (media {np.mean(t['mej']):.1f})",
          t["pac_obs"] == [t["pac"]] and t["fin_ep"] == int(round(np.mean(t["mej"]))) and t["noche"] is True)
if "flexible_emg_mlp" in T:
    e = T["flexible_emg_mlp"]
    check("EMG-MLP: 1,558,165 parámetros, 8 canales de entrada, final de 12 épocas con 121 grabaciones y 150,272 épocas",
          f"{e['params']:,}; {e['canales']}; {e['fin_ep']} épocas; {e['fin_grab']} grabaciones; {e['fin_n']:,} épocas",
          (e["params"], e["canales"], e["fin_ep"], e["fin_grab"], e["fin_n"]) == (1558165, 8, 12, 121, 150272))
print(f"\nArchivos en {sal}. {'Todo coincide.' if fallos == 0 else f'{fallos} no coinciden: revisar.'}")
