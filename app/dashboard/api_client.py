"""Comunicación del tablero con la API de inferencia.

Toda llamada HTTP del tablero pasa por aquí, para que los errores lleguen a
la interfaz como mensajes en lenguaje llano (``ErrorAPI``) y no como trazas.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from procesamiento import (
    CANALES_EEG,
    ESTADIOS_SUENO,
    bloques_secuencia,
    construir_bloque,
    todas_las_combinaciones,
)


class ErrorAPI(RuntimeError):
    """Fallo de la API explicado para el usuario."""

    def __init__(self, mensaje: str, detalle_tecnico: str = ""):
        super().__init__(mensaje)
        self.mensaje = mensaje
        self.detalle_tecnico = detalle_tecnico


@dataclass
class InfoModelo:
    nombre: str
    endpoint: str
    epocas_por_llamada: Optional[int]
    canales_modelo: List[str]
    clases: List[str]
    combinaciones: List[List[str]]
    origen: str = "config"  # "api" si se leyó de GET /models
    descripcion: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultadoInferencia:
    estadios: np.ndarray  # (n_epocas,) str
    confianzas: np.ndarray  # (n_epocas,) float
    probabilidades: np.ndarray  # (n_epocas, n_clases)
    clases: List[str]
    segundos: float


def combinaciones_desde_canales(canales_modelo: Sequence[str]) -> List[List[str]]:
    """Combinaciones de la propuesta cuyos canales acepta el modelo."""

    return [list(c) for c in todas_las_combinaciones() if set(c) <= set(canales_modelo)]


class ClienteAPI:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.url = config["api"]["url"]
        self.sesion = requests.Session()

    # ------------------------------------------------------------------ util
    def _get(self, ruta: str) -> requests.Response:
        try:
            return self.sesion.get(self.url + ruta, timeout=self.config["api"]["timeout_consulta_s"])
        except requests.RequestException as error:
            raise ErrorAPI(
                "No hay conexión con el servicio de clasificación. Intente de nuevo en unos "
                "minutos; si el problema continúa, avise al equipo de soporte.",
                f"GET {self.url}{ruta}: {error}",
            ) from error

    @staticmethod
    def _mensaje_error(respuesta: requests.Response) -> ErrorAPI:
        try:
            detalle = respuesta.json().get("detail", respuesta.text)
        except ValueError:
            detalle = respuesta.text
        texto = json.dumps(detalle, ensure_ascii=False) if not isinstance(detalle, str) else detalle
        if respuesta.status_code == 503:
            return ErrorAPI(
                "El modelo no está disponible en este momento. Intente de nuevo en unos minutos.",
                texto,
            )
        if respuesta.status_code == 422:
            return ErrorAPI(
                "El servicio no aceptó los datos enviados. Revise que las señales elegidas "
                "existan en el registro.",
                texto,
            )
        return ErrorAPI(f"El servicio respondió con un error ({respuesta.status_code}).", texto)

    # --------------------------------------------------------------- consultas
    def salud(self) -> Dict[str, Any]:
        respuesta = self._get("/health")
        if respuesta.status_code != 200:
            raise self._mensaje_error(respuesta)
        return respuesta.json()

    def info_modelo(self) -> InfoModelo:
        """Modelo campeón y combinaciones que acepta.

        Primero pregunta a GET /models; si la API aún no lo tiene (404) usa
        configs/app.yaml, que debe describir lo mismo que la API acepta.
        """

        respaldo = self.config["modelo"]
        respuesta = self._get("/models")
        if respuesta.status_code == 200:
            datos = respuesta.json()
            canales = datos.get("canales_modelo") or respaldo["canales_modelo"]
            return InfoModelo(
                nombre=datos.get("modelo", respaldo["nombre"]),
                endpoint=datos.get("endpoint", respaldo["endpoint"]),
                epocas_por_llamada=datos.get("epocas_por_llamada", respaldo["epocas_por_llamada"]),
                canales_modelo=list(canales),
                clases=list(datos.get("clases", respaldo["clases"])),
                combinaciones=[list(c) for c in datos.get("combinaciones", combinaciones_desde_canales(canales))],
                origen="api",
                descripcion=respaldo.get("descripcion", ""),
            )
        if respuesta.status_code != 404:
            raise self._mensaje_error(respuesta)
        return InfoModelo(
            nombre=respaldo["nombre"],
            endpoint=respaldo["endpoint"],
            epocas_por_llamada=respaldo["epocas_por_llamada"],
            canales_modelo=list(respaldo["canales_modelo"]),
            clases=list(respaldo["clases"]),
            combinaciones=combinaciones_desde_canales(respaldo["canales_modelo"]),
            origen="config",
            descripcion=respaldo.get("descripcion", ""),
        )

    # --------------------------------------------------------------- inferencia
    def _enviar_lote(self, modelo: InfoModelo, bloques: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
        canales = list(modelo.canales_modelo)
        canales += [f"presence {c}" for c in modelo.canales_modelo]
        epochs = np.stack(bloques) if len(bloques) > 1 else bloques[0]
        cuerpo = {
            "channel_names": canales,
            "sampling_rate_hz": self.config["senales"]["frecuencia_hz"],
            "signal_unit": "uV",
            # Mantener las muestras float32 del EDF sin redondearlas: así la
            # API recibe los mismos valores que se prepararon para el modelo.
            "epochs": epochs.astype(np.float32).tolist(),
        }
        ultimo_error: Optional[Exception] = None
        for intento in range(self.config["api"]["reintentos"] + 1):
            try:
                respuesta = self.sesion.post(
                    self.url + modelo.endpoint,
                    json=cuerpo,
                    timeout=self.config["api"]["timeout_inferencia_s"],
                )
            except requests.RequestException as error:
                ultimo_error = error
                time.sleep(1.5 * (intento + 1))
                continue
            if respuesta.status_code == 200:
                return respuesta.json()["predictions"]
            if respuesta.status_code in (502, 504) and intento < self.config["api"]["reintentos"]:
                time.sleep(1.5 * (intento + 1))
                continue
            raise self._mensaje_error(respuesta)
        raise ErrorAPI(
            "Se perdió la conexión con el servicio durante la clasificación. Intente de nuevo.",
            str(ultimo_error),
        )

    def clasificar(
        self,
        senales: Dict[str, np.ndarray],
        canales_usados: Sequence[str],
        modelo: InfoModelo,
        n_epocas: int,
        progreso: Callable[[int, int], None] | None = None,
    ) -> ResultadoInferencia:
        """Clasifica todas las épocas del registro, enviando bloques en paralelo."""

        if not any(c in CANALES_EEG for c in canales_usados):
            raise ErrorAPI("Se necesita al menos una señal EEG para clasificar.")
        frecuencia = self.config["senales"]["frecuencia_hz"]
        muestras = int(frecuencia * self.config["senales"]["duracion_epoca_s"])
        largo = modelo.epocas_por_llamada or 200
        if modelo.epocas_por_llamada:
            bloques = bloques_secuencia(n_epocas, largo)
        else:
            bloques = [(i, min(i + largo, n_epocas), i) for i in range(0, n_epocas, largo)]
        if not bloques:
            raise ErrorAPI(f"El registro es demasiado corto: se necesitan al menos {largo} épocas de 30 s.")

        secuencias_por_solicitud = (
            int(self.config["api"].get("secuencias_por_solicitud", 1))
            if modelo.epocas_por_llamada
            else 1
        )
        if not 1 <= secuencias_por_solicitud <= 4:
            raise ErrorAPI("api.secuencias_por_solicitud debe estar entre 1 y 4.")
        lotes = [
            bloques[inicio:inicio + secuencias_por_solicitud]
            for inicio in range(0, len(bloques), secuencias_por_solicitud)
        ]

        estadios = np.full(n_epocas, "?", dtype=object)
        confianzas = np.zeros(n_epocas, dtype=float)
        clases = list(modelo.clases)
        probabilidades = np.zeros((n_epocas, len(clases)), dtype=float)

        def tarea(lote: List[Tuple[int, int, int]]) -> Tuple[List[Tuple[int, int, int]], List[Dict[str, Any]]]:
            datos = [
                construir_bloque(senales, canales_usados, modelo.canales_modelo, inicio, fin, muestras)
                for inicio, fin, _ in lote
            ]
            return lote, self._enviar_lote(modelo, datos)

        comienzo = time.perf_counter()
        hechos = 0
        with ThreadPoolExecutor(max_workers=self.config["api"]["solicitudes_paralelas"]) as ejecutor:
            futuros = [ejecutor.submit(tarea, lote) for lote in lotes]
            try:
                for futuro in as_completed(futuros):
                    lote, predicciones = futuro.result()
                    for p in predicciones:
                        indice_lote = int(p["epoch_index"]) // largo
                        indice_secuencia = int(p["epoch_index"]) % largo
                        if indice_lote >= len(lote):
                            raise ErrorAPI("La API devolvió más predicciones de las secuencias enviadas.")
                        inicio, _fin, primera_nueva = lote[indice_lote]
                        indice = inicio + indice_secuencia
                        if indice < primera_nueva:
                            continue  # época ya clasificada por el bloque anterior
                        estadios[indice] = p["predicted_stage"]
                        confianzas[indice] = float(p["confidence"])
                        for k, clase in enumerate(clases):
                            probabilidades[indice, k] = float(p["probabilities"].get(clase, 0.0))
                        for clase in p["probabilities"]:
                            if clase not in clases:  # p. ej. "M" si el modelo pasa a 6 clases
                                clases.append(clase)
                                probabilidades = np.hstack([probabilidades, np.zeros((n_epocas, 1))])
                                probabilidades[indice, -1] = float(p["probabilities"][clase])
                    hechos += 1
                    if progreso:
                        progreso(hechos, len(lotes))
            except Exception:
                for f in futuros:
                    f.cancel()
                raise

        return ResultadoInferencia(
            estadios=estadios,
            confianzas=confianzas,
            probabilidades=probabilidades,
            clases=clases,
            segundos=time.perf_counter() - comienzo,
        )

    def clasificar_con_recorte_preliminar(
        self,
        senales: Dict[str, np.ndarray],
        canales_usados: Sequence[str],
        modelo: InfoModelo,
        n_epocas: int,
        progreso: Callable[[int, int], None] | None = None,
        margen_epocas: int = 60,
    ) -> ResultadoInferencia:
        """Ubica sueño en una pasada y refina la noche dentro de sus bordes.

        El entrenamiento recortó cada noche a 30 minutos antes del primer
        sueño y después del último. Sin hipnograma, se usa una predicción
        preliminar para estimar esos bordes y se vuelve a clasificar ese tramo.
        """
        largo = modelo.epocas_por_llamada or 15
        bloques_preliminares = len(bloques_secuencia(n_epocas, largo))
        secuencias_por_lote = int(self.config["api"].get("secuencias_por_solicitud", 1))
        pasos_preliminares = (
            bloques_preliminares + secuencias_por_lote - 1
        ) // secuencias_por_lote

        def progreso_preliminar(hechos: int, total: int) -> None:
            if progreso:
                progreso(
                    round(hechos / total * pasos_preliminares),
                    max(2 * pasos_preliminares, 1),
                )

        preliminar = self.clasificar(
            senales,
            canales_usados,
            modelo,
            n_epocas,
            progreso_preliminar if progreso else None,
        )
        indices_sueno = np.flatnonzero(np.isin(preliminar.estadios, ESTADIOS_SUENO))
        if not len(indices_sueno):
            raise ErrorAPI(
                "La pasada preliminar no detectó un periodo de sueño suficiente para "
                "recortar la noche y ejecutar la clasificación final."
            )

        inicio = max(0, int(indices_sueno[0]) - margen_epocas)
        fin = min(n_epocas, int(indices_sueno[-1]) + margen_epocas + 1)
        largo_bloque = int(modelo.epocas_por_llamada or 15)
        muestras_por_epoca = int(
            self.config["senales"]["frecuencia_hz"]
            * self.config["senales"]["duracion_epoca_s"]
        )
        inicio_muestra = inicio * muestras_por_epoca
        fin_muestra = fin * muestras_por_epoca
        senales_recortadas = {
            canal: señal[inicio_muestra:fin_muestra]
            for canal, señal in senales.items()
        }
        epocas_recortadas = fin - inicio
        bloques_refinados = len(bloques_secuencia(epocas_recortadas, largo_bloque))
        pasos_refinados = (
            bloques_refinados + secuencias_por_lote - 1
        ) // secuencias_por_lote
        total_pasos = pasos_preliminares + pasos_refinados

        def progreso_refinado(hechos: int, total: int) -> None:
            if progreso:
                hechos_refinados = round(hechos / total * pasos_refinados)
                progreso(pasos_preliminares + hechos_refinados, total_pasos)

        refinado = self.clasificar(
            senales_recortadas,
            canales_usados,
            modelo,
            epocas_recortadas,
            progreso_refinado if progreso else None,
        )

        # Solo se publican las predicciones del segundo pase, dentro del tramo
        # recortado. Fuera de él se marca "?" para no presentar como finales las
        # predicciones preliminares hechas sobre la noche completa.
        fin_refinado = inicio + bloques_refinados * largo_bloque
        estadios = np.full(n_epocas, "?", dtype=object)
        confianzas = np.full(n_epocas, np.nan, dtype=float)
        probabilidades = np.zeros_like(preliminar.probabilidades)
        if fin_refinado > inicio:
            n_refinadas = fin_refinado - inicio
            estadios[inicio:fin_refinado] = refinado.estadios[:n_refinadas]
            confianzas[inicio:fin_refinado] = refinado.confianzas[:n_refinadas]
            probabilidades[inicio:fin_refinado] = refinado.probabilidades[:n_refinadas]

        if progreso:
            progreso(total_pasos, total_pasos)
        return ResultadoInferencia(
            estadios=estadios,
            confianzas=confianzas,
            probabilidades=probabilidades,
            clases=preliminar.clases,
            segundos=preliminar.segundos + refinado.segundos,
        )
