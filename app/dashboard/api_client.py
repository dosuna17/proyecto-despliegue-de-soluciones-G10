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
