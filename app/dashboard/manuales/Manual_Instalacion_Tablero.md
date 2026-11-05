# Manual de Instalación — 🌙 Sleep Staging Assistant

**Dirigido a:** cualquier integrante del equipo (o evaluador) que necesite levantar el tablero y la API en su propia máquina.

**Alcance:** instalación local mediante Docker Compose (ruta recomendada) y alternativa de desarrollo sin Docker.

---

## 1. Requisitos previos

- **Sistema operativo:** Windows 10/11 (64 bits), macOS o Linux. Este manual documenta el procedimiento verificado en Windows con WSL2; en macOS/Linux el paso de Docker Desktop es más directo (no requiere WSL2).
- **Docker Desktop** instalado y corriendo.
- **Git** instalado, con acceso de lectura al repositorio del proyecto (`proyecto-desarrollo-de-soluciones-G5`).
- **El dataset Sleep-EDFx** descargado localmente (no se distribuye dentro del repositorio ni de las imágenes Docker por su tamaño; ver sección 4).
- Conexión a internet para que la API pueda descargar los artefactos de los modelos desde el servidor de MLflow en AWS.

No es necesario tener Python instalado en el equipo para la ruta con Docker, todo el entorno (dependencias, librerías de señales, modelos) vive dentro de los contenedores.

---

## 2. Instalar Docker Desktop (Windows)

Si Docker ya está instalado y `docker --version` funciona en su terminal, puede saltar a la sección 3.

1. **Instalar WSL2.** Abra PowerShell como administrador y ejecute:
   ```powershell
   wsl --install
   ```
   Reinicie el equipo cuando se lo solicite.

2. **Instalar Docker Desktop.** Descargue el instalador desde el sitio oficial de Docker y elija la build correspondiente a su arquitectura (verifique con `$env:PROCESSOR_ARCHITECTURE` en PowerShell — en la mayoría de los equipos será AMD64).

3. **Verificar la integración con WSL2.** Al abrir Docker Desktop por primera vez, confirme en Settings → Resources → WSL Integration que la distribución de WSL2 instalada está habilitada.

4. **Cerrar y reabrir completamente su editor/terminal.** Si acaba de instalar Docker, una nueva pestaña de terminal dentro del mismo editor puede no reconocer el comando `docker` todavía, cierre el editor por completo (no solo la pestaña) y vuelva a abrirlo para que el PATH del sistema se actualice.

5. **Confirmar la instalación:**
   ```powershell
   docker --version
   docker compose version
   ```
   Ambos comandos deben responder con un número de versión, sin errores de conexión.

---

## 3. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd proyecto-desarrollo-de-soluciones-G5
```

Sitúese en la raíz del repositorio (donde está el archivo `docker-compose.yml`) para todos los comandos de las siguientes secciones.

---

## 4. Configurar el acceso al dataset

El dataset Sleep-EDFx **no viaja dentro de la imagen Docker**, se monta desde una carpeta de su equipo hacia el contenedor en tiempo de ejecución. Antes de levantar los servicios, defina la variable de entorno que le indica a Docker Compose dónde está esa carpeta en su máquina:

**PowerShell (Windows):**
```powershell
$env:SLEEP_STAGING_DATA_ROOT_HOST = "C:\ruta\completa\a\sleep-edf-database-expanded-1.0.0"
```

**bash/zsh (macOS/Linux):**
```bash
export SLEEP_STAGING_DATA_ROOT_HOST="/ruta/completa/a/sleep-edf-database-expanded-1.0.0"
```

La carpeta debe contener la estructura estándar de Sleep-EDFx (subcarpetas `sleep-cassette/` y `sleep-telemetry/` con los pares `*-PSG.edf` / `*-Hypnogram.edf`).

> **Importante:** en PowerShell, las variables de entorno definidas con `$env:...` solo existen en la terminal donde las escribió, si abre una terminal nueva (o su editor se reinicia), debe volver a definirla antes de ejecutar cualquier comando `docker compose`. Si ve errores como `invalid spec: :/data/sleep-edfx: empty section between colons` o `Falta configurar SLEEP_STAGING_DATA_ROOT`, esta es la causa más común: vuelva a ejecutar el `$env:...` en esa terminal.

---

## 5. Levantar los servicios con Docker Compose

Con la variable de la sección anterior definida en su terminal actual:

```bash
docker compose up --build
```

Esto construye las imágenes de los dos servicios definidos en `docker-compose.yml` y los levanta:

| Servicio | Puerto | Función |
|---|---|---|
| `api` | `8000` | API REST (FastAPI) que expone `/health`, `/predict-tinysleepnet` (TinySleepNet) y `/predict-xgboost` |
| `dashboard` | `8501` | Tablero (Streamlit) que consume la API |

El servicio `dashboard` espera a que `api` reporte estar saludable (`condition: service_healthy`) antes de arrancar, así que es normal ver primero los logs de `api` y luego los de `dashboard`.

Una vez ambos contenedores estén arriba, abra en su navegador:

```
http://localhost:8501
```

Para apagar los servicios sin perder las imágenes ya construidas:

```bash
docker compose down
```

### 5.1. Notas sobre la comunicación entre contenedores

El tablero se conecta a la API usando el nombre del servicio dentro de la red interna de Docker Compose (`http://api:8000`), **no** `http://localhost:8000`, dentro de un contenedor, `localhost` se refiere al propio contenedor, no a sus vecinos. Esto ya está configurado en `docker-compose.yml`; se documenta aquí únicamente para quien necesite depurar un error de conexión entre `dashboard` y `api`.

### 5.2. Reconstruir después de cambios de código

- Si solo cambiaron variables de entorno en `docker-compose.yml`, basta con:
  ```bash
  docker compose up -d --force-recreate
  ```
- Si cambió el código fuente (`app/`, `src/`), reconstruya la imagen:
  ```bash
  docker compose build --no-cache dashboard
  docker compose up -d
  ```
  Este paso puede tardar varios minutos (instalación de dependencias de Python). Para desarrollo activo, el `docker-compose.yml` del repositorio monta la carpeta `app/` como volumen (`./app:/app/app`), por lo que los cambios en el código del tablero se reflejan de inmediato sin reconstruir la imagen, sí es necesario reconstruir si cambian `requirements.txt`, `src/` o los modelos en `modelos/`.

---

## 6. Verificar que la instalación quedó correcta

1. **Verificar la API directamente:**
   ```bash
   curl http://localhost:8000/health
   ```
   Debe responder con estado saludable.

2. **Verificar el tablero:** abra `http://localhost:8501`, seleccione un estudio (por ejemplo `SC4001E0`), active al menos las señales `EEG Fpz-Cz` y `EEG Pz-Oz`, y haga clic en "Analizar estudio". Debe ver el hipnograma y las predicciones sin mensajes de error de conexión.

3. **Verificar los logs si algo falla:**
   ```bash
   docker compose logs api --tail=50
   docker compose logs dashboard --tail=50
   ```

---

## 7. Alternativa: entorno local sin Docker (modo desarrollo)

Para desarrollo rápido de la API o el tablero sin reconstruir imágenes, es posible correr ambos servicios directamente con Python, siempre que tenga instaladas las dependencias del proyecto (`pip install -r requirements.txt`).

**Terminal 1 — API:**
```bash
$env:MODEL_PATH = "C:\ruta\al\checkpoint\tinysleepnet_final_eeg_eog.pth"   # opcional: usar un checkpoint local en vez de descargarlo de MLflow
uvicorn app.api.main:app --reload --port 8000
```

**Terminal 2 — Dashboard:**
```bash
$env:SLEEP_STAGING_DATA_ROOT = "C:\ruta\completa\a\sleep-edf-database-expanded-1.0.0"
$env:SLEEP_STAGING_API_URL = "http://localhost:8000"
streamlit run app/dashboard/app.py
```

En este modo, como ambos procesos corren directamente en el sistema operativo (no dentro de contenedores separados), `http://localhost:8000` sí es la dirección correcta para `SLEEP_STAGING_API_URL` a diferencia del modo Docker (sección 5.1), aquí no hay red interna que resolver por nombre de servicio.

> Recuerde que estas variables de entorno, igual que en Docker, solo persisten en la terminal donde las definió.

---

## 8. Solución de problemas comunes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `docker: command not found` en una terminal nueva | El editor no recargó el PATH tras instalar Docker Desktop | Cierre y reabra el editor completo (no solo la pestaña) |
| `ConnectionRefusedError` al llamar a la API desde el tablero (modo Docker) | `SLEEP_STAGING_API_URL` apunta a `localhost` en vez de `api` | Verifique la variable `SLEEP_STAGING_API_URL: http://api:8000` en `docker-compose.yml` |
| `RuntimeError: Falta configurar SLEEP_STAGING_DATA_ROOT` | La variable de entorno no está definida en la terminal actual | Redefina `SLEEP_STAGING_DATA_ROOT_HOST` (Docker) o `SLEEP_STAGING_DATA_ROOT` (local) antes de levantar el servicio |
| `invalid spec: :/data/sleep-edfx: empty section between colons` | Igual causa anterior, Compose necesita la variable incluso para comandos como `exec` o `logs` | Redefina la variable de host antes de cualquier comando `docker compose` |
| El tablero no refleja cambios de código recientes | La imagen se construyó antes de los cambios y no hay volumen de desarrollo montado | `docker compose build --no-cache dashboard && docker compose up -d`, o confirme que `./app:/app/app` está en `volumes` |
| La predicción tarda mucho o falla intermitentemente contra el modelo avanzado | El servidor de MLflow en AWS Academy usa credenciales temporales que rotan, es una limitación de la infraestructura académica, no del código local | Reintente en unos minutos; si persiste, contacte al responsable de mantener la instancia EC2 activa |

---

*Nota de versión: procedimiento verificado en Windows 11 con WSL2 y Docker Desktop, levantando ambos servicios (`api`, `dashboard`) con `docker compose up --build` y confirmando predicciones reales de los dos modelos disponibles (XGBoost y TinySleepNet) sobre el estudio SC4001E0.*
