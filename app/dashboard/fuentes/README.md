# Fuentes del informe PDF

El informe PDF usa **Century Gothic** si está instalada en el equipo donde
corre el tablero (viene con Microsoft Office en Windows y macOS).

En el servidor Linux (Docker) no está disponible y su licencia no permite
incluirla en el repositorio, así que se usa **URW Gothic**, una versión libre
del mismo diseño geométrico (ITC Avant Garde Gothic, en el que se basa
Century Gothic). Visualmente son casi idénticas.

- `URWGothic-Book.otf`, `URWGothic-Demi.otf`: de
  [ArtifexSoftware/urw-base35-fonts](https://github.com/ArtifexSoftware/urw-base35-fonts)
  (versión 20200910).
- Licencia: AGPL v3 con una excepción que permite incluir las fuentes en
  documentos PDF sin que la licencia se extienda al documento (ver `LICENSE`).

El archivo Excel declara la fuente Century Gothic; Excel la muestra con la
fuente instalada en el equipo de quien lo abre.
