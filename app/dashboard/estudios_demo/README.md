# Estudios de demostración

Esta carpeta contiene los registros que el tablero ofrece en la opción
**"Estudio de demostración"**. Los archivos `*-PSG.edf` y
`*-Hypnogram.edf` son copias exactas de las grabaciones originales completas
de Sleep Cassette (Sleep-EDFx), para las mismas personas del **conjunto de
prueba** que no se usaron para entrenar el modelo. El tablero lee estos EDF
directamente y aplica su flujo de carga habitual. Los `.npz` existentes se
conservan como copias recortadas y el tablero prefiere EDF cuando están ambos.

Se generan con:

```bash
python scripts/generar_estudios_demo.py \
    --datos data/sleep-edfx/sleep-cassette \
    --particion data/SC-subjects-split.xlsx \
    --sujetos 19 24 40 74 --formato edf
```

El modo `--formato edf` conserva los archivos PSG e hipnograma originales,
sin recorte ni conversión de señal, y crea un `.json` de descripción para
cada PSG. El modo predeterminado `--formato npz` sigue generando la versión
recortada y compacta. Si la carpeta está vacía, el tablero lo indica y permite
subir un registro propio.

Los archivos EDF crudos no se versionan en Git. Para desplegarlos, cópielos
por separado a esta carpeta en el servidor antes de reconstruir la imagen del
tablero.

## Fuente y atribución

Los estudios provienen de [Sleep-EDF Database Expanded v1.0.0 en PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/) y se distribuyen bajo la [Open Data Commons Attribution License v1.0](https://physionet.org/content/sleep-edfx/view-license/1.0.0/). El aviso de atribución de la base de datos es:

> Contains information from [Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) which is made available under the [ODC Attribution License v1.0](https://physionet.org/content/sleep-edfx/view-license/1.0.0/).

Referencias solicitadas por PhysioNet:

- Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. “Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG.” *IEEE Transactions on Biomedical Engineering*. 2000;47(9):1185–1194.
- Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. “PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals.” *Circulation*. 2000;101(23):e215–e220. [https://doi.org/10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215).
