# Estudios de demostración

Esta carpeta contiene los registros que el tablero ofrece en la opción
**"Estudio de demostración"**. Son registros de Sleep Cassette (Sleep-EDFx)
de personas del **conjunto de prueba**, es decir, que no se usaron para
entrenar el modelo.

Se generan con:

```bash
python scripts/generar_estudios_demo.py \
    --datos data/sleep-edfx/sleep-cassette \
    --particion data/SC-subjects-split.xlsx
```

Cada archivo `.npz` contiene las señales (µV, 100 Hz), el hipnograma del
experto y metadatos sin datos identificables (edad, sexo, noche). Pesa entre
15 y 20 MB. Si la carpeta está vacía, el tablero lo indica y permite subir un
registro propio.
