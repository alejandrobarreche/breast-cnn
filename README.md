# Breast-CNN

CNN para la predicción de respuesta patológica completa (**pCR**) en cáncer de mama a partir de imágenes **DCE-MRI**.

Este repositorio contiene el desarrollo de un proyecto académico de Deep Learning cuyo objetivo es estudiar si una red neuronal convolucional puede aprender patrones asociados a la respuesta al tratamiento a partir de tres fases de una resonancia magnética dinámica con contraste.

La red genera una probabilidad de pCR para cada corte. Posteriormente, las predicciones de los cortes pertenecientes a una misma paciente se agregan para obtener una predicción a nivel de paciente.

**Aplicación web:** <https://alejandrobarreche.github.io/breast-cnn/>. El modelo se ejecuta en el navegador; uso educativo, sin validez clínica.

---

## Objetivo

La **respuesta patológica completa (pCR)** indica la ausencia de enfermedad tumoral residual invasiva tras el tratamiento neoadyuvante.

El objetivo del proyecto es desarrollar un pipeline reproducible capaz de estimar la probabilidad de pCR de una paciente a partir de imágenes DCE-MRI adquiridas antes del inicio del tratamiento.

---

## Arquitectura

El proyecto utiliza una **CNN 2D desarrollada específicamente para este trabajo y entrenada desde cero**.

Cada arquitectura se describe en un fichero YAML dentro de `configs/`, de modo que puede leerse y modificarse sin tocar el código que la construye.

La arquitectura definitiva se determinará durante la fase experimental. Una vez fijada, su composición se documentará en la siguiente tabla:

| Capa / bloque | Operación | Kernel | Stride | Padding | Canales | Volumen de salida | Activación | Parámetros |
|---|---|---:|---:|---:|---:|---|---|---:|
| INPUT | — | — | — | — | 3 | 3 × 256 × 256 | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| — | — | — | — | — | — | — | — | — |
| OUTPUT | — | — | — | — | — | — | — | — |

---

## Dataset

Los datos utilizados proceden de **BreastDCEDL**, un dataset estandarizado de DCE-MRI de cáncer de mama preparado para tareas de Deep Learning ([Fridman et al., 2026](https://doi.org/10.1038/s41597-026-06589-6)).

Para este proyecto se utiliza una versión docente derivada de BreastDCEDL, formada por **1.273 pacientes con etiqueta pCR conocida**, procedentes de tres cohortes: **I-SPY1, I-SPY2 y Duke**.

| Cohorte | Train | Test | Total |
|---|---:|---:|---:|
| I-SPY1 | 104 | 35 | 139 |
| I-SPY2 | 784 | 99 | 883 |
| Duke | 209 | 42 | 251 |
| **Total** | **1.097** | **176** | **1.273** |

Los identificadores `ACRIN-6698-*` pertenecen a pacientes de la cohorte **I-SPY2** y no representan una cohorte adicional.

El conjunto contiene **12.703 cortes**: 10.945 en `train` y 1.758 en `test`.

Cada corte está representado por tres imágenes DCE-MRI de `256 × 256` píxeles correspondientes a diferentes momentos de la adquisición:

- **PRE**: antes de la administración de contraste.
- **EARLY**: fase temprana post-contraste.
- **LATE**: fase tardía post-contraste.

Cada muestra de entrada está formada por las tres fases del mismo corte:

```text
PRE + EARLY + LATE -> [3, 256, 256]
```

La etiqueta **pCR** pertenece a la paciente, por lo que todos los cortes de una misma paciente comparten la misma etiqueta.

### Organización de los datos

Las imágenes se organizan por paciente dentro de los conjuntos `train` y `test`:

```text
data/
├── train/
│   ├── ACRIN-6698-103939/
│   │   ├── ACRIN-6698-103939_z006_PRE.png
│   │   ├── ACRIN-6698-103939_z006_EARLY.png
│   │   ├── ACRIN-6698-103939_z006_LATE.png
│   │   ├── ACRIN-6698-103939_z007_PRE.png
│   │   └── ...
│   └── ...
└── test/
    ├── <patient_id>/
    └── ...
```

El identificador `zXXX` indica la posición axial del corte, mientras que `PRE`, `EARLY` y `LATE` indican la fase temporal.

### Ejemplo

Paciente `ACRIN-6698-103939`, perteneciente a **I-SPY2**, corte `z006`:

| PRE | EARLY | LATE |
|---|---|---|
| ![PRE](documentation/figures/ACRIN-6698-103939_z006_PRE.png) | ![EARLY](documentation/figures/ACRIN-6698-103939_z006_EARLY.png) | ![LATE](documentation/figures/ACRIN-6698-103939_z006_LATE.png) |

### Metadatos

La información asociada a las muestras y pacientes se encuentra en:

```text
metadata/
├── samples.csv
├── patients.csv
├── excluded_patients.csv
└── statistics.json
```

`samples.csv` contiene una fila por corte y las rutas de sus tres fases.

`patients.csv` contiene una fila por paciente con la cohorte de origen, la etiqueta pCR y metainformación clínica y de adquisición.

---

## Predicción por paciente

La CNN genera inicialmente una probabilidad de pCR para cada corte:

```text
corte 1 -> 0.21
corte 2 -> 0.27
corte 3 -> 0.35
...
corte n -> 0.29
```

Como la pCR es una propiedad de la paciente, las probabilidades de todos sus cortes se agregan mediante la media:

```text
                        p1 + p2 + ... + pn
P(pCR | paciente) = -------------------------
                                 n
```

La media es el punto de partida. Si se prueba otra agregación (por ejemplo, ponderar o descartar cortes), la elección se hará con la validación interna de `train`, nunca con `test`.

La evaluación final del modelo se realiza a nivel de paciente.

---

## Uso

Requisitos: Conda, Make y Poetry `>=2.3,<3.0`.

```bash
conda env create -f environment.yml
conda activate ml-cnn
make install-mac        # o install-cpu / install-windows-cuda / install-linux-cuda / install-linux-rocm
```

Datos y análisis exploratorio:

```bash
python scripts/download_dataset.py                  # descarga las imágenes en data/
python scripts/audit_dataset.py --check-images      # comprueba metadatos e imágenes
python scripts/check_data_pipeline.py               # recorre un batch del Dataset
python scripts/explore_train.py                     # EDA de train -> artifacts/
```

El notebook `notebooks/01_eda.ipynb` presenta el análisis exploratorio.

Arquitectura:

```bash
python scripts/describe_model.py configs/model/baseline.yaml   # tabla de capas y parámetros
```

Entrenamiento (validación cruzada con los 5 `fold` de `train`; un experimento por fichero de `configs/experiment/`):

```bash
python scripts/train.py +experiment=baseline_unweighted
python scripts/train.py +experiment=baseline_weighted
python scripts/train.py +experiment=augmented     # aumento geométrico conjunto
python scripts/train.py +experiment=regularized   # tasa de aprendizaje 3e-4, decaimiento 1e-2
python scripts/train.py +experiment=separable     # convoluciones separables en profundidad
python scripts/train.py +experiment=augmented_slow  # aumento geométrico y tasa de aprendizaje 1e-4
python scripts/train.py +experiment=patient_mean    # pérdida por paciente: media de logits de sus cortes
python scripts/train.py +experiment=patient_max     # pérdida por paciente: máximo de logits de sus cortes
python scripts/train.py +experiment=label_smoothing # objetivos 0,05 / 0,95
python scripts/train.py +experiment=patient_mean_large_batch  # patient_mean con 12 pacientes por lote
python scripts/train.py +experiment=crop128         # región central de 128 x 128 (recorte dentro del modelo)
python scripts/train.py +experiment=central_slices  # entrena con los 4 cortes centrales de cada paciente
python scripts/train.py +experiment=sgd             # SGD con momento 0,9 en lugar de AdamW
python scripts/train.py +experiment=wide_crop128_cosine_ema  # tasa de aprendizaje coseno y media móvil exponencial de los pesos, 40 épocas
python scripts/train.py +experiment=crop128_cosine_ema         # lo anterior con 8-128 canales (≈296.000 parámetros)
python scripts/train.py +experiment=narrow_crop128_cosine_ema  # lo anterior con 4-64 canales (≈74.000 parámetros)
python scripts/train.py +experiment=crop128_cosine_ema_decay   # crop128_cosine_ema con decaimiento de pesos 0,5
```

Varios experimentos seguidos con un solo comando (si uno falla, se informa y continúa el siguiente). Por ejemplo, la batería de entrada, capacidad y optimización (justificación en `documentation/research/pcr_input_and_training.md`):

```bash
python scripts/train.py --experiments crop128 crop160 pool2 central_slices wide deep sgd crop128_slow crop128_central_slow pool2_slow wide_slow wide_crop128_slow
```

Cada época imprime la pérdida y el AUC por paciente de train y validación. Si un entrenamiento se interrumpe, se continúa desde la última época completada:

```bash
python scripts/train.py --resume runs/<id>
```

Calibración y umbral del ensemble final. El run anidado entrena un modelo por cada par de folds con el protocolo final. Cada paciente recibe la media de los cuatro modelos que no vieron su fold, que es el mismo tipo de puntuación que da el ensemble en test. Sobre esa puntuación se ajustan el mapa de Platt y el umbral (`analysis.nested_run` en `configs/config.yaml`):

```bash
python scripts/train.py +experiment=wide_crop128_nested
python scripts/fit_calibration.py                 # por paciente (evaluación en test)
python scripts/fit_calibration.py --unit slice    # por corte (aplicación web)
python scripts/evaluate_test.py --correction      # reevalúa test con el umbral corregido
```

Tests y estilo:

```bash
python -m pytest
python -m ruff check .
```

Aplicación web (`web/`, Vite + TypeScript; el modelo se ejecuta en el navegador con ONNX Runtime Web):

```bash
python scripts/export_onnx.py       # ensemble calibrado -> artifacts/models/ensemble.onnx
python scripts/export_web.py        # modelo, manifiesto, ejemplos y resultados de los runs para web/
cd web && npm install && npm test   # incluye la paridad con las salidas de Python
npm run dev                         # http://localhost:5173
```

Se publica en <https://alejandrobarreche.github.io/breast-cnn/> con `.github/workflows/pages.yml` en cada push a `main` que toque `web/`. El modelo no está en Git: el workflow lo descarga de la Release `model-v2` y comprueba su SHA-256 contra `web/src/manifest.json`.

---

## Licencia y atribución

Las imágenes derivan de BreastDCEDL (Fridman et al., *Scientific Data*, 2026, [doi:10.1038/s41597-026-06589-6](https://doi.org/10.1038/s41597-026-06589-6)), construido a partir de colecciones de TCIA (I-SPY1, I-SPY2 y Duke), y se distribuyen bajo [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). El código propio se rige por [LICENSE](LICENSE).

El [enunciado original](documentation/Case-Breast.md) describe los requisitos del caso.

---

## Aviso

Proyecto académico y de investigación.

Los modelos y resultados obtenidos son experimentales y no tienen validez clínica.
