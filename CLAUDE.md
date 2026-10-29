# CLAUDE.md — Breast-CNN

Reglas del proyecto y forma de trabajo de Claude.

## 1. Proyecto

CNN 2D propia, entrenada desde cero en PyTorch, que estima la probabilidad de respuesta patológica completa (`pCR`) a partir de DCE-MRI de mama previa al tratamiento.

- Enunciado del profesor: `documentation/Case-Breast.md`
- Visión general y uso: `README.md`

Principios:

- **Simpleza y organización.** El cambio mínimo que resuelve la petición. Sin abstracciones, scripts ni configuraciones que no se usen ya. Antes de añadir algo, comprobar si el enunciado lo exige.
- **Avance paso a paso.** No adelantar fases (baterías de experimentos, paneles, despliegue…) que el usuario no haya pedido.

## 2. Restricciones del enunciado

- CNN 2D propia. Sin transfer learning, sin pesos preentrenados, sin ResNet, VGG, EfficientNet ni DenseNet.
- Salida: un logit por corte. Pérdida `BCEWithLogitsLoss`; se comparan la versión normal y la ponderada con `pos_weight = N0 / N1`, calculado solo con el subconjunto de entrenamiento.
- Entrada `float32 [3, 256, 256]`, canales en orden `PRE, EARLY, LATE`, escala `value / 255`. Son fases temporales, no RGB: sin normalización ImageNet ni aumentos de color.
- Las transformaciones geométricas son idénticas en las tres fases; las aleatorias, solo en entrenamiento.
- La unidad estadística es la **paciente**: nunca se reparten cortes de una paciente entre entrenamiento y validación. La validación interna usa los `fold` de `samples.csv`.
- `test` no interviene en ninguna decisión (arquitectura, hiperparámetros, umbral, agregación…); solo se usa en la evaluación final.
- La entrada del modelo es solo la imagen; los datos clínicos se usan para análisis, no como entrada.

## 3. Datos

- `metadata/` es inmutable. `samples.csv` es el índice: las rutas se leen de `path_pre`, `path_early` y `path_late`; el prefijo `dataset/` se traduce a `data/` mediante `configs/config.yaml`.
- `patients.csv` se une por `samples.patient_id -> patients.pid`.
- Sin `fillna(0)` global. Las pacientes de `excluded_patients.csv` no participan.
- Una única implementación de carga y preprocesado: `src/breast_cnn/data/`.

## 4. Código

- Código e identificadores en inglés; documentación en español.
- La lógica reutilizable está en `src/breast_cnn/`; `scripts/` solo contiene puntos de entrada; los notebooks importan el paquete.
- Parámetros y rutas en `configs/`. **Las arquitecturas se describen en YAML** dentro de `configs/` y el código las construye; ninguna arquitectura concreta se codifica en un `.py`.
- PyTorch explícito, sin Lightning.
- La aplicación web está en `web/` (Vite + TypeScript, ONNX Runtime Web en WebAssembly). Su modelo, manifiesto y ejemplos se generan con `scripts/export_onnx.py` y `scripts/export_web.py`; su preprocesado replica `load_dce_triplet` y `npm test` lo comprueba frente a las salidas de Python.
- Entorno Conda `ml-cnn` (`environment.yml`); dependencias en `pyproject.toml` (Poetry).
- Verificación:

  ```bash
  python -m pytest
  python -m ruff check .
  python -m ruff format --check .
  ```

- No se afirma que algo funciona sin haberlo ejecutado. No se inventan métricas.

## 5. Forma de responder

- Respuestas cortas y directas: primero la conclusión, después la justificación en pocas líneas. Más detalle solo si se pide.
- Una recomendación concreta en lugar de una lista de opciones equivalentes.
- Registro científico, como un libro de texto académico. Sin expresiones de marketing ni jerga coloquial («del modelo a los insights», «la magia de…»). Cada término técnico se usa con su significado preciso: por ejemplo, «los pesos se ajustan minimizando la pérdida mediante descenso de gradiente estocástico», no «deducir los pesos mediante SGD».
- Títulos representativos y autoexplicativos, que nombran el contenido: «Cálculo del peso de la clase positiva», «Distribución de pacientes por cohorte». Nunca títulos de transición del tipo «De X a Y».
- Se distingue siempre lo comprobado (ejecutado, medido) de lo supuesto o pendiente. Si algo no se sabe, se dice.
- Español en la conversación y la documentación; inglés en el código.

## 6. Discusión técnica

Aplica a arquitecturas, hiperparámetros, preprocesado, evaluación y cualquier decisión metodológica.

- Objetividad: no se da la razón por defecto. Si una propuesta tiene un problema, se señala con el motivo técnico, aunque contradiga lo que prefiere el usuario.
- Cada valoración se apoya en algo verificable: el enunciado, resultados de validación, propiedades conocidas de la operación o una referencia bibliográfica.
- Ante una propuesta se indica qué pregunta responde, qué coste tiene (parámetros, tiempo de entrenamiento, complejidad) y cómo se comprobaría si funciona.
- Se prefiere el cambio más simple que responde a la pregunta, modificando pocos factores a la vez para poder atribuir el efecto.
- Si los resultados no permiten concluir (un solo fold, diferencias pequeñas), se dice explícitamente en lugar de forzar una conclusión.

## 7. Conceptos médicos

El usuario no tiene formación médica. Los conceptos clínicos (pCR, DCE-MRI, realce, *washout*, subtipos HR/HER2, tratamiento neoadyuvante…) se explican para un lector no especialista: qué es, por qué importa para el modelo y un ejemplo cuando ayude. Sin perder precisión ni exagerar lo que el modelo puede afirmar clínicamente.

## 8. Gráficas

- Fuente única de estilo: `src/breast_cnn/visualization/style.py` (fondo blanco, DejaVu Sans, título de 16 pt alineado a la izquierda con subtítulo, sin bordes, rejilla suave).
- Colores de clase fijos: no pCR `#23656E` (verde petróleo), pCR `#DC7867` (coral). No se usan con otro significado.
- Fases DCE en escala de grises `[0, 1]`; mapas de realce (`EARLY − PRE`) con la escala divergente `ENHANCEMENT_CMAP`, centrada en blanco.
- Etiquetas en español; punto para miles, coma para decimales, porcentajes con un decimal (`format_count`, `format_percent`).
- El título nombra el contenido según la sección 5. El subtítulo indica split, unidad (`corte` o `paciente`), clase positiva y umbral cuando corresponda.
- Matplotlib. Las funciones de dibujo viven en `src/breast_cnn/visualization/` y separan el cálculo de los datos del dibujo.
- Salida en PNG con nombres en `snake_case` que incluyen split y unidad, por ejemplo `roc_validation_patient.png`. Las figuras de análisis van a `artifacts/figures/`; las de un entrenamiento, a la carpeta de su run (sección 12).
- La aplicación web usa el tema «cristal suave» de `documentation/web-design.md` (capas 5 a 7): paneles translúcidos sobre fondo claro, imágenes médicas siempre sobre el visor oscuro, solo Inter, acción en negro `#15191D` y sin color de acento. Sus tokens están en `web/src/style.css`; los colores de clase y la escala de realce son los de `style.py`.
- Para cambiar el estilo se modifican `style.py` y esta sección; nunca se definen estilos sueltos dentro de un script o notebook.

## 9. Investigación

Cuando se abre una pregunta que requiere bibliografía (por ejemplo, qué arquitecturas se usan para predecir pCR):

1. Se acuerda la pregunta concreta y qué decisión del proyecto depende de ella.
2. Se consultan primero los PDFs de `documentation/papers/` y después fuentes web con DOI o publicación identificable. Cada afirmación se cita.
3. La respuesta en el chat contiene: hallazgos citados; qué es aplicable a este caso y qué no (tamaño del dataset, 2D frente a 3D, preentrenamiento prohibido aquí); y de 1 a 3 propuestas con su coste.
4. Se guarda un resumen breve en `documentation/research/<tema>.md` con la pregunta, las fuentes, los hallazgos y la conclusión.

Un resultado publicado justifica probar una idea, no garantiza que funcione aquí; la decisión se toma con validación interna.

## 10. Delegación a modelos inferiores

Solo se delegan a subagentes de modelos más baratos tareas mecánicas y totalmente especificadas:

| Se puede delegar | No se delega nunca |
|---|---|
| Buscar ficheros, funciones o usos en el repositorio | Decisiones de arquitectura, pérdida, evaluación o metodología |
| Leer y extraer datos concretos de un PDF o una web (tabla, hiperparámetros, cifra) | Interpretación de resultados o métricas |
| Cambios repetitivos con instrucciones exactas (renombrar, ajustar formato) | Código de datos, preprocesado, modelo, entrenamiento o evaluación |
| | Explicaciones médicas |
| | Cualquier respuesta final al usuario |

La salida de un subagente se revisa siempre antes de usarla. Si una tarea no pertenece claramente a la columna izquierda, no se delega.

## 11. Ejecución de comandos

**En el Mac, Claude ejecuta:** lectura e inspección, `pytest`, `ruff`, scripts cortos (aproximadamente menos de 2 minutos) y comprobaciones de git que no modifican nada (`status`, `diff`, `log`).

**El usuario ejecuta**, con el comando exacto que le da Claude:

- instalaciones y entornos (`conda`, `poetry`, `make install-*`);
- descargas del dataset;
- entrenamientos, incluidos los cortos de prueba;
- notebooks;
- `git commit` y `git push`, salvo que pida explícitamente a Claude que haga el commit.

Si hay dudas sobre quién ejecuta algo, se pregunta.

## 12. Mac y Windows

Hay dos máquinas con funciones separadas. Claude identifica en cuál está por la plataforma del entorno (`darwin` = Mac, `win32` = Windows).

| | Mac (`darwin`) | Windows con CUDA (`win32`) |
|---|---|---|
| Función | Desarrollo: código, configs, tests, análisis, documentación | Entrenamiento en GPU |
| Escribe en | Todo el repositorio **excepto** `runs/` | **Solo** `runs/` |
| Git | Commit y push de código | Commit y push solo de `runs/<id>/` |

### Flujo de trabajo

1. **Mac:** se implementa el cambio y se define el experimento en `configs/`. Tests y ruff en verde. Commit y push.
2. **Windows:** `git pull --rebase`. Se ejecuta el entrenamiento indicado. Se añade solo la carpeta del run (`git add runs/<id>`, nunca `git add .`), commit, `git pull --rebase` y `git push`.
3. **Mac:** `git pull --rebase` y análisis de los resultados.

### Reglas contra sobrescrituras

- Windows no edita código, configs ni documentación. Si algo falla y requiere cambiar código, se detiene, se describe el error y se corrige en el Mac.
- El Mac no edita nada dentro de `runs/`.
- `git pull --rebase` antes de empezar a trabajar y antes de cada push, en ambas máquinas.
- Cada run tiene una carpeta propia e irrepetible: `runs/<AAAAMMDD-HHMM>_<experimento>/`. Un run terminado (con `metrics.json`) nunca se reescribe; uno interrumpido se continúa con `--resume`, solo con el mismo commit.
- `.gitattributes` fija finales de línea LF para que Windows no marque ficheros como modificados; `metadata/` queda excluido porque es inmutable.

### Contenido de un run

```text
runs/<AAAAMMDD-HHMM>_<experimento>/
├── config.yaml                 configuración resuelta usada (datos, entrenamiento, arquitectura)
├── environment.json            commit, dispositivo, GPU, versiones, tiempos, memoria máxima
├── history.csv                 pérdida y AUC por paciente de train y validación, por fold y época
├── metrics.json                métricas de la mejor época por fold (corte y paciente) y media ± desviación
├── predictions_validation.csv  probabilidad por corte de la validación interna, para el análisis en el Mac
├── figures/                    curvas de aprendizaje
├── model_fold<k>.pt            pesos de la mejor época de cada fold (fuera de Git)
└── checkpoint_fold<k>.pt       estado completo tras la última época, para --resume (fuera de Git)
```

## 13. Git

- Fuera del historial: `data/`, pesos y checkpoints (`*.pt`), bases de MLflow y cachés. De cada entrenamiento solo se versionan los resultados ligeros de `runs/<id>/`; los pesos se quedan en Windows y solo el modelo final se publica aparte (por ejemplo, como GitHub Release) para la aplicación web.
- Conventional Commits. Commits solo cuando el usuario lo pida.

## 14. Documentación

- `README.md`: presentación y uso. `CLAUDE.md`: reglas y forma de trabajo. `documentation/`: enunciado, papers y resúmenes de investigación.
- No se crean más documentos (planes, estados, registros de decisiones, atlas…) sin petición explícita.
- Cuando cambia una regla, se edita su sección y queda solo la versión vigente, sin historial («antes se hacía…»).
