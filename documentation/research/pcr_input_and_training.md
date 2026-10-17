# Entrada y entrenamiento en CNN para predecir pCR

## Pregunta

Diez experimentos con distintas pérdidas, regularización y aumento de datos dan un AUC de validación por paciente de 0,52–0,55, si se promedia sobre todas las épocas. Se busca qué cambios en la entrada (recorte, cortes usados, resolución) y en la capacidad u optimización proponen trabajos parecidos.

## Fuentes

- Fridman et al., *BreastDCEDL*, arXiv:2506.12190 (v3). Es el mismo conjunto de datos que el de este proyecto.
- Braman et al., *Deep learning-based prediction of response to HER2-targeted neoadjuvant chemotherapy from pre-treatment dynamic breast MRI*, arXiv:2001.08570 (2020).
- Khan et al., *Deep Learning Prediction of Pathologic Complete Response in Breast Cancer Using MRI and Other Clinical Data: A Systematic Review*, Tomography 8(6), 2022. DOI 10.3390/tomography8060232.
- Comprobación propia: el mapa medio de realce (`EARLY − PRE`), calculado sobre 400 cortes de entrenamiento por cohorte.

## Hallazgos

### Resultados de BreastDCEDL

- **Modelo.** Un ViT sobre las fases PRE, EARLY y LATE fusionadas como canales RGB.
- **AUC por grupo.**
  - 0,72 en su test completo (177 pacientes);
  - 0,54 en Duke;
  - entre 0,63 y 0,68 según el subtipo.
- **Detalles que el artículo no da.** No indica cuántos cortes usa por paciente, cómo agrega las predicciones de los cortes ni si el modelo está preentrenado.

### Braman et al. (2020)

**Diseño.**

- Usan una CNN 2D entrenada desde cero con unas 100 pacientes, con validación externa de AUC 0,77–0,85.
- La entrada es un **recorte ajustado al tumor**: un volumen de interés delimitado por una radióloga, de 156×156 píxeles con relleno de ceros.
- Cada fase pasa por **una rama convolucional propia**. Cada rama tiene 4 bloques con 32–64 filtros de 3×3 y dropout cada dos bloques.
- La mejor combinación de fases fue la previa al contraste y la tardía.

**Entrenamiento.**

- Optimizador SGD con tasa de aprendizaje 5e-4.
- Aumento de datos con rotación, volteo, traslación y cambio de escala.
- 80 épocas.

### Revisión sistemática de Khan et al. (2022)

La mayoría de los trabajos con AUC alto combinan la imagen con variables clínicas o con resonancias posteriores al tratamiento. En este proyecto ninguna de las dos cosas está permitida.

### Comprobación propia

- **El tumor está centrado.** En las tres cohortes el realce medio se concentra en el centro de la imagen, con un diámetro aproximado de 60–120 píxeles.
- **Reparto del realce.** El cuadrado central de 128×128 ocupa el 25 % del área y contiene el 37–51 % del realce. El resto corresponde a vasos y a otro tejido que también realza.
- **Diferencias entre cohortes.** El campo de visión es distinto en cada cohorte; en I-SPY1, por ejemplo, se ve el contorno de la mama. La red puede distinguir la cohorte por la imagen en sí, sin usar el tumor.
- **Cortes por paciente.** Cada paciente tiene 10 cortes consecutivos. En I-SPY se priorizan los de mayor superficie tumoral, según el enunciado.

## Aplicable a este caso

| Idea | Aplicable | Motivo |
|---|---|---|
| Recorte centrado en el tumor | Sí | El tumor está centrado. El recorte se implementa dentro del modelo, así que la entrada sigue siendo `[3, 256, 256]`. |
| Usar solo los cortes centrales | Sí, solo en entrenamiento | Reduce los cortes con poco tumor. La evaluación por paciente sigue usando todos los cortes. |
| Menor resolución | Sí | Reduce el detalle que se puede memorizar y amplía el campo receptivo relativo. |
| SGD con momento | Sí | Es la configuración de Braman et al. y la habitual en CNN con BatchNorm. |
| Una rama por fase | Posible | Exige otra estructura en el constructor. Se deja para después si el recorte funciona. |
| Variables clínicas como entrada | No | Lo prohíbe el enunciado. |
| Preentrenamiento | No | Lo prohíbe el enunciado. |

## Conclusión

La diferencia más clara con los trabajos que funcionan desde cero está en la **entrada**: usan un recorte ajustado al tumor, mientras que aquí el modelo recibe un campo de 256×256 en el que hay otras estructuras que realzan y diferencias entre cohortes.

La batería de experimentos se organiza así:

1. **Un factor cada vez respecto a `baseline_weighted`:** `crop128`, `crop160`, `pool2`, `central_slices`, `wide`, `deep` y `sgd`.
2. **Combinaciones con el régimen de `augmented_slow`:** `crop128_slow`, `crop128_central_slow`, `pool2_slow`, `wide_slow` y `wide_crop128_slow`.

Los resultados de otros trabajos justifican probar estas ideas, pero no garantizan que funcionen aquí. La decisión se toma con la media del AUC de validación sobre todas las épocas y con el AUC de entrenamiento, no con el máximo por época.
