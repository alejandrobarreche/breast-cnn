# Técnicas de Géron (caps. 11 y 14) aplicables al modelo base

**Pregunta.** ¿Qué técnicas del libro pueden reducir el sobreajuste del modelo base (AUC por paciente ≈ 0,57 en validación interna, pérdida de validación mínima en las épocas 1–4) sin violar el enunciado?

**Fuentes.** A. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*, 2.ª ed., O'Reilly, 2019, caps. 11 y 14 (`documentation/papers/`). Referencias primarias citadas en el libro: Chollet, «Xception», arXiv:1610.02357 (2016); Hu et al., «Squeeze-and-Excitation Networks», CVPR 2018; He et al., «Deep Residual Learning», arXiv:1512.03385 (2015); Smith, arXiv:1803.09820 (2018); Gal y Ghahramani (2016).

## Hallazgos y aplicabilidad

| Técnica (sección) | Lo que afirma el libro | Decisión aquí |
|---|---|---|
| Aumento de datos (cap. 14, «Data Augmentation») | Variantes realistas (desplazar, rotar, escalar, voltear) regularizan; el ruido blanco no ayuda | **Experimento `augmented`**. Sin volteo vertical (pondría la pared torácica arriba: no es realista) ni cambios de contraste por canal (prohibidos por el enunciado) |
| Regularización ℓ2 y planificación de la tasa de aprendizaje (cap. 11) | El decaimiento de pesos limita el tamaño de los pesos; *1cycle* suele converger antes y mejor | **Experimento `regularized`** (tasa 3e-4, decaimiento 1e-2). *1cycle* queda pendiente: añadiría un segundo factor |
| Convolución separable en profundidad (cap. 14, «Xception») | Menos parámetros y cómputo y, en general, igual o mejor rendimiento; evitarla tras capas con pocos canales | **Experimento `separable`**: 37.793 parámetros frente a 295.673; la etapa 1 se mantiene estándar |
| Conexiones residuales (cap. 14, «ResNet») | Permiten entrenar redes muy profundas | No: nuestra red ya alcanza pérdida de entrenamiento ≈ 0; el problema es la generalización, no la optimización. Además el enunciado prohíbe ResNet |
| Bloques SE (cap. 14, «SENet») | Recalibran canales; mejoran ResNet/Inception en ImageNet | Pospuesto: añade parámetros y no ataca el sobreajuste |
| Preentrenamiento en tarea auxiliar (cap. 11) | Reutilizar capas entrenadas en otra tarea cuando hay pocas etiquetas | No sin consultar al profesor: el enunciado prohíbe modelos preentrenados y *transfer learning* |
| MC Dropout (cap. 11) | Promediar predicciones con *dropout* activo estima la incertidumbre | Aplicable a la aplicación web (el enunciado exige mostrar incertidumbre), no al entrenamiento |

## Conclusión

Tres experimentos, cada uno con un único factor cambiado respecto a `baseline_weighted`: `augmented`, `regularized` y `separable`. Se decide con la validación interna de 5 folds; una diferencia menor que la variación entre folds (≈ 0,02–0,04 de AUC) no se considera concluyente.
