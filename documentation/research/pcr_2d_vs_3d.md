# Volumen 3D frente a cortes 2D para predecir pCR

## Pregunta

¿Hay evidencia de que un modelo sobre el volumen DCE-MRI completo prediga la pCR mejor que una CNN 2D sobre unos diez cortes por paciente? De la respuesta depende cómo se presenta el uso del volumen 3D como línea futura en la diapositiva 5.

## Fuentes

- Xu et al., *Deep Learning Models Based on Pretreatment MRI and Clinicopathological Data to Predict Responses to Neoadjuvant Systemic Therapy in Triple-Negative Breast Cancer*, Cancers 17(6), 2025. DOI 10.3390/cancers17060966.
- Braman et al., *Deep learning-based prediction of response to HER2-targeted neoadjuvant chemotherapy from pre-treatment dynamic breast MRI*, arXiv:2001.08570 (2020).
- Fridman et al., *BreastDCEDL*, arXiv:2506.12190.

## Hallazgos

- **No se ha encontrado una comparación directa** de modelos 2D y 3D entrenados con los mismos datos para predecir pCR.
- **Xu et al. (2025).** Modelo 3D sobre DCE-MRI previa al tratamiento en cáncer triple negativo (282 pacientes en el desarrollo). En la validación externa con I-SPY 2 obtiene un AUC de 0,72 (IC 95 % 0,57–0,84) solo con DCE. El modelo usa el volumen tumoral contorneado y una ResNeXt50, y se estudia un único subtipo.
- **Braman et al. (2020).** Una CNN 2D con recorte del tumor alcanza un AUC de 0,85 en validación externa (28 pacientes, solo HER2+). Un buen resultado no exige 3D si la entrada está centrada en el tumor.
- **BreastDCEDL.** El dataset original contiene volúmenes 3D de 2.070 pacientes; la versión docente extrae unos diez cortes por paciente.

## Conclusión

El volumen 3D aporta contexto entre cortes y permite delimitar el tumor completo, y los trabajos 3D publicados llegan a AUC de alrededor de 0,7 en validación externa. Esos trabajos, sin embargo, también segmentan el tumor, se limitan a un subtipo y usan arquitecturas que no están permitidas aquí. Por eso el uso del 3D se presenta como hipótesis justificada, no como mejora asegurada. Se comprobaría entrenando una CNN 3D y la CNN 2D actual con las mismas pacientes y los mismos folds y comparando el AUC por paciente con *bootstrap* pareado.
