# Diseño de la aplicación web y del panel de resultados

## Pregunta

Qué debe contener la aplicación web del proyecto (sección 9 y 12 del enunciado) y qué información sobre el modelo conviene mostrar, de forma que la demostración sea útil sin exagerar lo que el modelo puede afirmar (AUC en test del modelo final 0,522 [0,442; 0,603]).

## Fuentes

- Enunciado del caso, `documentation/Case-Breast.md`, secciones 9, 10 y 12.
- Mitchell et al., *Model Cards for Model Reporting*, FAT* 2019. arXiv:1810.03993. DOI 10.1145/3287560.3287596.
- Sendak et al., *Presenting machine learning model information to clinical end users with model facts labels*, npj Digital Medicine 3:41, 2020. DOI 10.1038/s41746-020-0253-3.
- Collins et al., *TRIPOD+AI statement*, BMJ 2024;385:e078378. DOI 10.1136/bmj-2023-078378.
- Vickers y Elkin, *Decision curve analysis: a novel method for evaluating prediction models*, Medical Decision Making 26(6):565–574, 2006. DOI 10.1177/0272989X06295361.
- Park et al., *Reliable and Trustworthy Machine Learning for Health Using Dataset Shift Detection*, NeurIPS 2021. arXiv:2110.14019.
- Evaluación de mapas de saliencia (incluido Grad-CAM) con aleatorización de parámetros y etiquetas en RSNA Pneumonia, arXiv:2006.00063 (2020).

## Hallazgos

- **Requisitos del enunciado.**
  - **Entrada:** carga las tres fases de *un corte*, o permite seleccionar un ejemplo.
  - **Visualización:** muestra PRE, EARLY, LATE y el mapa de realce.
  - **Resultado:** muestra la probabilidad, la clase según el umbral, el umbral, la versión o *checksum* del modelo, el dispositivo, el tiempo de respuesta, los parámetros de entrenamiento y las métricas internas.
  - **Errores:** gestiona archivos corruptos, imágenes que no son PNG de 256 × 256 y fases ausentes o duplicadas.
  - **Avisos y seguridad:** muestra un aviso de uso educativo. No acepta rutas del sistema ni ejecuta contenido subido.
  - **Despliegue:** la aplicación está desplegada y accesible por URL. Procesa cinco muestras nuevas sin reiniciarse y registra la probabilidad, el umbral y la clase de cada una.
- **Ficha del modelo.** Mitchell et al. y Sendak et al. proponen documentar en una sola página el uso previsto, los usos fuera de alcance, los datos de entrenamiento, las métricas por subgrupo y las limitaciones. TRIPOD+AI pide además informar sobre la calibración, el umbral y el rendimiento por subgrupo.
- **Utilidad clínica.** Las curvas de decisión comparan el beneficio neto del modelo con las estrategias de tratar a todas o a ninguna. Encajan con la comparación frente a «siempre sin pCR» que ya está en el informe de test.
- **Datos de otra procedencia.** Con imágenes ajenas a la distribución de entrenamiento, los modelos médicos dan predicciones poco razonables sin avisar. Un indicador visible de desplazamiento cambió cómo los usuarios confiaban en las predicciones (Park et al.).
- **Mapas de saliencia.** Grad-CAM y métodos afines localizan peor que los expertos y, en algunos estudios, no superan las pruebas de aleatorización. Con un AUC de 0,52, un mapa de calor sugeriría una explicación que el modelo no respalda.

## Conclusión

- **Núcleo:** la aplicación del enunciado, con predicción de un corte, ficha del modelo y registro de predicciones.
- **Panel de resultados:** figuras ya validadas (experimentos, validación interna y test, este último fijo).
- **Mapas de saliencia:** no se incluyen.
- **Datos nuevos:** se tratan como evaluación externa. No se reentrena desde la aplicación.
