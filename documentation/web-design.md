# Diseño de la aplicación web

Definición del rediseño de `web/`, decidida por capas, de lo estructural a lo visual. Cada capa se cierra antes de pasar a la siguiente; la implementación empieza cuando están cerradas las capas 1 a 7.

| # | Capa | Estado |
|---|---|---|
| 1 | Propósito y audiencia | Cerrada |
| 2 | Estructura | Cerrada |
| 3 | Niveles de lectura | Cerrada |
| 4 | Contenido | Cerrada |
| 5 | Sistema visual | Cerrada |
| 6 | Guía de gráficas | Cerrada |
| 7 | Componentes | Cerrada |
| 8 | Implementación | Hecha |

## 1. Propósito y audiencia

### Usos

La web cubre tres usos, por orden de prioridad:

1. **Herramienta de análisis.** Se introducen las tres fases DCE de un corte (o se elige un ejemplo) y la web devuelve la probabilidad de pCR estimada por el modelo, ejecutado en el navegador, junto con su interpretación (curva de captación, mapa de oclusión).
2. **Soporte de la defensa.** Se proyecta mientras el autor expone el trabajo: datos, arquitectura, experimentos, resultados y limitaciones.
3. **Exploración autónoma.** El profesor recorre la web sin el autor delante.

### Audiencia

- Profesor y tribunal: formación en aprendizaje automático, sin formación médica obligatoria.
- Los conceptos clínicos se explican para un lector no especialista (CLAUDE.md §7).

### Requisitos que se derivan

- **Texto autosuficiente.** Por el uso 3, cada bloque se entiende sin la exposición oral: dice qué muestra y qué conclusión se extrae.
- **Legible a distancia.** Por el uso 2, tamaños de letra y de gráfica pensados para proyector.
- **Herramienta siempre accesible.** Por el uso 1, el análisis de un corte se alcanza desde cualquier punto de la web con una sola acción.
- **Solo escritorio.** Sin diseño ni comprobaciones para móvil.

### Alcance

Rediseño completo: estructura, textos, sistema visual, gráficas y componentes. Se mantienen solo las restricciones técnicas y de significado:

- Inferencia con ONNX Runtime Web en el navegador; el preprocesado replica `load_dce_triplet` y `npm test` lo comprueba.
- Colores de clase compartidos con `style.py`: no pCR `#23656E`, pCR `#DC7867`, y escala de realce `ENHANCEMENT_CMAP`.
- Contenido obtenido del manifiesto y de los ficheros generados por `scripts/export_web.py`; ninguna métrica se escribe a mano.

El resto (tema, acento, tipografías, orden de secciones) puede cambiar; si cambia, se actualiza CLAUDE.md §8.

## 2. Estructura

### Pestañas

Tres pestañas, una por recorrido de lectura:

| Pestaña | Uso | Contenido |
|---|---|---|
| **Analizar** | Herramienta | Cargar un corte o elegir un ejemplo; probabilidad de pCR, curva de captación y mapa de oclusión |
| **El estudio** | Defensa y exploración | Relato en orden: problema y datos, modelo, experimentos, resultados |
| **Alcance** | Defensa y exploración | Qué permite afirmar el modelo, limitaciones, glosario clínico y técnico |

- La web abre en **Analizar**, que funciona como un portal de lectura con marco propio: primero la entrada (elegir o cargar un estudio) y, al abrir uno, el espacio de trabajo. El botón «Cómo funciona el modelo» lleva a El estudio.
- Una barra superior fija está siempre visible: en el portal, la marca y la ruta del estudio abierto; en El estudio y Alcance, las dos pestañas, su índice de secciones y el botón «Analizar un estudio».
- Cada pestaña tiene su dirección (`#analizar`, `#estudio`, `#alcance`), de modo que el botón «atrás» del navegador funciona y se puede enlazar directamente una pestaña o una sección.

### Secciones de cada pestaña

**Analizar**

1. Entrada: ejemplos o carga de las tres fases, con las tres imágenes visibles.
2. Resultado: probabilidad de pCR y clase según el umbral.
3. Interpretación: curva de captación y mapa de oclusión.
4. Recientes: los estudios leídos en la sesión, con su estimación y la clase real si se reveló; se pueden descargar en CSV.

**El estudio**

1. Problema: qué es la pCR, por qué importa predecirla antes del tratamiento y qué aporta la DCE-MRI. Abre con la animación del recorrido (adquisición, contraste, recorte, red, probabilidad).
2. Datos: cohorte, reparto de pacientes en folds y preprocesado de un corte.
3. Modelo: arquitectura por etapas, ensamblado de los cinco modelos de fold y calibración.
4. Experimentos: AUC por paciente en la validación interna y elección del número de épocas.
5. Resultados: AUC en test, curvas ROC, distribución de probabilidades por clase y clasificación con el umbral, incluida la explicación de la doble evaluación en test.

**Alcance**

1. Qué significa la probabilidad: cómo se lee el valor y qué no permite afirmar.
2. Limitaciones.
3. Ficha del modelo.
4. Glosario.

## 3. Niveles de lectura

### Un solo nivel

No hay modo resumen ni modo técnico, ni definiciones al pasar el ratón, ni desplegables. Todo el contenido está siempre visible y tiene nivel técnico suficiente para un lector con formación en aprendizaje automático. La separación entre la herramienta (Analizar) y el relato (El estudio, Alcance) ya evita que la explicación técnica interfiera con el uso de la herramienta.

### Jerarquía dentro de cada bloque

La densidad se controla con jerarquía visual, no ocultando contenido. Un bloque usa como máximo estos cuatro elementos, y solo los que necesita:

1. **Título** que nombra el contenido.
2. **Visual**: gráfica, imagen o diagrama. Cuando existe, es el elemento dominante.
3. **Conclusión**: una frase (unas 25 palabras como máximo), en tamaño grande, legible en proyector.
4. **Detalle técnico**: método, cifras exactas con su dispersión y decisiones con su justificación. Como máximo unas 80 palabras, en un tamaño claramente secundario.

Hay tres tipos de bloque:

- **Visual**: título, visual, conclusión y detalle. Se usa cuando hay datos que mostrar o un mecanismo que dibujar.
- **Titular**: título grande y una explicación breve en tamaño secundario, sin visual. Se usa cuando un dibujo no añade nada (una decisión, una definición, una lista de limitaciones).
- **Herramienta**: controles y visuales rotulados, sin conclusión redactada. Solo en Analizar.

### Explicación de conceptos

- **Antes que el texto, el visual.** Un concepto que se puede dibujar (realce, *washout*, reparto en folds, recorte, ensamblado) se explica con un diagrama SVG anotado y no con un párrafo.
- **Conceptos clínicos**: se explican una vez, en El estudio › Problema. El glosario de Alcance los recoge para consultarlos.
- **En Analizar** las explicaciones van rotuladas sobre los propios visuales; por ejemplo, la curva de captación señala las fases y el tipo de curva, sin párrafos.

## 4. Contenido

Inventario de bloques. Cada bloque indica su tipo (capa 3), lo que muestra, lo que dice y de dónde salen sus datos. Las conclusiones que dependen de resultados se redactan con las cifras de las fuentes, no antes; aquí se describe qué deben afirmar. Abreviaturas de las fuentes: `manifest` = `web/src/manifest.json`, `results` = `web/src/results.json`, `runs` = `web/src/runs.json`, todos generados por `scripts/export_web.py`; «navegador» = calculado en la web a partir de las imágenes.

### Analizar

**A1. Presentación de la herramienta** · Titular

- Título: Probabilidad de pCR a partir de un corte de DCE-MRI.
- Texto: qué entra (tres fases de un corte), qué sale (probabilidad calibrada), que el modelo se ejecuta en el navegador y enlace a El estudio.

**A2. Entrada** · Herramienta

- Visual: las tres fases en escala de grises, con rótulos PRE, EARLY y LATE, y el mapa de realce `EARLY − PRE`. Selector de ejemplos y carga de tres PNG.
- Rótulos: qué es cada fase en una línea (antes del contraste, poco después, minutos después).
- Con «Medir captación» desactivada, pulsar una vista la amplía a todo el visor; «Cuatro vistas» o Escape vuelve a la cuadrícula. Con la medida activa, pulsar fija el punto.
- Fuente: `manifest.examples` (corte, cohorte y ficheros; la etiqueta queda oculta hasta revelarla).

**A3. Resultado** · Herramienta

- Visual: barra fina de 0 a 0,6 dividida por el umbral en los tramos no pCR y pCR, con sus colores de clase, y la marca de la estimación; el tramo donde cae se intensifica. Clase resultante con su color.
- Debajo, «Predecir» y, en la misma fila, «Revelar clase real», que se sustituye por la clase real y si coincide.
- Rótulo: una línea que recuerda que el AUC en test es próximo al azar, con enlace a Alcance › Qué significa la probabilidad.
- Fuente: navegador (salida `calibrated`); `manifest.model.threshold`; `manifest.test_patient.auc`.

**A4. Curva de captación** · Herramienta

- Visual: intensidad en PRE, EARLY y LATE del punto seleccionado, con el área bajo la curva sombreada. El tipo de curva (persistente, meseta, lavado o sin realce) y el cambio de LATE frente a EARLY van en una etiqueta junto al título; los valores de cada fase, sobre las imágenes.
- Fuente: navegador (ventana de 5 × 5 píxeles; regla de ±10 % en LATE).

**A5. Mapa de oclusión** · Herramienta

- Visual: mapa superpuesto al corte; color según cuánto cambia la probabilidad al tapar cada región, con la escala rotulada.
- Leyenda en el panel del informe, junto a la de realce: escala de color con «hacia no pCR» y «hacia pCR», probabilidad sin tapar y una línea con el tamaño de parche, el relleno (cero) y el mayor cambio.
- Fuente: navegador.

**A6. Registro de la defensa** · Herramienta

- Visual: tabla de hasta cinco cortes con probabilidad, umbral, clase y etiqueta real revelada a demanda; recuento de aciertos; copia en CSV.
- Fuente: navegador.

### El estudio

**E1. Apertura** · Visual

- Visual: animación del recorrido (adquisición, contraste, realce, recorte, etapas de la red, probabilidad).
- Título: el del trabajo. Una frase que lo resume.

**E2. Momento de la predicción** · Visual *(visual nuevo)*

- Visual: línea temporal del tratamiento: diagnóstico → DCE-MRI previa → quimioterapia neoadyuvante → cirugía → análisis patológico (pCR o no pCR). Marca dónde actúa el modelo.
- Conclusión: se quiere estimar antes del tratamiento lo que hoy solo se conoce tras la cirugía.
- Detalle: qué es la pCR y por qué se asocia a mejor pronóstico.

**E3. Qué muestra la DCE-MRI** · Visual *(visual nuevo)*

- Visual: las tres fases de un mismo corte, el mapa de realce y las tres formas típicas de curva de captación (esquema).
- Conclusión: el contraste se acumula y se elimina de forma distinta en el tejido tumoral; la red recibe las tres fases como canales.
- Fuente: un ejemplo de `manifest.examples`.

**E4. Pregunta y restricciones** · Titular

- Título: Una CNN 2D entrenada desde cero que estima la pCR a partir de un corte.
- Texto: sin preentrenamiento; entrada solo imagen; unidad estadística la paciente; test reservado a la evaluación final.

**E5. Cohortes y reparto en entrenamiento y test** · Visual

- Visual: pacientes por cohorte (Duke, I-SPY1, I-SPY2) y clase, separando entrenamiento y test.
- Conclusión: la proporción de pCR cambia entre cohortes; eso permite predecir algo solo con la cohorte, y por eso se usa como referencia (E15).
- Detalle: pacientes y cortes por split, pacientes excluidas.
- Fuente: `manifest.split` (`cohorts`, `train`, `test`, `excluded_patients`).

**E6. Reparto de pacientes en cinco folds** · Visual

- Visual: pacientes de entrenamiento agrupadas por fold y coloreadas por clase.
- Conclusión: cada paciente pertenece a un único fold; la validación nunca contiene cortes de una paciente de entrenamiento.
- Detalle: pacientes y proporción de pCR por fold.
- Fuente: `manifest.split.train`.

**E7. Preprocesado de un corte** · Visual

- Visual: secuencia PNG → escala `value / 255` → tensor `[3, 256, 256]` → recorte central de 128 píxeles.
- Conclusión: la red recibe las tres fases alineadas como canales, sin normalización de color.
- Fuente: `manifest.architecture.input_crop`, un ejemplo de `manifest.examples`.

**E8. Aumentos geométricos** · Visual

- Visual: un corte con varias transformaciones aleatorias, idénticas en las tres fases.
- Conclusión: solo transformaciones geométricas, solo en entrenamiento.
- Detalle: rotación, traslación, escala y volteo horizontal con sus rangos.
- Fuente: `manifest.training`.

**E9. Arquitectura por etapas** · Visual

- Visual: las cinco etapas con canales, tamaño espacial y parámetros; promedio global, *dropout* y capa lineal hasta el logit.
- Conclusión: red pequeña, de unos 74.000 parámetros, adecuada al tamaño del conjunto.
- Fuente: `manifest.architecture`.

**E10. Entrenamiento** · Titular

- Título: Entrenamiento con pérdida ponderada por clase.
- Texto: `BCEWithLogitsLoss` con `pos_weight = N0 / N1`, optimizador, tasa de aprendizaje, programa de la tasa, EMA y número de épocas.
- Fuente: `manifest.training`.

**E11. Ensamblado y calibración** · Visual

- Visual: cinco modelos de fold → media de probabilidades → calibración → probabilidad final → umbral.
- Conclusión: la salida es la media de los cinco modelos, calibrada; el umbral se fija con la validación interna.
- Detalle: criterio del umbral (especificidad mínima) y método de calibración.
- Fuente: `manifest.model`.

**E12. AUC por paciente en la validación interna** · Visual

- Visual: los 28 experimentos frente a la referencia `baseline_weighted` y a la referencia por cohorte, con la dispersión entre folds.
- Conclusión: a redactar con los datos; debe decir si las diferencias superan la variabilidad entre folds.
- Fuente: `runs` (`experiments`, `reference`, `cohort_baseline`).

**E13. Número de épocas del modelo final** · Visual

- Visual: AUC de validación por época y fold, con la época elegida.
- Conclusión: cómo se eligió el número de épocas sin usar test.
- Fuente: `results.epochs`.

**E15. AUC por paciente en test** · Visual

- Visual: AUC con intervalo de confianza del modelo final y de la referencia por cohorte, en un mismo eje con el azar (0,5) marcado.
- Conclusión: a redactar con los datos; debe decir si el modelo supera a la referencia por cohorte.
- Detalle: número de pacientes, método del intervalo, diferencia con la referencia y que el test ya se había consultado antes con otro modelo candidato.
- Fuente: `manifest.test_patient`.

**E16. Curvas ROC por paciente** · Visual

- Visual: ROC de validación interna y de test, con el punto del umbral.
- Fuente: `results.validation`, `results.test`.

**E17. Probabilidad estimada por clase** · Visual

- Visual: distribución de la probabilidad por paciente en pCR y no pCR, con el umbral.
- Conclusión: a redactar con los datos; cuánto se solapan las dos clases.
- Fuente: `results.test`.

**E18. Clasificación con el umbral** · Visual

- Visual: matriz de confusión por paciente en test.
- Detalle: sensibilidad, especificidad y exactitud equilibrada.
- Fuente: `results.test`, `results.threshold`.

### Alcance

**S1. Qué significa la probabilidad** · Titular

- Título: Una estimación sin capacidad discriminativa demostrada.
- Texto: cómo se lee el valor, qué no permite afirmar (no es un diagnóstico ni orienta un tratamiento) y por qué, con referencia a E15.

**S2. Limitaciones** · Titular

- Lista breve: un corte en lugar del volumen, tamaño del conjunto, diferencias entre cohortes, doble evaluación en test, ausencia de validación externa.

**S3. Ficha del modelo** · Titular con tabla

- Tabla: datos de entrenamiento, arquitectura, parámetros, entrada, salida, umbral, métricas de validación y test, uso previsto y no previsto.
- Fuente: `manifest`, `results`.

**S4. Glosario** · Titular con lista

- Términos clínicos (pCR, DCE-MRI, realce, *washout*, neoadyuvante) y técnicos (AUC, fold, calibración, oclusión), con una definición de una o dos líneas cada uno.

### Visuales nuevos

- Nuevos: E2 (línea temporal del tratamiento), E3 (esquema de las curvas de captación), E5 (cohortes y clase por split) y E11 (ensamblado y calibración).
- Ampliado: E15 añade el modelo anterior a la figura actual (`auc.ts`), que solo muestra el modelo actual y la referencia por cohorte.
- El resto tiene ya una versión en `web/src/` que se rediseña con la guía de gráficas (capa 6).

### Pendiente de comprobar

- El umbral tiene dos valores por diseño: `manifest.model.threshold` (0,351) es el umbral por corte, que usa Analizar, y `results.threshold` (0,352) el umbral por paciente, que usan los resultados. Ambos se fijan con especificidad ≥ 0,9 en la validación anidada (`calibrate_and_threshold`, `unit="slice"` o `"patient"`). La web nombra siempre la unidad.
- El texto del mapa de oclusión toma el AUC de `manifest.test_patient.auc`.

## 5. Sistema visual

Dirección elegida: **cristal suave**. Paneles translúcidos sobre un fondo claro con tres manchas difusas, muy pocos colores y la acción en negro. La muestra de referencia está en el lienzo «Capa 5 · Direcciones visuales» (B · Analizar, B · E15 y Tokens de B), con cifras reales de la web y cortes simulados.

### Tema y superficies

| Token | Valor | Uso |
|---|---|---|
| Fondo base | `#EDEFF1` | Color de la página bajo las manchas |
| Manchas | `#D6E4E6` arriba a la izquierda, `#E6E0EE` arriba a la derecha, `#ECE5DC` abajo | Degradados radiales que se desvanecen al 70 %. No significan nada: dan profundidad al cristal |
| Panel | blanco al 55 %, borde blanco al 80 %, desenfoque 24 px, sombra `0 12px 40px rgba(30,40,50,.08)` | Contenedor de todo bloque |
| Superficie interna | blanco al 70 % | Notas, botones en reposo, etiqueta de clase, recuadros dentro de un panel |
| Visor | `#0E1113`, rótulos `#A4A9AE`, etiquetas sobre imagen en blanco al 14 % | Fondo de toda imagen médica |

- Las imágenes médicas van siempre sobre el visor oscuro, nunca directamente sobre el cristal.

### Color

| Token | Valor | Uso |
|---|---|---|
| Texto principal y acción | `#15191D` | Texto, botones activos, umbral, serie principal |
| Texto secundario | `#4A525B` | Explicaciones, rótulos, leyendas (7,9:1 sobre blanco) |
| Gris de referencia | `#8B9197` | Bordes discontinuos y series de referencia; nunca texto |
| Separador | `#15191D` al 8 % | Líneas entre partes de un panel |
| No pCR | `#23656E` | Fijo (capa 1) |
| pCR | `#DC7867` | Fijo (capa 1) |

- **Sin color de acento.** La acción es negra. La miel `#E2B55E` solo queda en la señal que recorre la red en la animación de apertura de «El estudio», sobre el visor oscuro.
- **Los colores de clase forman también la escala de realce** (`ENHANCEMENT_CMAP`: verde petróleo pierde contraste, coral gana). Por eso el mapa de realce lleva siempre su propia leyenda.

### Tipografía

Una sola familia en la web, **Inter**, cargada desde Google Fonts. Desaparecen IBM Plex Sans y Newsreader. Pesos 400, 500 y 600. Matplotlib sigue con DejaVu Sans (`style.py`), sin instalar nada en las máquinas.

| Tamaño | Peso | Uso |
|---|---|---|
| 56 px | 600 | Cifra principal (probabilidad) |
| 40 px | 600 | Título de página y de sección |
| 24 px | 600 | Conclusión de un bloque |
| 19 px | 400 | Cuerpo |
| 16 px | 600 / 400 | Título de tarjeta / detalle técnico y controles |
| 14 px | 500 | Rótulos, ejes, leyendas, etiquetas |

- Nada por debajo de 14 px, por el proyector.

### Cifras

- Cifras tabulares (`font-variant-numeric: tabular-nums`) en toda cifra, para que se alineen y no bailen al animarse.
- Probabilidades, umbral y AUC en escala 0–1 con coma decimal, nunca en porcentaje: probabilidad con 2 decimales, umbral y AUC con 3.
- El porcentaje se reserva para proporciones (29,4 % de pCR en entrenamiento, % de la fila en la matriz de confusión).

### Radios

| Radio | Uso |
|---|---|
| 14 px | Panel |
| 10 px | Bloque interno, visor, nota |
| 8 px | Botón, imagen, navegación, etiqueta de clase |
| 4–6 px | Etiqueta pequeña, barra, intervalo |

- Sin formas de píldora. El círculo queda solo para puntos de gráfica y la mira del visor.

### Retícula y espaciado

- **Página**: barra superior fija de 64 px y contenido de 1520 px como máximo; El estudio y Alcance, de 1280 px.
- **Contenido**: 12 columnas con separación de 24 px. En el portal, lista de estudios de 260 px, visor en el centro e informe de 380 px. En un bloque visual de El estudio, gráfica en 8 y texto en 4. En un titular, el título ocupa todo el ancho.
- **Espaciado**: 4, 8, 12, 16, 24, 32 y 48 px. Paneles con relleno de 24 px (32 en los bloques visuales de El estudio) y separados 24 px. 4 y 12 solo dentro de componentes pequeños.

### Navegación

- Barra superior fija con la marca «Lector pCR» a la izquierda. La marca, tres láminas apiladas, indica sin texto si el modelo está disponible: se iluminan con la descarga, quedan sólidas cuando está listo y discontinuas si falla (con el botón «Reintentar»).
- En el portal, la ruta «Estudios / estudio abierto» y el botón «Cómo funciona el modelo». En El estudio y Alcance, las dos pestañas, el índice de secciones con la visible marcada y el botón «Analizar un estudio».
- No se muestran el hash del modelo ni detalles de ejecución; el hash está en la ficha del modelo.

### Estados de interacción

| Estado | Aspecto |
|---|---|
| Reposo | Superficie interna (blanco al 70 %) |
| Al pasar el cursor | Blanco al 100 % |
| Seleccionado | Fondo `#15191D`, texto blanco |
| Foco de teclado | Contorno de 2 px `#15191D` separado 2 px |
| Desactivado | Opacidad al 40 % |

Se aplican a botones, ejemplos, pestañas y a los elementos que se pulsan dentro de las gráficas (fold, etapa de la red, experimento).

### Tinta de gráficas

| Elemento | Trazo |
|---|---|
| Rejilla | 1 px, `#15191D` al 6 % |
| Eje | 1 px, `#15191D` al 18 % |
| Azar | 2 px discontinuo, `#15191D` |
| Umbral | 2 px continuo, `#15191D` |
| Serie principal | 4–5 px continuo, `#15191D` |
| Serie de referencia | 3 px discontinuo, `#8B9197` |
| Intervalo principal | barra de 24 px, `#15191D` al 85 % |
| Intervalo secundario | barra de 24 px, `#15191D` al 35 % |

- Las referencias se distinguen por el trazo, no por el color; el color queda para la clase.
- Las gráficas se dibujan en la web, en SVG, a partir de los JSON de `scripts/export_web.py`. Las figuras de Matplotlib para notebooks y diapositivas usan los mismos valores.

### Movimiento

| Animación | Duración |
|---|---|
| Entrada de paneles | 600 ms, en cascada de 50 ms |
| Cifras que cuentan hasta su valor | 900 ms |
| Curvas que se dibujan, barras e intervalos que crecen | 700–1100 ms |

- Curva de salida suave `cubic-bezier(.2, .7, .2, 1)`.
- Nada se repite en bucle salvo la apertura E1.
- Todo se desactiva con `prefers-reduced-motion`.

### Comprobación en proyector

El cristal pierde contraste antes que un tema opaco. Antes de la implementación, se proyecta la muestra en el aula. Si se lava, el panel sube al 70 % de blanco y se quitan las manchas, sin cambiar ningún otro token.

## 6. Guía de gráficas

Las reglas comunes se aplican a toda gráfica, diagrama e imagen anotada de la web y a las figuras de Matplotlib; las fichas concretan cada visual del inventario (capa 4). Trazos, colores y duraciones son los tokens de la capa 5.

### Reglas comunes

1. **Una pregunta por gráfica.** El título del bloque la nombra y la conclusión la responde; la gráfica hace visible esa respuesta.
2. **Un protagonista.** El elemento que responde a la pregunta (el modelo final, la paciente seleccionada, el umbral) va en tinta principal o en color de clase y con el trazo más grueso. Todo lo demás pasa a gris de referencia o a opacidad baja. Si todo tiene el mismo peso, la gráfica no dice nada.
3. **Anotación directa, sin leyendas.** Cada serie, línea de referencia y región lleva su rótulo junto a ella («azar», «umbral 0,351», «referencia por cohorte»). La única leyenda separada es la escala de color de un mapa (realce, oclusión).
4. **La cifra clave, escrita.** El valor que cita la conclusión aparece en la gráfica junto a su marca, con su intervalo cuando lo tiene.
5. **Línea de contexto.** Encima de la gráfica, en rótulo de 14 px: split, unidad (corte o paciente), número de casos y umbral cuando interviene. Ejemplo: «Test · por paciente · 176 pacientes, 53 pCR».
6. **Incertidumbre siempre visible.** Ninguna métrica se muestra sin su dispersión: intervalo de confianza en test, valor de cada fold en la validación interna.
7. **Ejes mínimos.** Solo las marcas necesarias para leer los valores, con su nombre y unidad. Rejilla solo en el eje en que se leen valores. El rango se elige por los datos y no empieza en cero cuando no hay barras; las referencias (azar, umbral) quedan siempre dentro del rango.
8. **Color con significado.** Colores de clase solo para la clase; tinta principal para el protagonista; gris discontinuo para referencias. Ningún otro color.
9. **Nada esencial depende del cursor.** Lo necesario para entender la gráfica está escrito. El cursor solo identifica elementos individuales (una paciente, un experimento) y la selección muestra el detalle en la columna de texto del bloque.
10. **Animación de entrada, una vez.** La gráfica se dibuja al entrar en pantalla, en el orden en que se lee: ejes y referencias primero, protagonista al final. No se repite al volver a pasar.
11. **Accesible.** Cada SVG lleva `role="img"` y una descripción que enuncia la conclusión, no la forma.

### Fichas de los visuales

**A3. Resultado.** Barra horizontal con eje de 0 a 0,6 (las probabilidades calibradas no pasan de 0,5; máximo 0,475 en validación y 0,435 en test). Protagonista: la probabilidad calibrada, marca grande en tinta con su valor encima. Los cinco modelos de fold como puntos pequeños grises. Umbral en línea continua rotulada; la región a su derecha, con un tinte muy claro de pCR, rotulada «clasificado como pCR». Prevalencia de pCR en entrenamiento como marca gris rotulada. La marca se desplaza con un muelle al cambiar de corte.

**A4. Curva de captación.** Eje horizontal categórico PRE, EARLY y LATE, equiespaciado y sin escala de tiempo, porque los tiempos de adquisición no se conocen por corte. Eje vertical: intensidad relativa. Protagonista: la curva del punto seleccionado. Referencia: la media del recorte central, en gris discontinuo. En LATE se dibuja la banda de ±10 % respecto a EARLY, que es la regla de clasificación, y el tipo de curva (persistente, meseta, lavado) se rotula al final de la curva con su porcentaje.

**A5. Mapa de oclusión.** Superpuesto al corte en el visor, con transparencia regulable. Escala divergente centrada en cero: el cambio de probabilidad al tapar cada parche. Leyenda propia con los extremos rotulados por su efecto («tapar baja la probabilidad de pCR» y «tapar la sube»). La mira del visor marca el parche de mayor efecto.

**E2. Momento de la predicción.** Línea temporal horizontal de cinco estaciones. La DCE-MRI previa, donde actúa el modelo, en tinta y destacada; el resto en gris. Una llave entre la resonancia y el análisis patológico rotulada «lo que el modelo intenta anticipar». El final se bifurca en pCR y no pCR con sus colores. Sin duraciones numéricas salvo que se citen de una fuente. Las estaciones aparecen en orden.

**E3. Qué muestra la DCE-MRI.** A la izquierda, las tres fases de un ejemplo y su mapa de realce en el visor. A la derecha, las tres formas de curva de captación, rotuladas como esquema, con el mismo lenguaje visual que A4, para que la curva de Analizar se reconozca.

**E5. Cohortes.** Barras horizontales apiladas, una por cohorte, con el número de pacientes. En entrenamiento, segmentos por clase y proporción de pCR escrita al final de cada barra. En test, barra gris sin clase, porque sus etiquetas solo intervienen en la evaluación final. Protagonista: la diferencia de proporción de pCR entre cohortes.

**E6. Reparto en cinco folds.** Un punto por paciente, una columna por fold y otra para test (versión actual de `split.ts`). Bajo cada columna, pacientes y proporción de pCR. Al seleccionar un fold, sus pacientes pasan a ser la validación y el resto se atenúa.

**E7. Preprocesado.** Secuencia horizontal con la imagen real en cada paso y la forma del tensor rotulada debajo: PNG → `value / 255` → `[3, 256, 256]` → recorte central `[3, 128, 128]`. El recorte se marca sobre la imagen completa antes de aplicarse.

**E8. Aumentos.** Rejilla de cuatro transformaciones por tres fases, alineadas en columnas para que se vea que la geometría es idéntica en PRE, EARLY y LATE. Un botón genera otra muestra. Rangos de cada transformación rotulados en la línea de contexto.

**E9. Arquitectura.** Vista oblicua actual (`network.ts`), construida desde la tabla de capas. Protagonista: el tamaño espacial que decrece mientras crecen los canales, rotulados en cada etapa. Total de parámetros escrito. Al seleccionar una etapa, su detalle aparece en la columna de texto.

**E11. Ensamblado y calibración.** Diagrama con las cifras reales de un ejemplo de `manifest.examples`: cinco probabilidades de fold → media → probabilidad calibrada → comparación con el umbral. Cada paso aparece en orden y su cifra cuenta hasta su valor.

**E12. AUC en la validación interna.** Una fila por experimento, ordenadas por AUC medio. Cinco puntos pequeños grises por fila (un fold cada uno) y el AUC medio en tinta. Línea vertical discontinua de la referencia `baseline_weighted` y banda gris de la referencia por cohorte (media ± desviación entre folds). El modelo final destacado; el resto en gris. Al seleccionar una fila, la columna de texto muestra qué cambió respecto a la referencia.

**E13. Número de épocas.** AUC de validación por época: una línea gris fina por fold y la media en tinta. Línea vertical en la época elegida, rotulada con su número, y marcas en la época elegida para cada fold sin usarlo.

**E15. AUC en test.** Gráfico de intervalos horizontal con eje de 0,4 a 0,8. Dos filas: modelo actual (intervalo principal) y modelo anterior (intervalo secundario), con el AUC y el intervalo escritos a la derecha. Líneas verticales rotuladas para el azar (0,5) y la referencia por cohorte. Debajo, en un eje propio centrado en cero, la diferencia con la referencia y su intervalo.

**E16. Curvas ROC.** Curva de test en tinta y gruesa; las cinco de validación en gris fino. Diagonal del azar discontinua. Punto del umbral sobre la curva de test, rotulado con su sensibilidad y especificidad. AUC escrito junto a cada curva de referencia y a la de test.

**E17. Probabilidad por clase.** Un punto por paciente de test (enjambre de puntos), en dos filas por clase con sus colores. Umbral vertical rotulado. A cada lado del umbral, el número de pacientes de cada clase. Se prefiere al histograma porque con 176 pacientes se ven todos los casos.

**E18. Clasificación con el umbral.** Matriz 2 × 2 por paciente con el recuento grande y el porcentaje de la fila pequeño. Intensidad de la celda en tinta según el porcentaje de la fila. Clases rotuladas con su color en filas y columnas. Sensibilidad y especificidad escritas junto a su fila.

### Decisiones de las fichas

- **Prevalencia en el eje de probabilidad (A3).** En A3 la prevalencia de pCR en entrenamiento comparte eje con probabilidades, así que se escribe en escala 0–1 (0,294) y no en porcentaje, para no mezclar unidades en un eje. Fuera de ese eje sigue la regla de la capa 5.
- **Colores del mapa de oclusión (A5).** Usa la escala divergente de clase, porque el efecto de tapar una región se refiere a la clase (acerca o aleja la estimación de pCR). Como el mapa de realce usa los mismos colores con otro significado, cada mapa lleva su título y su leyenda propia.

## 7. Componentes

Lista cerrada: la web se construye solo con estas piezas. Tamaños, colores, radios y estados son los tokens de la capa 5; los estados de interacción se aplican a todo componente que se pulsa.

### Estructura de página

**Barra superior.** Descrita en la capa 5, «Navegación».

**Hoja del portal.** Una sola superficie de cristal para todo Analizar; sus partes se separan con filetes, no con paneles. Dos estados:

- **Entrada:** a la izquierda, las tres fases del primer ejemplo apiladas en perspectiva con un mismo punto y sus intensidades (`slices.ts`); a la derecha, el titular, la lista de estudios y la carga.
- **Estudio abierto:** lista de estudios (260 px), visor con herramientas y panel de informe (380 px).

**Cabecera de sección.** Número de sección en rótulo de 14 px, título de 40 px y, si hace falta, una línea de cuerpo en texto secundario. Sin panel: va directamente sobre el fondo.

**Panel.** Contenedor de cristal de todo bloque. Dos variantes de relleno: 24 px (Analizar, Alcance) y 32 px (bloques visuales de El estudio).

### Bloques

**Bloque visual.** Panel de 12 columnas: visual en 8 y texto en 4.

- Columna de texto, de arriba abajo: título (16 px, 600), conclusión (24 px, 600) y detalle técnico (16 px, 400, texto secundario).
- Encima del visual, la línea de contexto (14 px, 500, texto secundario).
- Cuando la gráfica admite selección (fold, etapa, experimento), el detalle de lo seleccionado sustituye al detalle técnico y un enlace devuelve al texto original.

**Titular.** Panel a todo el ancho: título (24 px, 600) y explicación (19 px, 400, texto secundario), con 60–70 caracteres por línea como máximo. Puede contener una tabla (ficha del modelo) o una lista de definiciones (glosario).

**Bloque de herramienta.** Panel de Analizar con título de tarjeta (16 px, 600), sus visuales rotulados y sus controles. Sin conclusión redactada.

### Controles

**Botón.**

- Variantes: principal (fondo en tinta, texto blanco; una sola por pantalla: «Predecir»), secundario (superficie interna) y pequeño (32 px de alto, 14 px; por ejemplo, «Copiar CSV» o «Revelar»).
- Alto de 40 px, texto de 16 px, radio de 8 px. Solo texto, sin iconos.

**Lista de estudios.** Selector «Disponibles / Recientes». Cada fila: miniatura de EARLY, nombre del estudio («Breast_MRI_725 · corte 25») y cohorte; si ya se predijo, su estimación con el punto de su clase. En Recientes, además, cuándo se leyó y la clase real si se reveló, con «Descargar» (CSV) y «Borrar». El estudio abierto, en estado seleccionado.

**Carga de un estudio.** Zona discontinua que abre el selector de ficheros; también se puede soltar los tres PNG en cualquier punto de la página. Cada fichero se asigna a su fase por el sufijo; los errores aparecen en una nota.

**Herramientas del visor.** «Medir captación» y «Recorte» se activan y desactivan; «Realce» y «Oclusión» se alternan en la cuarta imagen. Activar la oclusión calcula el mapa, con su progreso.

**Enlace.** Texto en tinta, subrayado de 1 px. Se usa para los saltos entre pestañas (por ejemplo, de A3 a Alcance › Qué significa la probabilidad).

### Contenido

**Visor.** Fondo `#0E1113`, radio de 10 px.

- Variante triple: PRE, EARLY y LATE y mapa de realce, con rótulos de fase sobre la imagen.
- Variante única: una imagen grande con capa superpuesta (oclusión) y mira para seleccionar el punto de la curva de captación.
- Las leyendas de escala de los mapas van dentro del visor.

**Cifra principal.** 56 px, 600, cifras tabulares, con su rótulo de 14 px encima. Cuenta hasta su valor al aparecer.

**Etiqueta de clase.** Superficie interna, radio de 8 px, punto del color de clase y texto «pCR» o «no pCR». Variantes: estimada (por el modelo) y real (revelada); la real lleva el rótulo «real».

**Leyenda de escala.** Barra de degradado con los extremos rotulados por su significado, no solo por su valor. Solo para los mapas de realce y oclusión.

**Tabla.** Filas de 48 px, separadores del 8 %, cabecera en rótulo de 14 px y texto secundario, cifras tabulares alineadas a la derecha. Se usa en el registro de la defensa y en la ficha del modelo.

**Nota.** Superficie interna, radio de 10 px, texto de 16 px. Una sola por bloque como máximo. Se usa para el aviso de A3 y para los mensajes de error.

**Estado de proceso.** Texto de 14 px y barra de progreso fina en tinta. Se usa al cargar el modelo y al calcular el mapa de oclusión. Mientras dura, los controles que dependen del proceso están desactivados.

### Mensajes de error

No hay color de error: el rojo se confundiría con el coral de pCR. Un error se muestra en una nota con texto en tinta, empezando por lo que falló y seguido de cómo corregirlo («EARLY: el fichero no es un PNG de 256 × 256. Carga la fase EARLY del mismo corte.»). El hueco afectado de la zona de carga marca el error en su propio texto.

### Lo que no se usa

Ventanas modales, desplegables, pestañas dentro de un bloque, definiciones al pasar el cursor, notificaciones emergentes, formas de píldora, iconos decorativos y logotipo.

## 8. Implementación

### Organización del código

- `web/index.html`: barra superior, el portal de Analizar y los contenedores de El estudio y Alcance.
- `web/src/main.ts`: navegación entre pestañas (`#pestaña/sección`), el portal (entrada, estudio abierto, herramientas, Recientes) y la carga y verificación del modelo.
- `web/src/slices.ts`: las fases apiladas de la entrada del portal.
- `web/src/study.ts` y `web/src/scope.ts`: El estudio y Alcance, construidos la primera vez que se abren.
- `web/src/page.ts`: componentes de bloque (cabecera de sección, bloque visual, titular), entrada animada de paneles e índice de secciones.
- Gráficas: `ruler.ts` (A3), `uptake.ts` (A4), `clinic.ts` (E2, E3), `split.ts` (E5, E6), `network.ts` (E9), `ensemble.ts` (E11), `lab.ts` (E12), `results.ts` (E13, E16, E17), `auc.ts` (E15). Toda gráfica se dibuja como máximo a escala 1:1, para que sus rótulos no bajen de 14 px.

### Diferencias con las capas 4 a 7

- **A3**: el modelo ONNX no exporta el logit; el detalle muestra la probabilidad sin calibrar y el rango de los cinco modelos.
- **A5**: el mapa de oclusión es la cuarta imagen del visor, no un bloque aparte, y no tiene deslizador de opacidad.
- **Registro de la defensa (A6)**: pasa a ser «Recientes» en la lista de estudios.
- **E3**: las cuatro imágenes en fila y el esquema de curvas debajo.
- **E9**: bloque a todo el ancho y sin la parte final de los cinco modelos, que explica E11; a ocho columnas, sus rótulos quedaban por debajo de 14 px.
- **E12**: sin la marca de la mejor época del propio fold; aparece en la tabla al seleccionar un experimento.
- **E13**: solo el AUC; la pérdida se cita en el detalle.

### Pendiente

- Proyectar la web en el aula (capa 5, comprobación en proyector).
