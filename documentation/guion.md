# Guion de la exposición

**Predicción de respuesta patológica completa con una CNN 2D sobre DCE-MRI de mama**

Alejandro Barreche · Aprendizaje Automático · 4 de noviembre de 2026

---

## Reparto del tiempo

La exposición dura **5 minutos**, demostración incluida.

| Diapositiva | Contenido | Tiempo |
|---|---|---|
| Portada | Título | 5 s |
| 1 | Problema clínico y solución | 35 s |
| 2 | Datos, separación por paciente y clases | 30 s |
| 3 | Arquitectura de la CNN | 45 s |
| 4 | Entrenamiento y resultados | 60 s |
| 5 | Demostración en directo | 60 s |
| 5 | Limitaciones, mejora y conclusión | 55 s |
| | **Total** | **≈ 4 min 50 s** |

---

## Antes de empezar

- La web **Lector pCR** abierta en una pestaña, ya cargada (el modelo tarda unos segundos la primera vez).
- La presentación en modo presentador, con las notas visibles.
- Un estudio de ejemplo elegido de antemano para la demostración.

---

## Portada · 5 s

> Buenos días. Presento Lector pCR: una red convolucional 2D, entrenada desde cero, que estima, a partir de la resonancia previa al tratamiento, la probabilidad de que una paciente con cáncer de mama alcance una respuesta patológica completa.

---

## Diapositiva 1 · Problema clínico y solución · 35 s

**En pantalla:** los tres puntos de «Lo que tenemos», la solución con la escala de probabilidad y la tira del proceso.

> Algunas pacientes con cáncer de mama reciben el tratamiento **antes** de la cirugía. Es lo que se llama tratamiento neoadyuvante.
>
> Cuando después se opera, el patólogo comprueba si queda tumor. Si no queda ninguno, hablamos de **respuesta patológica completa**, o pCR. Es una buena noticia, porque se asocia a una mayor supervivencia.
>
> El problema es que esto solo se sabe al operar, meses después de empezar el tratamiento.
>
> Contamos con la resonancia previa de 1.450 pacientes de tres cohortes, junto con su resultado. Solo unas 3 de cada 10 alcanzan la pCR.
>
> La pregunta es: ¿permite esa imagen anticipar la respuesta? Mi propuesta es **Lector pCR**, que estima la probabilidad de pCR de cada paciente y la compara con un umbral. Apoya al médico; no lo sustituye.

**Transición:** «Empiezo por los datos.»

---

## Diapositiva 2 · Datos, separación por paciente y clases · 30 s

**En pantalla:** las tres fases de la resonancia y el mapa de realce, la barra de clases y los cinco folds.

> Cada corte tiene tres imágenes de la resonancia: **antes** del contraste, **poco después** y **más tarde**. Son tres instantes de tiempo, no tres colores; por eso no uso la normalización de ImageNet.
>
> La resta entre la fase temprana y la previa muestra dónde capta contraste el tejido. El tumor suele captar más.
>
> Hay 1.097 pacientes de entrenamiento, con unos diez cortes cada una, repartidas en **cinco folds por paciente**: los cortes de una misma paciente nunca se separan entre entrenamiento y validación. El conjunto de test, de 176 pacientes, no interviene en ninguna decisión.
>
> Solo el 29,4 % alcanza pCR. Para compensar este desequilibrio, la pérdida pondera la clase positiva con N₀ entre N₁, aproximadamente 2,4, calculado en cada fold solo con sus pacientes de entrenamiento.

**Transición:** «Con estos datos, la red es esta.»

---

## Diapositiva 3 · Arquitectura de la CNN · 45 s

**En pantalla:** el esquema de la red, «Por qué así» y la tabla de capacidad.

> La red recibe las tres fases, de 256 × 256 píxeles, y recorta los 128 píxeles centrales, donde se concentra el tumor.
>
> Siguen **cinco bloques**. Cada uno aplica dos convoluciones 3 × 3 con normalización por lotes y ReLU, y después un max-pooling que reduce la imagen a la mitad. Mientras tanto, los canales se duplican: de 4 a 64.
>
> Al final, una media global, un dropout de 0,3 y una capa lineal producen **un único logit**. La sigmoide lo convierte en probabilidad.
>
> En total son **74.253 parámetros**. La anchura se eligió con validación, como muestra la tabla:
>
> - con 296.000 parámetros, la red memoriza: la brecha entre entrenamiento y validación llega a 0,37;
> - con 19.000, no aprende lo suficiente;
> - con 74.000 se obtiene el mejor AUC por paciente, 0,583.

**Transición:** «Así se entrenó y esto es lo que consigue.»

---

## Diapositiva 4 · Entrenamiento y resultados · 60 s

**En pantalla:** las curvas de entrenamiento por época, la configuración, el gráfico de AUC y la matriz de errores.

**Configuración y curvas de entrenamiento (20 s)**

> El entrenamiento usa la pérdida BCE ponderada y el optimizador AdamW con una tasa de 10⁻⁴, que decrece siguiendo un coseno, junto con una media exponencial de los pesos.
>
> Los aumentos son geométricos —giro de hasta 15°, traslación, escala y volteo— y se aplican igual en las tres fases.
>
> Las curvas son la media de los cinco folds. La pérdida de entrenamiento baja durante las 40 épocas, pero la de validación deja de bajar hacia la época 4 y después sube: es sobreajuste. El AUC de validación alcanza su máximo hacia la época 8. Por eso se entrenan 8 épocas, número elegido con los otros folds, sin mirar el fold evaluado.

**Gráfico de AUC (25 s)**

> El AUC mide si el modelo **ordena** bien a las pacientes: es la probabilidad de que una paciente con pCR reciba más puntuación que una sin pCR. Un 0,5 equivale al azar.
>
> La línea coral es una referencia que solo conoce el hospital de origen de la imagen.
>
> En validación, el modelo final llega a 0,582, pero la diferencia con la referencia no es significativa.
>
> En test, el modelo ancho obtuvo 0,591 y el estrecho, 0,522. Esta segunda cifra es una segunda evaluación del test y la presento como tal.

**Coste de equivocarse (25 s)**

> A la derecha está el coste de equivocarse.
>
> Un **falso positivo** anuncia una respuesta completa que no llega. Sería el error más grave si se usara para reducir el tratamiento.
>
> Un **falso negativo** no reconoce a una paciente que sí responde.
>
> Por eso fijé el umbral en validación para una especificidad del 90 %. En test, eso deja 9 falsos positivos, pero solo detecta 5 de las 53 pCR.

**Transición:** «Lo enseño funcionando.»

---

## Diapositiva 5 · Demostración en directo · 60 s

**Pasos:**

1. Cambiar a la pestaña de Lector pCR.
2. Elegir un estudio de la lista (o cargar las tres fases de uno propio).
3. Señalar la **probabilidad**, el **umbral** y la **decisión** (pCR o no pCR).
4. Volver a la presentación.

> Elijo un estudio. La aplicación carga las tres fases, aplica el mismo preprocesado que en el entrenamiento y, en menos de un segundo, da la probabilidad de pCR, el umbral y la decisión.
>
> Todo ocurre en el propio ordenador: las imágenes no salen del equipo.

---

## Diapositiva 5 · Limitaciones, mejora y conclusión · 55 s

**En pantalla:** los tres paneles y la barra de conclusión.

**Qué limita hoy a Lector pCR (15 s)**

> La aplicación funciona, pero tiene tres límites:
>
> 1. **Acierta poco.** En test, el AUC es 0,52, casi azar, y solo detecta 5 de 53 respuestas completas.
> 2. **Ve un corte, no el tumor.** Mira una imagen 2D cuando el tumor es un volumen.
> 3. **Aprende el hospital.** Parte de lo que aprende es de qué cohorte viene la imagen.

**Mejora propuesta (15 s)**

> La mejora que propongo tiene tres pasos:
>
> 1. Entrenar con el **volumen 3D completo**, que ya es público para 2.070 pacientes.
> 2. Que un radiólogo **marque el tumor** en unos mil estudios.
> 3. **Validar** el modelo con pacientes nuevas de tres hospitales.
>
> Cuesta unos 72.000 €, y el objetivo es un AUC de 0,75, cerca de lo publicado con 3D y tumor segmentado.

**Un hospital de referencia, un año (15 s)**

> ¿Para qué serviría? Imaginemos un hospital con 150 pacientes al año.
>
> En el ensayo PHERGain, un tercio de las pacientes HER2+ pudo evitar la quimioterapia porque una prueba de imagen, un PET, mostró pronto que respondían.
>
> Si Lector pCR hiciera ese papel con la resonancia que ya se hace, 15 pacientes al año evitarían la quimioterapia y no harían falta 45 PET. Son unos **200.000 € al año** por hospital: la mejora se pagaría en unos cuatro meses.
>
> Es un escenario: hoy el modelo no permite tomar ninguna de estas decisiones.

**Conclusión (10 s)**

> En conclusión: **el producto ya está listo; lo que falta es un modelo que vea el tumor entero y que se valide fuera de estos datos.**
>
> Muchas gracias.

---

## Supuestos del caso económico

Si preguntan por las cifras de la diapositiva 5:

| Cifra | Valor | Origen |
|---|---|---|
| Pacientes con neoadyuvancia en un hospital de referencia | 150 al año | Supuesto propio |
| Proporción de pacientes HER2+ | 30 % → 45 pacientes | Supuesto propio; la neoadyuvancia se usa sobre todo en HER2+ y triple negativo |
| HER2+ que evitan la quimioterapia | 1 de cada 3 → 15 pacientes | Ensayo PHERGain (Pérez-García et al., *Lancet*, 2024) |
| Quimioterapia evitada | 9.000 € por paciente | 6 ciclos × 1.500 €, extremo bajo del precio privado por sesión (1.500–4.500 €) |
| PET-TAC evitado | 1.500 € | Precio privado en España: 1.300–1.800 $ |
| Entrenamiento 3D en la nube | 2.000 € | Supuesto propio |
| Marcado del tumor | 10.000 € | 1.000 estudios × 10 min = 170 h × 60 €/h; supuesto propio |
| Validación en 3 hospitales | 60.000 € | Supuesto propio |
| Objetivo de AUC | 0,75 | Un modelo 3D con tumor segmentado obtuvo 0,72 en validación externa (Xu et al., 2025) |

**Cálculo:** 15 × 9.000 € + 45 × 1.500 € ≈ 135.000 € + 68.000 € ≈ **200.000 € al año**. La inversión de 72.000 € se recupera en unos 4 meses.

---

## Preguntas probables

**¿Qué error es más relevante?**

Depende del uso. Si el modelo sirviera para reducir el tratamiento, el falso positivo es el más grave, porque una paciente que no responde recibiría menos tratamiento del necesario. Por eso el umbral prioriza la especificidad (90 % en validación, 92,7 % en test), a costa de la sensibilidad (9,4 %).

**¿La red aprende biología tumoral o diferencias entre cohortes?**

En parte, cohortes. Una referencia que solo conoce la cohorte de origen obtiene un AUC de 0,544 en test, y el modelo no la supera de forma significativa (diferencia −0,022, IC 95 % de −0,102 a 0,059). Dentro de cada cohorte, el AUC en test es 0,59 en Duke, 0,36 en I-SPY1 y 0,55 en I-SPY2.

**¿Cómo afecta tener unos diez cortes correlacionados por paciente?**

Los cortes de una paciente no son observaciones independientes. Por eso la separación es siempre por paciente, la probabilidad de una paciente es la media de las de sus cortes, las métricas se calculan por paciente y los intervalos de confianza se obtienen con bootstrap sobre pacientes.

**¿Está calibrada la probabilidad?**

Se calibró con una regresión de Platt sobre puntuaciones fuera de fold, y el umbral se fijó en esa misma escala. Aun así, en test la escala de probabilidades no se mantiene del todo; la probabilidad debe leerse como orientativa.

**¿Por qué 8 épocas?**

Para cada fold, la época se eligió con la curva media de validación de los otros cuatro folds. Así no se escoge la mejor época mirando el fold que se evalúa, lo que inflaría el AUC.

**¿Por qué el recorte central de 128 píxeles?**

El tumor suele estar centrado, y el borde aporta sobre todo vasos y diferencias entre hospitales. Fue además el único cambio con un efecto consistente en comparaciones emparejadas (+0,01 a +0,02 de AUC).

**¿Por qué no una red más profunda o más ancha?**

Probé una variante profunda y otra ancha, y ninguna mejoró. Con más parámetros, la red memoriza más: la brecha entre entrenamiento y validación pasa de 0,19 a 0,37.

**¿Por qué no ayuda una penalización fuerte de los pesos?**

Con normalización por lotes después de cada convolución, la escala de los pesos de esa convolución no cambia la salida, de modo que penalizarla apenas regulariza.

**¿Por qué hay dos resultados de test?**

El primer modelo final fue el ancho (0,591). Después elegí el estrecho por su mejor validación y su menor sobreajuste, y lo evalué en test. Esa segunda cifra (0,522) es una segunda mirada al test, y la presento como tal. Ninguno de los dos supera de forma significativa a la referencia de cohorte.

**¿Qué evidencia faltaría para hablar de utilidad clínica?**

Una validación externa y prospectiva en hospitales nuevos, resultados por subgrupos (por ejemplo, receptores hormonales y HER2) para estudiar la equidad, y un modelo que use el volumen completo del tumor.

**¿Coincide la aplicación con el pipeline de Python?**

Sí. La web aplica el mismo preprocesado que el entrenamiento, y la diferencia entre las probabilidades de la web y las de Python es inferior a una millonésima. Se comprueba con tests automáticos y al arrancar la aplicación.
