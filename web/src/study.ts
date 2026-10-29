// El estudio (documentation/web-design.md, layer 4, blocks E1-E18): problem, data, model,
// experiments and results. Every figure and number comes from manifest.json, results.json and
// runs.json, written by scripts/export_web.py; conclusions that depend on results are phrased
// from those numbers.
import manifest from "./manifest.json";
import results from "./results.json";
import runs from "./runs.json";
import { drawAuc } from "./auc";
import { apply as augmentPhase, sample as sampleAugmentation } from "./augment";
import { drawCurveTypes, drawTimeline } from "./clinic";
import { drawEnsemble } from "./ensemble";
import { startHero } from "./hero";
import { describe, drawLab, mean, std, type Experiment } from "./lab";
import { steps as layerSteps } from "./layers";
import { drawNetwork } from "./network";
import { COHORTS, fit, make, sectionHead, spy, svgOf, titular, visualBlock } from "./page";
import { auc, confusion, drawDistribution, drawEpochs, drawRoc, roc, rocPoint, type FoldHistory } from "./results";
import { drawCohorts, drawSplit, positiveWeight, summarize, type Patient } from "./split";
import { decimal, percent, thousands } from "./svg";

const SECTIONS: [string, string][] = [["problema", "Problema"], ["datos", "Datos"], ["modelo", "Modelo"], ["experimentos", "Experimentos"], ["resultados", "Resultados"]];
const arch = manifest.architecture, t = manifest.training, s = manifest.test_patient, split = manifest.split;
const example = manifest.examples[0];
const exampleContext = `${example.sample_id} · ${COHORTS[example.cohort]} · un corte de test`;
const CROP = arch.input_crop ?? 256;
const SLICE_THRESHOLD = manifest.model.threshold, PATIENT_THRESHOLD = results.threshold;
const pct = (a: number, b: number) => `${percent(a / b)} %`;
const median = (v: number[]) => { const o = [...v].sort((a, b) => a - b), m = o.length >> 1; return o.length % 2 ? o[m] : (o[m - 1] + o[m]) / 2; };

function tile(source: HTMLCanvasElement, label: string, crop = false, zoom = false) {
  const box = make("div", "tile");
  const canvas = make("canvas");
  canvas.width = canvas.height = zoom ? CROP : source.width;
  const ctx = canvas.getContext("2d")!;
  if (zoom) ctx.drawImage(source, (source.width - CROP) / 2, (source.width - CROP) / 2, CROP, CROP, 0, 0, CROP, CROP);
  else ctx.drawImage(source, 0, 0);
  box.append(canvas, make("span", "chip-label", label));
  if (crop) {
    const frame = make("div", "crop-frame");
    frame.style.inset = `${((1 - CROP / 256) / 2) * 100}%`;
    box.append(frame);
  }
  return box;
}
function table(head: string[], rows: string[][], compact = true) {
  const node = make("table", `table${compact ? " compact" : ""}`);
  const tr = make("tr");
  head.forEach((h, i) => tr.append(make("th", i ? "num" : "", h)));
  const thead = make("thead"), tbody = make("tbody");
  thead.append(tr);
  for (const row of rows) {
    const r = make("tr");
    row.forEach((v, i) => r.append(make("td", i ? "num" : "", v)));
    tbody.append(r);
  }
  node.append(thead, tbody);
  return node;
}
const viewerOf = (...children: Node[]) => { const v = make("div", "viewer"); v.append(...children); return v; };
const enhancementLegend = () => {
  const legend = make("div", "legend");
  legend.append(make("span", "", "Pierde contraste"), make("div", "ramp"), make("span", "", "Gana contraste"));
  const foot = make("div", "viewer-foot");
  foot.append(legend);
  return foot;
};

export function buildStudy(root: HTMLElement, index: HTMLElement, views: HTMLCanvasElement[]) {
  const add = (...nodes: Node[]) => root.append(...nodes);
  const head = (n: number) => sectionHead(`estudio-${SECTIONS[n][0]}`, n + 1, SECTIONS[n][1]);

  /* ---------- 1. Problem ---------- */
  add(head(0));
  const opening = make("article", "panel study reveal");
  const heroCanvas = make("canvas", "hero-canvas");
  heroCanvas.setAttribute("role", "img");
  heroCanvas.setAttribute("aria-label", "Animación: se adquiere la fase PRE de un corte de test, llega el contraste y se muestra el realce, se recorta la región central y la señal atraviesa las cinco etapas de la red hasta la probabilidad de pCR.");
  const openingText = make("div", "titular");
  openingText.style.marginTop = "24px";
  openingText.append(
    make("h1", "page-title", "Estimación de la respuesta patológica completa a partir de una resonancia previa al tratamiento"),
    make("p", "", "Una red convolucional 2D entrenada desde cero lee un corte de DCE-MRI en sus tres fases y estima la probabilidad de que la paciente alcance una respuesta patológica completa tras la quimioterapia."),
  );
  opening.append(viewerOf(heroCanvas), openingText);
  add(opening);

  const timeline = svgOf("Línea temporal del tratamiento: diagnóstico, resonancia previa, quimioterapia neoadyuvante, cirugía y análisis patológico, que termina en pCR o no pCR");
  add(visualBlock({
    title: "Momento de la predicción",
    context: "Esquema del tratamiento neoadyuvante",
    figure: timeline,
    conclusion: "Se quiere estimar antes del tratamiento lo que hoy solo se conoce después de la cirugía.",
    detail: [
      "Algunas pacientes reciben quimioterapia antes de la cirugía (tratamiento neoadyuvante) para reducir el tumor. Después de operar, el patólogo examina el tejido extirpado: si no encuentra tumor invasivo residual, la respuesta es patológica completa (pCR).",
      "La pCR es un indicador temprano de buena respuesta, no una garantía de curación.",
    ],
  }).panel);

  const dce = make("div", "visual-figure");
  const phases = make("div", "tiles");
  phases.style.gridTemplateColumns = "repeat(4, minmax(0, 1fr))";
  phases.append(...["PRE", "EARLY", "LATE", "Realce"].map((name, k) => tile(views[k], name)));
  const curveTypes = svgOf("Esquema de las tres formas de la curva de captación: persistente, meseta y lavado");
  dce.append(viewerOf(phases, enhancementLegend()), curveTypes);
  add(visualBlock({
    title: "Qué muestra la DCE-MRI",
    context: exampleContext,
    figure: dce,
    conclusion: "El contraste entra y sale de cada tejido a ritmos distintos; la red recibe las tres fases como canales.",
    detail: [
      "PRE se adquiere antes de inyectar el contraste; EARLY, poco después; LATE, minutos después. Los tejidos con muchos vasos, como muchos tumores, lo captan pronto y brillan en EARLY. El mapa de realce colorea EARLY − PRE.",
      "Si en LATE la intensidad sigue subiendo, la curva es persistente; si apenas cambia, meseta; si baja, lavado (washout), un patrón frecuente en tumores malignos. A la red nadie le indica estos patrones.",
    ],
  }).panel);

  add(titular({
    title: "Una CNN 2D entrenada desde cero que estima la pCR a partir de un corte",
    body: [
      "Sin pesos preentrenados ni datos clínicos: la entrada es solo la imagen, en sus tres fases. La unidad estadística es la paciente, y sus cortes nunca se reparten entre entrenamiento y validación. Las decisiones se toman con validación cruzada de cinco folds; el test se reserva para la evaluación final.",
    ],
  }));

  /* ---------- 2. Data ---------- */
  add(head(1));
  const train = split.train as Patient[];
  const names = split.cohorts.map((c) => COHORTS[c]);
  const all = summarize(train);
  const rates = names.map((_, c) => { const own = summarize(train.filter((p) => p[2] === c)); return own.positives / own.patients; });
  const lo = rates.indexOf(Math.min(...rates)), hi = rates.indexOf(Math.max(...rates));
  const cohorts = svgOf("Pacientes por cohorte y clase en entrenamiento, y pacientes de test por cohorte");
  add(visualBlock({
    title: "Cohortes y reparto en entrenamiento y test",
    context: `Entrenamiento y test · por paciente · ${thousands(all.patients + split.test.patients)} pacientes`,
    figure: cohorts,
    conclusion: `La proporción de pCR va del ${percent(rates[lo])} % en ${names[lo]} al ${percent(rates[hi])} % en ${names[hi]}: la cohorte ya informa algo sobre la clase.`,
    detail: [
      `BreastDCEDL reúne resonancias de tres estudios. Su versión MinCrop recorta cada corte a 256 × 256 alrededor del tumor y conserva las fases PRE, EARLY y LATE; de cada paciente se usan unos diez cortes.`,
      `Entrenamiento: ${thousands(all.patients)} pacientes y ${thousands(all.slices)} cortes, ${pct(all.positives, all.patients)} con pCR. Test: ${split.test.patients} pacientes y ${thousands(split.test.slices)} cortes. ${split.excluded_patients} pacientes Duke quedaron excluidas porque faltaba su fase tardía.`,
      `Por la diferencia entre cohortes, los resultados se comparan con una referencia que solo conoce la cohorte de cada paciente.`,
    ],
  }).panel);

  const splitSvg = svgOf("Pacientes de entrenamiento por fold, coloreadas por clase, y pacientes de test");
  const folds = [0, 1, 2, 3, 4].map((f) => summarize(train.filter((p) => p[0] === f)));
  const foldRates = folds.map((f) => f.positives / f.patients);
  const weights = [0, 1, 2, 3, 4].map((f) => positiveWeight(summarize(train.filter((p) => p[0] !== f))));
  const splitBlock = visualBlock({
    title: "Reparto de pacientes en cinco folds",
    context: `Entrenamiento · por paciente · ${thousands(all.patients)} pacientes en 5 folds · pulsa un fold`,
    figure: splitSvg,
    conclusion: "Cada paciente pertenece a un único fold; la validación nunca contiene cortes de una paciente de entrenamiento.",
    detail: [
      `Cada fold actúa una vez como validación mientras los otros cuatro entrenan. La proporción de pCR por fold va del ${percent(Math.min(...foldRates))} al ${percent(Math.max(...foldRates))} %.`,
      `Si los cortes de una paciente estuvieran a ambos lados, la red podría reconocerla y la estimación sería optimista. Las pacientes de test aparecen en gris: su etiqueta solo interviene en la evaluación final.`,
    ],
  });
  add(splitBlock.panel);

  const read = make("div", "tiles");
  read.style.gridTemplateColumns = "repeat(4, minmax(0, 1fr))";
  const steps = [
    [tile(views[1], "PNG"), "Tres PNG de 8 bits, enteros 0–255"],
    [tile(views[1], "÷ 255"), "float32 [3, 256, 256] en [0, 1]"],
    [tile(views[1], "Recorte", true), `Región central de ${CROP} × ${CROP}`],
    [tile(views[1], `${CROP} × ${CROP}`, false, true), `[3, ${CROP}, ${CROP}]: lo que ve la red`],
  ] as const;
  for (const [node, caption] of steps) {
    const cell = make("div", "tool");
    cell.style.gap = "8px";
    cell.append(viewerOf(node), make("span", "context", caption));
    read.append(cell);
  }
  add(visualBlock({
    title: "Preprocesado de un corte",
    context: `${exampleContext} · fase EARLY`,
    figure: read,
    conclusion: "La red recibe las tres fases alineadas como canales, en escala value / 255 y sin normalización de color.",
    detail: [
      "El dataset ya recortó las intensidades a los percentiles 1 y 99 con una ventana común a las tres fases, de modo que EARLY − PRE mide realce. Normalizar cada fase por separado borraría esa diferencia.",
      `El recorte central forma parte de la red: la entrada sigue siendo [3, 256, 256] y el recorte es idéntico en entrenamiento e inferencia.`,
    ],
  }).panel);

  const augment = make("div", "tool");
  const grid = make("div", "tiles");
  grid.style.gridTemplateColumns = "repeat(4, minmax(0, 1fr))";
  const another = make("button", "btn small", "Otra muestra");
  another.style.justifySelf = "start";
  const drawAugment = () => {
    const samples = [0, 1, 2, 3].map(() => sampleAugmentation(t, 256));
    const nodes: HTMLElement[] = [];
    ["PRE", "EARLY", "LATE"].forEach((name, k) => samples.forEach((a, j) => nodes.push(tile(augmentPhase(views[k], a), j ? "" : name))));
    grid.replaceChildren(...nodes);
    for (const label of grid.querySelectorAll(".chip-label")) if (!label.textContent) label.remove();
  };
  another.addEventListener("click", drawAugment);
  augment.append(viewerOf(grid), another);
  add(visualBlock({
    title: "Aumentos geométricos",
    context: `Entrenamiento · rotación ±${t.rotation_degrees}°, escala ${decimal(t.scale_range[0], 1)}–${decimal(t.scale_range[1], 1)}, desplazamiento ≤ ${Math.round(t.max_translation * 100)} %, volteo p = ${decimal(t.horizontal_flip_probability, 1)}`,
    figure: augment,
    conclusion: "Solo transformaciones geométricas, idénticas en las tres fases y solo en entrenamiento.",
    detail: [
      "Cada columna es una transformación aleatoria aplicada a las tres fases a la vez, de modo que el realce EARLY − PRE sigue alineado. Los bordes que quedan sin imagen valen 0.",
      "No hay aumentos de color: las fases son instantes de tiempo, no canales RGB.",
    ],
  }).panel);

  /* ---------- 3. Model ---------- */
  add(head(2));
  const netSvg = svgOf("Arquitectura de la red, de las tres fases a la probabilidad de pCR");
  const STEPS = layerSteps(arch.stages, arch.kernel_size, 3, CROP, arch.models);
  const last = arch.stages[arch.stages.length - 1];
  const netBlock = visualBlock({
    wide: true,
    title: "Arquitectura por etapas",
    context: `Un modelo de los ${arch.models}, uno por fold · entrada [3, 256, 256] · pulsa una etapa`,
    figure: netSvg,
    conclusion: `Cinco etapas convolucionales: la resolución baja de ${arch.stages[0].size} a ${last.size / 2} píxeles mientras los canales suben de ${arch.stages[0].channels} a ${last.channels}; ${thousands(arch.params)} parámetros por modelo.`,
    detail: [
      `Cada etapa aplica ${arch.stages[0].convs} convoluciones ${arch.kernel_size} × ${arch.kernel_size}, cada una seguida de normalización por lotes y ReLU, y un max pooling 2 × 2 que reduce la resolución a la mitad.`,
      `Al final, el promedio global de los ${last.channels} mapas, dropout ${decimal(arch.dropout, 1)} solo en entrenamiento y una capa lineal producen un logit.`,
    ],
  });
  add(netBlock.panel);

  const posWeight = `${decimal(Math.min(...weights), 2)}–${decimal(Math.max(...weights), 2)}`;
  add(titular({
    title: t.loss === "BCE ponderada" ? "Entrenamiento con pérdida ponderada por clase" : "Entrenamiento con pérdida sin ponderar",
    body: [
      `BCEWithLogitsLoss sobre el logit de cada corte${t.loss === "BCE ponderada" ? `, con pos_weight = N0 / N1 calculado en cada fold solo con sus cortes de entrenamiento (${posWeight}): cada corte con pCR pesa más para compensar que haya menos` : ""}.`,
      `Optimizador ${t.optimizer === "adamw" ? "AdamW" : t.optimizer} con tasa de aprendizaje ${String(t.learning_rate).replace(".", ",")}, programa en coseno de ${t.lr_schedule_epochs} épocas detenido en la época ${t.epochs}, y media móvil exponencial de los pesos (${decimal(t.ema_decay)}).`,
    ],
  }));

  const ensembleSvg = svgOf("Probabilidades de los cinco modelos de fold para un corte de ejemplo, su media, la calibración y la comparación con el umbral");
  const e = example.expected;
  add(visualBlock({
    title: "Ensamblado y calibración",
    context: `${exampleContext} · salidas reales del modelo`,
    figure: ensembleSvg,
    conclusion: "La salida es la media de los cinco modelos, calibrada; el umbral se fija con la validación interna, sin test.",
    detail: [
      `La calibración de Platt ajusta σ(a · logit + b) sobre la media de las probabilidades fuera de fold de una validación anidada, para que el valor se pueda leer como probabilidad.`,
      `El umbral es el menor valor con especificidad de al menos ${decimal(manifest.model.min_specificity, 1)} en esa validación: ${decimal(SLICE_THRESHOLD)} por corte, que usa Analizar, y ${decimal(PATIENT_THRESHOLD)} por paciente, que usan los resultados.`,
    ],
  }).panel);

  /* ---------- 4. Experiments ---------- */
  add(head(3));
  const experiments = runs.experiments as Experiment[];
  const finalRun = experiments.find((x) => x.final)!;
  const cohortMean = mean(runs.cohort_baseline);
  const means = experiments.map((x) => mean(x.auc)), spreads = experiments.map((x) => std(x.auc));
  const best = Math.max(...means), gain = best - cohortMean, maxSpread = Math.max(...spreads);
  const labSvg = svgOf("AUC por paciente en validación de cada experimento, con el valor de cada fold");
  const labFigure = make("div", "visual-figure");
  const LAB_HINT = "Pasa el cursor por una fila para ver qué cambia respecto a la referencia; púlsala para ver sus folds.";
  const labLine = make("p", "detail", LAB_HINT);
  labLine.setAttribute("aria-live", "polite");
  labFigure.append(labSvg, labLine);
  const labBlock = visualBlock({
    title: "AUC por paciente en la validación interna",
    context: `Validación interna · por paciente · ${experiments.length} experimentos, 5 folds · pulsa una fila`,
    figure: labFigure,
    conclusion: `El mejor experimento obtiene ${decimal(best)}, ${decimal(gain, 2)} por encima de la referencia por cohorte` +
      (gain < maxSpread ? `; con desviaciones entre folds de hasta ${decimal(maxSpread, 2)}, los experimentos no se pueden ordenar con seguridad.` : "."),
    detail: [
      `Cada fila es un entrenamiento completo con validación cruzada; los puntos pequeños son los cinco folds y el grande, su media. El modelo final, ${finalRun.name}, obtiene ${decimal(mean(finalRun.auc))} ± ${decimal(std(finalRun.auc))}.`,
      `Cada fold se lee en una época elegida sin mirarlo (la que maximiza la curva media de los otros cuatro) o en la última, si las épocas son fijas. Elegir la mejor época del propio fold usaría la validación para decidir y para medir.`,
    ],
  });
  add(labBlock.panel);

  const epochSvg = svgOf("AUC por paciente en validación por fold y época, con la época elegida");
  const ep = results.epochs, histories = ep.folds as FoldHistory[];
  const common = Math.min(...histories.map((f) => f.epoch.length));
  const meanLoss = histories[0].epoch.slice(0, common).map((_, i) => mean(histories.map((f) => f.validation_loss[i])));
  const lowest = meanLoss.indexOf(Math.min(...meanLoss)) + 1;
  const chosenBy = ep.out_of_fold_epochs.filter((x: number) => x === ep.chosen).length;
  add(visualBlock({
    title: "Número de épocas del modelo final",
    context: `Validación interna · por paciente · ${ep.experiment}, 5 folds`,
    figure: epochSvg,
    conclusion: `Época ${ep.chosen}: la que maximiza la curva media de los otros cuatro folds en ${chosenBy} de 5 folds, sin usar test.`,
    detail: [
      `Los círculos blancos marcan, para cada fold, la época elegida con los otros cuatro. La pérdida de validación media es mínima en la época ${lowest} y después crece mientras la de entrenamiento sigue bajando: a partir de ahí la red se ajusta a particularidades de las pacientes de entrenamiento.`,
      `El modelo final repite la configuración y se detiene en la época ${ep.chosen}.`,
    ],
  }).panel);

  /* ---------- 5. Results ---------- */
  add(head(4));
  const earlier = results.earlier_test;
  add(titular({
    id: "estudio-dos-evaluaciones",
    title: "El test se evaluó dos veces",
    body: [
      `El primer modelo final, ${earlier.run.split("_").slice(1).join("_")}, obtuvo en test un AUC de ${decimal(earlier.auc.value)} (IC 95 % ${decimal(earlier.auc.lower)}–${decimal(earlier.auc.upper)}). Su umbral se corrigió después de esa evaluación.`,
      `Después se sustituyó por ${finalRun.name}, elegido solo con la validación cruzada. Su evaluación es la segunda consulta del mismo test: no es del todo independiente y no sirve para elegir entre los dos modelos.`,
    ],
  }));

  const aucSvg = svgOf("AUC por paciente en test de los dos modelos, con su intervalo de confianza, frente al azar y a la referencia por cohorte");
  const d = s.auc_difference;
  const chance = s.auc.lower <= 0.5, indistinct = d.lower <= 0;
  add(visualBlock({
    title: "AUC por paciente en test",
    context: `Test · por paciente · ${s.patients} pacientes, ${s.positives} pCR · IC 95 % por bootstrap`,
    figure: aucSvg,
    conclusion: `El modelo actual obtiene ${decimal(s.auc.value)}` + (chance && indistinct ? `: no se distingue del azar ni de la referencia por cohorte (${decimal(s.cohort_baseline_auc)}).` : chance ? ": su intervalo incluye el azar." : indistinct ? `: no se distingue de la referencia por cohorte (${decimal(s.cohort_baseline_auc)}).` : "."),
    detail: [
      "El AUC es la probabilidad de que una paciente con pCR reciba una puntuación mayor que una sin pCR: 0,5 equivale al azar y 1 a separarlas sin error.",
      `La referencia por cohorte puntúa a cada paciente con la proporción de pCR de su cohorte. Diferencia del modelo actual con ella: ${decimal(d.value)} (IC 95 % ${decimal(d.lower)} a ${decimal(d.upper)}).`,
    ],
  }).panel);

  const val = results.validation as number[][], test = results.test as number[][];
  const foldRoc = [0, 1, 2, 3, 4].map((f) => val.filter((p) => p[0] === f)).map((rows) => roc(rows.map((p) => p[1]), rows.map((p) => p[2])));
  const foldAuc = foldRoc.map(auc);
  const testRoc = roc(test.map((p) => p[0]), test.map((p) => p[1]));
  const testAuc = auc(testRoc);
  const point = rocPoint(test.map((p) => p[0]), test.map((p) => p[2]), PATIENT_THRESHOLD);
  const rocSvg = svgOf("Curvas ROC por paciente de los cinco folds de validación y de test, con el punto del umbral");
  const near = Math.max(testAuc, ...foldAuc) < 0.65;
  add(visualBlock({
    title: "Curvas ROC por paciente",
    context: "Validación interna (5 folds) y test · por paciente",
    figure: rocSvg,
    conclusion: near ? "Las seis curvas quedan cerca de la diagonal: la discriminación es escasa en validación y en test." : `AUC en test ${decimal(testAuc)}.`,
    detail: [
      "Cada curva recorre todos los umbrales posibles: sensibilidad frente a la proporción de pacientes sin pCR clasificadas como pCR. Una curva sobre la diagonal ordena a las pacientes al azar.",
      `El punto marca el umbral por paciente, ${decimal(PATIENT_THRESHOLD)}, fijado en validación con especificidad de al menos ${decimal(manifest.model.min_specificity, 1)}.`,
    ],
  }).panel);

  const byClass = (label: number) => test.filter((p) => p[0] === label).map((p) => p[2]);
  const distSvg = svgOf("Probabilidad calibrada de cada paciente de test por clase, con el umbral");
  add(visualBlock({
    title: "Probabilidad estimada por clase",
    context: `Test · por paciente · ${test.length} pacientes, ${byClass(1).length} pCR · umbral por paciente ${decimal(PATIENT_THRESHOLD)}`,
    figure: distSvg,
    conclusion: `Las dos clases ocupan casi el mismo intervalo: mediana ${decimal(median(byClass(0)), 2)} sin pCR y ${decimal(median(byClass(1)), 2)} con pCR.`,
    detail: [
      "Cada punto es una paciente: la media de la probabilidad calibrada de sus cortes según los cinco modelos. Los números a cada lado del umbral cuentan las pacientes de cada clase.",
    ],
  }).panel);

  const c = confusion(test.map((p) => p[0]), test.map((p) => p[2]), PATIENT_THRESHOLD);
  const sens = c.tp / (c.tp + c.fn), spec = c.tn / (c.tn + c.fp);
  const cm = make("div", "cm");
  const cell = (count: number, row: number) => {
    const share = count / row, node = make("div", "cell-cm");
    node.style.background = `rgba(21, 25, 29, ${0.06 + 0.8 * share})`;
    node.style.color = share > 0.45 ? "#fff" : "var(--ink)";
    node.append(make("b", "", String(count)), make("small", "", `${percent(share)} % de la fila`));
    return node;
  };
  const rowLabel = (name: string, positive: boolean) => { const r = make("span", "row"); const dot = make("i"); dot.style.background = `var(${positive ? "--pcr" : "--nopcr"})`; r.append(dot, name); return r; };
  cm.append(
    make("span"), make("span", "head", "Estimada no pCR"), make("span", "head", "Estimada pCR"),
    rowLabel("Real no pCR", false), cell(c.tn, c.tn + c.fp), cell(c.fp, c.tn + c.fp),
    rowLabel("Real pCR", true), cell(c.fn, c.tp + c.fn), cell(c.tp, c.tp + c.fn),
  );
  add(visualBlock({
    title: "Clasificación con el umbral",
    context: `Test · por paciente · umbral ${decimal(PATIENT_THRESHOLD)}`,
    figure: cm,
    conclusion: `Se detectan ${c.tp} de ${c.tp + c.fn} pacientes con pCR (sensibilidad ${decimal(sens, 2)}) y se clasifican bien ${c.tn} de ${c.tn + c.fp} sin pCR (especificidad ${decimal(spec, 2)}).`,
    detail: [
      `Exactitud equilibrada ${decimal((sens + spec) / 2, 2)}. El umbral prioriza no clasificar como pCR a quien no la alcanza; a cambio, la mayoría de las pacientes con pCR quedan por debajo.`,
    ],
  }).panel);

  /* ---------- Figures, drawn once their panels are in the page ---------- */
  drawTimeline(timeline);
  drawCurveTypes(curveTypes);
  drawCohorts(cohorts, names, train, split.test.by_cohort);
  const splitFigure = drawSplit(splitSvg, train, split.test, (fold) => {
    splitFigure.select(fold);
    const tr = summarize(train.filter((p) => p[0] !== fold)), va = folds[fold];
    splitBlock.select([
      make("b", "", `Fold ${fold + 1} como validación`),
      table(["", "Pacientes", "Cortes", "pCR"], [
        ["Entrenamiento", thousands(tr.patients), thousands(tr.slices), pct(tr.positives, tr.patients)],
        ["Validación", thousands(va.patients), thousands(va.slices), pct(va.positives, va.patients)],
      ]),
      `pos_weight = ${thousands(tr.slices - tr.positiveSlices)} / ${thousands(tr.positiveSlices)} = ${decimal(positiveWeight(tr), 2)}: cortes sin pCR entre cortes con pCR del entrenamiento de este fold.`,
    ]);
  });
  drawAugment();
  const network = drawNetwork(netSvg, arch.stages, CROP, () => {}, (stage) => {
    network.select(stage);
    const step = STEPS[stage + 1];
    netBlock.select([
      make("b", "", `${step.title}: ${thousands(step.params)} parámetros`),
      table(["Capa", "Salida", "Parámetros"], step.layers.map((l) => [l.name, l.shape, l.params ? thousands(l.params) : "—"])),
      `Campo receptivo de una unidad central: ${step.field} × ${step.field} píxeles${step.field > CROP ? ", más que el recorte" : ""}.`,
    ]);
  });
  network.setInput(views.slice(0, 3).map((v) => v.toDataURL()));
  drawEnsemble(ensembleSvg, e.fold_probabilities, e.probability[0], e.calibrated[0], SLICE_THRESHOLD);
  const lab = drawLab(labSvg, experiments, runs.reference, runs.cohort_baseline, (name) => {
    lab.select(name);
    const x = experiments.find((y) => y.name === name)!;
    const changes = describe(x.changes as Record<string, unknown>, runs.reference_settings as Record<string, unknown>);
    const list = make("ul");
    list.append(...(changes.length ? changes : ["Configuración de referencia"]).map((ch) => make("li", "", ch)));
    labBlock.select([
      make("b", "", x.name + (x.final ? " (modelo final)" : "")),
      list,
      table(["Fold", "AUC", "Mejor época del fold"], [
        ...x.auc.map((v, k) => [String(k + 1), decimal(v), decimal(x.best_auc[k])]),
        ["Media", `${decimal(mean(x.auc))} ± ${decimal(std(x.auc))}`, decimal(mean(x.best_auc))],
      ]),
      "La última columna elige la mejor época en el propio fold y es optimista.",
    ]);
  }, (name) => {
    if (!name) { labLine.textContent = LAB_HINT; return; }
    const x = experiments.find((y) => y.name === name)!;
    const changes = describe(x.changes as Record<string, unknown>, runs.reference_settings as Record<string, unknown>);
    labLine.textContent = `${name}: ${changes.length ? changes.join(", ") : "configuración de referencia"}.`;
  });
  drawEpochs(epochSvg, histories, ep.chosen, ep.out_of_fold_epochs);
  drawAuc(aucSvg, [
    { label: "Modelo actual", value: s.auc.value, lower: s.auc.lower, upper: s.auc.upper, primary: true },
    { label: "Modelo anterior", value: earlier.auc.value, lower: earlier.auc.lower, upper: earlier.auc.upper, primary: false },
  ], s.cohort_baseline_auc, [
    { label: "Modelo actual", value: d.value, lower: d.lower, upper: d.upper, primary: true },
    { label: "Modelo anterior", ...earlier.auc_difference, primary: false },
  ]);
  drawRoc(rocSvg, foldRoc, testRoc, point, {
    test: `test · AUC ${decimal(testAuc)}`,
    folds: `validación · AUC ${decimal(Math.min(...foldAuc))}–${decimal(Math.max(...foldAuc))}`,
    point: `umbral: sensibilidad ${decimal(point[1], 2)}, especificidad ${decimal(1 - point[0], 2)}`,
  });
  drawDistribution(distSvg, [
    { label: "no pCR", values: byClass(0), positive: false },
    { label: "pCR", values: byClass(1), positive: true },
  ], PATIENT_THRESHOLD);
  startHero(heroCanvas, {
    phases: views.slice(0, 3),
    enhancement: views[3],
    sampleId: example.sample_id,
    cohort: COHORTS[example.cohort],
    crop: CROP,
    stages: arch.stages,
    params: arch.params,
    models: arch.models,
    probability: e.calibrated[0],
  });

  fit(root);

  /* ---------- Section index with the section in view marked ---------- */
  const links = SECTIONS.map(([id, label], k) => {
    const a = make("a");
    a.href = `#estudio/${id}`;
    a.append(make("span", "", String(k + 1)), label);
    index.append(a);
    return a;
  });
  spy(links, SECTIONS.map(([id]) => document.getElementById(`estudio-${id}`)!));
}
