import "./style.css";
import manifest from "./manifest.json";
import { apply as augmentPhase, sample as sampleAugmentation, type Augmentation } from "./augment";
import { drawAuc } from "./auc";
import { describe, drawCurves, drawLab, mean, std, type Experiment } from "./lab";
import { steps as layerSteps } from "./layers";
import { occlusion, type OcclusionMap } from "./occlusion";
import runs from "./runs.json";
import { loadModel, type Model, type Prediction } from "./model";
import { drawNetwork } from "./network";
import { checkHeader, checkNames, MAX_BYTES, PHASES, SIZE, toTensor, type Phase } from "./preprocess";
import { startHero } from "./hero";
import { drawRuler } from "./ruler";
import { drawSplit, positiveWeight, summarize, type Patient } from "./split";
import { percent, thousands } from "./svg";
import { drawUptake, kinetics, MIN_ENHANCEMENT, windowMean } from "./uptake";

type Example = (typeof manifest.examples)[number];
interface Entry { time: Date; sampleId: string; probability: number; threshold: number; positive: boolean; label: number | null }

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const make = <K extends keyof HTMLElementTagNameMap>(tag: K, className = "", content = "") => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content) node.textContent = content;
  return node;
};
const COHORTS: Record<string, string> = { duke: "Duke", spy1: "I-SPY1", spy2: "I-SPY2" };
const THRESHOLD = manifest.model.threshold;
const BASE = import.meta.env.BASE_URL;
const arch = manifest.architecture;
const t = manifest.training, v = manifest.validation_slice, s = manifest.test_patient;
const CROP = arch.input_crop ?? SIZE;
const comma = (x: number, d = 3) => x.toFixed(d).replace(".", ",");
// Name and corner note of the four views of the viewer
const VIEWS: [string, string][] = [["PRE", "antes del contraste"], ["EARLY", "fase inicial"], ["LATE", "fase tardía"], ["Realce", "EARLY − PRE"]];

const state = {
  model: null as Model | null,
  modelError: null as string | null,
  pixels: null as Uint8ClampedArray[] | null,
  views: [] as HTMLCanvasElement[],
  probe: null as { x: number; y: number } | null,
  auto: null as { x: number; y: number } | null,
  pinned: false,
  sampleId: null as string | null,
  example: null as Example | null,
  log: [] as Entry[],
};

/* ---------- Presentation and static sections ---------- */
function renderSpec() {
  const sections: [string, [string, string][]][] = [
    ["Validación interna, por corte", [["Cortes", `${thousands(v.n)} (${thousands(v.positives)} pCR)`], ["AUC", comma(v.auc!)], ["Sensibilidad", comma(v.sensitivity!)], ["Especificidad", comma(v.specificity!)], ["Exactitud equilibrada", comma(v.balanced_accuracy!)]]],
    ["Test, por paciente", [["Pacientes", `${s.patients} (${s.positives} pCR)`], ["AUC", comma(s.auc.value)], ["IC 95 % (bootstrap)", `${comma(s.auc.lower)} – ${comma(s.auc.upper)}`], ["Solo cohorte", comma(s.cohort_baseline_auc)]]],
    ["Entrenamiento", [["Pérdida", t.loss], ["Optimizador", `${t.optimizer === "adamw" ? "AdamW" : t.optimizer}, lr ${String(t.learning_rate).replace(".", ",")}`], ["Épocas", `${t.epochs} fijas`], ["Rotación", `±${t.rotation_degrees}°`], ["Volteo horizontal", `p = ${comma(t.horizontal_flip_probability, 1)}`], ["Dropout", comma(arch.dropout, 1)]]],
    ["Modelo", [["Formato", "ONNX"], ["Modelos", "5 folds + Platt"], ["Tamaño", `${comma(manifest.model.bytes / 1e6, 1)} MB`], ["Experimento", t.experiment], ["SHA-256", manifest.model.sha256]]],
  ];
  $("spec").replaceChildren(...sections.map(([title, rows]) => {
    const block = make("div", "spec-block");
    const dl = make("dl");
    for (const [k, value] of rows) dl.append(make("dt", "", k), make("dd", k === "SHA-256" ? "hash" : "", value));
    block.append(make("h3", "", title), dl);
    return block;
  }));
  $("auc-explain").textContent =
    `Un AUC de 0,5 equivale a ordenar a las pacientes al azar y 1 a separarlas sin error. ` +
    `El modelo obtiene ${comma(s.auc.value)}; el intervalo no incluye 0,5, pero sí la referencia que solo usa la tasa de pCR de cada cohorte (${comma(s.cohort_baseline_auc)}). ` +
    `La capacidad de discriminación es modesta.`;
}

function renderLimits() {
  const items = [
    `Discriminación modesta: AUC de ${comma(s.auc.value)} en test, con un intervalo de ${comma(s.auc.lower)} a ${comma(s.auc.upper)}.`,
    `Con el umbral elegido, la sensibilidad en validación es ${comma(v.sensitivity!)}: la mayoría de los casos con pCR se clasifican como «sin pCR».`,
    "Cada predicción usa un único corte 2D, no el volumen completo de la resonancia ni datos clínicos como el subtipo tumoral.",
    "Las cinco muestras de la defensa sirven para comprobar que la aplicación funciona con datos nuevos, no para estimar su rendimiento.",
    "No hay validación prospectiva ni externa: el resultado no debe influir en ninguna decisión clínica.",
  ];
  $("limits").replaceChildren(...items.map((text) => make("li", "", text)));
}

/* ---------- Figures ---------- */
const netReadout = `${thousands(arch.params)} parámetros por modelo. Pulsa una etapa para ver sus capas.`;
$("net-readout").textContent = netReadout;
const network = drawNetwork($("net") as unknown as SVGSVGElement, arch.stages, arch.input_crop ?? 256, arch.models, (label) => {
  $("net-readout").textContent = label ?? netReadout;
}, (stage) => selectStep(stage + 1));
const uptake = drawUptake($("uptake") as unknown as SVGSVGElement);
const ruler = drawRuler($("ruler") as unknown as SVGSVGElement, THRESHOLD, manifest.model.train_pcr_rate, arch.models);
drawAuc($("auc") as unknown as SVGSVGElement, s.auc.value, s.auc.lower, s.auc.upper, s.cohort_baseline_auc);
$("f-threshold").textContent = `${percent(THRESHOLD)} %`;

// ENHANCEMENT_CMAP of style.py: no-pCR teal, paper white, pCR coral, on [-1, 1].
const STOPS = [[0x23, 0x65, 0x6e], [0xfa, 0xfa, 0xf8], [0xdc, 0x78, 0x67]];
function enhancementColour(value: number): number[] {
  const u0 = (Math.max(-1, Math.min(1, value)) + 1) / 2;
  const [a, b, u] = u0 < 0.5 ? [STOPS[0], STOPS[1], u0 * 2] : [STOPS[1], STOPS[2], (u0 - 0.5) * 2];
  return a.map((c, k) => Math.round(c + (b[k] - c) * u));
}
function imageData(pixels: Uint8ClampedArray) { return new ImageData(new Uint8ClampedArray(pixels), SIZE, SIZE); }
// EARLY − PRE over the PRE phase: the colour of ENHANCEMENT_CMAP, with an opacity that grows
// with |EARLY − PRE| and is full from a difference of FULL_AT. Pixels that do not change keep PRE.
const FULL_AT = 0.25;
function enhancement(pre: Uint8ClampedArray, early: Uint8ClampedArray) {
  const out = new ImageData(SIZE, SIZE);
  for (let p = 0; p < SIZE * SIZE * 4; p += 4) {
    const value = (early[p] - pre[p]) / 255;
    const colour = enhancementColour(value);
    const alpha = Math.min(1, Math.abs(value) / FULL_AT);
    for (let c = 0; c < 3; c++) out.data[p + c] = Math.round(pre[p] * (1 - alpha) + colour[c] * alpha);
    out.data[p + 3] = 255;
  }
  return out;
}
function canvasOf(data: ImageData) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = SIZE;
  canvas.getContext("2d")!.putImageData(data, 0, 0);
  return canvas;
}

/* ---------- Data: cohorts and patient split ---------- */
const split = manifest.split;
const trainPatients = split.train as Patient[];
const COHORT_NAMES = split.cohorts.map((c) => COHORTS[c]);
const pct = (a: number, b: number) => `${percent(a / b)} %`;
function renderData() {
  const train = summarize(trainPatients);
  const total = trainPatients.length + split.test.patients;
  const perCohort = split.cohorts.map((_, c) => trainPatients.filter((p) => p[2] === c).length + split.test.by_cohort[c]);
  $("datos-lead").textContent =
    `BreastDCEDL reúne resonancias DCE de ${thousands(total)} pacientes de tres estudios: ` +
    COHORT_NAMES.map((name, c) => `${name} (${thousands(perCohort[c])})`).join(", ").replace(/, ([^,]*)$/, " e $1") +
    `. La versión MinCrop recorta cada corte a 256 × 256 alrededor del tumor y conserva las fases PRE, EARLY y LATE; de cada paciente se usan unos diez cortes. ` +
    `Hay ${thousands(train.patients)} pacientes de entrenamiento, de las que el ${pct(train.positives, train.patients)} alcanzó pCR, y ${split.test.patients} de test. ` +
    `${split.excluded_patients} pacientes Duke quedaron excluidas porque faltaba su fase tardía.`;
  const rows = COHORT_NAMES.map((name, c) => {
    const own = trainPatients.filter((p) => p[2] === c);
    const s = summarize(own);
    return [name, thousands(s.patients), pct(s.positives, s.patients), String(split.test.by_cohort[c])];
  });
  table($("cohort-table") as HTMLTableElement, ["Cohorte", "Train", "pCR en train", "Test"], rows);
}
function table(node: HTMLTableElement, head: string[], rows: string[][]) {
  const thead = make("thead"), tr = make("tr");
  head.forEach((h, i) => tr.append(make("th", i ? "num" : "", h)));
  thead.append(tr);
  const tbody = make("tbody");
  for (const row of rows) {
    const r = make("tr");
    row.forEach((v, i) => r.append(make("td", i ? "num" : "", v)));
    tbody.append(r);
  }
  node.replaceChildren(thead, tbody);
}
const splitFigure = drawSplit($("split") as unknown as SVGSVGElement, trainPatients, split.test, (fold) => selectFold(fold));
function selectFold(fold: number) {
  splitFigure.select(fold);
  const train = summarize(trainPatients.filter((p) => p[0] !== fold));
  const valid = summarize(trainPatients.filter((p) => p[0] === fold));
  table($("fold-table") as HTMLTableElement, [`Fold ${fold + 1} como validación`, "Pacientes", "Cortes", "pCR"], [
    ["Entrenamiento", thousands(train.patients), thousands(train.slices), pct(train.positives, train.patients)],
    ["Validación", thousands(valid.patients), thousands(valid.slices), pct(valid.positives, valid.patients)],
  ]);
  $("fold-weight").textContent =
    `Peso de la clase positiva en la pérdida: pos_weight = N0 / N1 = ${thousands(train.slices - train.positiveSlices)} / ${thousands(train.positiveSlices)} = ${comma(positiveWeight(train), 2)}, ` +
    `calculado con los cortes de entrenamiento de este fold. Cada corte con pCR pesa ${comma(positiveWeight(train), 2)} veces más que uno sin pCR, lo que compensa que haya menos.`;
}

/* ---------- Data: preprocessing of the loaded slice ---------- */
let augmentation: Augmentation = sampleAugmentation(t, SIZE);
function tile(canvas: HTMLCanvasElement, label: string, crop = false) {
  const box = make("div", "tile");
  const copy = make("canvas");
  copy.width = copy.height = SIZE;
  copy.getContext("2d")!.drawImage(canvas, 0, 0);
  box.append(copy, make("span", "corner tl", label));
  if (crop) {
    const frame = make("div", "crop");
    frame.style.inset = `${((1 - CROP / SIZE) / 2) * 100}%`;
    box.append(frame);
  }
  return box;
}
function drawPipeline() {
  if (!state.pixels) return;
  const [pre, early, late] = state.views;
  $("pipe-read").replaceChildren(tile(pre, "PRE"), tile(early, "EARLY"), tile(late, "LATE"));
  const ranges = state.pixels.map((p) => { let lo = 255, hi = 0; for (let i = 0; i < p.length; i += 4) { lo = Math.min(lo, p[i]); hi = Math.max(hi, p[i]); } return [lo, hi]; });
  $("pipe-read-text").textContent =
    `Tres PNG en escala de grises de 8 bits, enteros de 0 a 255 (en este corte: ${ranges.map(([lo, hi], i) => `${PHASES[i]} ${lo}–${hi}`).join(", ")}). ` +
    `El dataset ya recortó las intensidades a los percentiles 1 y 99 con una ventana común para las tres fases, así que EARLY − PRE mide realce.`;
  const means = state.pixels.map((p) => windowMean(p, SIZE, SIZE / 2, SIZE / 2, SIZE));
  const scale = make("dl", "tensor");
  PHASES.forEach((phase, i) => scale.append(make("dt", "", `Canal ${i} (${phase})`), make("dd", "", `media ${comma(means[i], 3)}`)));
  scale.append(make("dt", "", "Forma"), make("dd", "", "[3, 256, 256]"), make("dt", "", "Tipo"), make("dd", "", "float32"));
  $("pipe-scale").replaceChildren(scale);
  drawAugmentation();
  $("pipe-crop").replaceChildren(tile(state.views[3], "Realce", true));
  $("pipe-crop-text").textContent =
    `La primera capa de la red conserva la región central de ${CROP} × ${CROP} píxeles, donde MinCrop sitúa el tumor; el resto no se usa. ` +
    `Al estar dentro del modelo, la entrada sigue siendo [3, 256, 256] y el recorte es idéntico en entrenamiento e inferencia.`;
}
function drawAugmentation() {
  if (!state.pixels) return;
  const phases = state.views.slice(0, 3).map((c) => augmentPhase(c, augmentation));
  const data = phases.map((c) => c.getContext("2d")!.getImageData(0, 0, SIZE, SIZE).data);
  const enhanced = canvasOf(enhancement(data[0], data[1]));
  $("pipe-augment").replaceChildren(tile(phases[0], "PRE"), tile(phases[1], "EARLY"), tile(phases[2], "LATE"), tile(enhanced, "Realce"));
  const a = augmentation;
  const signed = (x: string) => x.replace("-", "−");
  $("augment-params").textContent =
    `Rotación ${signed(comma(a.angle, 1))}°, escala ${comma(a.scale, 2)}, desplazamiento (${signed(String(Math.round(a.dx)))}, ${signed(String(Math.round(a.dy)))}) px, volteo ${a.flip ? "sí" : "no"}`;
  $("pipe-augment-text").textContent =
    `En cada época, cada corte se gira hasta ±${t.rotation_degrees}°, se escala entre ${comma(t.scale_range[0], 1)} y ${comma(t.scale_range[1], 1)}, ` +
    `se desplaza hasta un ${Math.round(t.max_translation * 100)} % y se voltea con probabilidad ${comma(t.horizontal_flip_probability, 1)}. ` +
    `La misma transformación se aplica a las tres fases, de modo que el mapa de realce sigue alineado. Los bordes que quedan sin imagen valen 0.`;
}
$("augment").addEventListener("click", () => { augmentation = sampleAugmentation(t, SIZE); drawAugmentation(); });

/* ---------- Network: step-by-step layers and receptive field ---------- */
const STEPS = layerSteps(arch.stages, arch.kernel_size, 3, CROP, arch.models);
const STEP_TEXT = [
  `La red recibe las tres fases como canales. Su primera operación recorta el centro de ${CROP} × ${CROP}; no tiene parámetros.`,
  ...arch.stages.map((stage, i) =>
    `${["Una convolución", "Dos convoluciones", "Tres convoluciones"][stage.convs - 1] ?? `${stage.convs} convoluciones`} ${arch.kernel_size} × ${arch.kernel_size} combinan cada píxel con sus vecinos en los ${i ? arch.stages[i - 1].channels : 3} canales de entrada y producen ${stage.channels} mapas de características. ` +
    `La normalización por lotes estabiliza la escala de las activaciones y ReLU anula los valores negativos. El max pooling reduce la resolución a la mitad, de ${stage.size} a ${stage.size / 2} píxeles, ` +
    `y cada unidad pasa a depender de una zona mayor de la imagen.`),
  `El promedio global resume cada uno de los ${arch.stages[arch.stages.length - 1].channels} mapas en un número. Durante el entrenamiento, el dropout anula al azar el ${Math.round(arch.dropout * 100)} % de esos valores para reducir el sobreajuste; en inferencia no actúa. Una capa lineal los combina en un logit.`,
  `Cada uno de los ${arch.models} modelos (uno por fold) convierte su logit en probabilidad con una recta de Platt ajustada en su validación, de modo que el valor se parezca a la frecuencia real de pCR. La salida es la media de las ${arch.models} probabilidades; el umbral de ${percent(THRESHOLD)} % decide la clase.`,
];
const stepButtons = STEPS.map((step, i) => {
  const button = make("button", "step-tab");
  button.setAttribute("role", "tab");
  button.append(make("span", "step-n", String(i + 1)), make("span", "", step.title));
  button.addEventListener("click", () => selectStep(i));
  $("step-list").append(button);
  return button;
});
let currentStep = 0;
$("step-list").addEventListener("keydown", (e) => {
  const move = e.key === "ArrowDown" ? 1 : e.key === "ArrowUp" ? -1 : 0;
  if (!move) return;
  e.preventDefault();
  selectStep((currentStep + move + STEPS.length) % STEPS.length);
  stepButtons[currentStep].focus();
});
function selectStep(i: number) {
  currentStep = i;
  stepButtons.forEach((b, k) => { b.setAttribute("aria-selected", String(k === i)); b.tabIndex = k === i ? 0 : -1; });
  const step = STEPS[i];
  $("step-title").textContent = step.title;
  $("step-explain").textContent = STEP_TEXT[i];
  const rows = step.layers.map((l) => [l.name, l.shape, l.params ? thousands(l.params) : "—"]);
  if (step.layers.length > 1 && step.params) rows.push(["Total", "", thousands(step.params)]);
  table($("step-table") as HTMLTableElement, ["Capa", "Salida", "Parámetros"], rows);
  drawField();
}
// Receptive field of one unit at the centre of the crop, over the crop of the loaded slice
function drawField() {
  const step = STEPS[currentStep];
  const canvas = $("field") as HTMLCanvasElement, ctx = canvas.getContext("2d")!;
  const k = canvas.width / CROP, o = (SIZE - CROP) / 2;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (state.views[0]) ctx.drawImage(state.views[0], o, o, CROP, CROP, 0, 0, canvas.width, canvas.height);
  const cells = step.outSize;
  if (currentStep > 0 && cells > 1) {
    ctx.strokeStyle = "rgba(232, 228, 218, 0.22)";
    ctx.lineWidth = 1;
    const cell = canvas.width / cells;
    ctx.beginPath();
    for (let c = 1; c < cells; c++) { ctx.moveTo(c * cell, 0); ctx.lineTo(c * cell, canvas.height); ctx.moveTo(0, c * cell); ctx.lineTo(canvas.width, c * cell); }
    ctx.stroke();
  }
  const side = Math.min(step.field, CROP) * k, x = (canvas.width - side) / 2;
  if (currentStep > 0) {
    ctx.fillStyle = "rgba(255, 252, 244, 0.16)";
    ctx.fillRect(x, x, side, side);
    ctx.strokeStyle = "rgba(255, 252, 244, 0.95)";
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 1, x + 1, side - 2, side - 2);
  }
  $("field-caption").textContent = currentStep === 0
    ? `Recorte central de ${CROP} × ${CROP} que recibe la red (fase PRE).`
    : currentStep <= arch.stages.length
      ? `Cuadrado: campo receptivo teórico de una unidad central, ${step.field} × ${step.field} píxeles${step.field > CROP ? `, mayor que el recorte: cada unidad puede depender de toda la imagen` : ""}. Rejilla: las ${cells} × ${cells} posiciones de salida de la etapa.`
      : `Tras el promedio global, la decisión depende de todo el recorte de ${CROP} × ${CROP}.`;
}

/* ---------- Lab: every training run on the internal validation ---------- */
const experiments = runs.experiments as Experiment[];
const cohortBaseline = mean(runs.cohort_baseline);
const finalRun = experiments.find((e) => e.final)!;
const labFigure = drawLab($("lab") as unknown as SVGSVGElement, experiments, cohortBaseline, (name) => selectExperiment(name));
function renderLab() {
  const honest = experiments.map((e) => mean(e.auc));
  const optimism = experiments.map((e) => mean(e.best_auc) - mean(e.auc));
  const spread = experiments.map((e) => std(e.auc));
  $("lab-lead").textContent =
    `${experiments.length} entrenamientos con validación cruzada por paciente en cinco folds. Cada uno cambia uno o varios factores respecto a la referencia (${runs.reference}). ` +
    `La métrica es el AUC por paciente: la probabilidad de que una paciente con pCR reciba una puntuación mayor que una sin pCR.`;
  const notes: [string, string][] = [
    ["Lo que permiten concluir",
      `Los experimentos obtienen entre ${comma(Math.min(...honest))} y ${comma(Math.max(...honest))}, y la desviación entre folds de cada uno va de ${comma(Math.min(...spread), 2)} a ${comma(Math.max(...spread), 2)}. ` +
      `Son diferencias del mismo orden que la variación entre folds: con cinco folds no se puede ordenar los experimentos con seguridad. ` +
      `Los mejores superan en unas ${comma(Math.max(...honest) - cohortBaseline, 2)} a la referencia que solo conoce la cohorte (${comma(cohortBaseline)}).`],
    ["Por qué no se elige la mejor época",
      `Elegir en cada fold la época con mayor AUC en ese mismo fold usa la validación para decidir y para medir. En estos experimentos eso añade entre ${comma(Math.min(...optimism), 2)} y ${comma(Math.max(...optimism), 2)} al AUC. ` +
      `Por eso cada fold se lee en la época que maximiza la curva media de los otros cuatro.`],
    ["Modelo final",
      `${finalRun.name} repite wide_crop128_slow (canales 16 → 256, recorte 128, aumento y tasa 0,0001) con 3 épocas fijas, la época que maximizaba la curva media de los otros folds, y guarda la última. ` +
      `Al reentrenarlo obtuvo ${comma(mean(finalRun.auc))}: el pico de ${comma(mean(experiments.find((e) => e.name === "wide_crop128_slow")!.auc))} no se reprodujo. ` +
      `El conjunto de test no intervino en ninguna de estas decisiones.`],
  ];
  $("lab-notes").replaceChildren(...notes.map(([title, body]) => {
    const div = make("div");
    div.append(make("h3", "", title), make("p", "", body));
    return div;
  }));
}
function selectExperiment(name: string) {
  const e = experiments.find((x) => x.name === name)!;
  labFigure.select(name);
  $("exp-title").textContent = e.name + (e.final ? " (modelo final)" : "");
  const changes = describe(e.changes as Record<string, unknown>, runs.reference_settings as Record<string, unknown>);
  $("exp-changes").replaceChildren(...(changes.length ? changes : ["Referencia: BCE ponderada, AdamW con tasa 0,001, canales 8 → 128, imagen completa, sin aumento geométrico"]).map((c) => make("li", "", c)));
  table($("exp-folds") as HTMLTableElement, ["Fold", e.selection === "last" ? "Última época" : "Época sin mirar el fold", "Mejor época del fold"], [
    ...e.auc.map((v, k) => [String(k + 1), comma(v), comma(e.best_auc[k])]),
    ["Media", `${comma(mean(e.auc))} ± ${comma(std(e.auc), 3)}`, comma(mean(e.best_auc))],
  ]);
  drawCurves($("curves") as unknown as SVGSVGElement, e, cohortBaseline);
  const hasTrain = e.curves.train.some((v) => v != null);
  $("curves-caption").textContent =
    `AUC por paciente, media de los folds en cada época. ` +
    (hasTrain ? "La distancia entre entrenamiento (discontinua) y validación (continua) mide el sobreajuste. " : "Este run no registró el AUC de entrenamiento. ") +
    "La línea de puntos es la referencia por cohorte.";
}

/* ---------- Occlusion map of the loaded slice ---------- */
// One inference at a time on the session: Predict waits while the map is computed
let occlusionRun = 0, occluding = false;
let occlusionMap: OcclusionMap | null = null;
function drawOcclusion() {
  const canvas = $("occlusion") as HTMLCanvasElement, ctx = canvas.getContext("2d")!;
  const o = (SIZE - CROP) / 2, k = canvas.width / CROP;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (state.views[0]) ctx.drawImage(state.views[0], o, o, CROP, CROP, 0, 0, canvas.width, canvas.height);
  const map = occlusionMap;
  if (!map) return;
  const top = Math.max(1e-6, ...Array.from(map.deltas, Math.abs));
  for (let i = 0; i < map.deltas.length; i++) {
    const d = map.deltas[i];
    // Covering a region that lowers P(pCR) means that region pushed towards pCR: coral
    const [r, g, b] = enhancementColour(-d / top);
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.15 + 0.6 * Math.abs(d) / top})`;
    ctx.fillRect((i % map.size) * map.patch * k, Math.floor(i / map.size) * map.patch * k, map.patch * k, map.patch * k);
  }
}
function resetOcclusion() {
  occlusionRun++;
  occlusionMap = null;
  ($("occlusion-run") as HTMLButtonElement).textContent = "Calcular mapa";
  $("occlusion-status").textContent = "";
  drawOcclusion();
}
$("occlusion-run").addEventListener("click", async () => {
  const button = $("occlusion-run") as HTMLButtonElement;
  if (button.textContent === "Cancelar") { resetOcclusion(); $("occlusion-status").textContent = "Cancelado."; return; }
  if (!state.model || !state.pixels) return;
  const run = ++occlusionRun, start = performance.now();
  button.textContent = "Cancelar";
  occluding = true;
  updateButtons();
  occlusionMap = null;
  const model = state.model;
  const map = await occlusion(toTensor(state.pixels), SIZE, CROP, 16, async (t) => (await model.predict(t)).calibrated,
    (done, total) => { if (run === occlusionRun) $("occlusion-status").textContent = `Calculando: ${done} de ${total}`; },
    () => run !== occlusionRun);
  occluding = false;
  updateButtons();
  if (!map || run !== occlusionRun) return;
  occlusionMap = map;
  button.textContent = "Calcular mapa";
  const top = Math.max(...Array.from(map.deltas, Math.abs));
  $("occlusion-status").textContent =
    `${map.deltas.length} inferencias en ${comma((performance.now() - start) / 1000, 1)} s. Probabilidad sin tapar ${percent(map.base)} %; el mayor cambio al tapar un cuadro es de ${comma(top * 100, 1)} puntos.`;
  drawOcclusion();
});

/* ---------- Viewer: four views with a shared measuring point ---------- */
interface Cell { canvas: HTMLCanvasElement; read: HTMLElement; error: HTMLElement; cross: HTMLElement }
const quad = $("quad");
const cells: Cell[] = VIEWS.map(([name, note], i) => {
  const cell = make("div", "cell");
  const canvas = make("canvas");
  canvas.width = canvas.height = SIZE;
  const cross = make("div", "cross");
  const crop = make("div", "crop");
  crop.style.inset = `${((1 - CROP / SIZE) / 2) * 100}%`;
  const read = make("span", "corner br");
  const error = make("span", "cell-error");
  cell.append(canvas, crop, cross, make("span", "corner tl", name), make("span", "corner tr", note), read, error);
  if (i === 0) cell.append(make("span", "corner bl slice-id", ""));
  quad.append(cell);
  return { canvas, read, error, cross };
});

const comma2 = (x: number) => x.toFixed(2).replace(".", ",");
// Brightest 5 x 5 enhancement inside the central crop: the starting point of the curve
function strongestEnhancement(pre: Uint8ClampedArray, early: Uint8ClampedArray) {
  const lo = (SIZE - CROP) / 2 + 2, hi = (SIZE + CROP) / 2 - 3;
  let best = { x: SIZE / 2, y: SIZE / 2 }, top = -Infinity;
  for (let y = lo; y <= hi; y += 2)
    for (let x = lo; x <= hi; x += 2) {
      const d = windowMean(early, SIZE, x, y) - windowMean(pre, SIZE, x, y);
      if (d > top) { top = d; best = { x, y }; }
    }
  return best;
}
function cropMean(pixels: Uint8ClampedArray) {
  const r = CROP / 2 - 1;
  return windowMean(pixels, SIZE, SIZE / 2, SIZE / 2, r);
}

function drawViewer(errors: Partial<Record<string, string>> = {}) {
  cells.forEach((cell, i) => {
    const context = cell.canvas.getContext("2d")!;
    context.fillStyle = "#000";
    context.fillRect(0, 0, SIZE, SIZE);
    if (state.views[i]) context.drawImage(state.views[i], 0, 0);
    cell.canvas.setAttribute("aria-label", `${VIEWS[i][0]} del corte ${state.sampleId ?? "sin cargar"}`);
    cell.error.textContent = errors[VIEWS[i][0]] ?? "";
  });
  quad.classList.toggle("empty", !state.views.length);
  quad.querySelector(".slice-id")!.textContent = state.sampleId ?? "Sin corte cargado";
  drawProbe();
}

function drawProbe() {
  const probe = state.probe;
  const pixels = state.pixels;
  cells.forEach((cell) => {
    cell.cross.style.opacity = probe && pixels ? "1" : "0";
    if (probe) cell.cross.style.setProperty("--x", `${((probe.x + 0.5) / SIZE) * 100}%`), cell.cross.style.setProperty("--y", `${((probe.y + 0.5) / SIZE) * 100}%`);
  });
  quad.classList.toggle("pinned", state.pinned);
  ($("probe-reset") as HTMLButtonElement).disabled = !pixels || probe === state.auto;
  if (!probe || !pixels) {
    cells.forEach((cell) => (cell.read.textContent = ""));
    uptake.set(null, null);
    $("probe-read").textContent = pixels ? "" : "Carga un corte para medir la captación de contraste.";
    return;
  }
  const values = pixels.map((p) => windowMean(p, SIZE, probe.x, probe.y));
  cells.forEach((cell, i) => (cell.read.textContent = i < 3 ? comma2(values[i]) : `${values[1] - values[0] >= 0 ? "+" : "−"}${comma2(Math.abs(values[1] - values[0]))}`));
  uptake.set(values, pixels.map(cropMean));
  const { pattern, delayed } = kinetics(values[0], values[1], values[2]);
  const where = state.pinned
    ? `Punto fijado en (${probe.x}, ${probe.y}). Haz clic de nuevo para soltarlo.`
    : probe === state.auto
      ? `Punto de mayor realce del recorte central, en (${probe.x}, ${probe.y}). Pasa el cursor por la imagen para medir otro.`
      : `Punto (${probe.x}, ${probe.y}). Haz clic para fijarlo.`;
  const change = `${delayed >= 0 ? "sube" : "baja"} un ${percent(Math.abs(delayed))} % respecto a EARLY`;
  const verdict =
    pattern === "sin realce"
      ? `Sin realce apreciable: EARLY supera a PRE en menos de ${comma2(MIN_ENHANCEMENT)}.`
      : `${pattern[0].toUpperCase()}${pattern.slice(1)}: en LATE la intensidad ${change}.`;
  $("probe-read").replaceChildren(
    make("span", "probe-values", `PRE ${comma2(values[0])}  →  EARLY ${comma2(values[1])}  →  LATE ${comma2(values[2])}`),
    make("b", "", verdict),
    make("span", "muted", where),
  );
}

function pointAt(event: PointerEvent) {
  const box = (event.target as HTMLElement).closest(".cell")!.getBoundingClientRect();
  const clamp = (v: number) => Math.max(0, Math.min(SIZE - 1, Math.floor(v * SIZE)));
  return { x: clamp((event.clientX - box.left) / box.width), y: clamp((event.clientY - box.top) / box.height) };
}
quad.addEventListener("pointermove", (event) => {
  if (state.pinned || !state.pixels || !(event.target as HTMLElement).closest(".cell")) return;
  state.probe = pointAt(event);
  drawProbe();
});
$("probe-reset").addEventListener("click", () => {
  state.probe = state.auto;
  state.pinned = false;
  drawProbe();
});
quad.addEventListener("click", (event) => {
  if (!state.pixels || !(event.target as HTMLElement).closest(".cell")) return;
  state.pinned = !state.pinned;
  state.probe = pointAt(event as PointerEvent);
  drawProbe();
});
quad.addEventListener("keydown", (event) => {
  const moves: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
  const move = moves[event.key];
  if (!move || !state.probe) return;
  event.preventDefault();
  const step = event.shiftKey ? 8 : 1;
  const clamp = (v: number) => Math.max(0, Math.min(SIZE - 1, v));
  state.probe = { x: clamp(state.probe.x + move[0] * step), y: clamp(state.probe.y + move[1] * step) };
  state.pinned = true;
  drawProbe();
});
($("crop-toggle") as HTMLInputElement).addEventListener("change", (e) => {
  quad.classList.toggle("no-crop", !(e.target as HTMLInputElement).checked);
});

/* ---------- Result and log ---------- */
function showResult(prediction: Prediction | null, animate = true) {
  ruler.set(prediction?.foldCalibrated ?? null, prediction?.calibrated ?? null, animate);
  network.setOutputs(prediction?.foldCalibrated ?? null, prediction?.calibrated ?? null);
  const pill = $("pill"), prob = $("prob");
  if (!prediction) {
    const ready = state.model != null;
    prob.className = "prob idle";
    prob.textContent = !ready ? (state.modelError ? "Modelo no disponible" : "Cargando el modelo") : state.pixels ? "Pulsa Predecir" : "Carga un corte";
    pill.className = "pill idle";
    pill.textContent = state.pixels ? "Lista" : "Sin muestra";
    $("reading").textContent = ready
      ? ""
      : state.modelError
        ? `No se pudo cargar el modelo: ${state.modelError} Recarga la página para intentarlo de nuevo.`
        : `Descargando y verificando el modelo (${comma(manifest.model.bytes / 1e6, 1)} MB). Predecir se activa cuando termine.`;
    $("f-spread").textContent = "—";
    $("f-time").textContent = "—";
    $("revealed").textContent = "";
    return;
  }
  const positive = prediction.calibrated >= THRESHOLD;
  prob.className = "prob";
  prob.innerHTML = `${percent(prediction.calibrated)}<small>%</small>`;
  pill.className = `pill ${positive ? "pos" : "neg"}`;
  pill.textContent = positive ? "pCR" : "Sin pCR";
  const f = prediction.foldCalibrated;
  const low = Math.min(...f), high = Math.max(...f);
  const split = low < THRESHOLD && high >= THRESHOLD;
  $("reading").textContent =
    `El modelo estima una probabilidad de pCR del ${percent(prediction.calibrated)} %, ` +
    `${positive ? "igual o superior" : "inferior"} al umbral de ${percent(THRESHOLD)} %. ` +
    (split
      ? `Los cinco modelos no coinciden en la clase (${percent(low)} – ${percent(high)} %): la estimación es incierta.`
      : `Los cinco modelos coinciden en la clase (${percent(low)} – ${percent(high)} %).`);
  $("f-spread").textContent = `${percent(low)} – ${percent(high)} %`;
  $("f-time").textContent = `${Math.max(1, Math.round(prediction.milliseconds))} ms`;
}

function renderLog() {
  const rows = $("rows");
  rows.replaceChildren();
  for (let i = 0; i < Math.max(5, state.log.length); i++) {
    const e = state.log[i];
    const tr = make("tr", e ? "" : "empty");
    const cell = (content = "", cls = "") => { const td = make("td", cls, content); tr.append(td); return td; };
    cell(String(i + 1), "n");
    if (!e) {
      cell("—", "sid"); cell("", "num"); cell("", "num"); cell(""); cell("");
    } else {
      cell(e.sampleId, "sid");
      cell(`${percent(e.probability)} %`, "num");
      cell(`${percent(e.threshold)} %`, "num muted");
      const cls = cell("");
      const tag = make("span", `tag ${e.positive ? "pos" : "neg"}`, e.positive ? "pCR" : "Sin pCR");
      cls.append(tag);
      const truth = cell("");
      if (e.label == null) truth.append(make("span", "muted", "oculta"));
      else {
        const hit = (e.label === 1) === e.positive;
        truth.append(make("span", "", e.label ? "pCR" : "Sin pCR"), make("span", `mark ${hit ? "hit" : "miss"}`, hit ? "acierto" : "fallo"));
      }
    }
    rows.append(tr);
  }
  const revealed = state.log.filter((e) => e.label != null);
  const hits = revealed.filter((e) => (e.label === 1) === e.positive).length;
  $("log-summary").textContent =
    `${state.log.length} de 5, ` + (revealed.length ? `${hits} ${hits === 1 ? "acierto" : "aciertos"} de ${revealed.length} ${revealed.length === 1 ? "revelada" : "reveladas"}` : "sin etiquetas reveladas");
}

function updateButtons() {
  ($("predict") as HTMLButtonElement).disabled = !(state.model && state.pixels) || occluding;
  ($("occlusion-run") as HTMLButtonElement).disabled = !(state.model && state.pixels);
  ($("clear") as HTMLButtonElement).disabled = !state.pixels;
}

/* ---------- Loading slices ---------- */
async function decode(blob: Blob): Promise<Uint8ClampedArray> {
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width; canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  return context.getImageData(0, 0, canvas.width, canvas.height).data;
}

async function loadExample(example: Example): Promise<Uint8ClampedArray[]> {
  const pixels = [];
  for (const phase of PHASES) pixels.push(await decode(await (await fetch(`${BASE}${example.files[phase]}`)).blob()));
  return pixels;
}

function setSlice(pixels: Uint8ClampedArray[] | null, sampleId: string | null, example: Example | null) {
  state.pixels = pixels; state.sampleId = sampleId; state.example = example;
  state.views = pixels ? [...pixels.map((p) => canvasOf(imageData(p))), canvasOf(enhancement(pixels[0], pixels[1]))] : [];
  network.setInput(pixels ? state.views.slice(0, 3).map((c) => c.toDataURL()) : null);
  state.auto = pixels ? strongestEnhancement(pixels[0], pixels[1]) : null;
  state.probe = state.auto;
  state.pinned = false;
  ($("reveal") as HTMLButtonElement).disabled = true;
  for (const card of document.querySelectorAll<HTMLElement>(".sample")) card.setAttribute("aria-current", String(card.dataset.id === sampleId && example != null));
  $("drop").classList.toggle("current", pixels != null && example == null);
  drawViewer();
  showResult(null);
  updateButtons();
  if (pixels) { drawPipeline(); drawField(); }
  resetOcclusion();
}

async function handleFiles(files: File[]) {
  const { byPhase, errors } = checkNames(files.map((f) => f.name));
  const onPlate: Partial<Record<string, string>> = {};
  const pixels: Uint8ClampedArray[] = [];
  for (const phase of PHASES) {
    const index = byPhase[phase as Phase];
    if (index === undefined) continue;
    const file = files[index];
    const fail = (message: string) => { errors.push(`${phase}: ${message}`); onPlate[phase] = message; };
    if (file.size > MAX_BYTES) { fail(`pesa ${(file.size / 1048576).toFixed(1).replace(".", ",")} MB; el máximo es 2 MB.`); continue; }
    const header = checkHeader(new Uint8Array(await file.slice(0, 33).arrayBuffer()));
    if (header) { fail(header); continue; }
    try { pixels[PHASES.indexOf(phase)] = await decode(file); }
    catch { fail("no se puede leer; el archivo puede estar dañado."); }
  }
  $("errors").replaceChildren(...errors.map((message) => make("li", "", message)));
  if (errors.length) { drawViewer(onPlate); return; }
  const name = files[byPhase.PRE!].name.replace(/_PRE\.png$/i, "");
  setSlice(pixels, name, null);
}

for (const example of manifest.examples) {
  const button = make("button", "sample");
  button.dataset.id = example.sample_id;
  button.setAttribute("role", "listitem");
  const strip = make("span", "strip");
  for (const phase of PHASES) {
    const image = make("img");
    image.src = `${BASE}${example.files[phase]}`;
    image.alt = "";
    strip.append(image);
  }
  const meta = make("span", "meta");
  meta.append(make("span", "cohort", COHORTS[example.cohort]), make("span", "sid", example.sample_id));
  button.setAttribute("aria-label", `Corte ${example.sample_id}, cohorte ${COHORTS[example.cohort]}, test`);
  button.append(strip, meta);
  button.addEventListener("click", async () => {
    $("errors").replaceChildren();
    setSlice(await loadExample(example), example.sample_id, example);
  });
  $("tray").append(button);
}

/* ---------- Actions ---------- */
$("predict").addEventListener("click", async () => {
  if (!state.model || !state.pixels) return;
  const prediction = await state.model.predict(toTensor(state.pixels));
  showResult(prediction);
  const positive = prediction.calibrated >= THRESHOLD;
  // One row per slice: predicting the same slice again replaces its row and keeps a revealed label
  const entry: Entry = { time: new Date(), sampleId: state.sampleId!, probability: prediction.calibrated, threshold: THRESHOLD, positive, label: null };
  const index = state.log.findIndex((e) => e.sampleId === entry.sampleId);
  if (index >= 0) state.log[index] = { ...entry, label: state.log[index].label };
  else state.log.push(entry);
  renderLog();
  ($("reveal") as HTMLButtonElement).disabled = state.example == null;
});
$("reveal").addEventListener("click", () => {
  const entry = state.log.find((e) => e.sampleId === state.sampleId);
  if (!state.example || !entry) return;
  entry.label = state.example.label;
  $("revealed").textContent = `Real: ${entry.label ? "pCR" : "sin pCR"}, ${(entry.label === 1) === entry.positive ? "acierto" : "fallo"}`;
  ($("reveal") as HTMLButtonElement).disabled = true;
  renderLog();
});
$("clear").addEventListener("click", () => setSlice(null, null, null));
$("copy-csv").addEventListener("click", async (event) => {
  const button = event.currentTarget as HTMLButtonElement;
  const rows = [["hora", "corte", "probabilidad", "umbral", "clase", "real"], ...state.log.map((e) => [e.time.toISOString(), e.sampleId, e.probability.toFixed(4), e.threshold.toFixed(4), e.positive ? 1 : 0, e.label ?? ""])];
  try { await navigator.clipboard.writeText(rows.map((r) => r.join(",")).join("\n")); button.textContent = "Copiado"; }
  catch { button.textContent = "No se pudo copiar"; }
  setTimeout(() => (button.textContent = "Copiar CSV"), 1600);
});
$("files").addEventListener("change", (e) => {
  const input = e.target as HTMLInputElement;
  handleFiles([...(input.files ?? [])]);
  input.value = "";
});
const catcher = $("catch");
let drags = 0;
addEventListener("dragenter", (e) => { if (e.dataTransfer?.types.includes("Files")) { drags++; catcher.classList.add("on"); } });
addEventListener("dragleave", () => { drags = Math.max(0, drags - 1); if (!drags) catcher.classList.remove("on"); });
addEventListener("dragover", (e) => e.preventDefault());
addEventListener("drop", (e) => { e.preventDefault(); drags = 0; catcher.classList.remove("on"); handleFiles([...(e.dataTransfer?.files ?? [])]); });

// Specular highlight of the glass buttons follows the pointer.
addEventListener("pointermove", (event) => {
  const button = (event.target as HTMLElement).closest<HTMLElement>(".btn");
  if (!button) return;
  const box = button.getBoundingClientRect();
  button.style.setProperty("--mx", `${event.clientX - box.left}px`);
  button.style.setProperty("--my", `${event.clientY - box.top}px`);
});

// The navigation marks the section in view; its glass thumb slides under that link and hides
// over the opening.
const links = [...document.querySelectorAll<HTMLAnchorElement>(".nav a")];
const thumb = document.querySelector<HTMLElement>(".nav-thumb")!;
function placeThumb() {
  const on = links.find((link) => link.classList.contains("on"));
  thumb.classList.toggle("on", on != null);
  if (!on) return;
  thumb.style.setProperty("--x", `${on.offsetLeft}px`);
  thumb.style.setProperty("--w", `${on.offsetWidth}px`);
}
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    const id = entry.target.id === "top" ? null : `#${entry.target.id}`;
    for (const link of links) link.classList.toggle("on", link.hash === id);
  }
  placeThumb();
}, { rootMargin: "-45% 0px -50% 0px" });
for (const link of links) observer.observe(document.querySelector(link.hash)!);
observer.observe($("top"));
document.fonts.ready.then(placeThumb);

/* ---------- Start: first example on screen, model loaded and checked against Python ---------- */
renderData();
selectFold(0);
renderLab();
selectExperiment(finalRun.name);
selectStep(0);
renderSpec();
renderLimits();
renderLog();
drawViewer();
(async () => {
  const first = manifest.examples[0];
  setSlice(await loadExample(first), first.sample_id, first);
  startHero($("hero-canvas") as HTMLCanvasElement, {
    phases: state.views.slice(0, 3),
    enhancement: state.views[3],
    sampleId: first.sample_id,
    cohort: COHORTS[first.cohort],
    crop: CROP,
    stages: arch.stages,
    params: arch.params,
    models: arch.models,
    probability: first.expected.calibrated[0],
  });
  const chip = $("model-chip"), stateLabel = $("model-state");
  try {
    const model = await loadModel(`${BASE}${manifest.model.file}`, manifest.model.sha256);
    // Parity in this browser: the first example must reproduce the outputs of Python.
    const check = await model.predict(toTensor(state.pixels!));
    const expected = first.expected;
    const difference = Math.max(
      ...check.foldProbabilities.map((value, k) => Math.abs(value - expected.fold_probabilities[k])),
      Math.abs(check.calibrated - expected.calibrated[0]),
    );
    if (difference > 1e-5) throw new Error(`paridad ${difference.toExponential(1)}`);
    state.model = model;
    chip.className = "chip ready";
    stateLabel.textContent = `${model.sha256.slice(0, 6)}…${model.sha256.slice(-4)}`;
    chip.title = `SHA-256 ${model.sha256} · diferencia máxima con Python ${difference.toExponential(1)}`;
  } catch (error) {
    chip.className = "chip error";
    stateLabel.textContent = "no disponible";
    state.modelError = error instanceof Error ? error.message : String(error);
    chip.title = state.modelError;
  }
  if ($("prob").classList.contains("idle")) showResult(null, false);
  updateButtons();
})();
