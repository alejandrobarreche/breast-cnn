import "./style.css";
import manifest from "./manifest.json";
import { drawAuc } from "./auc";
import { loadModel, type Model, type Prediction } from "./model";
import { drawNetwork } from "./network";
import { checkHeader, checkNames, MAX_BYTES, PHASES, SIZE, toTensor, type Phase } from "./preprocess";
import { startHero } from "./hero";
import { drawRuler } from "./ruler";
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
function renderSteps() {
  const stageParams = arch.stages.map((x) => x.channels).join(" → ");
  const steps: [string, string][] = [
    ["Entrada", `Las tres fases se apilan como tres canales de 256 × 256 con valores en [0, 1]. La red usa el recorte central de ${arch.input_crop} × ${arch.input_crop}: el dataset centra cada corte en el tumor.`],
    ["Extracción de características", `Cinco etapas de dos convoluciones 3 × 3, cada una con normalización por lotes y ReLU, seguidas de un max pooling 2 × 2. Los canales crecen ${stageParams} mientras la resolución baja de ${arch.input_crop} a ${arch.stages[arch.stages.length - 1].size}.`],
    ["Decisión", `Un promedio global por canal, dropout de ${comma(arch.dropout, 1)} y una capa lineal dan un logit por modelo. Cada uno se calibra con el método de Platt; la media de los cinco se compara con el umbral de ${percent(THRESHOLD)} %, elegido en validación para una especificidad de al menos ${comma(manifest.model.min_specificity, 2)}.`],
  ];
  $("steps").replaceChildren(...steps.map(([title, body]) => {
    const li = make("li");
    li.append(make("h3", "", title), make("p", "", body));
    return li;
  }));
}

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
const netReadout = `${thousands(arch.params)} parámetros por modelo. Pasa el cursor por una etapa para ver su tamaño.`;
$("net-readout").textContent = netReadout;
const network = drawNetwork($("net") as unknown as SVGSVGElement, arch.stages, arch.input_crop ?? 256, arch.models, (label) => {
  $("net-readout").textContent = label ?? netReadout;
});
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

/* ---------- Viewer: four views with a shared measuring point ---------- */
interface Cell { canvas: HTMLCanvasElement; read: HTMLElement; error: HTMLElement; cross: HTMLElement }
const quad = $("quad");
const CROP = arch.input_crop ?? SIZE;
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
  ($("predict") as HTMLButtonElement).disabled = !(state.model && state.pixels);
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
renderSteps();
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
