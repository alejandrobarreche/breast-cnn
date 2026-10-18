import "./style.css";
import manifest from "./manifest.json";
import { drawAuc } from "./auc";
import { loadModel, type Model, type Prediction } from "./model";
import { drawNetwork } from "./network";
import { checkHeader, checkNames, MAX_BYTES, PHASES, SIZE, toTensor, type Phase } from "./preprocess";
import { drawRuler } from "./ruler";
import { percent, thousands } from "./svg";

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
const VIEWS = ["PRE", "EARLY", "LATE", "Realce"];
const VIEW_TEXT = [
  "PRE: imagen adquirida antes de inyectar el contraste. Sirve de referencia para medir cuánto cambia cada tejido.",
  "EARLY: poco después de la inyección. Los tejidos con mucha vascularización, como suele serlo el tumor, captan contraste pronto y se ven más brillantes.",
  "LATE: más tarde. Compararla con EARLY indica si el tejido sigue captando contraste, lo mantiene o lo pierde (lavado).",
  "Realce: diferencia EARLY − PRE coloreada sobre PRE. La opacidad crece con la diferencia; donde no hay cambio se ve PRE.",
];

const state = {
  model: null as Model | null,
  pixels: null as Uint8ClampedArray[] | null,
  views: [] as HTMLCanvasElement[],
  view: 0,
  sampleId: null as string | null,
  example: null as Example | null,
  log: [] as Entry[],
};

/* ---------- Presentation and static sections ---------- */
function renderStats() {
  const items: [string, string][] = [
    [thousands(arch.params), "parámetros por modelo"],
    [String(arch.models), "modelos, uno por fold"],
    [comma(s.auc.value), `AUC en test · IC 95 % ${comma(s.auc.lower)} – ${comma(s.auc.upper)}`],
    [String(s.patients), `pacientes de test, ${s.positives} con pCR`],
  ];
  $("stats").replaceChildren(...items.map(([value, text]) => {
    const div = make("div");
    div.append(make("dd", "", value), make("dt", "", text));
    return div;
  }));
}

function renderSteps() {
  const stageParams = arch.stages.map((x) => x.channels).join(" → ");
  const steps: [string, string][] = [
    ["Entrada", `Las tres fases se apilan como tres canales de 256 × 256 con valores en [0, 1]. La red usa el recorte central de ${arch.input_crop} × ${arch.input_crop}: el dataset centra cada corte en el tumor.`],
    ["Extracción de características", `Cinco etapas de dos convoluciones 3 × 3, cada una con normalización por lotes y ReLU, seguidas de un max pooling 2 × 2. Los canales crecen ${stageParams} mientras la resolución baja de ${arch.input_crop} a ${arch.stages[arch.stages.length - 1].size}.`],
    ["Decisión", `Un promedio global por canal, dropout de ${comma(arch.dropout, 1)} y una capa lineal dan un logit por modelo. Cada uno se calibra con el método de Platt; la media de los cinco se compara con el umbral de ${percent(THRESHOLD)} %, elegido en validación para una especificidad de al menos ${comma(manifest.model.min_specificity, 2)}.`],
  ];
  $("steps").replaceChildren(...steps.map(([title, body], i) => {
    const li = make("li");
    li.append(make("span", "label", `Paso ${i + 1}`), make("h3", "", title), make("p", "", body));
    return li;
  }));
}

function renderSpec() {
  const sections: [string, [string, string][]][] = [
    ["Validación interna · corte", [["Cortes", `${thousands(v.n)} (${thousands(v.positives)} pCR)`], ["AUC", comma(v.auc!)], ["Sensibilidad", comma(v.sensitivity!)], ["Especificidad", comma(v.specificity!)], ["Exactitud equilibrada", comma(v.balanced_accuracy!)]]],
    ["Test · paciente", [["Pacientes", `${s.patients} (${s.positives} pCR)`], ["AUC", comma(s.auc.value)], ["IC 95 % (bootstrap)", `${comma(s.auc.lower)} – ${comma(s.auc.upper)}`], ["Solo cohorte", comma(s.cohort_baseline_auc)]]],
    ["Entrenamiento", [["Pérdida", t.loss], ["Optimizador", `${t.optimizer === "adamw" ? "AdamW" : t.optimizer}, lr ${String(t.learning_rate).replace(".", ",")}`], ["Épocas", `${t.epochs} fijas`], ["Rotación", `±${t.rotation_degrees}°`], ["Volteo horizontal", `p = ${comma(t.horizontal_flip_probability, 1)}`], ["Dropout", comma(arch.dropout, 1)]]],
    ["Modelo", [["Formato", "ONNX"], ["Modelos", "5 folds + Platt"], ["Tamaño", `${comma(manifest.model.bytes / 1e6, 1)} MB`], ["Experimento", t.experiment], ["SHA-256", manifest.model.sha256]]],
  ];
  $("spec").replaceChildren(...sections.map(([title, rows]) => {
    const card = make("article", "card spec-card");
    const dl = make("dl");
    for (const [k, value] of rows) dl.append(make("dt", "", k), make("dd", k === "SHA-256" ? "hash" : "", value));
    card.append(make("h3", "title", title), dl);
    return card;
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
$("net-readout").textContent = `${thousands(arch.params)} parámetros por modelo`;
const network = drawNetwork($("net") as unknown as SVGSVGElement, arch.stages, arch.input_crop ?? 256, arch.models, (label) => {
  $("net-readout").textContent = label ?? `${thousands(arch.params)} parámetros por modelo`;
});
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

/* ---------- Viewer ---------- */
const seg = $("seg");
const segButtons = [...seg.querySelectorAll<HTMLButtonElement>("button")];
function cropBox() {
  const box = make("div", "crop");
  const side = ((arch.input_crop ?? SIZE) / SIZE) * 100;
  box.style.inset = `${(100 - side) / 2}%`;
  box.append(make("span", "", `${arch.input_crop} × ${arch.input_crop}`));
  return box;
}
function drawViewer(errors: Partial<Record<string, string>> = {}) {
  const main = $("main-view"), thumbs = $("thumbs");
  main.replaceChildren();
  thumbs.replaceChildren();
  if (state.views.length) {
    const big = canvasOf(state.views[state.view].getContext("2d")!.getImageData(0, 0, SIZE, SIZE));
    big.setAttribute("aria-label", `Vista ${VIEWS[state.view]} del corte ${state.sampleId}`);
    main.append(big, cropBox());
  } else {
    main.append(make("div", "empty-view", "Sin corte cargado"));
  }
  VIEWS.forEach((name, i) => {
    const button = make("button", "thumb");
    button.setAttribute("aria-pressed", String(i === state.view));
    if (state.views[i]) button.append(state.views[i]);
    else button.append(make("div", "empty"));
    const caption = make("span", "label", name);
    button.append(caption);
    if (errors[name]) button.append(make("span", "err", errors[name]!));
    button.addEventListener("click", () => setView(i));
    thumbs.append(button);
  });
  $("legend").classList.toggle("on", state.view === 3);
  $("view-explain").textContent = VIEW_TEXT[state.view];
}
function setView(i: number) {
  state.view = i;
  segButtons.forEach((b, k) => b.setAttribute("aria-selected", String(k === i)));
  seg.style.setProperty("--i", String(i));
  drawViewer();
}
segButtons.forEach((b, i) => b.addEventListener("click", () => setView(i)));
seg.addEventListener("keydown", (e) => {
  const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
  if (!step) return;
  const next = (state.view + step + VIEWS.length) % VIEWS.length;
  setView(next);
  segButtons[next].focus();
});
($("crop-toggle") as HTMLInputElement).addEventListener("change", (e) => {
  $("main-view").classList.toggle("no-crop", !(e.target as HTMLInputElement).checked);
});

/* ---------- Result and log ---------- */
function showResult(prediction: Prediction | null, animate = true) {
  ruler.set(prediction?.foldCalibrated ?? null, prediction?.calibrated ?? null, animate);
  network.setOutputs(prediction?.foldCalibrated ?? null, prediction?.calibrated ?? null);
  const pill = $("pill"), prob = $("prob");
  if (!prediction) {
    prob.className = "prob idle";
    prob.textContent = state.pixels ? "Pulsa Predecir" : "Carga un corte";
    pill.className = "pill idle";
    pill.textContent = state.pixels ? "Lista" : "Sin muestra";
    $("reading").textContent = "";
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
  for (let i = 0; i < 5; i++) {
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
    `${Math.min(5, state.log.length)} de 5 · ` + (revealed.length ? `${hits} ${hits === 1 ? "acierto" : "aciertos"} de ${revealed.length} ${revealed.length === 1 ? "revelada" : "reveladas"}` : "sin etiquetas reveladas");
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
  $("viewer-sample").textContent = sampleId ?? "—";
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
  const image = make("img");
  image.src = `${BASE}${example.files.EARLY}`;
  image.alt = `Fase EARLY de ${example.sample_id}`;
  const meta = make("span", "meta");
  meta.append(make("span", "label", `${COHORTS[example.cohort]} · test`), make("span", "sid", example.sample_id));
  button.append(image, meta);
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
  state.log.push({ time: new Date(), sampleId: state.sampleId!, probability: prediction.calibrated, threshold: THRESHOLD, positive, label: null });
  renderLog();
  ($("reveal") as HTMLButtonElement).disabled = state.example == null;
});
$("reveal").addEventListener("click", () => {
  const entry = state.log[state.log.length - 1];
  if (!state.example || !entry) return;
  entry.label = state.example.label;
  $("revealed").textContent = `real ${entry.label ? "pCR" : "sin pCR"} · ${(entry.label === 1) === entry.positive ? "acierto" : "fallo"}`;
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

// The navigation marks the section in view.
const links = [...document.querySelectorAll<HTMLAnchorElement>(".nav a")];
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    for (const link of links) link.classList.toggle("on", link.hash === `#${entry.target.id}`);
  }
}, { rootMargin: "-45% 0px -50% 0px" });
for (const link of links) observer.observe(document.querySelector(link.hash)!);

/* ---------- Start: first example on screen, model loaded and checked against Python ---------- */
renderStats();
renderSteps();
renderSpec();
renderLimits();
renderLog();
drawViewer();
(async () => {
  const first = manifest.examples[0];
  setSlice(await loadExample(first), first.sample_id, first);
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
    chip.title = error instanceof Error ? error.message : String(error);
  }
  updateButtons();
})();
