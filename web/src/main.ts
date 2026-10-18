import "./style.css";
import manifest from "./manifest.json";
import { loadModel, type Model, type Prediction } from "./model";
import { drawNetwork } from "./network";
import { checkHeader, checkNames, MAX_BYTES, PHASES, SIZE, toTensor, type Phase } from "./preprocess";
import { drawRuler } from "./ruler";
import { percent, thousands, token } from "./svg";

type Example = (typeof manifest.examples)[number];
interface Entry { time: Date; sampleId: string; probability: number; threshold: number; positive: boolean; label: number | null }

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const COHORTS: Record<string, string> = { duke: "Duke", spy1: "I-SPY1", spy2: "I-SPY2" };
const THRESHOLD = manifest.model.threshold;
const BASE = import.meta.env.BASE_URL;
const arch = manifest.architecture;

const state = {
  model: null as Model | null,
  pixels: null as Uint8ClampedArray[] | null,
  sampleId: null as string | null,
  example: null as Example | null,
  log: [] as Entry[],
};

/* ---------- Figures ---------- */
const totalParams = `${thousands(arch.params)} parámetros por modelo`;
$("net-readout").textContent = totalParams;
const network = drawNetwork($("net") as unknown as SVGSVGElement, arch.stages, arch.input_crop ?? 256, arch.models, (label) => {
  $("net-readout").textContent = label ?? totalParams;
});
const ruler = drawRuler($("ruler") as unknown as SVGSVGElement, THRESHOLD, manifest.model.train_pcr_rate, arch.models);
$("f-threshold").textContent = `${percent(THRESHOLD)} %`;

// ENHANCEMENT_CMAP of style.py: no-pCR teal, paper white, pCR coral, on [-1, 1].
const STOPS = [[0x23, 0x65, 0x6e], [0xfa, 0xfa, 0xf8], [0xdc, 0x78, 0x67]];
function enhancementColour(value: number): number[] {
  const t = (Math.max(-1, Math.min(1, value)) + 1) / 2;
  const [a, b, u] = t < 0.5 ? [STOPS[0], STOPS[1], t * 2] : [STOPS[1], STOPS[2], (t - 0.5) * 2];
  return a.map((c, k) => Math.round(c + (b[k] - c) * u));
}
function imageData(pixels: Uint8ClampedArray) { return new ImageData(new Uint8ClampedArray(pixels), SIZE, SIZE); }
function enhancement(pre: Uint8ClampedArray, early: Uint8ClampedArray) {
  const out = new ImageData(SIZE, SIZE);
  for (let p = 0; p < SIZE * SIZE * 4; p += 4) {
    const [r, g, b] = enhancementColour((early[p] - pre[p]) / 255);
    out.data[p] = r; out.data[p + 1] = g; out.data[p + 2] = b; out.data[p + 3] = 255;
  }
  return out;
}
function canvasOf(data: ImageData) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = SIZE;
  canvas.getContext("2d")!.putImageData(data, 0, 0);
  return canvas;
}

function drawPhases(errors: Partial<Record<string, string>> = {}) {
  const quad = $("quad");
  quad.replaceChildren();
  const views = state.pixels ? [...state.pixels.map(imageData), enhancement(state.pixels[0], state.pixels[1])] : [];
  ["PRE", "EARLY", "LATE", "EARLY − PRE"].forEach((name, i) => {
    const figure = document.createElement("figure");
    if (views[i]) figure.appendChild(canvasOf(views[i]));
    else { const empty = document.createElement("div"); empty.className = "empty"; figure.appendChild(empty); }
    if (errors[name]) { const e = document.createElement("div"); e.className = "err"; e.textContent = errors[name]!; figure.appendChild(e); }
    const caption = document.createElement("figcaption");
    caption.className = "label";
    caption.textContent = name;
    figure.appendChild(caption);
    quad.appendChild(figure);
  });
}

/* ---------- Result and log ---------- */
function showResult(prediction: Prediction | null, animate = true) {
  ruler.set(prediction?.foldCalibrated ?? null, prediction?.calibrated ?? null, animate);
  network.setOutputs(prediction?.foldCalibrated ?? null, prediction?.calibrated ?? null);
  const pill = $("pill");
  if (!prediction) {
    $("prob").innerHTML = "—<small>%</small>";
    pill.className = "pill idle";
    pill.textContent = state.pixels ? "Lista" : "Sin muestra";
    $("f-spread").textContent = "—";
    $("f-time").textContent = "—";
    $("revealed").textContent = "";
    return;
  }
  const positive = prediction.calibrated >= THRESHOLD;
  $("prob").innerHTML = `${percent(prediction.calibrated)}<small>%</small>`;
  pill.className = `pill ${positive ? "pos" : "neg"}`;
  pill.textContent = positive ? "pCR" : "Sin pCR";
  const f = prediction.foldCalibrated;
  $("f-spread").textContent = `${percent(Math.min(...f))} – ${percent(Math.max(...f))} %`;
  $("f-time").textContent = `WASM · ${Math.max(1, Math.round(prediction.milliseconds))} ms`;
}

function renderLog() {
  const rows = $("rows");
  rows.replaceChildren();
  for (let i = 0; i < 5; i++) {
    const e = state.log[i];
    const li = document.createElement("li");
    const cell = (cls: string, content = "") => { const s = document.createElement("span"); s.className = cls; s.textContent = content; li.appendChild(s); return s; };
    cell("n", String(i + 1));
    if (!e) {
      li.className = "empty";
      cell("sid", "—"); cell("p"); cell("truth");
    } else {
      cell("sid", e.sampleId);
      cell("p", `${percent(e.probability)} %`).style.color = token(e.positive ? "--pcr-ink" : "--nopcr-ink");
      const truth = e.label == null ? "real oculta" : `real ${e.label ? "pCR" : "sin pCR"} · ${(e.label === 1) === e.positive ? "acierto" : "fallo"}`;
      cell(`truth${e.label == null ? "" : " ok"}`, truth);
    }
    rows.appendChild(li);
  }
  $("log-count").textContent = `${Math.min(5, state.log.length)} / 5`;
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
  network.setInput(pixels ? pixels.map((p) => canvasOf(imageData(p)).toDataURL()) : null);
  $("net-sample").textContent = sampleId ?? "—";
  ($("reveal") as HTMLButtonElement).disabled = true;
  drawPhases();
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
  $("errors").replaceChildren(...errors.map((message) => { const li = document.createElement("li"); li.textContent = message; return li; }));
  if (errors.length) { drawPhases(onPlate); openSheet("upload"); return; }
  const name = files[byPhase.PRE!].name.replace(/_PRE\.png$/i, "");
  setSlice(pixels, name, null);
  closeSheets();
}

/* ---------- Dock and sheets ---------- */
const dockButtons = [...document.querySelectorAll<HTMLButtonElement>(".dock button")];
function openSheet(name: string) {
  for (const button of dockButtons) {
    const on = button.dataset.sheet === name;
    const sheet = $(`sheet-${button.dataset.sheet}`);
    button.setAttribute("aria-expanded", String(on));
    if (on) {
      // the sheet grows out of the button that opened it
      const box = button.getBoundingClientRect();
      sheet.style.transformOrigin = `${box.left + box.width / 2 - (innerWidth / 2 - sheet.offsetWidth / 2)}px 100%`;
    }
    sheet.classList.toggle("open", on);
  }
}
function closeSheets() {
  for (const button of dockButtons) { button.setAttribute("aria-expanded", "false"); $(`sheet-${button.dataset.sheet}`).classList.remove("open"); }
}
for (const button of dockButtons) {
  button.addEventListener("click", () => (button.getAttribute("aria-expanded") === "true" ? closeSheets() : openSheet(button.dataset.sheet!)));
}
addEventListener("keydown", (e) => { if (e.key === "Escape") closeSheets(); });
document.addEventListener("pointerdown", (e) => { if (!(e.target as Element).closest(".sheet, .dock")) closeSheets(); });

for (const example of manifest.examples) {
  const button = document.createElement("button");
  button.className = "sample";
  const image = document.createElement("img");
  image.src = `${BASE}${example.files.EARLY}`;
  image.alt = `Fase EARLY de ${example.sample_id}`;
  const cohort = document.createElement("span"); cohort.className = "label"; cohort.textContent = `${COHORTS[example.cohort]} · test`;
  const id = document.createElement("span"); id.className = "sid"; id.textContent = example.sample_id;
  button.append(image, cohort, id);
  button.addEventListener("click", async () => { setSlice(await loadExample(example), example.sample_id, example); closeSheets(); });
  $("tray").appendChild(button);
}

function renderSpec() {
  const t = manifest.training, v = manifest.validation_slice, s = manifest.test_patient;
  const comma = (x: number, d = 3) => x.toFixed(d).replace(".", ",");
  const sections: [string, [string, string][]][] = [
    ["Arquitectura", [["Red", "CNN 2D desde cero"], ["Entrada", `3 × 256², recorte ${arch.input_crop}²`], ["Etapas", arch.stages.map((x) => x.channels).join(" · ")], ["Parámetros", `${thousands(arch.params)} × ${arch.models}`]]],
    ["Entrenamiento", [["Pérdida", t.loss], ["Optimizador", `${t.optimizer === "adamw" ? "AdamW" : t.optimizer}, lr ${t.learning_rate}`], ["Épocas", `${t.epochs} fijas`], ["Calibración", "Platt por fold"]]],
    ["Validación · corte", [["AUC", comma(v.auc!)], ["Sensibilidad", comma(v.sensitivity!)], ["Especificidad", comma(v.specificity!)]]],
    ["Test · paciente", [["AUC", `${comma(s.auc.value)} [${comma(s.auc.lower)}; ${comma(s.auc.upper)}]`], ["Solo cohorte", comma(s.cohort_baseline_auc)], ["Pacientes", `${s.patients} (${s.positives} pCR)`]]],
  ];
  $("spec").replaceChildren(...sections.map(([title, rows]) => {
    const section = document.createElement("section");
    const h = document.createElement("h3"); h.className = "label"; h.textContent = title;
    const dl = document.createElement("dl");
    for (const [k, value] of rows) { const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = value; dl.append(dt, dd); }
    section.append(h, dl);
    return section;
  }));
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
$("files").addEventListener("change", (e) => handleFiles([...((e.target as HTMLInputElement).files ?? [])]));
const catcher = $("catch");
let drags = 0;
addEventListener("dragenter", (e) => { if (e.dataTransfer?.types.includes("Files")) { drags++; catcher.classList.add("on"); } });
addEventListener("dragleave", () => { drags = Math.max(0, drags - 1); if (!drags) catcher.classList.remove("on"); });
addEventListener("dragover", (e) => e.preventDefault());
addEventListener("drop", (e) => { e.preventDefault(); drags = 0; catcher.classList.remove("on"); handleFiles([...(e.dataTransfer?.files ?? [])]); });

/* ---------- Start: first example on screen, model loaded and checked against Python ---------- */
renderSpec();
renderLog();
drawPhases();
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
      ...check.foldProbabilities.map((v, k) => Math.abs(v - expected.fold_probabilities[k])),
      Math.abs(check.calibrated - expected.calibrated[0]),
    );
    if (difference > 1e-5) throw new Error(`paridad ${difference.toExponential(1)}`);
    state.model = model;
    chip.className = "chip ready";
    stateLabel.textContent = `${model.sha256.slice(0, 6)}…${model.sha256.slice(-4)}`;
    chip.title = `SHA-256 ${model.sha256} · diferencia máxima con Python ${difference.toExponential(1)}`;
    if (state.example === first) showResult(check, false);
  } catch (error) {
    chip.className = "chip error";
    stateLabel.textContent = "no disponible";
    chip.title = error instanceof Error ? error.message : String(error);
  }
  updateButtons();
})();
