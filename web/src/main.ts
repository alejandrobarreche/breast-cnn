import "./style.css";
import manifest from "./manifest.json";
import { decode, enhancementColour, loadPhases, viewsOf } from "./images";
import { loadModel, type Model, type Prediction } from "./model";
import { occlusion, type OcclusionMap } from "./occlusion";
import { $, COHORTS, countUp, fit, make, reveal } from "./page";
import { checkHeader, checkNames, MAX_BYTES, PHASES, SIZE, toTensor, type Phase } from "./preprocess";
import { drawRuler } from "./ruler";
import { buildScope } from "./scope";
import { drawSlices } from "./slices";
import { buildStudy } from "./study";
import { decimal, percent } from "./svg";
import { drawUptake, kinetics, MIN_ENHANCEMENT, windowMean } from "./uptake";

type Example = (typeof manifest.examples)[number];
interface Entry { time: Date; sampleId: string; cohort: string | null; probability: number; threshold: number; positive: boolean; label: number | null }

const THRESHOLD = manifest.model.threshold; // slice threshold: the portal scores one slice
const BASE = import.meta.env.BASE_URL;
const CROP = manifest.architecture.input_crop ?? SIZE;
const testAuc = manifest.test_patient.auc;
// Name and corner note of the four views; the fourth shows the enhancement or the occlusion map
const VIEWS: [string, string][] = [["PRE", "antes del contraste"], ["EARLY", "poco después"], ["LATE", "minutos después"], ["Realce", "EARLY − PRE"]];
const OCCLUSION_VIEW: [string, string] = ["Oclusión", "recorte central"];
/** "Breast_MRI_725_z025" -> "Breast_MRI_725 · corte 25" */
const studyName = (id: string) => id.replace(/_z0*(\d+)$/, " · corte $1");

const state = {
  model: null as Model | null,
  modelError: null as string | null,
  open: false,
  list: "available" as "available" | "recent",
  pixels: null as Uint8ClampedArray[] | null,
  views: [] as HTMLCanvasElement[],
  probe: null as { x: number; y: number } | null,
  auto: null as { x: number; y: number } | null,
  pinned: false,
  measuring: true,
  fourth: "enhancement" as "enhancement" | "occlusion",
  sampleId: null as string | null,
  example: null as Example | null,
  // The open study once Predecir has been pressed on it
  lastPredicted: null as string | null,
  log: [] as Entry[],
};

/* ---------- Tabs: #analizar, #estudio, #alcance and #tab/section ---------- */
const TABS = ["analizar", "estudio", "alcance"] as const;
type Tab = (typeof TABS)[number];
let first: Promise<HTMLCanvasElement[]> | null = null;
const built = new Set<Tab>(["analizar"]);
let current: Tab | null = null;
const scrolls: Partial<Record<Tab, number>> = {};
const TITLES: Record<Tab, string> = { analizar: "Analizar", estudio: "El estudio", alcance: "Alcance" };

async function build(tab: Tab) {
  if (built.has(tab)) return;
  built.add(tab);
  const node = $(`tab-${tab}`);
  if (tab === "estudio") buildStudy(node, $("index-estudio"), await first!);
  else buildScope(node, $("index-alcance"));
  reveal(node);
}

async function route() {
  const [name, section] = location.hash.slice(1).split("/");
  const tab: Tab = (TABS as readonly string[]).includes(name) ? (name as Tab) : "analizar";
  const changed = tab !== current;
  if (current && changed) scrolls[current] = scrollY;
  current = tab;
  document.title = `${TITLES[tab]} · Lector pCR`;
  for (const t of TABS) $(`tab-${t}`).hidden = t !== tab;
  for (const link of document.querySelectorAll<HTMLElement>(".doc-tab")) {
    if (link.dataset.tab === tab) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  for (const part of document.querySelectorAll<HTMLElement>("[data-for]")) part.hidden = !part.dataset.for!.split(" ").includes(tab);
  await build(tab);
  if (tab === "analizar") return routeStudy(section);
  const target = section ? document.getElementById(`${tab}-${section}`) : null;
  if (target) target.scrollIntoView({ block: "start" });
  // Back on a tab, the reading goes on where it was left
  else if (changed) scrollTo({ top: scrolls[tab] ?? 0, behavior: "instant" });
}
addEventListener("hashchange", route);

/* ---------- Model mark: its three layers light up with the download ---------- */
const mark = $("mark") as unknown as SVGSVGElement;
const layers = [...mark.querySelectorAll("path")];
function markState(kind: "loading" | "ready" | "error", fraction = 0) {
  mark.setAttribute("class", `mark ${kind}`);
  layers.forEach((layer, k) => layer.classList.toggle("lit", kind === "loading" && fraction >= (k + 1) / 3));
  mark.setAttribute("aria-label", kind === "ready" ? "Modelo disponible" : kind === "error" ? "Modelo no disponible" : "Preparando el modelo");
  $("retry").hidden = kind !== "error";
}
$("retry").addEventListener("click", () => location.reload());

/* ---------- Viewer: four views with a shared measuring point ---------- */
interface Cell { canvas: HTMLCanvasElement; label: HTMLElement; note: HTMLElement; read: HTMLElement; cross: HTMLElement }
const viewsNode = $("views");
const cells: Cell[] = VIEWS.map(([name, note]) => {
  const cell = make("div", "cell");
  const canvas = make("canvas");
  canvas.width = canvas.height = 2 * SIZE;
  const cross = make("div", "cross");
  const frame = make("div", "crop-frame");
  frame.style.inset = `${((1 - CROP / SIZE) / 2) * 100}%`;
  const label = make("span", "chip-label", name), noteNode = make("span", "chip-note", note), read = make("span", "chip-read");
  cell.append(canvas, frame, cross, label, noteNode, read);
  viewsNode.append(cell);
  return { canvas, label, note: noteNode, read, cross };
});
const uptake = drawUptake($("uptake") as unknown as SVGSVGElement);
const ruler = drawRuler($("ruler") as unknown as SVGSVGElement, THRESHOLD, manifest.model.train_pcr_rate, manifest.architecture.models);

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
const cropMean = (pixels: Uint8ClampedArray) => windowMean(pixels, SIZE, SIZE / 2, SIZE / 2, CROP / 2 - 1);

/* ---------- Occlusion map: the fourth view when its tool is on ---------- */
// One inference at a time on the session: Predict waits while the map is computed
let occlusionRun = 0, occluding = false;
let occlusionMap: OcclusionMap | null = null;
const OCCLUSION_NOTE = `Muestra qué cambia la salida de este modelo, no dónde está el tumor. ` +
  `Con un AUC por paciente en test de ${decimal(testAuc.value)}, el mapa puede reflejar sobre todo ruido.`;

function drawFourth() {
  const cell = cells[3], ctx = cell.canvas.getContext("2d")!, k = cell.canvas.width / SIZE;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, cell.canvas.width, cell.canvas.height);
  const [name, note] = state.fourth === "occlusion" ? OCCLUSION_VIEW : VIEWS[3];
  cell.label.textContent = name;
  cell.note.textContent = note;
  if (!state.views.length) return;
  if (state.fourth === "enhancement" || !occlusionMap) {
    ctx.drawImage(state.views[state.fourth === "enhancement" ? 3 : 0], 0, 0, cell.canvas.width, cell.canvas.height);
    return;
  }
  const map = occlusionMap;
  ctx.drawImage(state.views[0], 0, 0, cell.canvas.width, cell.canvas.height);
  const top = Math.max(1e-6, ...Array.from(map.deltas, Math.abs));
  for (let i = 0; i < map.deltas.length; i++) {
    const d = map.deltas[i];
    // Covering a region that lowers P(pCR) means that region pushed towards pCR: coral
    const [r, g, b] = enhancementColour(-d / top);
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.2 + 0.7 * Math.abs(d) / top})`;
    ctx.fillRect((map.offset + (i % map.size) * map.patch) * k, (map.offset + Math.floor(i / map.size) * map.patch) * k, map.patch * k, map.patch * k);
  }
}

function viewDetail() {
  const detail = $("view-detail");
  if (state.fourth === "enhancement") {
    detail.textContent = "Realce: EARLY − PRE sobre la fase PRE. Coral donde el tejido gana contraste, verde petróleo donde lo pierde.";
  } else if (occlusionMap) {
    const top = Math.max(...Array.from(occlusionMap.deltas, Math.abs));
    detail.textContent = `Oclusión: cada parche de 16 × 16 píxeles del recorte se tapa con ceros. Coral: taparlo baja la probabilidad, así que empuja hacia pCR; verde petróleo, al contrario. ` +
      `Probabilidad sin tapar ${decimal(occlusionMap.base, 2)}; mayor cambio ${decimal(top, 3)}. ${OCCLUSION_NOTE}`;
  } else detail.textContent = "";
}

async function computeOcclusion() {
  if (!state.model || !state.pixels || occlusionMap || occluding) return;
  const run = ++occlusionRun, model = state.model;
  occluding = true;
  updateButtons();
  $("occlusion-progress").hidden = false;
  const map = await occlusion(toTensor(state.pixels), SIZE, CROP, 16, async (t) => (await model.predict(t)).calibrated,
    (done, total) => {
      if (run !== occlusionRun) return;
      $("occlusion-status").textContent = `Calculando el mapa de oclusión: ${done} de ${total} inferencias`;
      $("occlusion-bar").style.width = `${(done / total) * 100}%`;
    },
    () => run !== occlusionRun);
  if (run !== occlusionRun) return;
  occluding = false;
  $("occlusion-progress").hidden = true;
  updateButtons();
  if (!map) return;
  occlusionMap = map;
  drawFourth();
  viewDetail();
}

function setFourth(view: "enhancement" | "occlusion") {
  state.fourth = view;
  toggle("tool-enhancement", view === "enhancement");
  toggle("tool-occlusion", view === "occlusion");
  drawFourth();
  viewDetail();
  drawProbe();
  if (view === "occlusion") computeOcclusion();
}

function drawViewer() {
  cells.slice(0, 3).forEach((cell, i) => {
    const context = cell.canvas.getContext("2d")!;
    context.fillStyle = "#000";
    context.fillRect(0, 0, cell.canvas.width, cell.canvas.height);
    if (state.views[i]) context.drawImage(state.views[i], 0, 0, cell.canvas.width, cell.canvas.height);
    cell.canvas.setAttribute("aria-label", `${VIEWS[i][0]} del corte ${state.sampleId ?? "sin cargar"}`);
  });
  drawFourth();
  viewDetail();
  $("study-name").textContent = state.sampleId ? studyName(state.sampleId) : "";
  $("study-meta").textContent = state.example ? `${COHORTS[state.example.cohort]}, test` : "cargado desde archivo";
  drawProbe();
}

function drawProbe() {
  const probe = state.measuring ? state.probe : null, pixels = state.pixels;
  cells.forEach((cell) => {
    cell.cross.style.opacity = probe && pixels ? "1" : "0";
    if (probe) cell.cross.style.setProperty("--x", `${((probe.x + 0.5) / SIZE) * 100}%`), cell.cross.style.setProperty("--y", `${((probe.y + 0.5) / SIZE) * 100}%`);
  });
  viewsNode.classList.toggle("pinned", state.pinned);
  if (!probe || !pixels) {
    cells.forEach((cell) => (cell.read.textContent = ""));
    uptake.set(null, null);
    $("probe-context").textContent = "";
    $("uptake-text").textContent = pixels ? "Activa «Medir captación» y pasa el cursor por la imagen." : "";
    return;
  }
  const values = pixels.map((p) => windowMean(p, SIZE, probe.x, probe.y));
  cells.forEach((cell, i) => (cell.read.textContent = i < 3 ? decimal(values[i], 2) : state.fourth === "enhancement" ? `${values[1] - values[0] >= 0 ? "+" : "−"}${decimal(Math.abs(values[1] - values[0]), 2)}` : ""));
  uptake.set(values, pixels.map(cropMean));
  const { pattern, delayed } = kinetics(values[0], values[1], values[2]);
  $("uptake-text").textContent = pattern === "sin realce"
    ? `Sin realce apreciable: EARLY supera a PRE en menos de ${decimal(MIN_ENHANCEMENT, 2)}.`
    : `${pattern[0].toUpperCase()}${pattern.slice(1)}: en LATE, ${delayed >= 0 ? "+" : "−"}${percent(Math.abs(delayed))} % frente a EARLY.`;
  $("probe-context").textContent = state.pinned ? "Punto fijado" : probe === state.auto ? "Punto de mayor realce" : "Punto bajo el cursor";
}

function pointAt(event: PointerEvent) {
  const box = (event.target as HTMLElement).closest(".cell")!.getBoundingClientRect();
  const clamp = (v: number) => Math.max(0, Math.min(SIZE - 1, Math.floor(v * SIZE)));
  return { x: clamp((event.clientX - box.left) / box.width), y: clamp((event.clientY - box.top) / box.height) };
}
viewsNode.addEventListener("pointermove", (event) => {
  if (!state.measuring || state.pinned || !state.pixels || !(event.target as HTMLElement).closest(".cell")) return;
  state.probe = pointAt(event);
  drawProbe();
});
viewsNode.addEventListener("click", (event) => {
  if (!state.measuring || !state.pixels || !(event.target as HTMLElement).closest(".cell")) return;
  state.pinned = !state.pinned;
  state.probe = pointAt(event as PointerEvent);
  drawProbe();
});
viewsNode.addEventListener("keydown", (event) => {
  const moves: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
  const move = moves[event.key];
  if (!move || !state.probe || !state.measuring) return;
  event.preventDefault();
  const step = event.shiftKey ? 8 : 1;
  const clamp = (v: number) => Math.max(0, Math.min(SIZE - 1, v));
  state.probe = { x: clamp(state.probe.x + move[0] * step), y: clamp(state.probe.y + move[1] * step) };
  state.pinned = true;
  drawProbe();
});

/* ---------- Tools ---------- */
function toggle(id: string, on: boolean) { $(id).setAttribute("aria-pressed", String(on)); }
$("tool-probe").addEventListener("click", () => {
  state.measuring = !state.measuring;
  if (state.measuring) { state.probe = state.auto; state.pinned = false; }
  toggle("tool-probe", state.measuring);
  drawProbe();
});
$("tool-crop").addEventListener("click", () => toggle("tool-crop", !viewsNode.classList.toggle("no-crop")));
$("tool-enhancement").addEventListener("click", () => setFourth("enhancement"));
$("tool-occlusion").addEventListener("click", () => setFourth("occlusion"));

/* ---------- Result ---------- */
function classTag(node: HTMLElement, positive: boolean, note: string) {
  node.hidden = false;
  node.className = `class-tag ${positive ? "pos" : "neg"}`;
  node.replaceChildren(make("i"), positive ? "pCR" : "no pCR", ...(note ? [make("small", "", note)] : []));
}
function showResult(prediction: Prediction | null) {
  ruler.set(prediction?.foldCalibrated ?? null, prediction?.calibrated ?? null, true);
  const prob = $("prob");
  $("real-tag").hidden = true;
  if (!prediction) {
    prob.className = "figure-big idle";
    prob.textContent = state.modelError ? "Modelo no disponible" : "Pulsa Predecir";
    $("class-tag").hidden = true;
    return;
  }
  const positive = prediction.calibrated >= THRESHOLD;
  prob.className = "figure-big";
  countUp(prob, prediction.calibrated, (v) => decimal(v, 2));
  classTag($("class-tag"), positive, "");
}

function updateButtons() {
  const entry = state.log.find((e) => e.sampleId === state.sampleId);
  const read = entry != null && state.lastPredicted === state.sampleId;
  $("action-hint").textContent =
    !state.model ? (state.modelError ? "El modelo no se ha podido cargar; pulsa Reintentar en la barra superior." : "Predecir se activa cuando el modelo esté preparado.")
    : occluding ? "Predecir espera a que termine el mapa de oclusión."
    : !read ? ""
    : state.example == null ? "Los estudios cargados desde archivo no tienen clase real que revelar."
    : entry!.label != null ? "Clase real revelada; está en Recientes." : "";
  ($("predict") as HTMLButtonElement).disabled = !(state.model && state.pixels) || occluding;
  ($("reveal") as HTMLButtonElement).disabled = !read || state.example == null || entry!.label != null;
  const occlusionTool = $("tool-occlusion") as HTMLButtonElement;
  occlusionTool.disabled = !state.model;
  occlusionTool.title = state.model ? "" : "Disponible cuando el modelo esté preparado";
}

/* ---------- Studies: available and recent ---------- */
// The recent studies survive a reload of the tab (sessionStorage); they are lost when it closes
const LOG_KEY = "lector-pcr-log";
function saveLog() {
  try { sessionStorage.setItem(LOG_KEY, JSON.stringify(state.log)); } catch { /* storage unavailable */ }
}
function restoreLog() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(LOG_KEY) ?? "[]") as (Omit<Entry, "time"> & { time: string })[];
    state.log = saved.map((e) => ({ ...e, time: new Date(e.time) }));
  } catch { state.log = []; }
}
const ago = (time: Date) => {
  const minutes = Math.round((Date.now() - time.getTime()) / 60000);
  return minutes < 1 ? "ahora" : minutes < 60 ? `hace ${minutes} min` : `hace ${Math.round(minutes / 60)} h`;
};

function studyRow(o: { id: string; detail: string; entry?: Entry; onOpen?: () => void; compact: boolean }) {
  const item = make("div");
  item.setAttribute("role", "listitem");
  const row = make(o.onOpen ? "button" : "div", `study-row${o.onOpen ? "" : " static"}`);
  row.dataset.id = o.id;
  item.append(row);
  if (o.id === state.sampleId && state.open) row.setAttribute("aria-current", "true");
  const example = manifest.examples.find((e) => e.sample_id === o.id);
  const thumb = make("img");
  thumb.alt = "";
  if (example) thumb.src = `${BASE}${example.files.EARLY}`;
  const text = make("span", "study-text");
  const name = studyName(o.id);
  const detail = make("span", "", o.detail);
  text.append(make("b", "", o.compact ? name.replace(/^(Breast_MRI_|ISPY1_|ISPY2-|ACRIN-6698-)/, "") : name), detail);
  row.append(thumb, text);
  if (o.entry) {
    const value = make("span", "study-value");
    value.append(make("i", `dot ${o.entry.positive ? "pos" : "neg"}`), decimal(o.entry.probability, 2));
    // In the narrow list the estimate goes under the name, which then keeps its slice number
    if (o.compact) detail.append(value); else row.append(value);
  }
  if (o.onOpen) row.addEventListener("click", o.onOpen);
  return item;
}

function renderLists() {
  saveLog();
  for (const button of document.querySelectorAll<HTMLElement>("[data-list]")) button.setAttribute("aria-pressed", String(button.dataset.list === state.list));
  for (const actions of document.querySelectorAll<HTMLElement>("[data-recent-actions]")) actions.hidden = state.list !== "recent" || !state.log.length;
  document.querySelectorAll<HTMLElement>("[data-studies]").forEach((list) => {
    const compact = list.closest(".worklist") != null;
    list.setAttribute("role", "list");
    if (state.list === "available") {
      list.replaceChildren(...manifest.examples.map((example) => studyRow({
        id: example.sample_id, detail: COHORTS[example.cohort], compact,
        entry: state.log.find((e) => e.sampleId === example.sample_id),
        onOpen: () => openFromList(example),
      })));
      return;
    }
    if (!state.log.length) {
      const empty = make("p", "empty", cleared ? "Recientes borrados." : "Los estudios que leas aparecerán aquí con su estimación.");
      if (cleared) {
        const undo = make("button", "back", "Deshacer");
        undo.addEventListener("click", () => { state.log = cleared!.log; clearTimeout(cleared!.timer); cleared = null; renderLists(); updateButtons(); });
        empty.append(undo);
      }
      list.replaceChildren(empty);
      return;
    }
    list.replaceChildren(...[...state.log].reverse().map((entry) => {
      const example = manifest.examples.find((e) => e.sample_id === entry.sampleId);
      const real = entry.label == null ? "clase real sin revelar" : `clase real: ${entry.label ? "pCR" : "no pCR"}`;
      return studyRow({
        id: entry.sampleId, compact, entry,
        detail: !example ? `cargado desde archivo · ${ago(entry.time)}` : compact ? ago(entry.time) : `${COHORTS[example.cohort]} · ${ago(entry.time)} · ${real}`,
        onOpen: example ? () => openFromList(example) : undefined,
      });
    }));
  });
}
for (const button of document.querySelectorAll<HTMLElement>("[data-list]"))
  button.addEventListener("click", () => { state.list = button.dataset.list as "available" | "recent"; renderLists(); });
for (const button of document.querySelectorAll("[data-download]"))
  button.addEventListener("click", () => {
    const rows = [["hora", "corte", "cohorte", "probabilidad", "umbral", "clase", "real"], ...state.log.map((e) =>
      [e.time.toISOString(), e.sampleId, e.cohort ?? "", e.probability.toFixed(4), e.threshold.toFixed(4), e.positive ? 1 : 0, e.label ?? ""])];
    const link = make("a");
    link.href = URL.createObjectURL(new Blob([rows.map((r) => r.join(",")).join("\n")], { type: "text/csv" }));
    link.download = "lector-pcr-recientes.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  });
let cleared: { log: Entry[]; timer: number } | null = null;
for (const button of document.querySelectorAll("[data-clear-recent]"))
  button.addEventListener("click", () => {
    if (cleared) clearTimeout(cleared.timer);
    cleared = { log: state.log, timer: window.setTimeout(() => { cleared = null; renderLists(); }, 6000) };
    state.log = [];
    renderLists();
    updateButtons();
  });

/* ---------- Opening a study ---------- */
function showOpen(open: boolean) {
  state.open = open;
  $("entry").hidden = open;
  $("work").hidden = !open;
  $("crumb-home").hidden = $("crumb-sep").hidden = !open;
  $("crumb-study").textContent = open && state.sampleId ? studyName(state.sampleId) : "";
  renderLists();
}
$("crumb-home").addEventListener("click", () => { location.hash = "#analizar"; });

/** #analizar shows the entry; #analizar/<sample> opens that example; #analizar/archivo keeps
 * a study loaded from files while it is in memory. */
async function routeStudy(id: string | undefined) {
  const example = manifest.examples.find((e) => e.sample_id === id);
  if (example) { if (!state.open || state.sampleId !== id) await openExample(example); return; }
  if (id === "archivo" && state.pixels && !state.example) { if (!state.open) showOpen(true); return; }
  if (id) history.replaceState(null, "", "#analizar");
  if (state.open) showOpen(false);
}

function openStudy(pixels: Uint8ClampedArray[], sampleId: string, example: Example | null) {
  state.pixels = pixels; state.sampleId = sampleId; state.example = example;
  state.views = viewsOf(pixels);
  state.auto = strongestEnhancement(pixels[0], pixels[1]);
  state.probe = state.auto;
  state.pinned = false;
  state.lastPredicted = null;
  occlusionRun++;
  occlusionMap = null;
  occluding = false;
  $("occlusion-progress").hidden = true;
  state.fourth = "enhancement";
  toggle("tool-enhancement", true);
  toggle("tool-occlusion", false);
  for (const note of document.querySelectorAll<HTMLElement>("[data-upload-note]")) note.hidden = true;
  showOpen(true);
  drawViewer();
  showResult(null);
  updateButtons();
  ((state.model ? $("predict") : viewsNode) as HTMLElement).focus();
  // A study already in Recientes shows its estimate again, without a new entry
  if (state.model && state.log.some((e) => e.sampleId === sampleId)) predict(false);
}
async function openExample(example: Example) {
  for (const row of document.querySelectorAll<HTMLElement>(`.study-row[data-id="${example.sample_id}"]`)) row.classList.add("busy");
  openStudy(await loadPhases(BASE, example.files), example.sample_id, example);
}
const openFromList = (example: Example) => { location.hash = `#analizar/${example.sample_id}`; };

async function handleFiles(files: File[]) {
  const { byPhase, errors } = checkNames(files.map((f) => f.name));
  const pixels: Uint8ClampedArray[] = [];
  for (const phase of PHASES) {
    const index = byPhase[phase as Phase];
    if (index === undefined) continue;
    const file = files[index];
    if (file.size > MAX_BYTES) { errors.push(`${phase}: pesa ${decimal(file.size / 1048576, 1)} MB; el máximo es 2 MB.`); continue; }
    const header = checkHeader(new Uint8Array(await file.slice(0, 33).arrayBuffer()));
    if (header) { errors.push(`${phase}: ${header}`); continue; }
    try { pixels[PHASES.indexOf(phase)] = await decode(file); }
    catch { errors.push(`${phase}: no se puede leer; el archivo puede estar dañado.`); }
  }
  for (const note of document.querySelectorAll<HTMLElement>("[data-upload-note]")) {
    note.hidden = !errors.length;
    note.textContent = errors.length ? `No se ha podido abrir el estudio. ${errors.join(" ")}` : "";
  }
  if (errors.length) return;
  openStudy(pixels, files[byPhase.PRE!].name.replace(/_PRE\.png$/i, ""), null);
  if (location.hash !== "#analizar/archivo") location.hash = "#analizar/archivo";
}

/* ---------- Actions ---------- */
async function predict(record = true) {
  if (!state.model || !state.pixels) return;
  const prediction = await state.model.predict(toTensor(state.pixels));
  showResult(prediction);
  state.lastPredicted = state.sampleId;
  if (!record) {
    const label = state.log.find((e) => e.sampleId === state.sampleId)?.label;
    if (label != null) classTag($("real-tag"), label === 1, (label === 1) === prediction.calibrated >= THRESHOLD ? "real · acierto" : "real · fallo");
    updateButtons();
    return;
  }
  // One entry per study: predicting the same study again replaces it and keeps a revealed label
  const entry: Entry = { time: new Date(), sampleId: state.sampleId!, cohort: state.example?.cohort ?? null, probability: prediction.calibrated, threshold: THRESHOLD, positive: prediction.calibrated >= THRESHOLD, label: null };
  const index = state.log.findIndex((e) => e.sampleId === entry.sampleId);
  if (index >= 0) entry.label = state.log.splice(index, 1)[0].label;
  state.log.push(entry);
  if (entry.label != null) classTag($("real-tag"), entry.label === 1, (entry.label === 1) === entry.positive ? "real · acierto" : "real · fallo");
  renderLists();
  updateButtons();
}
$("predict").addEventListener("click", () => predict());
$("reveal").addEventListener("click", () => {
  const entry = state.log.find((e) => e.sampleId === state.sampleId);
  if (!state.example || !entry) return;
  entry.label = state.example.label;
  classTag($("real-tag"), entry.label === 1, (entry.label === 1) === entry.positive ? "real · acierto" : "real · fallo");
  renderLists();
  updateButtons();
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
addEventListener("drop", (e) => {
  e.preventDefault(); drags = 0; catcher.classList.remove("on");
  if (current !== "analizar") location.hash = "#analizar";
  handleFiles([...(e.dataTransfer?.files ?? [])]);
});

/* ---------- Start: entry on screen, model loaded and checked against Python ---------- */
$("entry-art").addEventListener("click", () => openFromList(manifest.examples[0]));
$("entry-art").setAttribute("aria-label", `Abrir ${studyName(manifest.examples[0].sample_id)}`);
const note = $("auc-note");
note.append(`AUC por paciente en test ${decimal(testAuc.value)}: la estimación apenas separa las clases. `);
const link = make("a", "", "Qué significa");
link.href = "#alcance/probabilidad";
note.append(link);
restoreLog();
renderLists();
showResult(null);
updateButtons();
fit(document);
reveal(document);

first = (async () => {
  const example = manifest.examples[0];
  const pixels = await loadPhases(BASE, example.files);
  const point = strongestEnhancement(pixels[0], pixels[1]);
  const art = $("slices") as unknown as SVGSVGElement;
  drawSlices(art, PHASES.map((p) => `${BASE}${example.files[p]}`), point, pixels.map((p) => windowMean(p, SIZE, point.x, point.y)));
  requestAnimationFrame(() => requestAnimationFrame(() => art.classList.add("in")));
  return viewsOf(pixels);
})();
route();
(async () => {
  try {
    const model = await loadModel(`${BASE}${manifest.model.file}`, manifest.model.sha256, manifest.model.bytes, (fraction) => markState("loading", fraction));
    // Parity in this browser: the first example must reproduce the outputs of Python.
    const example = manifest.examples[0];
    const check = await model.predict(toTensor(await loadPhases(BASE, example.files)));
    const expected = example.expected;
    const difference = Math.max(
      ...check.foldProbabilities.map((value, k) => Math.abs(value - expected.fold_probabilities[k])),
      Math.abs(check.calibrated - expected.calibrated[0]),
    );
    if (difference > 1e-5) throw new Error(`paridad ${difference.toExponential(1)}`);
    state.model = model;
    markState("ready");
  } catch (error) {
    state.modelError = error instanceof Error ? error.message : String(error);
    markState("error");
    mark.setAttribute("aria-label", `Modelo no disponible: ${state.modelError}`);
  }
  if ($("prob").classList.contains("idle")) showResult(null);
  updateButtons();
})();
