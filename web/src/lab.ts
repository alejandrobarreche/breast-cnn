// Experiment lab (web/src/runs.json, written by scripts/export_web.py): what each run changed
// with respect to the reference, and its validation patient AUC per fold. The honest value
// reads each fold at an epoch chosen without it (or the last epoch of fixed-epoch runs); the
// best-epoch value picks the maximum on the evaluated fold itself and is optimistic.
import { el, text, token } from "./svg";

type Settings = Record<string, unknown>;
interface Stage { channels: number; convs: number; conv: string }

export interface Experiment {
  name: string;
  final: boolean;
  selection: string;
  changes: Settings;
  auc: number[];
  best_auc: number[];
  curves: { epoch: number[]; train: (number | null)[]; validation: number[] };
}

export const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
export const std = (v: number[]) => Math.sqrt(v.reduce((a, b) => a + (b - mean(v)) ** 2, 0) / (v.length - 1));
const comma = (x: number, d = 3) => x.toFixed(d).replace(".", ",");
const decimal = (x: number) => String(x).replace(".", ",");

/** Plain-language list of what a run changed with respect to the reference settings. */
export function describe(changes: Settings, reference: Settings): string[] {
  const out: string[] = [];
  const c = changes;
  if (c.weighted_loss === false) out.push("BCE sin ponderar");
  if (c.loss_unit === "patient")
    out.push(`pérdida por paciente sobre el ${(c.patient_pooling ?? reference.patient_pooling) === "max" ? "máximo" : "promedio"} de los logits de sus cortes`);
  if ("patients_per_batch" in c) out.push(`${c.patients_per_batch} pacientes por lote`);
  if ("label_smoothing" in c) out.push(`suavizado de etiquetas ${decimal(c.label_smoothing as number)}`);
  if ("rotation_degrees" in c || "max_translation" in c || "scale_range" in c) out.push("aumento geométrico");
  if (c.optimizer === "sgd") out.push("SGD con momento");
  if ("learning_rate" in c) out.push(`tasa de aprendizaje ${decimal(c.learning_rate as number)}`);
  if ("weight_decay" in c) out.push(`weight decay ${decimal(c.weight_decay as number)}`);
  if ("train_slices_per_patient" in c) out.push(`${c.train_slices_per_patient} cortes centrales por paciente en entrenamiento`);
  if (c.epoch_selection === "last") out.push(`${c.epochs} épocas fijas, se guarda la última`);
  if ("model.input_crop" in c) out.push(`recorte central ${c["model.input_crop"]} × ${c["model.input_crop"]}`);
  if ("model.input_pool" in c) out.push(`imagen completa a 1/${c["model.input_pool"]} de resolución`);
  if ("model.dropout" in c) out.push(`dropout ${decimal(c["model.dropout"] as number)}`);
  if ("model.stages" in c) {
    const own = c["model.stages"] as Stage[], ref = reference["model.stages"] as Stage[];
    const channels = (s: Stage[]) => s.map((x) => x.channels).join(",");
    if (channels(own) !== channels(ref)) out.push(`canales ${own[0].channels} → ${own[own.length - 1].channels} (referencia ${ref[0].channels} → ${ref[ref.length - 1].channels})`);
    if (own[0].convs !== ref[0].convs) out.push(`${own[0].convs} convoluciones por etapa`);
    if (own.some((x) => x.conv === "separable")) out.push("convoluciones separables");
  }
  return out;
}

const X0 = 230, X1 = 760, LO = 0.4, HI = 0.7, ROW = 25, TOP = 42;
const x = (v: number) => X0 + ((Math.max(LO, Math.min(HI, v)) - LO) / (HI - LO)) * (X1 - X0);

/** Dot plot: one row per experiment, sorted by honest AUC. Returns the row order. */
export function drawLab(svg: SVGSVGElement, experiments: Experiment[], baseline: number, onSelect: (name: string) => void) {
  svg.replaceChildren();
  const rows = [...experiments].sort((a, b) => mean(b.auc) - mean(a.auc));
  const bottom = TOP + rows.length * ROW;
  for (let v = LO; v <= HI + 1e-9; v += 0.05) {
    el("line", { x1: x(v), y1: TOP - 10, x2: x(v), y2: bottom, stroke: token("--hair") }, svg);
    text(svg, x(v), bottom + 22, v.toFixed(2).replace(".", ","), { "text-anchor": "middle", "font-size": 14 });
  }
  el("line", { x1: x(0.5), y1: TOP - 14, x2: x(0.5), y2: bottom, stroke: token("--t4"), "stroke-dasharray": "3 4" }, svg);
  text(svg, x(0.5), TOP - 20, "azar", { "text-anchor": "middle", "font-size": 14 });
  el("line", { x1: x(baseline), y1: TOP - 14, x2: x(baseline), y2: bottom, stroke: token("--muted"), "stroke-width": 1.5 }, svg);
  text(svg, x(baseline) + 6, TOP - 20, `solo cohorte ${comma(baseline)}`, { "font-size": 14, fill: token("--ink") });

  const groups = new Map<string, SVGGElement>();
  rows.forEach((e, i) => {
    const y = TOP + i * ROW + ROW / 2;
    const g = el("g", { class: `lab-row${e.final ? " final" : ""}`, tabindex: 0, role: "button", "aria-label": `${e.name}, AUC ${comma(mean(e.auc))}` }, svg);
    el("rect", { class: "lab-hit", x: 0, y: y - ROW / 2, width: X1 + 30, height: ROW, rx: 4 }, g);
    text(g, 8, y + 5, e.name, { "font-size": 15, class: "lab-name" });
    const m = mean(e.auc), s = std(e.auc);
    el("line", { x1: x(m - s), y1: y, x2: x(m + s), y2: y, stroke: token("--t4"), "stroke-width": 2, "stroke-linecap": "round" }, g);
    for (const v of e.auc) el("circle", { cx: x(v), cy: y, r: 2.2, fill: token("--muted"), opacity: 0.7 }, g);
    el("circle", { cx: x(mean(e.best_auc)), cy: y, r: 4.5, fill: "none", stroke: token("--muted"), "stroke-width": 1.3 }, g);
    el("circle", { cx: x(m), cy: y, r: 5, fill: token("--ink") }, g);
    g.addEventListener("click", () => onSelect(e.name));
    g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); onSelect(e.name); } });
    groups.set(e.name, g);
  });
  svg.setAttribute("viewBox", `0 0 ${X1 + 40} ${bottom + 34}`);
  return {
    select(name: string) { for (const [n, g] of groups) g.classList.toggle("on", n === name); },
  };
}

/** Mean learning curves over folds: train and validation patient AUC per epoch. */
export function drawCurves(svg: SVGSVGElement, e: Experiment, baseline: number) {
  svg.replaceChildren();
  const W = 420, H = 230, L = 42, R = 12, T = 14, B = 32;
  const last = Math.max(...e.curves.epoch);
  const px = (ep: number) => L + ((ep - 1) / Math.max(1, last - 1)) * (W - L - R);
  const py = (v: number) => T + (1 - (Math.max(0.4, Math.min(1, v)) - 0.4) / 0.6) * (H - T - B);
  for (const v of [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) {
    el("line", { x1: L, y1: py(v), x2: W - R, y2: py(v), stroke: token("--hair") }, svg);
    text(svg, L - 6, py(v) + 4, v.toFixed(1).replace(".", ","), { "text-anchor": "end", "font-size": 13 });
  }
  const step = last > 20 ? 10 : last > 6 ? 2 : 1;
  for (let ep = 1; ep <= last; ep += ep === 1 && step > 1 ? step - 1 : step) text(svg, px(ep), H - 8, String(ep), { "text-anchor": "middle", "font-size": 13 });
  el("line", { x1: L, y1: py(baseline), x2: W - R, y2: py(baseline), stroke: token("--muted"), "stroke-dasharray": "2 4" }, svg);
  const path = (values: (number | null)[]) =>
    values.map((v, i) => (v == null ? "" : `${i && values[i - 1] != null ? "L" : "M"}${px(e.curves.epoch[i])},${py(v)}`)).join(" ");
  const label = (values: (number | null)[], name: string, dy: number) => {
    const i = values.length - 1, v = values[i];
    if (v != null) text(svg, Math.min(px(e.curves.epoch[i]), W - R) - 4, py(v) + dy, name, { "text-anchor": "end", "font-size": 14, fill: token("--ink") });
  };
  if (e.curves.train.some((v) => v != null)) {
    el("path", { d: path(e.curves.train), fill: "none", stroke: token("--muted"), "stroke-width": 1.8, "stroke-dasharray": "5 4" }, svg);
    label(e.curves.train, "entrenamiento", -8);
  }
  el("path", { d: path(e.curves.validation), fill: "none", stroke: token("--ink"), "stroke-width": 2.2 }, svg);
  label(e.curves.validation, "validación", 16);
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
}
