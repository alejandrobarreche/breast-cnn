// Experiment lab (web/src/runs.json, written by scripts/export_web.py): what each run changed
// with respect to the reference, and its validation patient AUC per fold. The honest value
// reads each fold at an epoch chosen without it (or the last epoch of fixed-epoch runs); the
// best-epoch value picks the maximum on the evaluated fold itself and is optimistic.
import { decimal, el, fade, text, token } from "./svg";

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
const plain = (x: number) => String(x).replace(".", ",");

/** Plain-language list of what a run changed with respect to the reference settings. */
export function describe(changes: Settings, reference: Settings): string[] {
  const out: string[] = [];
  const c = changes;
  if (c.weighted_loss === false) out.push("BCE sin ponderar");
  if (c.loss_unit === "patient")
    out.push(`pérdida por paciente sobre el ${(c.patient_pooling ?? reference.patient_pooling) === "max" ? "máximo" : "promedio"} de los logits de sus cortes`);
  if ("patients_per_batch" in c) out.push(`${c.patients_per_batch} pacientes por lote`);
  if ("label_smoothing" in c) out.push(`suavizado de etiquetas ${plain(c.label_smoothing as number)}`);
  if ("rotation_degrees" in c || "max_translation" in c || "scale_range" in c) out.push("aumento geométrico");
  if (c.optimizer === "sgd") out.push("SGD con momento");
  if ("learning_rate" in c) out.push(`tasa de aprendizaje ${plain(c.learning_rate as number)}`);
  if ("weight_decay" in c) out.push(`weight decay ${plain(c.weight_decay as number)}`);
  if (c.lr_schedule === "cosine") out.push(`tasa en coseno${c.lr_schedule_epochs ? ` de ${c.lr_schedule_epochs} épocas` : ""}`);
  if (c.ema_decay != null) out.push(`media móvil exponencial de los pesos (${plain(c.ema_decay as number)})`);
  if ("train_slices_per_patient" in c) out.push(`${c.train_slices_per_patient} cortes centrales por paciente en entrenamiento`);
  if (c.epoch_selection === "last") out.push(`${c.epochs} épocas fijas, se guarda la última`);
  if ("model.input_crop" in c) out.push(`recorte central ${c["model.input_crop"]} × ${c["model.input_crop"]}`);
  if ("model.input_pool" in c) out.push(`imagen completa a 1/${c["model.input_pool"]} de resolución`);
  if ("model.dropout" in c) out.push(`dropout ${plain(c["model.dropout"] as number)}`);
  if ("model.stages" in c) {
    const own = c["model.stages"] as Stage[], ref = reference["model.stages"] as Stage[];
    const channels = (s: Stage[]) => s.map((x) => x.channels).join(",");
    if (channels(own) !== channels(ref)) out.push(`canales ${own[0].channels} → ${own[own.length - 1].channels} (referencia ${ref[0].channels} → ${ref[ref.length - 1].channels})`);
    if (own[0].convs !== ref[0].convs) out.push(`${own[0].convs} convoluciones por etapa`);
    if (own.some((x) => x.conv === "separable")) out.push("convoluciones separables");
  }
  return out;
}

const X0 = 260, X1 = 650, LO = 0.4, HI = 0.7, ROW = 28, TOP = 48;
const x = (v: number) => X0 + ((Math.max(LO, Math.min(HI, v)) - LO) / (HI - LO)) * (X1 - X0);

/** E12 dot plot: one row per experiment sorted by mean AUC, a small grey dot per fold, the mean
 * (in ink for the final model), the reference run, the cohort-only band and chance. */
export function drawLab(svg: SVGSVGElement, experiments: Experiment[], reference: string, cohort: number[], onSelect: (name: string) => void, onHover: (name: string | null) => void) {
  svg.replaceChildren();
  const ink = token("--ink"), ref = token("--ref");
  const rows = [...experiments].sort((a, b) => mean(b.auc) - mean(a.auc));
  const bottom = TOP + rows.length * ROW;
  const c = mean(cohort), s = std(cohort);
  // References share one baseline above the plot and start where the rows start
  el("rect", { x: x(c - s), y: TOP - 8, width: x(c + s) - x(c - s), height: bottom - TOP + 8, fill: ref, "fill-opacity": 0.16, ...fade(0) }, svg);
  text(svg, x(c + s) + 8, TOP - 18, `referencia por cohorte ${decimal(c)} ± ${decimal(s)}`);
  for (let v = LO; v <= HI + 1e-9; v += 0.05) {
    el("line", { x1: x(v), y1: TOP - 8, x2: x(v), y2: bottom, stroke: token("--grid") }, svg);
    text(svg, x(v), bottom + 26, decimal(v, 2), { "text-anchor": "middle" });
  }
  el("line", { x1: X0, y1: bottom, x2: X1, y2: bottom, stroke: token("--axis") }, svg);
  el("line", { x1: x(0.5), y1: TOP - 8, x2: x(0.5), y2: bottom, stroke: ink, "stroke-width": 2, "stroke-dasharray": "5 5", ...fade(100) }, svg);
  text(svg, x(0.5) - 8, TOP - 18, "azar", { "text-anchor": "end", fill: ink });

  const groups = new Map<string, SVGGElement>();
  rows.forEach((e, i) => {
    const y = TOP + i * ROW + ROW / 2;
    const g = el("g", { class: `lab-row${e.final ? " final" : ""}`, tabindex: 0, role: "button", "aria-label": `${e.name}, AUC ${decimal(mean(e.auc))}` }, svg);
    el("rect", { class: "lab-hit", x: 0, y: y - ROW / 2 + 1, width: X1 + 70, height: ROW - 2, rx: 6 }, g);
    text(g, 10, y + 5, e.name + (e.name === reference ? " (referencia)" : ""), { class: "lab-name", fill: e.final ? ink : token("--ink-2"), "font-weight": e.final ? 600 : 500 });
    const inner = el("g", fade(300 + i * 25), g);
    for (const v of e.auc) el("circle", { cx: x(v), cy: y, r: 3, fill: ref, opacity: 0.7 }, inner);
    el("circle", { cx: x(mean(e.auc)), cy: y, r: e.final ? 7 : 5.5, fill: e.final ? ink : token("--ink-2"), stroke: "#fff", "stroke-width": 1.5 }, inner);
    text(inner, X1 + 16, y + 5, decimal(mean(e.auc)), { fill: e.final ? ink : token("--ink-2"), "font-weight": e.final ? 600 : 500 });
    g.addEventListener("click", () => onSelect(e.name));
    for (const type of ["pointerenter", "focus"]) g.addEventListener(type, () => onHover(e.name));
    for (const type of ["pointerleave", "blur"]) g.addEventListener(type, () => onHover(null));
    g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); onSelect(e.name); } });
    groups.set(e.name, g);
  });
  svg.setAttribute("viewBox", `0 0 ${X1 + 80} ${bottom + 40}`);
  return {
    select(name: string) { for (const [n, g] of groups) g.classList.toggle("on", n === name); },
  };
}
