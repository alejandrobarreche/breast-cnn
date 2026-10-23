// Results charts (web/src/results.json, written by scripts/export_web.py): patient ROC curves of
// the five validation folds and of test, the calibrated probability of each patient by class
// with the threshold, the confusion matrix at that threshold, and the per-fold curves of the run
// that fixed the number of epochs. Computation is kept apart from drawing.
import { el, text, token } from "./svg";

export interface Confusion { tp: number; fn: number; tn: number; fp: number }
export interface FoldHistory { epoch: number[]; train_loss: number[]; validation_loss: number[]; validation_patient_auc: number[] }

const comma = (x: number, d = 2) => x.toFixed(d).replace(".", ",");

/** ROC points [false positive rate, true positive rate], tied scores in one step. */
export function roc(labels: number[], scores: number[]): [number, number][] {
  const order = scores.map((s, i) => [s, labels[i]]).sort((a, b) => b[0] - a[0]);
  const P = labels.filter((y) => y === 1).length, N = labels.length - P;
  const points: [number, number][] = [[0, 0]];
  let tp = 0, fp = 0;
  for (let i = 0; i < order.length; i++) {
    if (order[i][1] === 1) tp++; else fp++;
    if (i === order.length - 1 || order[i + 1][0] !== order[i][0]) points.push([fp / N, tp / P]);
  }
  return points;
}

/** Area under the ROC points (trapezoids), equal to the Mann-Whitney AUC with ties as 1/2. */
export const auc = (points: [number, number][]) =>
  points.slice(1).reduce((a, [x, y], i) => a + (x - points[i][0]) * (y + points[i][1]) / 2, 0);

/** Counts at a threshold: a patient is predicted pCR when its score is at least the threshold. */
export function confusion(labels: number[], scores: number[], threshold: number): Confusion {
  const c = { tp: 0, fn: 0, tn: 0, fp: 0 };
  labels.forEach((y, i) => {
    const positive = scores[i] >= threshold;
    if (y === 1) positive ? c.tp++ : c.fn++; else positive ? c.fp++ : c.tn++;
  });
  return c;
}

/** Five thin validation curves, one thick test curve and the diagonal of chance. */
export function drawRoc(svg: SVGSVGElement, folds: [number, number][][], test: [number, number][]) {
  svg.replaceChildren();
  const S = 400, L = 52, T = 12;
  const x = (v: number) => L + v * S, y = (v: number) => T + (1 - v) * S;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    el("line", { x1: x(v), y1: y(0), x2: x(v), y2: y(1), stroke: token("--hair") }, svg);
    el("line", { x1: x(0), y1: y(v), x2: x(1), y2: y(v), stroke: token("--hair") }, svg);
    text(svg, x(v), y(0) + 24, comma(v), { "text-anchor": "middle", "font-size": 14 });
    text(svg, x(0) - 10, y(v) + 5, comma(v), { "text-anchor": "end", "font-size": 14 });
  }
  text(svg, x(0.5), y(0) + 52, "Tasa de falsos positivos (1 − especificidad)", { "text-anchor": "middle", "font-size": 15 });
  text(svg, 14, y(0.5), "Sensibilidad", { "text-anchor": "middle", "font-size": 15, transform: `rotate(-90 14 ${y(0.5)})` });
  el("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: token("--t4"), "stroke-dasharray": "3 4" }, svg);
  const path = (points: [number, number][]) => points.map(([a, b], i) => `${i ? "L" : "M"}${x(a)},${y(b)}`).join("");
  for (const points of folds) el("path", { d: path(points), fill: "none", stroke: token("--muted"), "stroke-width": 1.4, opacity: 0.75 }, svg);
  el("path", { d: path(test), fill: "none", stroke: token("--lit"), "stroke-width": 3 }, svg);
  svg.setAttribute("viewBox", `0 0 ${L + S + 16} ${T + S + 64}`);
}

/** One row of dots per split and class on the calibrated probability, with the threshold. */
export function drawDistribution(
  svg: SVGSVGElement,
  rows: { label: string; split: string; values: number[]; positive: boolean }[],
  threshold: number,
) {
  svg.replaceChildren();
  const L = 190, W = 1040, ROW = 64, T = 34, HI = 0.7;
  const x = (v: number) => L + (Math.min(v, HI) / HI) * (W - L - 20);
  const bottom = T + rows.length * ROW;
  for (let v = 0; v <= HI + 1e-9; v += 0.1) {
    el("line", { x1: x(v), y1: T - 6, x2: x(v), y2: bottom, stroke: token("--hair") }, svg);
    text(svg, x(v), bottom + 24, comma(v, 1), { "text-anchor": "middle", "font-size": 14 });
  }
  text(svg, x(HI / 2), bottom + 54, "Probabilidad calibrada de pCR por paciente", { "text-anchor": "middle", "font-size": 15 });
  rows.forEach((row, r) => {
    const cy = T + r * ROW + ROW / 2;
    if (r === 2) el("line", { x1: 0, y1: T + r * ROW, x2: W, y2: T + r * ROW, stroke: token("--t3") }, svg);
    if (r % 2 === 0) text(svg, 0, cy + ROW / 2 + 5, row.split, { "font-size": 15, fill: token("--ink"), "font-weight": 600 });
    text(svg, 96, cy + 5, row.label, { "font-size": 14.5 });
    const colour = token(row.positive ? "--pcr" : "--nopcr");
    // Deterministic jitter: golden-ratio sequence over the row height
    row.values.forEach((v, i) => {
      const jitter = ((i * 0.6180339887) % 1 - 0.5) * (ROW - 22);
      el("circle", { cx: x(v), cy: cy + jitter, r: 2.6, fill: colour, opacity: 0.75 }, svg);
    });
    const sorted = [...row.values].sort((a, b) => a - b), median = sorted[Math.floor(sorted.length / 2)];
    el("line", { x1: x(median), y1: cy - ROW / 2 + 6, x2: x(median), y2: cy + ROW / 2 - 6, stroke: token("--lit"), "stroke-width": 2.5 }, svg);
  });
  el("line", { x1: x(threshold), y1: T - 14, x2: x(threshold), y2: bottom, stroke: token("--lit"), "stroke-width": 1.5, "stroke-dasharray": "6 4" }, svg);
  text(svg, x(threshold) + 8, T - 14, `umbral ${comma(threshold, 3)}`, { "font-size": 14, fill: token("--ink") });
  svg.setAttribute("viewBox", `0 0 ${W} ${bottom + 66}`);
}

/** Per-fold curves of one history column against the epoch, with the chosen epoch marked; the
 * vertical axis covers every value shown, in multiples of `step`. */
export function drawEpochs(
  svg: SVGSVGElement,
  folds: FoldHistory[],
  key: "validation_loss" | "validation_patient_auc",
  chosen: number,
  step: number,
  train?: boolean,
) {
  svg.replaceChildren();
  const W = 560, H = 300, L = 48, R = 16, T = 18, B = 40;
  const last = Math.max(...folds.map((f) => f.epoch[f.epoch.length - 1]));
  const px = (ep: number) => L + ((ep - 1) / (last - 1)) * (W - L - R);
  const shown = folds.flatMap((f) => [...f[key], ...(train ? f.train_loss : [])]);
  const lo = Math.floor(Math.min(...shown) / step) * step, hi = Math.ceil(Math.max(...shown) / step) * step;
  const py = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const steps = Math.round((hi - lo) / step);
  for (let k = 0; k <= steps; k++) {
    const v = lo + k * step;
    el("line", { x1: L, y1: py(v), x2: W - R, y2: py(v), stroke: token("--hair") }, svg);
    text(svg, L - 8, py(v) + 5, comma(v, 1), { "text-anchor": "end", "font-size": 13.5 });
  }
  for (const ep of [1, 5, 10, 15, 20, 25].filter((e) => e <= last)) text(svg, px(ep), H - B + 22, String(ep), { "text-anchor": "middle", "font-size": 13.5 });
  text(svg, W - R, H - 2, "época", { "text-anchor": "end", "font-size": 13.5 });
  const line = (epochs: number[], values: number[]) => epochs.map((e, i) => `${i ? "L" : "M"}${px(e)},${py(values[i])}`).join("");
  for (const f of folds) el("path", { d: line(f.epoch, f[key]), fill: "none", stroke: token("--muted"), "stroke-width": 1.3, opacity: 0.7 }, svg);
  // Mean over the epochs that every fold reached
  const common = Math.min(...folds.map((f) => f.epoch.length));
  const epochs = folds[0].epoch.slice(0, common);
  const meanOf = (k: keyof FoldHistory) => epochs.map((_, i) => folds.reduce((a, f) => a + f[k][i], 0) / folds.length);
  if (train) el("path", { d: line(epochs, meanOf("train_loss")), fill: "none", stroke: token("--ink"), "stroke-width": 2, "stroke-dasharray": "6 4" }, svg);
  el("path", { d: line(epochs, meanOf(key)), fill: "none", stroke: token("--lit"), "stroke-width": 3 }, svg);
  el("line", { x1: px(chosen), y1: T - 6, x2: px(chosen), y2: H - B, stroke: token("--ink"), "stroke-width": 1.5, "stroke-dasharray": "2 3" }, svg);
  text(svg, px(chosen) + 6, T + 8, `época ${chosen}`, { "font-size": 14, fill: token("--ink") });
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
}
