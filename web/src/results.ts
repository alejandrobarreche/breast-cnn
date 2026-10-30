// Results charts (web/src/results.json, written by scripts/export_web.py): patient ROC curves of
// the five validation folds and of test with the threshold point (E16), the calibrated
// probability of each test patient by class with the threshold (E17), and the per-fold AUC
// curves of the run that fixed the number of epochs (E13). Computation is kept apart from drawing.
import { decimal, draw, el, fade, text, token } from "./svg";

export interface Confusion { tp: number; fn: number; tn: number; fp: number }
export interface FoldHistory { epoch: number[]; train_loss: number[]; validation_loss: number[]; validation_patient_auc: number[] }


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

/** Point of a score threshold on the ROC plane: [false positive rate, sensitivity]. */
export function rocPoint(labels: number[], scores: number[], threshold: number): [number, number] {
  const c = confusion(labels, scores, threshold);
  return [c.fp / (c.fp + c.tn), c.tp / (c.tp + c.fn)];
}

/** Five thin grey validation curves, the thick test curve, chance and the threshold point. */
export function drawRoc(svg: SVGSVGElement, folds: [number, number][][], test: [number, number][], point: [number, number], labels: { test: string; folds: string; point: string }) {
  svg.replaceChildren();
  const S = 440, L = 64, T = 16;
  const ink = token("--ink"), ref = token("--ref");
  const x = (v: number) => L + v * S, y = (v: number) => T + (1 - v) * S;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    el("line", { x1: x(0), y1: y(v), x2: x(1), y2: y(v), stroke: token("--grid") }, svg);
    text(svg, x(v), y(0) + 26, decimal(v, 2), { "text-anchor": "middle" });
    text(svg, x(0) - 12, y(v) + 5, decimal(v, 2), { "text-anchor": "end" });
  }
  el("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(0), stroke: token("--axis") }, svg);
  el("line", { x1: x(0), y1: y(0), x2: x(0), y2: y(1), stroke: token("--axis") }, svg);
  text(svg, x(0.5), y(0) + 56, "1 − especificidad", { "text-anchor": "middle" });
  text(svg, 16, y(0.5), "Sensibilidad", { "text-anchor": "middle", transform: `rotate(-90 16 ${y(0.5)})` });
  el("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: ink, "stroke-width": 2, "stroke-dasharray": "5 5", ...fade(0) }, svg);
  text(svg, x(0.82), y(0.74), "azar", { fill: ink, transform: `rotate(-45 ${x(0.82)} ${y(0.74)})` });
  const path = (points: [number, number][]) => points.map(([a, b], i) => `${i ? "L" : "M"}${x(a)},${y(b)}`).join("");
  folds.forEach((points, k) => el("path", { d: path(points), fill: "none", stroke: ref, "stroke-width": 1.5, opacity: 0.8, ...draw(150 + k * 60, 900) }, svg));
  text(svg, x(0.03), y(0.9), labels.folds);
  el("path", { d: path(test), fill: "none", stroke: ink, "stroke-width": 4, "stroke-linejoin": "round", ...draw(600, 1100) }, svg);
  text(svg, x(0.03), y(0.96), labels.test, { fill: ink, "font-size": 16, "font-weight": 600 });
  const g = el("g", fade(1600), svg);
  el("circle", { cx: x(point[0]), cy: y(point[1]), r: 7, fill: ink, stroke: "#fff", "stroke-width": 2.5 }, g);
  text(g, x(point[0]) + 14, y(point[1]) - 10, labels.point, { fill: ink });
  svg.setAttribute("viewBox", `0 0 ${L + S + 24} ${T + S + 72}`);
}

/** Beeswarm offsets: each dot takes the smallest vertical shift that does not overlap the
 * dots already placed, in value order. */
export function swarm(xs: number[], radius: number): number[] {
  const order = xs.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
  const placed: [number, number][] = [];
  const out = new Array<number>(xs.length);
  const d = 2 * radius + 1;
  for (const [x, i] of order) {
    const near = placed.filter(([px]) => Math.abs(px - x) < d);
    let best = 0;
    for (let k = 0; ; k++) {
      const offset = (k % 2 ? 1 : -1) * Math.ceil(k / 2) * (d / 2);
      if (near.every(([px, py]) => (px - x) ** 2 + (py - offset) ** 2 >= d * d)) { best = offset; break; }
    }
    placed.push([x, best]);
    out[i] = best;
  }
  return out;
}

/** One swarm row per class on the calibrated probability of each test patient, with the
 * threshold and the number of patients of each class on each side of it. */
export function drawDistribution(svg: SVGSVGElement, rows: { label: string; values: number[]; positive: boolean }[], threshold: number) {
  svg.replaceChildren();
  const L = 90, W = 720, ROW = 180, T = 40, HI = 0.5, R = 4;
  const ink = token("--ink");
  const x = (v: number) => L + (Math.min(v, HI) / HI) * (W - L - 20);
  const bottom = T + rows.length * ROW;
  for (let k = 0; k <= 5; k++) {
    el("line", { x1: x(k / 10), y1: T, x2: x(k / 10), y2: bottom, stroke: token("--grid") }, svg);
    text(svg, x(k / 10), bottom + 24, decimal(k / 10, 1), { "text-anchor": "middle" });
  }
  el("line", { x1: L, y1: bottom, x2: W - 20, y2: bottom, stroke: token("--axis") }, svg);
  text(svg, x(HI / 2), bottom + 54, "Probabilidad calibrada de pCR por paciente", { "text-anchor": "middle" });
  rows.forEach((row, r) => {
    const cy = T + r * ROW + ROW / 2;
    const colour = token(row.positive ? "--pcr" : "--nopcr");
    text(svg, 0, cy + 6, row.label, { fill: colour, "font-size": 16, "font-weight": 600 });
    const offsets = swarm(row.values.map(x), R);
    row.values.forEach((v, i) => el("circle", { cx: x(v), cy: cy + 12 + Math.max(-ROW / 2 + 30, Math.min(ROW / 2 - 18, offsets[i])), r: R, fill: colour, ...fade(200 + r * 300) }, svg));
    const below = row.values.filter((v) => v < threshold).length;
    text(svg, x(threshold) - 12, cy - ROW / 2 + 20, String(below), { "text-anchor": "end", fill: ink, "font-size": 16, "font-weight": 600, ...fade(900) });
    text(svg, x(threshold) + 12, cy - ROW / 2 + 20, String(row.values.length - below), { fill: ink, "font-size": 16, "font-weight": 600, ...fade(900) });
  });
  el("line", { x1: x(threshold), y1: T - 16, x2: x(threshold), y2: bottom, stroke: ink, "stroke-width": 2, ...draw(700, 600) }, svg);
  text(svg, x(threshold) + 8, T - 20, `umbral por paciente ${decimal(threshold)}`, { fill: ink });
  svg.setAttribute("viewBox", `0 0 ${W} ${bottom + 66}`);
}

/** Validation patient AUC per epoch: a thin grey line per fold, the mean in ink, the chosen
 * epoch and, per fold, the epoch chosen from the other four. */
export function drawEpochs(svg: SVGSVGElement, folds: FoldHistory[], chosen: number, outOfFold: number[]) {
  svg.replaceChildren();
  const W = 720, H = 380, L = 56, R = 24, T = 36, B = 52;
  const ink = token("--ink"), ref = token("--ref");
  const key = "validation_patient_auc";
  const last = Math.max(...folds.map((f) => f.epoch[f.epoch.length - 1]));
  const px = (ep: number) => L + ((ep - 1) / (last - 1)) * (W - L - R);
  const shown = folds.flatMap((f) => f[key]);
  const lo = Math.floor(Math.min(...shown) * 10) / 10, hi = Math.ceil(Math.max(...shown) * 10) / 10;
  const py = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  for (let v = lo; v <= hi + 1e-9; v += 0.1) {
    el("line", { x1: L, y1: py(v), x2: W - R, y2: py(v), stroke: token("--grid") }, svg);
    text(svg, L - 10, py(v) + 5, decimal(v, 1), { "text-anchor": "end" });
  }
  el("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, stroke: token("--axis") }, svg);
  for (const ep of [1, 5, 10, 15, 20, 25, 30, 35, 40].filter((e) => e <= last)) text(svg, px(ep), H - B + 24, String(ep), { "text-anchor": "middle" });
  text(svg, W - R, H - 4, "época", { "text-anchor": "end" });
  text(svg, L, T - 18, "AUC por paciente en validación", { "text-anchor": "start" });
  const line = (epochs: number[], values: number[]) => epochs.map((e, i) => `${i ? "L" : "M"}${px(e)},${py(values[i])}`).join("");
  folds.forEach((f, k) => el("path", { d: line(f.epoch, f[key]), fill: "none", stroke: ref, "stroke-width": 1.5, opacity: 0.8, ...draw(100 + k * 60, 900) }, svg));
  const common = Math.min(...folds.map((f) => f.epoch.length));
  const epochs = folds[0].epoch.slice(0, common);
  const mean = epochs.map((_, i) => folds.reduce((a, f) => a + f[key][i], 0) / folds.length);
  el("line", { x1: px(chosen), y1: T - 4, x2: px(chosen), y2: H - B, stroke: ink, "stroke-width": 2, ...draw(400, 500) }, svg);
  text(svg, px(chosen) + 8, T + 12, `época ${chosen}`, { fill: ink, "font-size": 16, "font-weight": 600 });
  el("path", { d: line(epochs, mean), fill: "none", stroke: ink, "stroke-width": 4, "stroke-linejoin": "round", ...draw(700, 1100) }, svg);
  text(svg, px(epochs[epochs.length - 1]), py(mean[mean.length - 1]) - 12, "media de los 5 folds", { "text-anchor": "end", fill: ink });
  // Epoch chosen for each fold without looking at it
  outOfFold.forEach((ep, k) => {
    const f = folds[k], i = f.epoch.indexOf(ep);
    if (i >= 0) el("circle", { cx: px(ep), cy: py(f[key][i]), r: 5, fill: "#fff", stroke: ink, "stroke-width": 2, ...fade(1500) }, svg);
  });
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
}
