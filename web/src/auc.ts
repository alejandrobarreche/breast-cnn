// E15: test AUC per patient with its 95 % bootstrap interval, against chance (0.5) and the
// cohort-only reference on a 0.4-0.8 scale; below, on its own axis centred on zero, the
// difference with that reference.
import { decimal, draw, el, fade, grow, text, token } from "./svg";

export interface Interval { label: string; value: number; lower: number; upper: number; primary: boolean }

const L = 150, R = 490, ROW = 56;

function rows(svg: SVGSVGElement, items: Interval[], top: number, lo: number, hi: number, delay: number) {
  const x = (v: number) => L + ((v - lo) / (hi - lo)) * (R - L);
  const ink = token("--ink");
  items.forEach((item, i) => {
    const cy = top + i * ROW + ROW / 2;
    text(svg, 0, cy + 5, item.label, { fill: item.primary ? ink : token("--ink-2"), "font-size": 16, "font-weight": item.primary ? 600 : 500 });
    el("rect", { x: x(item.lower), y: cy - 12, width: x(item.upper) - x(item.lower), height: 24, rx: 6, fill: ink, opacity: item.primary ? 0.85 : 0.35, ...grow(delay + i * 150) }, svg);
    el("line", { x1: x(item.value), y1: cy - 12, x2: x(item.value), y2: cy + 12, stroke: "#fff", "stroke-width": 3, ...fade(delay + 500 + i * 150) }, svg);
    text(svg, R + 20, cy + 5, `${decimal(item.value)}  [${decimal(item.lower)}; ${decimal(item.upper)}]`, { fill: item.primary ? ink : token("--ink-2"), "font-size": 16, ...fade(delay + 600 + i * 150) });
  });
  return x;
}

function axis(svg: SVGSVGElement, x: (v: number) => number, y: number, ticks: number[], digits: number) {
  el("line", { x1: L, y1: y, x2: R, y2: y, stroke: token("--axis") }, svg);
  for (const v of ticks) {
    el("line", { x1: x(v), y1: y, x2: x(v), y2: y + 6, stroke: token("--axis") }, svg);
    text(svg, x(v), y + 24, decimal(v, digits), { "text-anchor": "middle" });
  }
}

export function drawAuc(svg: SVGSVGElement, models: Interval[], baseline: number, differences: Interval[]) {
  svg.replaceChildren();
  const ink = token("--ink"), ref = token("--ref");
  const TOP = 36, bottom = TOP + models.length * ROW;
  const x = rows(svg, models, TOP, 0.4, 0.8, 300);
  for (let k = 0; k <= 8; k++) el("line", { x1: x(0.4 + k * 0.05), y1: TOP, x2: x(0.4 + k * 0.05), y2: bottom, stroke: token("--grid") }, svg);
  axis(svg, x, bottom, [0.4, 0.5, 0.6, 0.7, 0.8], 1);
  el("line", { x1: x(0.5), y1: TOP - 14, x2: x(0.5), y2: bottom, stroke: ink, "stroke-width": 2, "stroke-dasharray": "5 5", ...fade(0) }, svg);
  text(svg, x(0.5) - 6, TOP - 18, "azar 0,5", { "text-anchor": "end", fill: ink });
  el("line", { x1: x(baseline), y1: TOP - 14, x2: x(baseline), y2: bottom, stroke: ref, "stroke-width": 3, "stroke-dasharray": "6 6", ...fade(100) }, svg);
  text(svg, x(baseline) + 6, TOP - 18, `referencia por cohorte ${decimal(baseline)}`);

  // Differences with the cohort-only reference
  const DTOP = bottom + 92, span = 0.15;
  text(svg, 0, DTOP - 20, "Diferencia con la referencia por cohorte", { fill: ink, "font-size": 16, "font-weight": 600 });
  const dx = rows(svg, differences, DTOP, -span, span, 900);
  const dbottom = DTOP + differences.length * ROW;
  axis(svg, dx, dbottom, [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15], 2);
  el("line", { x1: dx(0), y1: DTOP - 4, x2: dx(0), y2: dbottom, stroke: ink, "stroke-width": 2, ...draw(800, 600) }, svg);
  svg.setAttribute("viewBox", `0 0 ${R + 220} ${dbottom + 36}`);
}
