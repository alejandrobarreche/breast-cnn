// Fig. 4: test AUC per patient with its 95 % bootstrap interval, against chance (0.5) and the
// cohort-only reference, on a 0.4-0.8 scale.
import { el, text, token } from "./svg";

const X0 = 24, X1 = 536, LO = 0.4, HI = 0.8;
const comma = (x: number) => x.toFixed(3).replace(".", ",");

export function drawAuc(svg: SVGSVGElement, value: number, lower: number, upper: number, baseline: number) {
  svg.replaceChildren();
  const x = (v: number) => X0 + ((v - LO) / (HI - LO)) * (X1 - X0);
  const AXIS = 150, MODEL = 62, REF = 112;

  // Axis with ticks every 0.05
  el("line", { x1: X0, y1: AXIS, x2: X1, y2: AXIS, stroke: token("--t3") }, svg);
  for (let k = 0; k <= 8; k++) {
    const v = LO + k * 0.05, major = k % 2 === 0;
    el("line", { x1: x(v), y1: AXIS, x2: x(v), y2: AXIS + (major ? 8 : 5), stroke: major ? token("--muted") : token("--t3") }, svg);
    if (major) text(svg, x(v), AXIS + 28, v.toFixed(1).replace(".", ","), { "text-anchor": "middle", "font-size": 14 });
  }

  // Chance
  el("line", { x1: x(0.5), y1: 22, x2: x(0.5), y2: AXIS, stroke: token("--t4"), "stroke-dasharray": "3 4" }, svg);
  text(svg, x(0.5) - 8, 30, "azar", { "text-anchor": "end", "font-size": 13 });

  // Model: interval and point
  el("line", { x1: x(lower), y1: MODEL, x2: x(upper), y2: MODEL, stroke: token("--ink"), "stroke-width": 3, "stroke-linecap": "round" }, svg);
  for (const v of [lower, upper]) el("line", { x1: x(v), y1: MODEL - 8, x2: x(v), y2: MODEL + 8, stroke: token("--ink"), "stroke-width": 2 }, svg);
  el("circle", { cx: x(value), cy: MODEL, r: 7, fill: token("--lit"), stroke: token("--bench"), "stroke-width": 2 }, svg);
  text(svg, x(value), MODEL - 18, `modelo ${comma(value)}`, { "text-anchor": "middle", "font-size": 14, fill: token("--ink") });
  text(svg, x(upper) + 12, MODEL + 5, "IC 95 %", { "font-size": 13 });

  // Cohort-only reference
  el("path", { d: `M${x(baseline)} ${REF - 7} l7 7 l-7 7 l-7 -7 z`, fill: token("--muted") }, svg);
  text(svg, x(baseline) + 14, REF + 5, `solo cohorte ${comma(baseline)}`, { "font-size": 14 });
}
