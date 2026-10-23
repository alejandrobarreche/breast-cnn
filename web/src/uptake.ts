// Uptake curve of one point: mean intensity of a 5 x 5 window in PRE, EARLY and LATE, against
// the mean of the central crop. The delayed pattern follows the usual ±10 % rule on LATE
// relative to EARLY; with intensities normalised per volume it is only indicative.
import { el, text, token } from "./svg";

export type Pattern = "sin realce" | "persistente" | "meseta" | "lavado";
export const MIN_ENHANCEMENT = 0.05;

/** Delayed pattern of a curve of intensities in [0, 1], and the relative change LATE vs EARLY. */
export function kinetics(pre: number, early: number, late: number): { pattern: Pattern; delayed: number } {
  const delayed = early > 0 ? (late - early) / early : 0;
  if (early - pre < MIN_ENHANCEMENT) return { pattern: "sin realce", delayed };
  return { pattern: delayed > 0.1 ? "persistente" : delayed < -0.1 ? "lavado" : "meseta", delayed };
}

/** Mean intensity in [0, 1] of a window of side 2r + 1 around (x, y) of an RGBA image of side n. */
export function windowMean(rgba: ArrayLike<number>, n: number, x: number, y: number, r = 2): number {
  let sum = 0, count = 0;
  for (let j = Math.max(0, y - r); j <= Math.min(n - 1, y + r); j++)
    for (let i = Math.max(0, x - r); i <= Math.min(n - 1, x + r); i++) { sum += rgba[4 * (j * n + i)]; count++; }
  return sum / count / 255;
}

const X = [70, 200, 330], Y0 = 180, Y1 = 24;
const y = (v: number) => Y0 - v * (Y0 - Y1);

export function drawUptake(svg: SVGSVGElement) {
  svg.replaceChildren();
  for (const v of [0, 0.5, 1]) {
    el("line", { x1: 40, y1: y(v), x2: 360, y2: y(v), stroke: token("--hair"), "stroke-width": 1 }, svg);
    text(svg, 32, y(v) + 4, v.toFixed(1).replace(".", ","), { "text-anchor": "end", "font-size": 12 });
  }
  ["PRE", "EARLY", "LATE"].forEach((name, k) => text(svg, X[k], 208, name, { "text-anchor": "middle", "font-size": 13 }));
  const reference = el("polyline", { fill: "none", stroke: token("--t4"), "stroke-width": 1.5, "stroke-dasharray": "4 4" }, svg);
  const referenceLabel = text(svg, 0, 0, "media del recorte", { "font-size": 12, fill: token("--muted") });
  const curve = el("polyline", { fill: "none", stroke: token("--ink"), "stroke-width": 2.2, "stroke-linejoin": "round" }, svg);
  const dots = X.map(() => el("circle", { r: 4.5, fill: token("--ink") }, svg));
  const points = (v: number[]) => v.map((value, k) => `${X[k]},${y(value)}`).join(" ");

  return {
    set(point: number[] | null, mean: number[] | null) {
      svg.style.opacity = point ? "1" : "0.35";
      reference.setAttribute("points", mean ? points(mean) : "");
      referenceLabel.setAttribute("x", String(X[2] - 4));
      referenceLabel.setAttribute("y", String(mean ? y(mean[2]) + (point && point[2] < mean[2] ? -10 : 18) : -20));
      referenceLabel.setAttribute("text-anchor", "end");
      curve.setAttribute("points", point ? points(point) : "");
      dots.forEach((dot, k) => {
        dot.setAttribute("cx", String(X[k]));
        dot.setAttribute("cy", String(point ? y(point[k]) : -20));
      });
    },
  };
}
