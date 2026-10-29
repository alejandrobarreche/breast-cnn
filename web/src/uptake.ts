// Uptake curve of one point: mean intensity of a 5 x 5 window in PRE, EARLY and LATE, against
// the mean of the central crop. The delayed pattern follows the usual ±10 % rule on LATE
// relative to EARLY, drawn as a band at LATE; with intensities normalised per volume it is only
// indicative. The horizontal axis is the order of the phases, not time. The pattern is written
// next to the chart, by its caller.
import { decimal, el, text, token } from "./svg";

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

const X = [52, 172, 292], Y0 = 196, Y1 = 28;
const y = (v: number) => Y0 - Math.max(0, Math.min(1, v)) * (Y0 - Y1);

export function drawUptake(svg: SVGSVGElement) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", "0 0 332 236");
  const ink = token("--ink"), ref = token("--ref");
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    el("line", { x1: 44, y1: y(v), x2: 316, y2: y(v), stroke: token("--grid") }, svg);
    text(svg, 36, y(v) + 5, decimal(v, 2), { "text-anchor": "end" });
  }
  ["PRE", "EARLY", "LATE"].forEach((name, k) => text(svg, X[k], 226, name, { "text-anchor": "middle", fill: ink }));
  // Fan from EARLY to ±10 % of EARLY at LATE: a curve that ends inside it is a plateau
  const band = el("polygon", { fill: ink, "fill-opacity": 0.07 }, svg);
  const bandLabel = text(svg, X[2] - 8, 0, "meseta ±10 %", { "text-anchor": "end" });
  const reference = el("polyline", { fill: "none", stroke: ref, "stroke-width": 3, "stroke-dasharray": "6 6", "stroke-linejoin": "round" }, svg);
  const referenceLabel = text(svg, 0, 0, "media del recorte");
  const curve = el("polyline", { fill: "none", stroke: ink, "stroke-width": 4, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  const dots = X.map(() => el("circle", { r: 6, fill: ink, stroke: "#fff", "stroke-width": 2 }, svg));
  const points = (v: number[]) => v.map((value, k) => `${X[k]},${y(value)}`).join(" ");

  return {
    set(point: number[] | null, mean: number[] | null) {
      svg.style.opacity = point ? "1" : "0.35";
      reference.setAttribute("points", mean ? points(mean) : "");
      referenceLabel.setAttribute("x", String(X[1]));
      referenceLabel.setAttribute("y", String(mean ? y(mean[1]) + (point && point[1] < mean[1] ? -14 : 24) : -20));
      referenceLabel.setAttribute("text-anchor", "middle");
      curve.setAttribute("points", point ? points(point) : "");
      dots.forEach((dot, k) => {
        dot.setAttribute("cx", String(X[k]));
        dot.setAttribute("cy", String(point ? y(point[k]) : -20));
      });
      if (!point) {
        band.setAttribute("points", "");
        bandLabel.textContent = "";
        return;
      }
      band.setAttribute("points", `${X[1]},${y(point[1])} ${X[2]},${y(point[1] * 1.1)} ${X[2]},${y(point[1] * 0.9)}`);
      bandLabel.textContent = "meseta ±10 %";
      // Below the fan when the curve leaves it upwards, above it otherwise
      bandLabel.setAttribute("y", String(point[2] > point[1] * 1.1 ? y(point[1] * 0.9) + 20 : y(point[1] * 1.1) - 10));
    },
  };
}
