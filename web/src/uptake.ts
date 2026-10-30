// Uptake curve of one point: mean intensity of a 5 x 5 window in PRE, EARLY and LATE; the
// values themselves are written on the viewer. The delayed pattern follows the usual ±10 % rule
// on LATE relative to EARLY; with intensities normalised per volume it is only indicative. The
// horizontal axis is the order of the phases, not time. The pattern is written next to the
// chart, by its caller.
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

const X = [36, 166, 296], Y0 = 92, Y1 = 10;
const y = (v: number) => Y0 - Math.max(0, Math.min(1, v)) * (Y0 - Y1);

export function drawUptake(svg: SVGSVGElement) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", "0 0 332 118");
  const ink = token("--ink");
  for (const v of [0, 0.5, 1]) el("line", { x1: 8, y1: y(v), x2: 324, y2: y(v), stroke: token("--grid") }, svg);
  ["PRE", "EARLY", "LATE"].forEach((name, k) => text(svg, X[k], 114, name, { "text-anchor": "middle" }));
  // Faint area under the curve, the curve and one dot per phase
  const area = el("polygon", { fill: ink, "fill-opacity": 0.06 }, svg);
  const curve = el("polyline", { fill: "none", stroke: ink, "stroke-width": 2.5, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  const dots = X.map((x) => el("circle", { cx: x, r: 4.5, fill: ink, stroke: "#fff", "stroke-width": 2 }, svg));

  return {
    set(point: number[] | null) {
      svg.style.opacity = point ? "1" : "0.35";
      const line = point ? point.map((v, k) => `${X[k]},${y(v)}`).join(" ") : "";
      curve.setAttribute("points", line);
      area.setAttribute("points", point ? `${X[0]},${Y0} ${line} ${X[2]},${Y0}` : "");
      dots.forEach((dot, k) => {
        dot.setAttribute("cy", String(point ? y(point[k]) : 0));
        dot.setAttribute("visibility", point ? "visible" : "hidden");
      });
    },
  };
}
