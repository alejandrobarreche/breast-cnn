// A3 scale: calibrated probability of one slice on a thin track from 0 to 0.6 (calibrated
// slice probabilities stay below 0.5), split by the slice threshold into the no pCR and pCR
// segments in their class colours; the segment the estimate falls in strengthens. Its value is
// the large figure above the scale. The mark moves on a critically damped spring (damping 1,
// response 0.4 s) that keeps its velocity when the target changes.
import { decimal, el, text, token } from "./svg";

export interface Ruler {
  set(calibrated: number | null, animate: boolean): void;
}

const X0 = 8, X1 = 324, MAX = 0.6, TRACK = 14;
const OMEGA = (2 * Math.PI) / 0.4;
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

export function drawRuler(svg: SVGSVGElement, threshold: number): Ruler {
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 332 46`);
  const ink = token("--ink");
  const rx = (v: number) => X0 + (Math.max(0, Math.min(MAX, v)) / MAX) * (X1 - X0);

  const segment = (x0: number, x1: number, fill: string) =>
    el("rect", { class: "zone", x: x0, y: TRACK - 4, width: x1 - x0, height: 8, fill, rx: 4 }, svg);
  const zones = [segment(rx(0), rx(threshold) - 2, token("--nopcr")), segment(rx(threshold) + 2, rx(MAX), token("--pcr"))];
  el("line", { x1: rx(threshold), y1: TRACK - 10, x2: rx(threshold), y2: TRACK + 10, stroke: ink, "stroke-width": 2, "stroke-linecap": "round" }, svg);
  text(svg, X0, 42, "0", { "text-anchor": "start" });
  text(svg, rx(threshold), 42, `umbral ${decimal(threshold)}`, { "text-anchor": "middle", fill: ink });
  text(svg, X1, 42, decimal(MAX, 1), { "text-anchor": "end" });

  const marker = el("g", { opacity: 0 }, svg);
  el("circle", { cx: 0, cy: TRACK, r: 7, fill: ink, stroke: "#fff", "stroke-width": 2.5 }, marker);
  const mover = { x: rx(0), v: 0, target: rx(0) };
  const apply = () => marker.setAttribute("transform", `translate(${mover.x} 0)`);
  let frame = 0, last = 0;
  const tick = (now: number) => {
    const dt = Math.min(0.032, (now - last) / 1000 || 0.016);
    last = now;
    mover.v += (-OMEGA * OMEGA * (mover.x - mover.target) - 2 * OMEGA * mover.v) * dt;
    mover.x += mover.v * dt;
    apply();
    frame = Math.abs(mover.x - mover.target) > 0.05 || Math.abs(mover.v) > 0.05 ? requestAnimationFrame(tick) : 0;
  };

  return {
    set(calibrated, animate) {
      marker.setAttribute("opacity", calibrated != null ? "1" : "0");
      zones.forEach((zone, k) => zone.classList.toggle("on", calibrated != null && (k === 1) === calibrated >= threshold));
      if (calibrated == null) return;
      mover.target = rx(calibrated);
      if (!animate || reduced) {
        mover.x = mover.target; mover.v = 0; apply();
        return;
      }
      if (!frame) { last = performance.now(); frame = requestAnimationFrame(tick); }
    },
  };
}
