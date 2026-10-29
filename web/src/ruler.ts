// A3 scale: calibrated probability of one slice on 0-0.6 (calibrated slice probabilities stay
// below 0.5), the slice threshold with the region it classifies as pCR, the pCR rate of the
// training slices, one grey dot per fold model and the ensemble mark with its value. Markers
// move on critically damped springs (damping 1, response 0.4 s) that keep their velocity
// when the target changes.
import { decimal, el, text, token } from "./svg";

export interface Ruler {
  set(foldCalibrated: number[] | null, calibrated: number | null, animate: boolean): void;
}

const X0 = 12, X1 = 318, MAX = 0.6, AXIS = 104;
const OMEGA = (2 * Math.PI) / 0.4;
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

interface Mover { x: number; v: number; target: number; apply(x: number): void }

export function drawRuler(svg: SVGSVGElement, threshold: number, trainRate: number, models: number): Ruler {
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 332 168`);
  const ink = token("--ink"), ref = token("--ref");
  const rx = (v: number) => X0 + (Math.max(0, Math.min(MAX, v)) / MAX) * (X1 - X0);

  // Region classified as pCR, axis and ticks every 0.1
  el("rect", { x: rx(threshold), y: AXIS - 70, width: rx(MAX) - rx(threshold), height: 70, fill: token("--pcr"), opacity: 0.1, rx: 4 }, svg);
  text(svg, rx(MAX) - 8, AXIS - 52, "pCR", { "text-anchor": "end" });
  el("line", { x1: X0, y1: AXIS, x2: X1, y2: AXIS, stroke: token("--axis") }, svg);
  for (let k = 0; k <= 6; k++) {
    el("line", { x1: rx(k / 10), y1: AXIS, x2: rx(k / 10), y2: AXIS + 6, stroke: token("--axis") }, svg);
    text(svg, rx(k / 10), AXIS + 24, decimal(k / 10, 1), { "text-anchor": "middle" });
  }

  // Threshold and the pCR rate of the training slices
  el("line", { x1: rx(threshold), y1: AXIS - 84, x2: rx(threshold), y2: AXIS, stroke: ink, "stroke-width": 2 }, svg);
  text(svg, rx(threshold), AXIS - 90, `umbral por corte ${decimal(threshold)}`, { "text-anchor": "middle", fill: ink });
  el("path", { d: `M${rx(trainRate)} ${AXIS + 34} l-6 9 h12 z`, fill: ref }, svg);
  text(svg, rx(trainRate), AXIS + 60, `pCR en entrenamiento ${decimal(trainRate)}`, { "text-anchor": "middle" });

  const dots = Array.from({ length: models }, () => el("circle", { cx: rx(0), cy: AXIS - 14, r: 4, fill: ref, opacity: 0 }, svg));
  const marker = el("g", { opacity: 0 }, svg);
    el("circle", { cx: 0, cy: AXIS, r: 9, fill: ink, stroke: "#fff", "stroke-width": 2.5 }, marker);
  const value = text(marker, 0, AXIS - 44, "", { "text-anchor": "middle", fill: ink, "font-size": 16, "font-weight": 600 });

  const movers: Mover[] = [
    ...dots.map((dot) => ({ x: rx(0), v: 0, target: rx(0), apply: (x: number) => dot.setAttribute("cx", String(x)) })),
    { x: rx(0), v: 0, target: rx(0), apply: (x: number) => marker.setAttribute("transform", `translate(${x} 0)`) },
  ];
  let frame = 0, last = 0;
  const tick = (now: number) => {
    const dt = Math.min(0.032, (now - last) / 1000 || 0.016);
    last = now;
    let moving = false;
    for (const m of movers) {
      m.v += (-OMEGA * OMEGA * (m.x - m.target) - 2 * OMEGA * m.v) * dt;
      m.x += m.v * dt;
      if (Math.abs(m.x - m.target) > 0.05 || Math.abs(m.v) > 0.05) moving = true;
      m.apply(m.x);
    }
    frame = moving ? requestAnimationFrame(tick) : 0;
  };

  return {
    set(foldCalibrated, calibrated, animate) {
      const visible = foldCalibrated != null && calibrated != null;
      dots.forEach((dot) => dot.setAttribute("opacity", visible ? "1" : "0"));
      marker.setAttribute("opacity", visible ? "1" : "0");
      if (!visible) return;
      value.textContent = decimal(calibrated, 2);
      // Alternate two rows in value order so that close dots stay apart
      foldCalibrated.map((v, k) => [v, k] as const).sort((a, b) => a[0] - b[0]).forEach(([, k], j) => {
        dots[k].setAttribute("cy", String(AXIS - 18 - (j % 2) * 9));
      });
      [...foldCalibrated, calibrated].forEach((v, i) => (movers[i].target = rx(v)));
      if (!animate || reduced) {
        for (const m of movers) { m.x = m.target; m.v = 0; m.apply(m.x); }
        return;
      }
      if (!frame) { last = performance.now(); frame = requestAnimationFrame(tick); }
    },
  };
}
