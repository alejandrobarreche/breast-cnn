// Fig. 3 scale: calibrated probability on 0-60 %, the threshold, the train pCR rate, one dot
// per fold model and the mean. Markers move on critically damped springs (damping 1,
// response 0.4 s) that keep their velocity when the target changes.
import { el, text, token } from "./svg";

export interface Ruler {
  set(foldCalibrated: number[] | null, calibrated: number | null, animate: boolean): void;
}

const X0 = 14, X1 = 386, MAX = 0.6, BAR = 66;
const OMEGA = (2 * Math.PI) / 0.4;
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

interface Mover { x: number; v: number; target: number; apply(x: number): void }

export function drawRuler(svg: SVGSVGElement, threshold: number, trainRate: number, models: number): Ruler {
  svg.replaceChildren();
  const rx = (v: number) => X0 + (Math.max(0, Math.min(MAX, v)) / MAX) * (X1 - X0);
  el("rect", { x: rx(0), y: BAR - 4, width: rx(threshold) - rx(0), height: 8, fill: token("--nopcr") }, svg);
  el("rect", { x: rx(threshold), y: BAR - 4, width: rx(MAX) - rx(threshold), height: 8, fill: token("--pcr"), opacity: 0.85 }, svg);
  for (let k = 0; k <= 30; k++) {
    const v = k * 0.02, major = k % 5 === 0;
    el("line", { x1: rx(v), y1: BAR + 8, x2: rx(v), y2: BAR + (major ? 18 : 13), stroke: major ? token("--muted") : token("--t3"), "stroke-width": major ? 1.2 : 0.8 }, svg);
    if (major) text(svg, rx(v), BAR + 36, String(Math.round(v * 100)), { "text-anchor": "middle", "font-size": 14 });
  }
  el("line", { x1: rx(threshold), y1: BAR - 46, x2: rx(threshold), y2: BAR + 8, stroke: token("--ink"), "stroke-width": 2 }, svg);
  text(svg, rx(threshold) + 6, BAR - 38, "umbral", { "font-size": 13, fill: token("--ink") });
  el("path", { d: `M${rx(trainRate)} ${BAR + 8} l-5 8 h10 z`, fill: token("--muted") }, svg);
  text(svg, rx(trainRate), BAR + 54, "pCR en train", { "text-anchor": "middle", "font-size": 12, fill: token("--muted") });

  const dots = Array.from({ length: models }, () => el("circle", { cx: rx(0), cy: BAR - 16, r: 4, fill: token("--t4"), opacity: 0 }, svg));
  const marker = el("g", { opacity: 0 }, svg);
  el("line", { x1: 0, y1: BAR - 14, x2: 0, y2: BAR + 14, stroke: token("--lit"), "stroke-width": 3, "stroke-linecap": "round" }, marker);
  el("circle", { cx: 0, cy: BAR, r: 6.5, fill: token("--lit"), stroke: token("--bench"), "stroke-width": 2 }, marker);

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
      // Alternate two rows in value order so that close dots stay apart
      foldCalibrated.map((v, k) => [v, k] as const).sort((a, b) => a[0] - b[0]).forEach(([, k], j) => {
        dots[k].setAttribute("cy", String(BAR - 16 - (j % 2) * 9));
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
