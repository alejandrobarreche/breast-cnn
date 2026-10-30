// Entry of the portal: the three phases of a real slice stacked in an oblique view, with one
// point followed through them and its intensity in each phase. The layers separate once when
// the drawing enters; with reduced motion they are drawn in place.
import { decimal, el, text } from "./svg";

const S = 270, CX = 280, TOP = 100, GAP = 135, PHASES = ["PRE", "EARLY", "LATE"];

/** Image point (px, py) of a 256 px slice on layer k of the stack. */
function project(px: number, py: number, k: number): [number, number] {
  const u = (px / 256 - 0.5) * S, v = (py / 256 - 0.5) * S;
  return [CX + (u - v) * Math.SQRT1_2, TOP + k * GAP + (u + v) * Math.SQRT1_2 * 0.5];
}

export function drawSlices(svg: SVGSVGElement, urls: string[], point: { x: number; y: number }, values: number[]) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", "0 0 560 470");
  const defs = el("defs", {}, svg);
  // Bottom layer first, so the upper ones lie on top
  for (const k of [2, 1, 0]) {
    const clip = el("clipPath", { id: `slice-clip-${k}` }, defs);
    el("rect", { x: -S / 2, y: -S / 2, width: S, height: S, rx: 6 }, clip);
    const layer = el("g", { class: "slice-layer", style: `--k:${k}` }, svg);
    const g = el("g", { transform: `translate(${CX} ${TOP + k * GAP}) scale(1 0.5) rotate(45)`, opacity: 0.9 }, layer);
    el("image", { href: urls[k], x: -S / 2, y: -S / 2, width: S, height: S, "clip-path": `url(#slice-clip-${k})`, preserveAspectRatio: "none" }, g);
    el("rect", { x: -S / 2, y: -S / 2, width: S, height: S, rx: 6, fill: "none", stroke: "rgba(255,255,255,.4)", "stroke-width": 1.4, "vector-effect": "non-scaling-stroke" }, g);
    const [lx, ly] = project(0, 256, k);
    text(layer, lx - 16, ly + 5, PHASES[k], { "text-anchor": "end", fill: "#fff", "font-size": 15, "font-weight": 600 });
  }
  const points = [0, 1, 2].map((k) => project(point.x, point.y, k));
  const probe = el("g", { class: "slice-probe" }, svg);
  el("path", { d: `M${points.map((p) => p.join(",")).join("L")}`, stroke: "#fff", "stroke-width": 2, fill: "none", "stroke-dasharray": "4 4" }, probe);
  points.forEach(([x, y], k) => {
    el("circle", { cx: x, cy: y, r: 5, fill: "#fff", stroke: "#0e1113", "stroke-width": 2 }, probe);
    el("rect", { x: x + 12, y: y - 13, width: 46, height: 24, rx: 5, fill: "#0e1113", opacity: 0.85 }, probe);
    text(probe, x + 35, y + 5, decimal(values[k], 2), { "text-anchor": "middle", fill: "#fff", "font-size": 15, "font-weight": 600 });
  });
}
