// E11: how one slice is scored, with the real outputs of an example from the manifest: the
// probabilities of the five fold models, their mean, the Platt map over that mean and the
// comparison with the slice threshold. Each step appears in reading order.
import { decimal, el, fade, text, token } from "./svg";

export function drawEnsemble(svg: SVGSVGElement, folds: number[], mean: number, calibrated: number, threshold: number) {
  svg.replaceChildren();
  const ink = token("--ink"), ref = token("--ref");
  const CY = 150, FX = 0, MX = 250, CX = 430, TX = 580;
  const box = (parent: Element, x: number, y: number, w: number, label: string, value: string, strong = false) => {
    el("rect", { x, y: y - 24, width: w, height: 48, rx: 8, fill: strong ? ink : "#fff", stroke: strong ? ink : token("--axis") }, parent);
    text(parent, x + w / 2, y + 6, value, { "text-anchor": "middle", fill: strong ? "#fff" : ink, "font-size": 18, "font-weight": 600 });
    text(parent, x + w / 2, y - 34, label, { "text-anchor": "middle" });
  };
  folds.forEach((p, k) => {
    const y = CY - 112 + k * 56;
    const g = el("g", fade(k * 80), svg);
    el("rect", { x: FX, y: y - 18, width: 130, height: 36, rx: 8, fill: "#fff", stroke: token("--axis") }, g);
    text(g, FX + 14, y + 5, `fold ${k + 1}`);
    text(g, FX + 118, y + 5, decimal(p, 3), { "text-anchor": "end", fill: ink, "font-size": 16, "font-weight": 600 });
    el("path", { d: `M${FX + 130} ${y} C${FX + 180} ${y}, ${MX - 60} ${CY}, ${MX - 8} ${CY}`, fill: "none", stroke: ref, "stroke-width": 1.5 }, g);
  });
  const m = el("g", fade(600), svg);
  box(m, MX, CY, 110, "media", decimal(mean, 3));
  el("line", { x1: MX + 110, y1: CY, x2: CX - 8, y2: CY, stroke: ref, "stroke-width": 1.5 }, m);
  text(m, (MX + 110 + CX) / 2, CY + 26, "Platt", { "text-anchor": "middle", fill: ink });
  const c = el("g", fade(900), svg);
  box(c, CX, CY, 110, "calibrada", decimal(calibrated, 2), true);
  el("line", { x1: CX + 110, y1: CY, x2: TX - 8, y2: CY, stroke: ref, "stroke-width": 1.5 }, c);
  const positive = calibrated >= threshold;
  const t = el("g", fade(1200), svg);
  text(t, TX + 65, CY - 34, `umbral ${decimal(threshold)}`, { "text-anchor": "middle" });
  el("rect", { x: TX, y: CY - 24, width: 130, height: 48, rx: 8, fill: "rgba(255,255,255,.7)", stroke: token("--axis") }, t);
  el("circle", { cx: TX + 22, cy: CY, r: 7, fill: token(positive ? "--pcr" : "--nopcr") }, t);
  text(t, TX + 38, CY + 6, positive ? "pCR" : "no pCR", { fill: ink, "font-size": 18, "font-weight": 600 });
  svg.setAttribute("viewBox", "0 0 720 300");
}
