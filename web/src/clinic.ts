// Clinical diagrams of El estudio: the moment of the prediction along the treatment (E2) and the
// three textbook shapes of an uptake curve (E3), drawn as schemes, not from data.
import { draw, el, fade, text, token } from "./svg";

/** E2: diagnosis → pre-treatment DCE-MRI → neoadjuvant chemotherapy → surgery → pathology,
 * which ends in pCR or no pCR. The MRI, where the model acts, is the protagonist. */
export function drawTimeline(svg: SVGSVGElement) {
  svg.replaceChildren();
  const ink = token("--ink"), ref = token("--ref");
  const Y = 120, X = [55, 180, 305, 420, 520];
  const stations = [
    ["Diagnóstico", ""],
    ["DCE-MRI previa", "aquí actúa el modelo"],
    ["Quimioterapia", "neoadyuvante"],
    ["Cirugía", ""],
    ["Patología", "análisis del tejido"],
  ];
  el("line", { x1: X[0], y1: Y, x2: X[4], y2: Y, stroke: ref, "stroke-width": 3, ...draw(0, 1100) }, svg);
  stations.forEach(([name, note], k) => {
    const main = k === 1, d = 150 + k * 180;
    const g = el("g", fade(d), svg);
    el("circle", { cx: X[k], cy: Y, r: main ? 12 : 8, fill: main ? ink : "#fff", stroke: main ? "#fff" : ref, "stroke-width": main ? 3 : 2.5 }, g);
    text(g, X[k], Y + 40, name, { "text-anchor": "middle", fill: main ? ink : token("--ink-2"), "font-size": 16, "font-weight": main ? 600 : 500 });
    if (note) text(g, X[k], Y + 62, note, { "text-anchor": "middle" });
  });
  // What the model tries to anticipate: from the MRI to the pathology result
  const b = el("g", fade(1200), svg);
  el("path", { d: `M${X[1]} ${Y - 30} v-12 H${X[4]} v12`, fill: "none", stroke: ink, "stroke-width": 2 }, b);
  text(b, (X[1] + X[4]) / 2, Y - 54, "lo que el modelo intenta anticipar", { "text-anchor": "middle", fill: ink, "font-size": 16, "font-weight": 600 });
  // Outcome: pCR or no pCR
  const o = el("g", fade(1500), svg);
  const end = X[4];
  el("path", { d: `M${end} ${Y} C${end + 40} ${Y}, ${end + 40} ${Y - 40}, ${end + 80} ${Y - 40}`, fill: "none", stroke: token("--pcr"), "stroke-width": 3 }, o);
  el("path", { d: `M${end} ${Y} C${end + 40} ${Y}, ${end + 40} ${Y + 40}, ${end + 80} ${Y + 40}`, fill: "none", stroke: token("--nopcr"), "stroke-width": 3 }, o);
  text(o, end + 90, Y - 34, "pCR", { fill: token("--pcr"), "font-size": 16, "font-weight": 600 });
  text(o, end + 90, Y + 46, "no pCR", { fill: token("--nopcr"), "font-size": 16, "font-weight": 600 });
  svg.setAttribute("viewBox", `0 0 720 210`);
}

/** E3: the three delayed shapes of an uptake curve after the early rise, labelled as a scheme
 * with the same visual language as the curve of Analizar. */
export function drawCurveTypes(svg: SVGSVGElement) {
  svg.replaceChildren();
  const ink = token("--ink");
  const X = [50, 180, 310], y = (v: number) => 200 - v * 160;
  el("line", { x1: 30, y1: y(0), x2: 330, y2: y(0), stroke: token("--axis") }, svg);
  ["PRE", "EARLY", "LATE"].forEach((name, k) => text(svg, X[k], 228, name, { "text-anchor": "middle", fill: ink }));
  const shapes: [string, number, number][] = [["persistente", 0.82, 0.18], ["meseta", 0.7, 0], ["lavado", 0.6, -0.18]];
  shapes.forEach(([name, early, delta], k) => {
    const pts = [[X[0], y(0.12)], [X[1], y(early)], [X[2], y(early + delta)]];
    el("path", { d: `M${pts.map((p) => p.join(",")).join("L")}`, fill: "none", stroke: ink, "stroke-width": 4 - k * 0.5, opacity: 1 - k * 0.2, "stroke-linejoin": "round", ...draw(200 + k * 250, 900) }, svg);
    text(svg, X[2] + 14, y(early + delta) + 5, name, { fill: ink, "font-size": 16, "font-weight": 600, ...fade(900 + k * 250) });
  });
  text(svg, 30, 18, "Esquema, no datos", {});
  svg.setAttribute("viewBox", "0 0 440 240");
}
