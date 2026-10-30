// Patient split: one dot per patient, a column per fold plus the test set. Train patients are
// coloured by class and ordered by cohort and class; test patients stay grey because their
// labels take part only in the final evaluation. Selecting a fold shows it as validation.
import { el, fade, grow, percent, text, thousands, token } from "./svg";

/** [fold, pCR, cohort index, slices] of one train patient. */
export type Patient = [number, number, number, number];

export interface FoldSummary {
  patients: number;
  positives: number;
  slices: number;
  positiveSlices: number;
}

const PER_ROW = 10, PITCH = 9, R = 3.3, COLUMN = PER_ROW * PITCH, GAP = 26, TOP = 34;

/** Patients, pCR patients, slices and pCR slices of a set of train patients. */
export function summarize(patients: Patient[]): FoldSummary {
  const out = { patients: patients.length, positives: 0, slices: 0, positiveSlices: 0 };
  for (const [, label, , slices] of patients) {
    out.positives += label;
    out.slices += slices;
    out.positiveSlices += label * slices;
  }
  return out;
}

/** pos_weight = N0 / N1 over the training slices, as in training/loop.py. */
export const positiveWeight = (s: FoldSummary) => (s.slices - s.positiveSlices) / s.positiveSlices;

export function drawSplit(
  svg: SVGSVGElement,
  patients: Patient[],
  test: { patients: number; by_cohort: number[] },
  onSelect: (fold: number) => void,
) {
  svg.replaceChildren();
  const folds = Math.max(...patients.map((p) => p[0])) + 1;
  const columns: SVGGElement[] = [];
  let height = 0;

  // A new row starts at each cohort so that the cohorts read as bands
  const place = (g: SVGGElement, items: [number, string][], x0: number) => {
    let row = 0, col = 0, last = items[0]?.[0];
    for (const [cohort, fill] of items) {
      if (cohort !== last || col === PER_ROW) { row += cohort !== last ? 1.6 : 1; col = 0; last = cohort; }
      el("circle", { cx: x0 + col * PITCH + R, cy: TOP + row * PITCH + R, r: R, fill }, g);
      col++;
    }
    height = Math.max(height, TOP + (row + 1) * PITCH);
  };

  const nopcr = token("--nopcr"), pcr = token("--pcr");
  for (let f = 0; f < folds; f++) {
    const x0 = f * (COLUMN + GAP);
    const g = el("g", { class: "fold", tabindex: 0, role: "button", "aria-label": `Usar el fold ${f + 1} como validación` }, svg);
    el("rect", { class: "fold-frame", x: x0 - 8, y: 6, width: COLUMN + 16, height: 10, rx: 4 }, g);
    text(g, x0 + COLUMN / 2, 24, `Fold ${f + 1}`, { "text-anchor": "middle", class: "fold-label" });
    place(g, patients.filter((p) => p[0] === f).map((p) => [p[2], p[1] ? pcr : nopcr]), x0);
    g.addEventListener("click", () => onSelect(f));
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(f); } });
    columns.push(g);
  }
  const tx = folds * (COLUMN + GAP) + GAP;
  const divider = el("line", { x1: tx - GAP, y1: 10, x2: tx - GAP, y2: 10, stroke: token("--axis") }, svg);
  const tg = el("g", { class: "test" }, svg);
  text(tg, tx + COLUMN / 2, 24, "Test", { "text-anchor": "middle" });
  place(tg, test.by_cohort.flatMap((n, c) => Array.from({ length: n }, () => [c, token("--ref")] as [number, string])), tx);

  for (const g of columns) g.querySelector(".fold-frame")!.setAttribute("height", String(height + 2));
  divider.setAttribute("y2", String(height));
  for (let f = 0; f < folds; f++) {
    const own = summarize(patients.filter((p) => p[0] === f)), cx = f * (COLUMN + GAP) + COLUMN / 2;
    text(columns[f], cx, height + 30, `${own.patients} pacientes`, { "text-anchor": "middle" });
    text(columns[f], cx, height + 50, `${percent(own.positives / own.patients)} % pCR`, { "text-anchor": "middle", fill: token("--ink") });
  }
  text(tg, tx + COLUMN / 2, height + 30, `${test.patients} pacientes`, { "text-anchor": "middle" });
  text(tg, tx + COLUMN / 2, height + 50, "etiqueta oculta", { "text-anchor": "middle" });
  svg.setAttribute("viewBox", `-10 0 ${tx + COLUMN + 20} ${height + 60}`);

  return {
    select(fold: number) {
      columns.forEach((g, f) => {
        g.classList.toggle("on", f === fold);
        g.querySelector(".fold-label")!.textContent = f === fold ? "Validación" : `Fold ${f + 1}`;
      });
    },
  };
}

/** E5: one stacked bar per cohort with its train patients by class and the pCR proportion,
 * and a grey bar with its test patients, whose labels are not shown before the results. */
export function drawCohorts(svg: SVGSVGElement, names: string[], patients: Patient[], test: number[]) {
  svg.replaceChildren();
  const L = 90, R = 420, BAR = 28, GROUP = 96, TOP = 34;
  const ink = token("--ink");
  const counts = names.map((_, c) => summarize(patients.filter((p) => p[2] === c)));
  const top = Math.max(...counts.map((s) => s.patients));
  const w = (n: number) => (n / top) * (R - L);
  text(svg, L, TOP - 14, "Entrenamiento", { fill: ink });
  names.forEach((name, c) => {
    const y = TOP + c * GROUP, s = counts[c], d = c * 150;
    text(svg, 0, y + BAR / 2 + 6, name, { fill: ink, "font-size": 16, "font-weight": 600 });
    el("rect", { x: L, y, width: w(s.patients - s.positives), height: BAR, rx: 4, fill: token("--nopcr"), ...grow(d) }, svg);
    el("rect", { x: L + w(s.patients - s.positives), y, width: w(s.positives), height: BAR, rx: 4, fill: token("--pcr"), ...grow(d + 300) }, svg);
    text(svg, L + w(s.patients) + 12, y + BAR / 2 + 6, `${percent(s.positives / s.patients)} % pCR`, { fill: ink, "font-size": 16, "font-weight": 600, ...fade(d + 700) });
    text(svg, L + w(s.patients) + 112, y + BAR / 2 + 6, `${thousands(s.patients)} pacientes`, fade(d + 700));
    el("rect", { x: L, y: y + BAR + 8, width: w(test[c]), height: 14, rx: 4, fill: token("--ref"), opacity: 0.6, ...grow(d + 200) }, svg);
    text(svg, L + w(test[c]) + 12, y + BAR + 20, `${test[c]} de test`, fade(d + 700));
  });
  svg.setAttribute("viewBox", `0 0 ${R + 240} ${TOP + names.length * GROUP}`);
}
