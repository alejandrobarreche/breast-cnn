// Page components of documentation/web-design.md (layer 7) built from code: section header,
// visual block, titular block, and the entrance of panels as they come into view.
export const COHORTS: Record<string, string> = { duke: "Duke", spy1: "I-SPY1", spy2: "I-SPY2" };

export const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

export function make<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", content = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content) node.textContent = content;
  return node;
}

export const svgOf = (label: string) => {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", label);
  return svg;
};

/** Paragraphs from strings, or nodes as they are. */
const paragraphs = (body: (string | Node)[]) => body.map((b) => (typeof b === "string" ? make("p", "", b) : b));

export function sectionHead(id: string, n: number, title: string, lead = "") {
  const head = make("header", "section-head");
  head.id = id;
  const h2 = make("h2");
  const number = make("span", "n", String(n));
  number.setAttribute("aria-hidden", "true");
  h2.append(number, title);
  head.append(h2);
  if (lead) head.append(make("p", "", lead));
  return head;
}

export interface Visual {
  panel: HTMLElement;
  /** Replace the technical detail with the detail of a selection, with a link back. */
  select(nodes: (string | Node)[]): void;
}

/** Title, context line, visual, conclusion and technical detail: visual in 8 columns, text in 4. */
export function visualBlock(o: { id?: string; wide?: boolean; title: string; context: string; figure: Node; conclusion: string; detail: (string | Node)[] }): Visual {
  const panel = make("article", `panel study reveal visual${o.wide ? " wide" : ""}`);
  if (o.id) panel.id = o.id;
  const figure = make("div", "visual-figure");
  figure.append(make("span", "context", o.context), o.figure);
  const text = make("div", "visual-text");
  const detail = make("div", "detail");
  const original = paragraphs(o.detail);
  detail.append(...original);
  text.append(make("h3", "card-title", o.title), make("p", "conclusion", o.conclusion), detail);
  panel.append(figure, text);
  return {
    panel,
    select(nodes) {
      const back = make("button", "back", "Volver a la explicación");
      back.addEventListener("click", () => detail.replaceChildren(...original));
      detail.replaceChildren(...paragraphs(nodes), back);
    },
  };
}

/** Charts are drawn at 1:1 at most, so that their 14 px labels never grow; they shrink only
 * when the column is narrower than the drawing. */
export function fit(root: ParentNode) {
  for (const svg of root.querySelectorAll<SVGSVGElement>("svg.chart")) {
    const width = svg.viewBox.baseVal?.width;
    if (width) svg.style.maxWidth = `${width}px`;
  }
}

/** Large title and a short explanation in secondary text, without a visual. */
export function titular(o: { id?: string; title: string; body: (string | Node)[] }) {
  const panel = make("article", "panel study titular reveal");
  if (o.id) panel.id = o.id;
  panel.append(make("h3", "", o.title), ...paragraphs(o.body));
  return panel;
}

/** Panels fade and rise as they come into view, once; those entering together cascade by 50 ms. */
const seen = new IntersectionObserver((entries) => {
  let k = 0;
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    const node = entry.target as HTMLElement;
    node.style.setProperty("--delay", `${k++ * 50}ms`);
    node.classList.add("in");
    seen.unobserve(node);
  }
}, { rootMargin: "0px 0px -8% 0px" });
export function reveal(root: ParentNode) {
  for (const node of root.querySelectorAll<HTMLElement>(".reveal:not(.in)")) seen.observe(node);
}

/** Marks the index link of the last section whose start has passed the upper third of the
 * window; only while that section's tab is visible. */
export function spy(links: HTMLElement[], targets: HTMLElement[]) {
  let frame = 0;
  const update = () => {
    frame = 0;
    if (!targets[0].offsetParent) return;
    let k = 0;
    targets.forEach((t, j) => { if (t.getBoundingClientRect().top < innerHeight / 3) k = j; });
    links.forEach((a, j) => a.classList.toggle("on", j === k));
  };
  addEventListener("scroll", () => { if (!frame) frame = requestAnimationFrame(update); }, { passive: true });
  update();
}

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
/** A number that counts up to its value in 900 ms. */
export function countUp(node: HTMLElement, value: number, format: (v: number) => string) {
  if (reduced) { node.textContent = format(value); return; }
  const start = performance.now();
  const step = (now: number) => {
    const u = Math.min(1, (now - start) / 900), eased = 1 - Math.pow(1 - u, 3);
    node.textContent = format(value * eased);
    if (u < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
