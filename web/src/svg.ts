// Helpers shared by every chart: SVG elements, labels, number formats and the entrance
// animation classes of style.css (draw: a stroke that draws itself; grow: a bar that grows
// from its start; fade: an element that appears). The delay orders the reading: axes and
// references first, the protagonist last.
const NS = "http://www.w3.org/2000/svg";

export const token = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function el<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number> = {},
  parent?: Element,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  parent?.appendChild(node);
  return node;
}

/** Label of 14 px in the secondary text colour unless the attributes say otherwise. */
export function text(
  parent: Element,
  x: number,
  y: number,
  content: string,
  attrs: Record<string, string | number> = {},
): SVGTextElement {
  const node = el("text", { x, y, "font-size": 14, "font-weight": 500, fill: token("--ink-2"), ...attrs }, parent);
  node.textContent = content;
  return node;
}

/** Entrance animation attributes; `delay` in milliseconds. */
export const draw = (delay = 0, duration = 900) => ({ class: "draw", pathLength: 1, style: `--d:${delay}ms;--t:${duration}ms` });
export const grow = (delay = 0, duration = 900) => ({ class: "grow", style: `--d:${delay}ms;--t:${duration}ms` });
export const fade = (delay = 0) => ({ class: "fade", style: `--d:${delay}ms` });

/** Decimal comma with a fixed number of decimals: 0.3487 -> "0,349". */
export const decimal = (value: number, digits = 3) => value.toFixed(digits).replace(".", ",").replace(/^-/, "−");

/** Percent with a decimal comma: 0.3487 -> "34,9". */
export const percent = (value: number) => (value * 100).toFixed(1).replace(".", ",");

/** Thousands with a dot: 1180017 -> "1.180.017". */
export const thousands = (value: number) => value.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
