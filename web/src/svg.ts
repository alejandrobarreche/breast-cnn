// Small helpers shared by the network drawing and the ruler.
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

export function text(
  parent: Element,
  x: number,
  y: number,
  content: string,
  attrs: Record<string, string | number> = {},
): SVGTextElement {
  const node = el("text", { x, y, "font-family": token("--font-ui"), "font-size": 15, fill: token("--muted"), ...attrs }, parent);
  node.textContent = content;
  return node;
}

/** Percent with a decimal comma: 0.3487 -> "34,9". */
export const percent = (value: number) => (value * 100).toFixed(1).replace(".", ",");

/** Thousands with a dot: 1180017 -> "1.180.017". */
export const thousands = (value: number) => value.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
