// E9: the network drawn from its real layer table, in an oblique view whose front faces
// stay square, so the input images read flat. Face side ~ log2(spatial size), depth ~ log2(channels).
import { el, text, thousands, token } from "./svg";

export interface Stage {
  channels: number;
  convs: number;
  size: number;
  params: number;
}

export interface Network {
  setInput(urls: string[] | null): void;
  select(stage: number | null): void;
}

const DX = Math.cos(Math.PI / 6) * 0.8;
const DY = Math.sin(Math.PI / 6) * 0.8;
const AXIS = 210;
const side = (size: number) => 26 + 17 * Math.log2(size / 4);
const depth = (channels: number) => 10 + 13 * Math.log2(channels / 8);
const PHASES = ["PRE", "EARLY", "LATE"];

export function drawNetwork(
  svg: SVGSVGElement,
  stages: Stage[],
  crop: number,
  onStage: (label: string | null) => void,
  onPick: (stage: number) => void,
): Network {
  svg.replaceChildren();
  const tones = ["#FFFFFF", "#E4E7EA", "#CDD2D7"];
  const stroke = { stroke: token("--ink"), "stroke-opacity": 0.35, "stroke-width": 1, "stroke-linejoin": "round" };
  const g = el("g", {}, svg);
  const clip = el("clipPath", { id: "crop-clip" }, el("defs", {}, svg));
  const blocks: SVGGElement[] = [];

  const prism = (parent: Element, x: number, s: number, d: number, lift = 0) => {
    const y0 = AXIS - s / 2 - lift, y1 = AXIS + s / 2 - lift, ox = d * DX, oy = -d * DY;
    el("polygon", { points: `${x},${y0} ${x + s},${y0} ${x + s + ox},${y0 + oy} ${x + ox},${y0 + oy}`, fill: tones[1], ...stroke }, parent);
    el("polygon", { points: `${x + s},${y0} ${x + s + ox},${y0 + oy} ${x + s + ox},${y1 + oy} ${x + s},${y1}`, fill: tones[2], ...stroke }, parent);
    el("rect", { x, y: y0, width: s, height: s, fill: tones[0], class: "face", ...stroke }, parent);
  };
  const caption = (cx: number, first: string, second: string) => {
    text(g, cx, AXIS + 112, first, { "text-anchor": "middle", fill: token("--ink"), "font-size": 16, "font-weight": 600 });
    text(g, cx, AXIS + 134, second, { "text-anchor": "middle" });
  };
  const link = (x0: number, x1: number) => {
    el("line", { x1: x0, y1: AXIS, x2: x1, y2: AXIS, stroke: token("--ref"), "stroke-width": 1.5 }, g);
    el("circle", { cx: x0, cy: AXIS, r: 2.5, fill: token("--ref") }, g);
  };

  // Input: the three phases stacked in depth, PRE at the back
  let x = 10;
  const sIn = side(256);
  const inputs: SVGImageElement[] = [];
  for (let k = 2; k >= 0; k--) {
    const off = k * 18, px = x + off * DX, py = AXIS - sIn / 2 - off * DY;
    el("rect", { x: px - 1, y: py - 1, width: sIn + 2, height: sIn + 2, fill: token("--viewer"), stroke: token("--axis"), rx: 2 }, g);
    inputs[k] = el("image", { x: px, y: py, width: sIn, height: sIn, preserveAspectRatio: "none" }, g);
  }
  caption(x + sIn / 2 + 18, "3 × 256²", PHASES.join(", "));
  const fraction = crop / 256;
  const cx0 = x + (sIn * (1 - fraction)) / 2, cy0 = AXIS - (sIn * fraction) / 2, cs = sIn * fraction;
  el("rect", { x: cx0, y: cy0, width: cs, height: cs, fill: "none", stroke: "#fff", "stroke-width": 1.5, "stroke-dasharray": "4 3" }, g);
  x += sIn + 44;

  // The central crop, enlarged as the network sees it
  const sC = side(crop);
  el("line", { x1: cx0 + cs, y1: cy0, x2: x, y2: AXIS - sC / 2, stroke: token("--ref"), "stroke-width": 1 }, g);
  el("line", { x1: cx0 + cs, y1: cy0 + cs, x2: x, y2: AXIS + sC / 2, stroke: token("--ref"), "stroke-width": 1 }, g);
  el("rect", { x, y: AXIS - sC / 2, width: sC, height: sC }, clip);
  el("rect", { x: x - 1, y: AXIS - sC / 2 - 1, width: sC + 2, height: sC + 2, fill: token("--viewer"), stroke: token("--axis") }, g);
  const full = sC / fraction;
  const cropImage = el("image", { x: x - (full - sC) / 2, y: AXIS - full / 2, width: full, height: full, preserveAspectRatio: "none", "clip-path": "url(#crop-clip)" }, g);
  caption(x + sC / 2, `3 × ${crop}²`, "recorte central");
  x += sC + 42;

  // Convolutional stages: the convolutions (drawn as two slabs), then 2 x 2 max pooling
  stages.forEach((stage, i) => {
    const s = side(stage.size), d = depth(stage.channels), half = d / 2 - 2;
    link(x - 26, x);
    const block = el("g", { class: "stage", tabindex: 0, role: "button", "aria-label": `Etapa ${i + 1}` }, g);
    blocks.push(block);
    el("rect", { x: x - 6, y: AXIS - s / 2 - d * DY - 6, width: s + d * DX + 12, height: s + d * DY + 12, fill: "transparent" }, block);
    prism(block, x + (half + 4) * DX, s, half, (half + 4) * DY);
    prism(block, x, s, half);
    const label = `Etapa ${i + 1}: ${stage.channels} × ${stage.size}², ${thousands(stage.params)} parámetros`;
    block.addEventListener("pointerenter", () => onStage(label));
    block.addEventListener("pointerleave", () => onStage(null));
    block.addEventListener("click", () => onPick(i));
    block.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick(i); } });
    caption(x + s / 2 + (d * DX) / 2, `${stage.channels} × ${stage.size}²`, `conv ×${stage.convs} y pool`);
    x += s + d * DX + 40;
  });

  // Global average pooling, then the logit of one model
  const last = stages[stages.length - 1].channels;
  x += 20;
  link(x - 26, x);
  for (let k = 0; k < 16; k++) {
    el("rect", { x, y: AXIS - 50 + k * 6.25, width: 10, height: 4.75, fill: k % 4 === 0 ? token("--ref") : "#CDD2D7" }, g);
  }
  caption(x + 5, String(last), "promedio");
  x += 80;
  link(x - 54, x - 8);
  el("circle", { cx: x, cy: AXIS, r: 8, fill: "#fff", stroke: token("--ink"), "stroke-width": 2 }, g);
  caption(x, "1", "logit");

  const box = svg.getBBox();
  svg.setAttribute("viewBox", `${box.x - 8} ${box.y - 12} ${box.width + 16} ${box.height + 20}`);

  return {
    setInput(urls) {
      inputs.forEach((node, k) => (urls ? node.setAttribute("href", urls[k]) : node.removeAttribute("href")));
      if (urls) cropImage.setAttribute("href", urls[2]);
      else cropImage.removeAttribute("href");
    },
    select(stage) {
      blocks.forEach((b, k) => b.classList.toggle("on", k === stage));
    },
  };
}
