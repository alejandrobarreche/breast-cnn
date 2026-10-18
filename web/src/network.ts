// Fig. 1: the network drawn from its real layer table, in an oblique view whose front faces
// stay square, so the input images read flat. Face side ~ log2(spatial size), depth ~ log2(channels).
import { el, percent, text, thousands, token } from "./svg";

export interface Stage {
  channels: number;
  size: number;
  params: number;
}

export interface Network {
  setInput(urls: string[] | null): void;
  setOutputs(foldCalibrated: number[] | null, calibrated: number | null): void;
}

const DX = Math.cos(Math.PI / 6) * 0.8;
const DY = Math.sin(Math.PI / 6) * 0.8;
const AXIS = 210;
const side = (size: number) => 30 + 21 * Math.log2(size / 4);
const depth = (channels: number) => 10 + 16 * Math.log2(channels / 8);
const PHASES = ["PRE", "EARLY", "LATE"];

export function drawNetwork(
  svg: SVGSVGElement,
  stages: Stage[],
  crop: number,
  models: number,
  onStage: (label: string | null) => void,
): Network {
  svg.replaceChildren();
  const tones = [token("--t3"), token("--t4"), token("--t2")];
  const stroke = { stroke: "rgba(0,0,0,.6)", "stroke-width": 0.8, "stroke-linejoin": "round" };
  const g = el("g", {}, svg);
  const clip = el("clipPath", { id: "crop-clip" }, el("defs", {}, svg));

  const prism = (parent: Element, x: number, s: number, d: number, lift = 0) => {
    const y0 = AXIS - s / 2 - lift, y1 = AXIS + s / 2 - lift, ox = d * DX, oy = -d * DY;
    el("polygon", { points: `${x},${y0} ${x + s},${y0} ${x + s + ox},${y0 + oy} ${x + ox},${y0 + oy}`, fill: tones[1], ...stroke }, parent);
    el("polygon", { points: `${x + s},${y0} ${x + s + ox},${y0 + oy} ${x + s + ox},${y1 + oy} ${x + s},${y1}`, fill: tones[2], ...stroke }, parent);
    el("rect", { x, y: y0, width: s, height: s, fill: tones[0], class: "face", ...stroke }, parent);
    el("line", { x1: x + 0.5, y1: y0 + 0.8, x2: x + s - 0.5, y2: y0 + 0.8, stroke: "rgba(255,255,255,.09)", "stroke-width": 1 }, parent);
  };
  const caption = (cx: number, first: string, second: string) => {
    text(g, cx, AXIS + 112, first, { "text-anchor": "middle", fill: token("--ink"), "font-size": 16 });
    text(g, cx, AXIS + 134, second, { "text-anchor": "middle", "font-size": 14, fill: token("--muted") });
  };
  const link = (x0: number, x1: number) => {
    el("line", { x1: x0, y1: AXIS, x2: x1, y2: AXIS, stroke: token("--t4"), "stroke-width": 1 }, g);
    el("circle", { cx: x0, cy: AXIS, r: 2.2, fill: token("--t4") }, g);
  };

  // Input: the three phases stacked in depth, PRE at the back
  let x = 10;
  const sIn = side(256);
  const inputs: SVGImageElement[] = [];
  for (let k = 2; k >= 0; k--) {
    const off = k * 18, px = x + off * DX, py = AXIS - sIn / 2 - off * DY;
    el("rect", { x: px - 1, y: py - 1, width: sIn + 2, height: sIn + 2, fill: token("--bench"), stroke: token("--t3"), rx: 2 }, g);
    inputs[k] = el("image", { x: px, y: py, width: sIn, height: sIn, preserveAspectRatio: "none" }, g);
  }
  caption(x + sIn / 2 + 18, "3 × 256²", PHASES.join(" · "));
  const fraction = crop / 256;
  const cx0 = x + (sIn * (1 - fraction)) / 2, cy0 = AXIS - (sIn * fraction) / 2, cs = sIn * fraction;
  el("rect", { x: cx0, y: cy0, width: cs, height: cs, fill: "none", stroke: token("--lit"), "stroke-width": 1.2, "stroke-dasharray": "4 3" }, g);
  x += sIn + 62;

  // The central crop, enlarged as the network sees it
  const sC = side(crop);
  el("line", { x1: cx0 + cs, y1: cy0, x2: x, y2: AXIS - sC / 2, stroke: token("--t4"), "stroke-width": 0.8 }, g);
  el("line", { x1: cx0 + cs, y1: cy0 + cs, x2: x, y2: AXIS + sC / 2, stroke: token("--t4"), "stroke-width": 0.8 }, g);
  el("rect", { x, y: AXIS - sC / 2, width: sC, height: sC }, clip);
  el("rect", { x: x - 1, y: AXIS - sC / 2 - 1, width: sC + 2, height: sC + 2, fill: token("--bench"), stroke: token("--t3") }, g);
  const full = sC / fraction;
  const cropImage = el("image", { x: x - (full - sC) / 2, y: AXIS - full / 2, width: full, height: full, preserveAspectRatio: "none", "clip-path": "url(#crop-clip)" }, g);
  caption(x + sC / 2, `3 × ${crop}²`, "recorte central");
  x += sC + 54;

  // Convolutional stages: two convolutions (two slabs), then 2 x 2 max pooling
  stages.forEach((stage, i) => {
    const s = side(stage.size), d = depth(stage.channels), half = d / 2 - 2;
    link(x - 38, x);
    const block = el("g", { class: "block" }, g);
    el("rect", { x: x - 6, y: AXIS - s / 2 - d * DY - 6, width: s + d * DX + 12, height: s + d * DY + 12, fill: "transparent" }, block);
    prism(block, x + (half + 4) * DX, s, half, (half + 4) * DY);
    prism(block, x, s, half);
    const label = `Etapa ${i + 1} · ${stage.channels} × ${stage.size}² · ${thousands(stage.params)} parámetros`;
    block.addEventListener("pointerenter", () => onStage(label));
    block.addEventListener("pointerleave", () => onStage(null));
    caption(x + s / 2 + (d * DX) / 2, `${stage.channels} × ${stage.size}²`, "conv ×2 · pool");
    x += s + d * DX + 56;
  });

  // Global average pooling, then the logit of one model
  const last = stages[stages.length - 1].channels;
  link(x - 40, x);
  for (let k = 0; k < 16; k++) {
    el("rect", { x, y: AXIS - 50 + k * 6.25, width: 10, height: 4.75, fill: k % 4 === 0 ? token("--t4") : token("--t3") }, g);
  }
  caption(x + 5, String(last), "promedio");
  x += 96;
  link(x - 46, x - 8);
  el("circle", { cx: x, cy: AXIS, r: 8, fill: token("--t2"), stroke: token("--t4") }, g);
  caption(x, "1", "logit");

  // The same network trained on each fold, calibrated, and the mean of the five
  const fx = x + 104, mx = fx + 156;
  const values: SVGTextElement[] = [];
  for (let k = 0; k < models; k++) {
    const y = AXIS - 64 + k * 32;
    el("line", { x1: x + 8, y1: AXIS, x2: fx - 6, y2: y, stroke: token("--t3") }, g);
    el("circle", { cx: fx, cy: y, r: 6, fill: token("--t2"), stroke: token("--t4") }, g);
    values.push(text(g, fx + 14, y + 5, "—", { "font-size": 14 }));
    el("line", { x1: fx + 70, y1: y, x2: mx - 11, y2: AXIS, stroke: token("--t3") }, g);
  }
  caption(fx + 30, `×${models}`, "folds");
  el("circle", { cx: mx, cy: AXIS, r: 11, fill: token("--lit") }, g);
  const mean = text(g, mx, AXIS - 24, "—", { "text-anchor": "middle", "font-size": 22, fill: token("--ink"), "font-weight": 500 });
  caption(mx, "media", "calibrada");

  const box = svg.getBBox();
  svg.setAttribute("viewBox", `${box.x - 8} ${box.y - 12} ${box.width + 16} ${box.height + 20}`);

  return {
    setInput(urls) {
      inputs.forEach((node, k) => (urls ? node.setAttribute("href", urls[k]) : node.removeAttribute("href")));
      if (urls) cropImage.setAttribute("href", urls[2]);
      else cropImage.removeAttribute("href");
    },
    setOutputs(foldCalibrated, calibrated) {
      values.forEach((node, k) => (node.textContent = foldCalibrated ? `${percent(foldCalibrated[k])} %` : "—"));
      mean.textContent = calibrated == null ? "—" : `${percent(calibrated)} %`;
    },
  };
}
