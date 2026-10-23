// Opening sequence, drawn on a canvas from a real test slice and the real layer table: the PRE
// phase is read out line by line, the contrast arrives as a wave that reveals EARLY − PRE, the
// central crop closes in and pulses cross the five stages to the calibrated output. It does not
// react to the pointer; afterwards a few more pulses cross the network and the drawing stops on
// its final frame. With reduced motion only that frame is drawn. It pauses while off screen or
// in a hidden tab.
import { percent, thousands } from "./svg";

export interface HeroInput {
  phases: HTMLCanvasElement[]; // PRE, EARLY, LATE
  enhancement: HTMLCanvasElement;
  sampleId: string;
  cohort: string;
  crop: number;
  stages: { channels: number; size: number }[];
  params: number;
  models: number;
  probability: number;
}

// Timeline in milliseconds
const READ = 2600, WAVE = 1700, CROP = 900;
const T_WAVE = READ, T_CROP = T_WAVE + WAVE, T_NET = T_CROP + CROP;
const HOP = 340; // time of a pulse between two columns
const BANDS = 64; // readout lines of 4 image rows
const LINKS = 5; // connections drawn between two columns

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const ease = (u: number) => 1 - Math.pow(1 - Math.min(1, Math.max(0, u)), 3);
const BONE = "232, 228, 218";
// Colour of the travelling signal: the accent of the page (CSS --signal)
const css = (name: string, fallback: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

interface Column { x: number; cy: number; side: number; sheets: number; gap: number; width: number; label: string; kind: "input" | "stage" | "vector" | "output" }
interface Pulse { column: number; link: number; start: number }

export function startHero(canvas: HTMLCanvasElement, input: HeroInput) {
  const ctx = canvas.getContext("2d")!;
  const SIGNAL = css("--signal", "255, 252, 244");
  const font = getComputedStyle(document.documentElement).getPropertyValue("--font-ui").trim();
  let W = 0, H = 0, dpr = 1;
  let img = { x: 0, y: 0, s: 0 };
  let columns: Column[] = [];
  let links: [number, number, number, number][][] = []; // per column gap: segments x0 y0 x1 y1
  const activation: number[] = [];
  const pulses: Pulse[] = [];
  let outputSeen = false, nextIdle = 0, idleLeft = 3, finished = false;

  function layout() {
    const box = canvas.getBoundingClientRect();
    dpr = Math.min(2, devicePixelRatio || 1);
    W = box.width; H = box.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    // Same 1520 px grid as the rest of the page
    const left = Math.max(64, (W - 1520) / 2 + 64), right = W - left;
    const s = Math.min(H * 0.72, (right - left) * 0.32);
    img = { x: left, y: (H - s) / 2 - 12, s };
    const cy = img.y + s / 2;

    // Sheet side ~ log2(spatial size); sheets per stack ~ log2(channels)
    const unit = Math.min(H * 0.5, s * 0.78);
    const side = (size: number) => unit * (0.16 + 0.84 * Math.log2(size / 8) / 4);
    const spec: Omit<Column, "x" | "width">[] = [
      { cy, side: side(input.crop), sheets: 3, gap: 9, label: `3 × ${input.crop}²`, kind: "input" },
      ...input.stages.map((st) => ({ cy, side: side(st.size), sheets: 2 + 2 * Math.log2(st.channels / 16), gap: 4, label: `${st.channels} × ${st.size}²`, kind: "stage" as const })),
      { cy, side: unit * 0.34, sheets: 1, gap: 0, label: String(input.stages[input.stages.length - 1].channels), kind: "vector" },
      { cy, side: 16, sheets: 1, gap: 0, label: "P(pCR)", kind: "output" },
    ];
    const widthOf = (c: Omit<Column, "x" | "width">) =>
      c.kind === "vector" ? 8 : c.kind === "output" ? 16 : c.side * 0.36 + (c.sheets - 1) * c.gap;
    const x0 = img.x + s + (right - left) * 0.08, x1 = right - 24;
    const total = spec.reduce((a, c) => a + widthOf(c), 0);
    const space = (x1 - x0 - total) / (spec.length - 1);
    let x = x0;
    columns = spec.map((c) => {
      const width = widthOf(c);
      const col = { ...c, x, width };
      x += width + space;
      return col;
    });
    while (activation.length < columns.length) activation.push(0);

    // Connections from the right edge of each column to the left edge of the next
    links = columns.slice(0, -1).map((a, i) => {
      const b = columns[i + 1];
      const ra = edge(a, "right"), lb = edge(b, "left");
      return Array.from({ length: LINKS }, (_, k) => {
        const u = (k + 0.5) / LINKS, w = ((k * 7) % LINKS + 0.5) / LINKS;
        return [ra.x, ra.top + (ra.bottom - ra.top) * u, lb.x, lb.top + (lb.bottom - lb.top) * w] as [number, number, number, number];
      });
    });
  }

  // Front sheet of a stack: the unit square mapped to a parallelogram seen at an angle
  const sheetBox = (c: Column, k: number) => {
    const w = c.side * 0.36, x = c.x + k * c.gap, top = c.cy - c.side / 2 + k * c.gap * 0.35;
    return { x, top, w, h: c.side, skew: -c.side * 0.16 };
  };
  function edge(c: Column, which: "left" | "right") {
    if (c.kind === "vector") return { x: which === "left" ? c.x : c.x + 8, top: c.cy - c.side / 2, bottom: c.cy + c.side / 2 };
    if (c.kind === "output") return { x: c.x + 8, top: c.cy, bottom: c.cy };
    const b = sheetBox(c, which === "left" ? 0 : c.sheets - 1);
    const dx = which === "left" ? 0 : b.w, dy = which === "left" ? 0 : b.skew;
    return { x: b.x + dx, top: b.top + dy + b.h * 0.12, bottom: b.top + dy + b.h * 0.88 };
  }

  const cropRect = (u: number) => {
    const f = input.crop / 256, side = img.s * (1 - (1 - f) * u);
    return { x: img.x + (img.s - side) / 2, y: img.y + (img.s - side) / 2, s: side };
  };

  function text(value: string, x: number, y: number, size: number, alpha: number, align: CanvasTextAlign = "left") {
    ctx.font = `400 ${size}px ${font}`;
    ctx.textAlign = align;
    ctx.fillStyle = `rgba(${BONE}, ${alpha})`;
    ctx.fillText(value, x, y);
  }

  function glow(x: number, y: number, r: number, alpha: number) {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(${SIGNAL}, ${alpha})`);
    g.addColorStop(0.25, `rgba(${SIGNAL}, ${alpha * 0.45})`);
    g.addColorStop(1, `rgba(${SIGNAL}, 0)`);
    ctx.fillStyle = g;
    ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
  }

  function drawImage(t: number) {
    const { x, y, s } = img;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.strokeStyle = `rgba(${BONE}, 0.14)`;
    ctx.lineWidth = 1;
    ctx.strokeRect(x - 0.5, y - 0.5, s + 1, s + 1);
    const [pre] = input.phases;
    if (t < T_WAVE) {
      // Readout: rows above the current band are complete, the band fills from the left
      const p = t / READ * BANDS, band = Math.floor(p), part = p - band, rows = 256 / BANDS;
      const done = band * rows;
      if (done > 0) ctx.drawImage(pre, 0, 0, 256, done, x, y, s, (done / 256) * s);
      if (band < BANDS) {
        const by = y + (done / 256) * s, bh = (rows / 256) * s;
        ctx.drawImage(pre, 0, done, 256 * part, rows, x, by, s * part, bh);
        ctx.fillStyle = `rgba(${BONE}, 0.07)`;
        ctx.fillRect(x, by, s, bh);
        glow(x + s * part, by + bh / 2, 26, 0.9);
      }
      return;
    }
    ctx.drawImage(pre, x, y, s, s);
    // Contrast: a wavefront from the centre reveals the enhancement map over PRE
    const u = reduced ? 1 : ease((t - T_WAVE) / WAVE);
    const r = u * s * 0.75;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, s, s);
    ctx.clip();
    ctx.beginPath();
    ctx.arc(x + s / 2, y + s / 2, r, 0, Math.PI * 2);
    ctx.save();
    ctx.clip();
    ctx.drawImage(input.enhancement, x, y, s, s);
    ctx.restore();
    if (u < 1) {
      ctx.strokeStyle = `rgba(255, 252, 244, ${0.75 * (1 - u)})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawCrop(t: number) {
    if (t < T_CROP) return;
    const u = reduced ? 1 : ease((t - T_CROP) / CROP);
    const c = cropRect(u);
    ctx.save();
    ctx.fillStyle = `rgba(0, 0, 0, ${0.55 * u})`;
    ctx.beginPath();
    ctx.rect(img.x, img.y, img.s, img.s);
    ctx.rect(c.x, c.y, c.s, c.s);
    ctx.fill("evenodd");
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = `rgba(${BONE}, ${0.4 + 0.5 * u})`;
    ctx.lineWidth = 1.2;
    ctx.strokeRect(c.x, c.y, c.s, c.s);
    ctx.restore();
    if (u >= 1) text(`${input.crop} × ${input.crop}`, c.x + 6, c.y + 16, 12, 0.85);
  }

  function drawSheet(c: Column, k: number, light: number, image?: HTMLCanvasElement) {
    const b = sheetBox(c, k);
    const corners = [[b.x, b.top], [b.x + b.w, b.top + b.skew], [b.x + b.w, b.top + b.skew + b.h], [b.x, b.top + b.h]];
    ctx.beginPath();
    corners.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
    ctx.closePath();
    if (image) {
      ctx.save();
      ctx.clip();
      ctx.setTransform(dpr * b.w, dpr * b.skew, 0, dpr * b.h, dpr * b.x, dpr * b.top);
      const f = input.crop / 256, o = (256 * (1 - f)) / 2;
      ctx.drawImage(image, o, o, 256 * f, 256 * f, 0, 0, 1, 1);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.restore();
    } else {
      ctx.fillStyle = `rgba(${32 + 40 * light}, ${37 + 40 * light}, ${40 + 38 * light}, 0.96)`;
      ctx.fill();
    }
    ctx.strokeStyle = `rgba(${BONE}, ${0.2 + 0.7 * light})`;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  function drawNetwork(t: number) {
    const shown = reduced ? 1 : ease((t - T_NET + 500) / 700);
    if (shown <= 0) return;
    ctx.globalAlpha = shown;
    // Projection of the crop onto the input stack
    const c = cropRect(1), first = sheetBox(columns[0], 0);
    ctx.strokeStyle = `rgba(${BONE}, 0.16)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(c.x + c.s, c.y); ctx.lineTo(first.x, first.top);
    ctx.moveTo(c.x + c.s, c.y + c.s); ctx.lineTo(first.x, first.top + first.h);
    ctx.stroke();
    // Connections
    ctx.strokeStyle = `rgba(${BONE}, 0.09)`;
    ctx.beginPath();
    for (const group of links) for (const [x0, y0, x1, y1] of group) { ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); }
    ctx.stroke();
    // Columns
    columns.forEach((col, i) => {
      const light = activation[i];
      if (col.kind === "input") for (let k = 0; k < 3; k++) drawSheet(col, k, light, input.phases[k]);
      else if (col.kind === "stage") for (let k = 0; k < col.sheets; k++) drawSheet(col, k, light * (0.4 + 0.6 * (k + 1) / col.sheets));
      else if (col.kind === "vector") {
        const n = 12, h = col.side / n;
        for (let k = 0; k < n; k++) {
          ctx.fillStyle = `rgba(${BONE}, ${0.16 + 0.7 * light * ((k * 5) % 7) / 7})`;
          ctx.fillRect(col.x, col.cy - col.side / 2 + k * h + 1, 8, h - 2);
        }
      } else {
        const on = outputSeen ? 1 : 0;
        if (light > 0.02) glow(col.x + 8, col.cy, 34, 0.55 * light);
        ctx.beginPath();
        ctx.arc(col.x + 8, col.cy, 8, 0, Math.PI * 2);
        ctx.fillStyle = on ? `rgb(${BONE})` : "rgb(40, 46, 50)";
        ctx.fill();
        ctx.strokeStyle = `rgba(${BONE}, 0.5)`;
        ctx.stroke();
      }
      const labelX = col.kind === "output" ? col.x + 8 : col.x + col.width / 2;
      text(col.label, labelX, col.cy + Math.max(col.side / 2, 20) + 30, 13, 0.62, "center");
    });
    if (outputSeen) {
      const out = columns[columns.length - 1];
      text(`${percent(input.probability)} %`, out.x + 8, out.cy - 26, 22, 0.95, "center");
    }
    ctx.globalAlpha = 1;
  }

  function stepPulses(t: number) {
    for (let i = pulses.length - 1; i >= 0; i--) {
      const p = pulses[i];
      const u = (t - p.start) / HOP;
      if (u < 0) continue;
      if (u >= 1) {
        activation[p.column + 1] = 1;
        if (p.column + 1 === columns.length - 1) outputSeen = true;
        if (p.column + 1 < links.length) pulses[i] = { column: p.column + 1, link: (p.link * 3 + 1) % LINKS, start: p.start + HOP };
        else pulses.splice(i, 1);
        continue;
      }
      const [x0, y0, x1, y1] = links[p.column][p.link];
      const e = u * u * (3 - 2 * u);
      const hx = x0 + (x1 - x0) * e, hy = y0 + (y1 - y0) * e;
      const tail = Math.max(0, e - 0.35);
      const grad = ctx.createLinearGradient(x0 + (x1 - x0) * tail, y0 + (y1 - y0) * tail, hx, hy);
      grad.addColorStop(0, `rgba(${SIGNAL}, 0)`);
      grad.addColorStop(1, `rgba(${SIGNAL}, 0.9)`);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(x0 + (x1 - x0) * tail, y0 + (y1 - y0) * tail);
      ctx.lineTo(hx, hy);
      ctx.stroke();
      glow(hx, hy, 12, 0.9);
    }
  }

  function caption(t: number) {
    const f = input.crop;
    const line =
      t < T_WAVE ? "Fase PRE: imagen adquirida antes del contraste"
      : t < T_CROP ? "Llegada del contraste: diferencia EARLY − PRE"
      : t < T_NET ? `Recorte central de ${f} × ${f} píxeles que recibe la red`
      : !outputSeen ? `Cinco etapas convolucionales, ${thousands(input.params)} parámetros por modelo`
      : `Media calibrada de ${input.models} modelos: probabilidad de pCR del ${percent(input.probability)} %`;
    text(line, img.x, img.y + img.s + 32, 14, 0.82);
    text(input.sampleId, img.x + 8, img.y + 18, 12, 0.7);
    text(`${input.cohort}, test`, img.x + img.s - 8, img.y + 18, 12, 0.7, "right");
    const phase = t < T_WAVE ? "PRE" : t < T_CROP ? "EARLY − PRE" : "Realce sobre PRE";
    text(phase, img.x + 8, img.y + img.s - 10, 12, 0.7);
  }

  let start = 0, clock = 0, last = 0, frame = 0, visible = true;
  function draw(now: number) {
    if (!start) { start = now; last = now; }
    const dt = Math.min(64, now - last);
    clock += dt;
    last = now;
    const t = reduced ? T_NET + 4000 : clock;
    if (reduced) { activation.fill(0.5); outputSeen = true; }

    // Pulses: a first train when the network appears, then one every few seconds
    if (!reduced && t >= T_NET && t >= nextIdle && idleLeft > 0) {
      const train = nextIdle === 0 ? 4 : 1;
      if (nextIdle !== 0) idleLeft--;
      for (let k = 0; k < train; k++) pulses.push({ column: 0, link: (k * 2 + Math.floor(t / 997)) % LINKS, start: t + k * 140 });
      activation[0] = 1;
      nextIdle = t + (nextIdle === 0 ? 4200 : 2600 + (Math.floor(t) % 1400));
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);
    drawImage(t);
    drawCrop(t);
    drawNetwork(t);
    if (!reduced) stepPulses(t);
    caption(t);
    if (!reduced) for (let i = 0; i < activation.length; i++) activation[i] *= Math.exp(-dt / 500);
    finished = reduced || (idleLeft === 0 && pulses.length === 0 && activation.every((a) => a < 0.01));
    frame = visible && !finished ? requestAnimationFrame(draw) : 0;
  }

  layout();
  new ResizeObserver(() => { layout(); if (!frame) draw(performance.now()); }).observe(canvas);
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting && !document.hidden;
    if (visible && !frame && !finished) { last = performance.now(); frame = requestAnimationFrame(draw); }
  }).observe(canvas);
  document.addEventListener("visibilitychange", () => {
    visible = !document.hidden;
    if (visible && !frame && !finished) { last = performance.now(); frame = requestAnimationFrame(draw); }
  });
  draw(performance.now());
}
