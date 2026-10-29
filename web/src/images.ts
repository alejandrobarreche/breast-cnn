// Image helpers shared by Analizar and El estudio: PNG decoding without colour conversion, the
// views of a slice and the enhancement map EARLY − PRE coloured with ENHANCEMENT_CMAP of style.py.
import { PHASES, SIZE } from "./preprocess";

// ENHANCEMENT_CMAP of style.py: no-pCR teal, paper white, pCR coral, on [-1, 1].
const STOPS = [[0x23, 0x65, 0x6e], [0xfa, 0xfa, 0xf8], [0xdc, 0x78, 0x67]];
export function enhancementColour(value: number): number[] {
  const u0 = (Math.max(-1, Math.min(1, value)) + 1) / 2;
  const [a, b, u] = u0 < 0.5 ? [STOPS[0], STOPS[1], u0 * 2] : [STOPS[1], STOPS[2], (u0 - 0.5) * 2];
  return a.map((c, k) => Math.round(c + (b[k] - c) * u));
}

// EARLY − PRE over the PRE phase: the colour of ENHANCEMENT_CMAP, with an opacity that grows
// with |EARLY − PRE| and is full from a difference of FULL_AT. Pixels that do not change keep PRE.
const FULL_AT = 0.25;
export function enhancement(pre: ArrayLike<number>, early: ArrayLike<number>) {
  const out = new ImageData(SIZE, SIZE);
  for (let p = 0; p < SIZE * SIZE * 4; p += 4) {
    const value = (early[p] - pre[p]) / 255;
    const colour = enhancementColour(value);
    const alpha = Math.min(1, Math.abs(value) / FULL_AT);
    for (let c = 0; c < 3; c++) out.data[p + c] = Math.round(pre[p] * (1 - alpha) + colour[c] * alpha);
    out.data[p + 3] = 255;
  }
  return out;
}

export function canvasOf(data: ImageData) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = data.width;
  canvas.getContext("2d")!.putImageData(data, 0, 0);
  return canvas;
}

/** PRE, EARLY, LATE and the enhancement map of a slice, as canvases. */
export function viewsOf(pixels: Uint8ClampedArray[]) {
  return [...pixels.map((p) => canvasOf(new ImageData(new Uint8ClampedArray(p), SIZE, SIZE))), canvasOf(enhancement(pixels[0], pixels[1]))];
}

export async function decode(blob: Blob): Promise<Uint8ClampedArray> {
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width; canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  return context.getImageData(0, 0, canvas.width, canvas.height).data;
}

export async function loadPhases(base: string, files: Record<string, string>): Promise<Uint8ClampedArray[]> {
  const pixels = [];
  for (const phase of PHASES) pixels.push(await decode(await (await fetch(`${base}${files[phase]}`)).blob()));
  return pixels;
}
