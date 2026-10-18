// Input contract of the model, mirrored from breast_cnn.data.transforms.load_dce_triplet:
// three 8-bit grayscale PNG of 256 x 256, channels PRE, EARLY, LATE, values / 255.
// The web parity test (test/parity.test.ts) checks that this gives the Python outputs.

export const SIZE = 256;
export const PHASES = ["PRE", "EARLY", "LATE"] as const;
export type Phase = (typeof PHASES)[number];
export const MAX_BYTES = 2 * 1024 * 1024;

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/** Phase and slice name from a file name such as ISPY1_1001_z012_EARLY.png. */
export function parseName(name: string): { sample: string; phase: Phase } | null {
  const match = name.match(/^(.+)_(PRE|EARLY|LATE)\.png$/i);
  return match ? { sample: match[1], phase: match[2].toUpperCase() as Phase } : null;
}

/** Problems of a set of file names, in the order a person would fix them. */
export function checkNames(names: string[]): { byPhase: Partial<Record<Phase, number>>; errors: string[] } {
  const byPhase: Partial<Record<Phase, number>> = {};
  const errors: string[] = [];
  const samples = new Set<string>();
  names.forEach((name, index) => {
    const parsed = parseName(name);
    if (!parsed) {
      errors.push(`${name}: el nombre debe terminar en _PRE.png, _EARLY.png o _LATE.png.`);
      return;
    }
    if (byPhase[parsed.phase] !== undefined) {
      errors.push(`Hay dos archivos de la fase ${parsed.phase}; deja solo uno.`);
      return;
    }
    byPhase[parsed.phase] = index;
    samples.add(parsed.sample);
  });
  for (const phase of PHASES) if (byPhase[phase] === undefined) errors.push(`Falta la fase ${phase}.`);
  if (samples.size > 1) errors.push(`Los archivos son de cortes distintos: ${[...samples].join(", ")}.`);
  return { byPhase, errors };
}

/** Header problem of a PNG file (signature and IHDR), or null when it is 8-bit grayscale 256 x 256. */
export function checkHeader(bytes: Uint8Array): string | null {
  if (bytes.length < 33 || PNG_SIGNATURE.some((b, i) => bytes[i] !== b)) return "no es un PNG válido.";
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const width = view.getUint32(16), height = view.getUint32(20);
  const bitDepth = bytes[24], colourType = bytes[25];
  if (width !== SIZE || height !== SIZE) return `mide ${width} × ${height}; se esperan 256 × 256.`;
  if (colourType !== 0 || bitDepth !== 8) return "debe ser escala de grises de 8 bits.";
  return null;
}

/** Model input [1, 3, 256, 256] from the RGBA pixels of PRE, EARLY and LATE. */
export function toTensor(phases: ArrayLike<number>[]): Float32Array {
  if (phases.length !== 3) throw new Error("Three phases are required");
  const n = SIZE * SIZE;
  const out = new Float32Array(3 * n);
  phases.forEach((rgba, c) => {
    if (rgba.length !== 4 * n) throw new Error(`Phase ${PHASES[c]} is not 256 x 256 RGBA`);
    for (let p = 0; p < n; p++) out[c * n + p] = rgba[4 * p] / 255;
  });
  return out;
}
