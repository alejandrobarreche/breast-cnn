// Occlusion sensitivity: the central crop is covered patch by patch with zeros (the value the
// augmentation leaves at uncovered borders) and the calibrated probability is computed again.
// The map is the change with respect to the uncovered slice; it shows what moves this model's
// output, not where the tumour is or what causes a response.

export interface OcclusionMap { size: number; patch: number; offset: number; deltas: Float32Array; base: number }

export async function occlusion(
  tensor: Float32Array,
  side: number,
  crop: number,
  patch: number,
  predict: (input: Float32Array) => Promise<number>,
  onProgress: (done: number, total: number) => void,
  cancelled: () => boolean,
): Promise<OcclusionMap | null> {
  const n = crop / patch, offset = (side - crop) / 2, plane = side * side;
  const base = await predict(tensor);
  const deltas = new Float32Array(n * n);
  const input = new Float32Array(tensor.length);
  for (let i = 0; i < n * n; i++) {
    if (cancelled()) return null;
    input.set(tensor);
    const top = offset + Math.floor(i / n) * patch, left = offset + (i % n) * patch;
    for (let c = 0; c < 3; c++)
      for (let y = top; y < top + patch; y++) input.fill(0, c * plane + y * side + left, c * plane + y * side + left + patch);
    deltas[i] = (await predict(input)) - base;
    onProgress(i + 1, n * n);
  }
  return { size: n, patch, offset, deltas, base };
}
