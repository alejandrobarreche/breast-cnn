// Training augmentation of breast_cnn.data.transforms, drawn with canvas transforms: one random
// rotation, isotropic scale and translation (RandomAffine) and then a horizontal flip, applied
// with the same geometry to PRE, EARLY and LATE. affine_grid maps an output point p to the
// input point c + R(θ)/s · (p − c) + d, so the input is drawn with the inverse transform.
// Bilinear sampling and black (zero) borders, as grid_sample in training; for display only.

export interface Augmentation { angle: number; scale: number; dx: number; dy: number; flip: boolean }

export interface AugmentationRanges { rotation_degrees: number; max_translation: number; scale_range: number[]; horizontal_flip_probability: number }

/** Draw one set of parameters with the distributions used in training. */
export function sample(r: AugmentationRanges, size: number, random = Math.random): Augmentation {
  const uniform = (a: number, b: number) => a + random() * (b - a);
  return {
    angle: uniform(-r.rotation_degrees, r.rotation_degrees),
    scale: uniform(r.scale_range[0], r.scale_range[1]),
    // A shift of 2·t in [-1, 1] grid units is t of the width
    dx: uniform(-r.max_translation, r.max_translation) * size,
    dy: uniform(-r.max_translation, r.max_translation) * size,
    flip: random() < r.horizontal_flip_probability,
  };
}

/** The phase drawn on a new canvas of the same size with the augmentation applied. */
export function apply(source: HTMLCanvasElement, a: Augmentation): HTMLCanvasElement {
  const size = source.width, c = size / 2;
  const out = document.createElement("canvas");
  out.width = out.height = size;
  const ctx = out.getContext("2d")!;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingEnabled = true;
  if (a.flip) { ctx.translate(size, 0); ctx.scale(-1, 1); }
  ctx.translate(c, c);
  ctx.scale(a.scale, a.scale);
  ctx.rotate((-a.angle * Math.PI) / 180);
  ctx.translate(-c - a.dx, -c - a.dy);
  ctx.drawImage(source, 0, 0);
  return out;
}
