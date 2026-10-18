// The web preprocessing, applied to the example PNG files, must give the outputs that
// ONNX Runtime gave in Python (scripts/export_web.py) for the same model.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import * as ort from "onnxruntime-node";
import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";
import manifest from "../src/manifest.json";
import { PHASES, SIZE, toTensor } from "../src/preprocess";

const PUBLIC = join(__dirname, "../public");
const MODEL = join(PUBLIC, manifest.model.file);
const TOLERANCE = 1e-5;

describe.skipIf(!existsSync(MODEL))("parity with Python", () => {
  it("reproduces every example", async () => {
    const session = await ort.InferenceSession.create(MODEL);
    for (const example of manifest.examples) {
      // pngjs expands grayscale to RGBA, as the browser canvas does
      const pixels = PHASES.map((phase) => PNG.sync.read(readFileSync(join(PUBLIC, example.files[phase]))).data);
      const input = new ort.Tensor("float32", toTensor(pixels), [1, 3, SIZE, SIZE]);
      const out = await session.run({ images: input });
      for (const [name, expected] of Object.entries(example.expected)) {
        const actual = Array.from(out[name].data as Float32Array);
        actual.forEach((value, k) => expect(Math.abs(value - expected[k])).toBeLessThan(TOLERANCE));
      }
    }
  });
});
