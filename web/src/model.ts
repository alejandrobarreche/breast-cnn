// The calibrated ensemble (scripts/export_onnx.py) run with ONNX Runtime on WebAssembly.
// One thread: deterministic, and no cross-origin isolation needed on static hosting.
import * as ort from "onnxruntime-web/wasm";
import { SIZE } from "./preprocess";

ort.env.wasm.numThreads = 1;
// Absolute URL: a relative path would resolve against the bundled script in assets/.
ort.env.wasm.wasmPaths = new URL(`${import.meta.env.BASE_URL}ort/`, document.baseURI).href;

export interface Prediction {
  foldProbabilities: number[];
  probability: number;
  calibrated: number;
  foldCalibrated: number[];
  milliseconds: number;
}

export interface Model {
  sha256: string;
  predict(tensor: Float32Array): Promise<Prediction>;
}

const hex = (buffer: ArrayBuffer) =>
  [...new Uint8Array(buffer)].map((b) => b.toString(16).padStart(2, "0")).join("");

/** Download the model, check its SHA-256 against the manifest and open a session. */
export async function loadModel(url: string, expectedSha256: string): Promise<Model> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`No se pudo descargar el modelo (${response.status}).`);
  const bytes = await response.arrayBuffer();
  const sha256 = hex(await crypto.subtle.digest("SHA-256", bytes));
  if (sha256 !== expectedSha256) throw new Error("El modelo descargado no coincide con su checksum.");
  const session = await ort.InferenceSession.create(new Uint8Array(bytes), {
    executionProviders: ["wasm"],
  });
  return {
    sha256,
    async predict(tensor) {
      const start = performance.now();
      const input = new ort.Tensor("float32", tensor, [1, 3, SIZE, SIZE]);
      const out = await session.run({ images: input });
      const milliseconds = performance.now() - start;
      const values = (name: string) => Array.from(out[name].data as Float32Array);
      return {
        foldProbabilities: values("fold_probabilities"),
        probability: values("probability")[0],
        calibrated: values("calibrated")[0],
        foldCalibrated: values("fold_calibrated"),
        milliseconds,
      };
    },
  };
}
