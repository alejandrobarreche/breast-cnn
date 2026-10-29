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
export async function loadModel(url: string, expectedSha256: string, size: number, onProgress: (fraction: number) => void): Promise<Model> {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`No se pudo descargar el modelo (${response.status}).`);
  // Read the stream to report the download as a fraction of the size in the manifest
  const reader = response.body.getReader(), chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(Math.min(1, received / size));
  }
  const bytes = new Uint8Array(received);
  chunks.reduce((offset, chunk) => (bytes.set(chunk, offset), offset + chunk.length), 0);
  const sha256 = hex(await crypto.subtle.digest("SHA-256", bytes));
  if (sha256 !== expectedSha256) throw new Error("El modelo descargado no coincide con su checksum.");
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
  });
  // A session runs one inference at a time: calls wait for the previous one (prediction and
  // occlusion map may be requested together)
  let queue: Promise<unknown> = Promise.resolve();
  return {
    sha256,
    predict(tensor) {
      const run = queue.then(() => infer(tensor));
      queue = run.catch(() => {});
      return run;
    },
  };

  async function infer(tensor: Float32Array): Promise<Prediction> {
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
  }
}
