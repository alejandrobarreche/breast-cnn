// Copy the onnxruntime-web WebAssembly runtime into public/ort/, served as static files.
import { copyFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(root, "node_modules/onnxruntime-web/dist");
const target = join(root, "public/ort");
mkdirSync(target, { recursive: true });
const files = readdirSync(source).filter((name) => /^ort-wasm.*\.(wasm|mjs)$/.test(name));
if (files.length === 0) throw new Error(`No ort-wasm files in ${source}; run npm install`);
for (const name of files) copyFileSync(join(source, name), join(target, name));
console.log(`onnxruntime-web: ${files.length} files in public/ort/`);
