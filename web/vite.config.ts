import { defineConfig } from "vite";

// Relative base so the build works from any GitHub Pages path.
// onnxruntime-web loads its WebAssembly from public/ort/ (scripts/copy-ort.mjs), so Vite
// must not pre-bundle it.
export default defineConfig({
  base: "./",
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "es2022" },
});
