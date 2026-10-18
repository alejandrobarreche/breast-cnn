import { describe, expect, it } from "vitest";
import { checkHeader, checkNames, parseName, SIZE, toTensor } from "../src/preprocess";

function header(width: number, height: number, bitDepth: number, colourType: number): Uint8Array {
  const bytes = new Uint8Array(33);
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const view = new DataView(bytes.buffer);
  view.setUint32(8, 13);
  bytes.set([0x49, 0x48, 0x44, 0x52], 12); // IHDR
  view.setUint32(16, width);
  view.setUint32(20, height);
  bytes[24] = bitDepth;
  bytes[25] = colourType;
  return bytes;
}

describe("file names", () => {
  it("reads slice and phase", () => {
    expect(parseName("ISPY1_1001_z012_early.png")).toEqual({ sample: "ISPY1_1001_z012", phase: "EARLY" });
    expect(parseName("slice.png")).toBeNull();
  });

  it("reports missing, duplicated and mixed phases", () => {
    expect(checkNames(["a_PRE.png", "a_EARLY.png", "a_LATE.png"]).errors).toEqual([]);
    expect(checkNames(["a_PRE.png", "a_EARLY.png"]).errors).toEqual(["Falta la fase LATE."]);
    expect(checkNames(["a_PRE.png", "b_PRE.png", "a_EARLY.png", "a_LATE.png"]).errors[0]).toMatch(/dos archivos de la fase PRE/);
    expect(checkNames(["a_PRE.png", "b_EARLY.png", "a_LATE.png"]).errors).toEqual(["Los archivos son de cortes distintos: a, b."]);
  });
});

describe("PNG header", () => {
  it("accepts only 8-bit grayscale 256 x 256", () => {
    expect(checkHeader(header(256, 256, 8, 0))).toBeNull();
    expect(checkHeader(header(512, 512, 8, 0))).toMatch(/512 × 512/);
    expect(checkHeader(header(256, 256, 8, 2))).toMatch(/escala de grises/);
    expect(checkHeader(header(256, 256, 16, 0))).toMatch(/8 bits/);
    expect(checkHeader(new Uint8Array(40))).toMatch(/no es un PNG/);
  });
});

describe("tensor", () => {
  it("stacks PRE, EARLY, LATE and divides by 255", () => {
    const n = SIZE * SIZE;
    const phase = (value: number) => new Uint8ClampedArray(4 * n).fill(value);
    const tensor = toTensor([phase(0), phase(51), phase(255)]);
    expect(tensor.length).toBe(3 * n);
    expect([tensor[0], tensor[n], tensor[2 * n + n - 1]]).toEqual([0, Math.fround(51 / 255), 1]);
  });
});
