import { describe, expect, it } from "vitest";
import { kinetics, windowMean } from "../src/uptake";

describe("uptake curve", () => {
  it("classifies the delayed pattern with the ±10 % rule", () => {
    expect(kinetics(0.2, 0.6, 0.7).pattern).toBe("persistente");
    expect(kinetics(0.2, 0.6, 0.62).pattern).toBe("meseta");
    expect(kinetics(0.2, 0.6, 0.5).pattern).toBe("lavado");
    expect(kinetics(0.3, 0.32, 0.1).pattern).toBe("sin realce");
  });

  it("averages a window clipped at the border", () => {
    const n = 4, rgba = new Uint8Array(4 * n * n);
    rgba[0] = 255; // pixel (0, 0)
    expect(windowMean(rgba, n, 0, 0, 1)).toBeCloseTo(1 / 4);
  });
});
