import { describe, expect, it } from "vitest";
import manifest from "../src/manifest.json";
import { steps } from "../src/layers";
import { positiveWeight, summarize, type Patient } from "../src/split";

const arch = manifest.architecture;

describe("layer table", () => {
  const table = steps(arch.stages, arch.kernel_size, 3, arch.input_crop, arch.models);

  it("reproduces the parameter counts exported from PyTorch", () => {
    arch.stages.forEach((stage, i) => expect(table[i + 1].params).toBe(stage.params));
    expect(table.reduce((a, s) => a + s.params, 0)).toBe(arch.params);
  });

  it("grows the theoretical receptive field to 156 pixels", () => {
    expect(table.slice(1, 6).map((s) => s.field)).toEqual([6, 16, 36, 76, 156]);
  });
});

describe("patient split", () => {
  it("gives the pos_weight that training used on fold 0", () => {
    // runs/20261018-1330_wide_crop128_fixed/metrics.json, fold 0
    const train = (manifest.split.train as Patient[]).filter((p) => p[0] !== 0);
    expect(positiveWeight(summarize(train))).toBeCloseTo(2.4002329192546585, 12);
  });
});
