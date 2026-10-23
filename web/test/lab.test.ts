import { describe as group, expect, it } from "vitest";
import { describe, mean, type Experiment } from "../src/lab";
import runs from "../src/runs.json";

const experiments = runs.experiments as Experiment[];
const changes = (name: string) =>
  describe(experiments.find((e) => e.name === name)!.changes as Record<string, unknown>, runs.reference_settings as Record<string, unknown>);

group("experiment lab", () => {
  it("names what each run changed with respect to the reference", () => {
    expect(changes("baseline_weighted")).toEqual([]);
    expect(changes("baseline_unweighted")).toEqual(["BCE sin ponderar"]);
    expect(changes("patient_max")).toEqual(["pérdida por paciente sobre el máximo de los logits de sus cortes"]);
    expect(changes("wide")).toEqual(["canales 16 → 256 (referencia 8 → 128)"]);
    expect(changes("separable")).toEqual(["convoluciones separables"]);
  });

  it("marks one final run, read at its last epoch", () => {
    const final = experiments.filter((e) => e.final);
    expect(final.map((e) => [e.name, e.selection])).toEqual([["wide_crop128_fixed", "last"]]);
    expect(mean(final[0].auc)).toBeCloseTo(0.559, 3);
  });
});
