import { describe, expect, it } from "vitest";
import results from "../src/results.json";
import { auc, confusion, roc } from "../src/results";

const test = results.test as number[][], val = results.validation as number[][];

describe("results charts", () => {
  it("reproduces the test patient AUC of evaluate_test.py", () => {
    // artifacts/metrics/test_evaluation_patient.json, auc.value
    expect(auc(roc(test.map((p) => p[0]), test.map((p) => p[1])))).toBeCloseTo(0.5905813775118883, 12);
  });

  it("reproduces the per-fold validation AUC of fit_calibration.py", () => {
    // artifacts/metrics/calibration_validation_patient.json, folds.0.auc (nested out-of-fold ensemble)
    const fold = val.filter((p) => p[0] === 0);
    expect(auc(roc(fold.map((p) => p[1]), fold.map((p) => p[2])))).toBeCloseTo(0.5339717741935484, 12);
  });

  it("reproduces the test confusion matrix at the threshold", () => {
    // artifacts/metrics/test_evaluation_patient_corrected.json, at_threshold
    expect(confusion(test.map((p) => p[0]), test.map((p) => p[2]), results.threshold)).toEqual({ tp: 12, fn: 41, tn: 109, fp: 14 });
  });
});
