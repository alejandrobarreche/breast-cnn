from pathlib import Path

import pandas as pd
import pytest

from breast_cnn.data.exploration import explore_train, select_examples
from breast_cnn.data.metadata import load_patients, load_samples


def test_train_eda_does_not_include_test_and_preserves_missing(dataset_project: Path) -> None:
    samples, patients = load_samples(dataset_project), load_patients(dataset_project)
    patients["age"] = [30.0, float("nan"), 99999.0]
    eda = explore_train(samples, patients)
    assert set(eda.patients.pid) == {"p1", "p2"}
    assert set(eda.samples.patient_id) == {"p1", "p2"}
    assert eda.tables["class_balance"]["count"].tolist() == [1, 1, 1, 1]
    age = eda.tables["numeric_summary"].set_index("column").loc["age"]
    assert age["count"] == 1 and age["max"] == 30
    assert eda.tables["missing_values"].set_index("column").loc["age", "missing"] == 1
    assert eda.patients.age.isna().sum() == 1
    assert set(eda.tables["cohort_summary"]["data"]) == {"cohort_a"}
    pd.testing.assert_frame_equal(samples, load_samples(dataset_project))
    # Modifying test clinical values cannot affect any descriptive table.
    patients.loc[patients.split.eq("test"), ["age", "data"]] = [0, "unseen"]
    second = explore_train(samples, patients)
    for name, frame in eda.tables.items():
        pd.testing.assert_frame_equal(frame, second.tables[name])


def test_example_selection_is_train_only_and_independent_of_row_order(
    dataset_project: Path,
) -> None:
    samples, patients = load_samples(dataset_project), load_patients(dataset_project)
    first = select_examples(explore_train(samples, patients), seed=10)
    second = select_examples(explore_train(samples.iloc[::-1], patients.iloc[::-1]), seed=10)
    assert first.sample_id.tolist() == second.sample_id.tolist()
    assert set(first.split) == {"train"}
    assert len(first) == 2


@pytest.mark.parametrize("problem", ["duplicate", "orphan", "label"])
def test_train_join_rejects_invalid_metadata(dataset_project: Path, problem: str) -> None:
    samples, patients = load_samples(dataset_project), load_patients(dataset_project)
    if problem == "duplicate":
        patients = pd.concat([patients, patients.iloc[[0]]])
    elif problem == "orphan":
        patients = patients.iloc[1:]
    else:
        patients.loc[0, "pCR"] = 1
    with pytest.raises(ValueError):
        explore_train(samples, patients)
