from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import SequentialSampler

from breast_cnn.configuration import PipelineConfig, load_phase_one_config
from breast_cnn.data.dataset import BreastDCEDataset
from breast_cnn.data.loaders import central_slices, create_data_loaders
from breast_cnn.data.metadata import load_samples, summarize_folds
from breast_cnn.data.transforms import RandomAffine, RandomHorizontalFlip, load_dce_triplet
from breast_cnn.data.validation import validate_sample_frame


def test_dataset_preserves_order_scaling_ids_and_single_batch(dataset_project: Path) -> None:
    samples = load_samples(dataset_project).iloc[[1, 0]]
    row = samples.iloc[0]
    for column, value in zip(("path_pre", "path_early", "path_late"), (0, 128, 255), strict=True):
        Image.new("L", (256, 256), value).save(
            dataset_project / row[column].replace("dataset/", "data/", 1)
        )
    dataset = BreastDCEDataset(samples, dataset_project, path_aliases={"dataset": "data"})
    item = dataset[0]
    assert item["sample_id"] == row.sample_id
    assert item["patient_id"] == row.patient_id
    assert item["slice_index"] == row.slice_index
    assert item["label"].dtype == torch.float32
    assert item["label"].shape == ()
    assert item["label"].item() == 1
    assert item["image"].dtype == torch.float32
    assert item["image"].shape == (3, 256, 256)
    torch.testing.assert_close(item["image"][:, 0, 0], torch.tensor([0, 128 / 255, 1]))
    # Copying the frame prevents later caller edits from changing label/image correspondence.
    samples.loc[samples.index[0], "pCR"] = 0
    assert dataset[0]["label"].item() == 1
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    assert batch["image"].shape == (1, 3, 256, 256)
    assert batch["label"].shape == (1,)


@pytest.mark.parametrize("kind", ["corrupt", "rgb", "size", "16bit", "jpeg", "missing"])
def test_dataset_rejects_bad_images_with_sample_and_phase(dataset_project: Path, kind: str) -> None:
    path = dataset_project / "data/train/p1/p1_z001_EARLY.png"
    if kind == "corrupt":
        path.write_bytes(b"invalid PNG")
    elif kind == "missing":
        path.unlink()
    elif kind == "16bit":
        Image.fromarray(np.ones((256, 256), dtype=np.uint16)).save(path)
    else:
        Image.new("RGB" if kind == "rgb" else "L", (32, 32) if kind == "size" else (256, 256)).save(
            path, format="JPEG" if kind == "jpeg" else "PNG"
        )
    dataset = BreastDCEDataset(
        load_samples(dataset_project).iloc[[0]], dataset_project, path_aliases={"dataset": "data"}
    )
    with pytest.raises(ValueError, match="p1_z001.*EARLY"):
        dataset[0]


def test_preprocessing_rejects_duplicate_phases(dataset_project: Path) -> None:
    path = dataset_project / "data/train/p1/p1_z001_PRE.png"
    with pytest.raises(ValueError, match="distinct"):
        load_dce_triplet(path, path, path)


def test_synthetic_geometry_keeps_phases_and_signed_enhancement_aligned() -> None:
    pattern = torch.arange(256 * 256).reshape(256, 256).float() / (256 * 256)
    image = torch.stack([pattern, pattern * 0.5, pattern * 0.25])
    transformed = RandomHorizontalFlip(1)(image)
    torch.testing.assert_close(transformed[0], pattern.flip(-1))
    torch.testing.assert_close(transformed[1], transformed[0] * 0.5)
    torch.testing.assert_close(transformed[2], transformed[0] * 0.25)
    torch.testing.assert_close(transformed[1] - transformed[0], (image[1] - image[0]).flip(-1))
    assert (transformed[1] - transformed[0]).min() < 0
    assert torch.equal(RandomHorizontalFlip(0)(image), image)


def test_validation_forbids_augmentation_and_training_forbids_test(dataset_project: Path) -> None:
    samples = load_samples(dataset_project)
    with pytest.raises(ValueError, match="Augmentation is forbidden"):
        BreastDCEDataset(samples.iloc[[0]], dataset_project, horizontal_flip_probability=0.5)
    with pytest.raises(ValueError, match="only public train"):
        BreastDCEDataset(samples, dataset_project, training=True)


@pytest.mark.parametrize(
    "column,value",
    [
        ("pCR", 2),
        ("fold", -1),
        ("slice_index", -1),
        ("slice_index", 2),
        ("patient_id", "wrong"),
        ("path_early", "dataset/train/p1/p1_z001_PRE.png"),
    ],
)
def test_invalid_index_fails_before_image_loading(
    dataset_project: Path, column: str, value
) -> None:
    samples = load_samples(dataset_project)
    samples.loc[0, column] = value
    with pytest.raises(ValueError):
        BreastDCEDataset(samples, dataset_project)


@pytest.mark.parametrize("column", ["split", "fold", "pCR"])
def test_inconsistent_patient_assignment_is_rejected(dataset_project: Path, column: str) -> None:
    samples = load_samples(dataset_project)
    extra = samples.iloc[[0]].copy()
    extra["sample_id"] = "p1_z002"
    extra["slice_index"] = 2
    for name in ("path_pre", "path_early", "path_late"):
        extra[name] = extra[name].str.replace("z001", "z002")
    extra[column] = {"split": "test", "fold": 1, "pCR": 1}[column]
    if column == "split":
        extra["fold"] = -1
    with pytest.raises(ValueError, match="constant per patient"):
        validate_sample_frame(pd.concat([samples, extra], ignore_index=True))


def five_fold_index() -> pd.DataFrame:
    rows = []
    for fold in range(5):
        for slice_index in range(3):
            pid = f"p{fold}"
            sample = f"{pid}_z{slice_index:03}"
            rows.append(
                {
                    "sample_id": sample,
                    "patient_id": pid,
                    "slice_index": slice_index,
                    "split": "train",
                    "fold": fold,
                    "pCR": fold % 2,
                    **{
                        f"path_{phase.lower()}": f"dataset/train/{pid}/{sample}_{phase}.png"
                        for phase in ("PRE", "EARLY", "LATE")
                    },
                }
            )
    return pd.DataFrame(rows)


def test_five_fold_coverage_and_missing_fold_error() -> None:
    samples = five_fold_index()
    summary = summarize_folds(samples)
    assert summary.validation_patients.tolist() == [1] * 5
    assert summary.validation_slices.tolist() == [3] * 5
    assert summary.training_patients.tolist() == [4] * 5
    assert summary.training_slices.tolist() == [12] * 5
    with pytest.raises(ValueError, match="all five"):
        summarize_folds(samples[samples.fold.ne(4)])


@pytest.mark.parametrize("num_workers", [0, 1])
def test_loaders_keep_last_batch_validation_order_and_seed(
    tmp_path: Path, num_workers: int
) -> None:
    samples = five_fold_index()
    pattern = np.zeros((256, 256), dtype=np.uint8)
    pattern[:, :40] = 220
    for row in samples.itertuples():
        for column in ("path_pre", "path_early", "path_late"):
            path = tmp_path / getattr(row, column)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pattern).save(path)

    def collect():
        training, validation = create_data_loaders(
            samples,
            tmp_path,
            validation_fold=0,
            batch_size=2,
            seed=123,
            num_workers=num_workers,
            horizontal_flip_probability=0.5,
        )
        assert not training.drop_last and not validation.drop_last
        assert isinstance(validation.sampler, SequentialSampler)
        train_batches = list(training)
        first, second = list(validation), list(validation)
        assert [item for batch in first for item in batch["sample_id"]] == samples.loc[
            samples.fold.eq(0), "sample_id"
        ].tolist()
        assert len(first[-1]["label"]) == 1
        for a, b in zip(first, second, strict=True):
            assert torch.equal(a["image"], b["image"])
            assert torch.equal(a["label"], b["label"])
        # Validation does not use the training flip even when that option is active.
        assert first[0]["image"][0, 0, 0, 0] > first[0]["image"][0, 0, 0, -1]
        return train_batches

    first, second = collect(), collect()
    for a, b in zip(first, second, strict=True):
        assert a["sample_id"] == b["sample_id"]
        assert torch.equal(a["image"], b["image"])


def test_loader_excludes_test_even_if_its_files_are_missing(dataset_project: Path) -> None:
    samples = load_samples(dataset_project)
    for path in (dataset_project / "data/test/p3").iterdir():
        path.unlink()
    training, validation = create_data_loaders(
        samples,
        dataset_project,
        validation_fold=0,
        batch_size=8,
        seed=1,
        path_aliases={"dataset": "data"},
    )
    assert next(iter(training))["patient_id"] == ["p2"]
    assert next(iter(validation))["patient_id"] == ["p1"]


def test_invalid_configuration_fails_before_loading() -> None:
    from pydantic import ValidationError

    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValidationError):
        load_phase_one_config(root, ["pipeline.batch_size=0"])
    with pytest.raises(ValidationError):
        load_phase_one_config(root, ["data.expected_image_size=[128,128]"])
    with pytest.raises(ValidationError):
        PipelineConfig(
            validation_fold=True,
            batch_size=1,
            num_workers=0,
            seed=1,
            horizontal_flip_probability=0.0,
            report_path="unused.json",
        )


def test_random_affine_uses_one_grid_for_all_phases() -> None:
    torch.manual_seed(0)
    base = torch.rand(1, 256, 256)
    image = torch.cat([base, 0.5 * base, 0.25 * base])
    affine = RandomAffine(rotation_degrees=15, max_translation=0.1, scale_range=(0.9, 1.1))
    output = affine(image)
    assert output.shape == image.shape
    assert not torch.equal(output, image)
    torch.testing.assert_close(output[1], 0.5 * output[0])
    torch.testing.assert_close(output[2], 0.25 * output[0])
    assert torch.equal(RandomAffine()(image), image)


def test_affine_augmentation_is_rejected_outside_training(dataset_project: Path) -> None:
    samples = load_samples(dataset_project).iloc[[0]]
    with pytest.raises(ValueError, match="Augmentation is forbidden"):
        BreastDCEDataset(samples, dataset_project, affine=RandomAffine(rotation_degrees=10))


def test_central_slices_keeps_middle_of_each_patient() -> None:
    samples = pd.DataFrame(
        {"patient_id": ["a"] * 5 + ["b"] * 2, "slice_index": [9, 5, 6, 7, 8, 3, 4]}
    )
    kept = central_slices(samples, 3)
    assert sorted(kept.loc[kept.patient_id == "a", "slice_index"]) == [6, 7, 8]
    assert sorted(kept.loc[kept.patient_id == "b", "slice_index"]) == [3, 4]
