from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest
import requests
from PIL import Image

from breast_cnn.data.download import (
    DownloadItem,
    build_download_plan,
    download_image,
    download_images,
)
from breast_cnn.data.metadata import load_samples, resolve_image_path
from breast_cnn.data.validation import audit_dataset


class FakeResponse:
    def __init__(self, content: bytes, *, interrupted: bool = False) -> None:
        self.content = content
        self.interrupted = interrupted

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        yield self.content
        if self.interrupted:
            raise requests.ConnectionError("interrupted transfer")


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.urls: list[str] = []

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.urls.append(url)
        return self.response

    def mount(self, *args) -> None:
        pass

    def close(self) -> None:
        pass


def png_bytes() -> bytes:
    content = io.BytesIO()
    Image.new("L", (256, 256), color=64).save(content, format="PNG")
    return content.getvalue()


def test_plan_maps_remote_paths_to_existing_loader_layout(dataset_project: Path) -> None:
    samples = load_samples(dataset_project)
    originals = (dataset_project / "metadata/samples.csv").read_bytes()
    items = build_download_plan(samples, dataset_project, {"dataset": "data"})

    assert len(items) == 9
    assert items[0].remote_path == "dataset/train/p1/p1_z001_PRE.png"
    for item in items:
        assert item.destination == resolve_image_path(
            item.remote_path, dataset_project, {"dataset": "data"}
        )
    assert (dataset_project / "metadata/samples.csv").read_bytes() == originals


def test_plan_filters_patients_after_split(dataset_project: Path) -> None:
    items = build_download_plan(
        load_samples(dataset_project),
        dataset_project,
        {"dataset": "data"},
        split="test",
        patients=1,
    )
    assert len(items) == 3
    assert all(item.remote_path.startswith("dataset/test/p3/") for item in items)


@pytest.mark.parametrize("path", ["../outside.png", "/tmp/PRE.png", "dataset/test/p1/a_PRE.png"])
def test_plan_rejects_invalid_source_paths(dataset_project: Path, path: str) -> None:
    samples = load_samples(dataset_project)
    samples.loc[0, "path_pre"] = path
    with pytest.raises(ValueError, match="Invalid image path"):
        build_download_plan(samples, dataset_project, {"dataset": "data"})


def test_plan_rejects_alias_outside_project(dataset_project: Path) -> None:
    with pytest.raises(ValueError, match="relative path"):
        build_download_plan(load_samples(dataset_project), dataset_project, {"dataset": "../data"})


def test_plan_rejects_second_dataset_layout(dataset_project: Path) -> None:
    direct = dataset_project / "dataset/train/p1/p1_z001_PRE.png"
    direct.parent.mkdir(parents=True)
    direct.write_bytes(png_bytes())
    with pytest.raises(ValueError, match="outside the configured layout"):
        build_download_plan(load_samples(dataset_project), dataset_project, {"dataset": "data"})


def test_download_on_empty_machine_then_resume_without_network(tmp_path: Path) -> None:
    item = DownloadItem("dataset/train/p1/p1_z001_PRE.png", tmp_path / "data/train/p1/a.png")
    session = FakeSession(FakeResponse(png_bytes()))
    options = {
        "base_url": "https://example.org/breastdcedl",
        "expected_size": (256, 256),
        "timeout": 5,
    }

    assert download_image(item, session, **options) == "downloaded"
    assert session.urls == ["https://example.org/breastdcedl/dataset/train/p1/p1_z001_PRE.png"]
    assert item.destination.read_bytes() == png_bytes()
    assert download_image(item, session, **options) == "skipped"
    assert len(session.urls) == 1
    assert not list(tmp_path.rglob("*.part"))
    assert not (tmp_path / "breastdcedl").exists()


@pytest.mark.parametrize("interrupted", [False, True])
def test_failed_download_never_publishes_partial_image(tmp_path: Path, interrupted: bool) -> None:
    item = DownloadItem("dataset/train/p1/p1_z001_PRE.png", tmp_path / "data/p1/PRE.png")
    session = FakeSession(FakeResponse(b"invalid or incomplete PNG", interrupted=interrupted))

    with pytest.raises((OSError, requests.RequestException)):
        download_image(
            item, session, base_url="https://example.org", expected_size=(256, 256), timeout=5
        )
    assert not item.destination.exists()
    assert not list(tmp_path.rglob("*.part"))


def test_existing_corrupt_image_is_reported_and_preserved(tmp_path: Path) -> None:
    target = tmp_path / "PRE.png"
    target.write_bytes(b"original corrupt file")
    report = download_images(
        [DownloadItem("dataset/train/p1/p1_z001_PRE.png", target)],
        base_url="https://example.org",
        workers=1,
    )
    assert len(report.failures) == 1
    assert "preserved without overwriting" in report.failures[0]
    assert report.downloaded == report.skipped == 0
    assert target.read_bytes() == b"original corrupt file"


def test_fresh_checkout_download_passes_existing_audit(
    dataset_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "fresh-checkout"
    shutil.copytree(dataset_project / "metadata", checkout / "metadata")
    monkeypatch.setattr(
        "breast_cnn.data.download.requests.Session", lambda: FakeSession(FakeResponse(png_bytes()))
    )
    items = build_download_plan(load_samples(checkout), checkout, {"dataset": "data"})
    report = download_images(items, base_url="https://example.org", workers=2)

    assert report.downloaded == 9
    assert not report.failures
    audit = audit_dataset(checkout, path_aliases={"dataset": "data"}, check_images=True)
    assert audit.ok, audit.errors
    assert audit.details["image_inspection"]["checked"] == 9
    assert not (checkout / "breastdcedl").exists()
    assert not (checkout / "dataset").exists()
