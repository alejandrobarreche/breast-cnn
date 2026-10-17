"""Download public images into the configured layout without replacing source files.

Adapted from the standalone BreastDCEDL downloader supplied with the teaching case.
The versioned sample index controls both selection and remote object paths.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from urllib.parse import quote

import pandas as pd
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from breast_cnn.data.validation import PHASE_PATH_COLUMNS


@dataclass(frozen=True)
class DownloadItem:
    remote_path: str
    destination: Path


@dataclass
class DownloadReport:
    downloaded: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)


def build_download_plan(
    samples: pd.DataFrame,
    project_root: Path,
    path_aliases: Mapping[str, str],
    *,
    split: str = "all",
    patients: int | None = None,
) -> list[DownloadItem]:
    """Map selected CSV paths to safe destinations, including files not yet present."""
    if split not in {"all", "train", "test"}:
        raise ValueError("split must be all, train or test")
    if patients is not None and patients < 1:
        raise ValueError("patients must be a positive integer")
    columns = ["patient_id", "split", *PHASE_PATH_COLUMNS.values()]
    if not set(columns).issubset(samples.columns) or samples[columns].isna().any().any():
        raise ValueError("Sample index contains missing download columns or values")
    if not samples["split"].isin(["train", "test"]).all():
        raise ValueError("Only public train and test samples may be downloaded")
    selected = samples if split == "all" else samples.loc[samples["split"].eq(split)]
    if patients is not None:
        chosen = sorted(selected["patient_id"].unique())[:patients]
        selected = selected.loc[selected["patient_id"].isin(chosen)]
    if selected.empty:
        raise ValueError("No samples match the requested selection")

    root = project_root.resolve()
    items: list[DownloadItem] = []
    destinations: set[Path] = set()
    for row in selected.to_dict("records"):
        for phase, column in PHASE_PATH_COLUMNS.items():
            remote = str(row[column])
            path = PurePosixPath(remote)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in remote
                or len(path.parts) != 4
                or path.parts[1:3] != (row["split"], row["patient_id"])
                or not path.name.endswith(f"_{phase}.png")
            ):
                raise ValueError(f"Invalid image path for {row['patient_id']}: {remote}")
            prefix = path.parts[0]
            if prefix not in path_aliases:
                raise ValueError(f"No configured destination alias for {prefix!r}")
            alias = Path(path_aliases[prefix])
            if alias.is_absolute() or ".." in alias.parts:
                raise ValueError(f"Destination alias must be a relative path: {alias}")
            image_root = (root / alias).resolve()
            destination = (image_root / Path(*path.parts[1:])).resolve()
            if (
                not image_root.is_relative_to(root)
                or image_root == root
                or not destination.is_relative_to(image_root)
            ):
                raise ValueError(
                    f"Image destination escapes the configured image directory: {remote}"
                )
            direct = (root / Path(*path.parts)).resolve()
            if direct != destination and direct.exists():
                raise ValueError(
                    f"An image already exists outside the configured layout: {direct}. "
                    "Resolve the duplicate layout before downloading."
                )
            if destination in destinations:
                raise ValueError(f"Repeated image destination: {remote}")
            destinations.add(destination)
            items.append(DownloadItem(remote, destination))
    return items


def validate_downloaded_image(path: Path, expected_size: tuple[int, int]) -> None:
    """Decode a file before considering it a complete grayscale PNG."""
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "L" or image.size != expected_size:
            raise ValueError(
                f"Expected grayscale PNG {expected_size}, found "
                f"{image.format}, mode={image.mode}, size={image.size}"
            )


def download_image(
    item: DownloadItem,
    session: requests.Session,
    *,
    base_url: str,
    expected_size: tuple[int, int],
    timeout: float,
) -> str:
    """Skip valid originals; download to a temporary file and publish only after validation."""
    if item.destination.exists():
        try:
            validate_downloaded_image(item.destination, expected_size)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Existing image is invalid; preserved without overwriting: {item.destination}"
            ) from error
        return "skipped"

    item.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        url = f"{base_url.rstrip('/')}/{quote(item.remote_path, safe='/')}"
        with session.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            with NamedTemporaryFile(
                dir=item.destination.parent, prefix=".download-", suffix=".part", delete=False
            ) as output:
                temporary = Path(output.name)
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    output.write(chunk)
        validate_downloaded_image(temporary, expected_size)
        if item.destination.exists():
            raise FileExistsError(f"Destination appeared during download: {item.destination}")
        temporary.replace(item.destination)
        return "downloaded"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_images(
    items: list[DownloadItem],
    *,
    base_url: str,
    workers: int = 16,
    timeout: float = 60,
    expected_size: tuple[int, int] = (256, 256),
) -> DownloadReport:
    """Download concurrently with one reusable HTTP session per worker and bounded retries."""
    if workers < 1 or timeout <= 0:
        raise ValueError("workers and timeout must be positive")
    if not base_url.startswith("https://"):
        raise ValueError("Download base_url must use HTTPS")
    state = threading.local()
    sessions: list[requests.Session] = []
    session_lock = threading.Lock()

    def transfer(item: DownloadItem) -> tuple[str, str]:
        if not hasattr(state, "session"):
            state.session = requests.Session()
            retry = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=frozenset({"GET"}),
            )
            state.session.mount("https://", HTTPAdapter(max_retries=retry))
            with session_lock:
                sessions.append(state.session)
        try:
            status = download_image(
                item,
                state.session,
                base_url=base_url,
                expected_size=expected_size,
                timeout=timeout,
            )
            return status, ""
        except (requests.RequestException, OSError, ValueError) as error:
            return "failed", f"{item.remote_path}: {error}"

    report = DownloadReport()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for status, error in tqdm(pool.map(transfer, items), total=len(items), desc="Images"):
                if status == "downloaded":
                    report.downloaded += 1
                elif status == "skipped":
                    report.skipped += 1
                else:
                    report.failures.append(error)
    finally:
        for session in sessions:
            session.close()
    return report
