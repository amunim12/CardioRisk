"""Data acquisition for the combined UCI Heart Disease dataset.

The canonical UCI repository splits the data across four sources
(Cleveland, Hungary, Switzerland, VA Long Beach). The popular single-file
Kaggle export (303 rows) is only the Cleveland subset and has almost no
missing values. To make the "handle missing values" task meaningful, this
module downloads and concatenates all four ``processed.*.data`` files
(~920 rows) with genuine missingness, then caches the result.

Network failures degrade gracefully: if a cache exists it is used; otherwise
a clear, actionable error is raised.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

logger = logging.getLogger(__name__)

# 14 columns of the UCI "processed" files, in order.
COLUMN_NAMES: list[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# The four source databases that make up the combined dataset.
SOURCES: dict[str, str] = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

# Candidate base URLs, tried in order (UCI has migrated paths over time).
BASE_URLS: tuple[str, ...] = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/",
    "https://archive.ics.uci.edu/static/public/45/data/",
)


def _download_text(filename: str, timeout: float = 30.0) -> str | None:
    """Fetch a single source file as text, trying each base URL."""
    for base in BASE_URLS:
        url = base + filename
        try:
            with urlopen(url, timeout=timeout) as response:
                logger.info("Downloaded %s", url)
                return response.read().decode("utf-8")
        except (URLError, TimeoutError, OSError) as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
    return None


def _download_combined() -> pd.DataFrame:
    """Download all four sources and concatenate them into one frame."""
    frames: list[pd.DataFrame] = []
    for source_name, filename in SOURCES.items():
        text = _download_text(filename)
        if text is None:
            continue
        frame = pd.read_csv(
            io.StringIO(text),
            header=None,
            names=COLUMN_NAMES,
            na_values="?",
        )
        frame["source"] = source_name
        frames.append(frame)

    if not frames:
        raise ConnectionError(
            "Could not download any UCI Heart Disease source file. "
            "Check your network connection, or place a pre-downloaded CSV at "
            "the configured raw_path."
        )
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %d rows from %d sources", len(combined), len(frames))
    return combined


def load_raw_data(raw_path: str | Path, *, force_download: bool = False) -> pd.DataFrame:
    """Load the combined dataset, using a local cache when available.

    Args:
        raw_path: path to the cached CSV. Created if missing.
        force_download: re-download even if the cache exists.

    Returns:
        The combined dataset including the ``num`` severity column and a
        ``source`` column identifying the originating database.

    Raises:
        ConnectionError: if no cache exists and the download fails.
    """
    cache = Path(raw_path)
    if cache.is_file() and not force_download:
        logger.info("Loading cached dataset from %s", cache)
        return pd.read_csv(cache)

    df = _download_combined()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    logger.info("Cached dataset to %s", cache)
    return df


def feature_target_split(
    df: pd.DataFrame, target_source: str = "num"
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the raw frame into model features and the raw severity target.

    The ``source`` bookkeeping column is dropped from the features so it does
    not leak into the model.
    """
    drop_cols = [c for c in (target_source, "source") if c in df.columns]
    features = df.drop(columns=drop_cols)
    target = df[target_source]
    return features, target
