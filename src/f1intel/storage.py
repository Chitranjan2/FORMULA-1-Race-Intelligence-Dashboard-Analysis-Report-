"""Table persistence helpers.

Prefers Parquet; transparently falls back to gzipped CSV when a Parquet engine
(``pyarrow``) is unavailable, so the pipeline runs on a bare install.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR


def save_table(frame: pd.DataFrame, name: str) -> Path:
    if frame.empty:
        path = PROCESSED_DIR / f"{name}.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        return path
    try:
        path = PROCESSED_DIR / f"{name}.parquet"
        frame.to_parquet(path, index=False)
    except (ImportError, ValueError):
        path = PROCESSED_DIR / f"{name}.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
    return path


def load_table(name: str) -> pd.DataFrame:
    parquet = PROCESSED_DIR / f"{name}.parquet"
    csv = PROCESSED_DIR / f"{name}.csv.gz"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"No processed table '{name}' in {PROCESSED_DIR}. Run: python -m f1intel build"
    )


def table_exists(name: str) -> bool:
    return (PROCESSED_DIR / f"{name}.parquet").exists() or (PROCESSED_DIR / f"{name}.csv.gz").exists()
