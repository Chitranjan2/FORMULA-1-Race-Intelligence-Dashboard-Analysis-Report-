"""Repository-wide paths and tunable constants."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# HTTP response cache. Kept outside version control; safe to delete.
CACHE_DIR = Path(os.environ.get("F1_CACHE_DIR", RAW_DIR / "cache"))

for _d in (RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Seasons to ingest by default. 2023 is the first season OpenF1 covers.
DEFAULT_SEASONS: tuple[int, ...] = tuple(
    int(x) for x in os.environ.get("F1_SEASONS", "2023,2024,2025,2026").split(",") if x.strip()
)

OPENF1_BASE = "https://api.openf1.org/v1"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

USER_AGENT = (
    "f1-race-intelligence/2.0 "
    "(+https://github.com/DhyeyTeraiya/f1-race-intelligence-analytics)"
)
HTTP_TIMEOUT = 60
MAX_RETRIES = 4
REQUEST_PAUSE = 0.15  # seconds between uncached requests, to be a polite API citizen

# Lap-analysis filters.
MIN_LAP_TIME = 60.0     # discard absurdly short laps (pit-lane artefacts)
MAX_LAP_TIME = 240.0    # discard absurdly long laps (red flags, in-laps)
PIT_OUT_TOLERANCE = 2    # laps after a pit-out before a lap counts as clean

# Model settings.
MODEL_TARGET = "podium_finish"
HOLDOUT_SEASONS = (2025, 2026)  # walk-forward holdout
RANDOM_STATE = 42
