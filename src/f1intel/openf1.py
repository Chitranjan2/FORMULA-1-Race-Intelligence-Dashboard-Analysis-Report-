"""OpenF1 client — official timing-derived session data (no credentials needed)."""

from __future__ import annotations

import urllib.error
from typing import Any

from .config import OPENF1_BASE
from .http import fetch_json


def _get(endpoint: str, **params: Any) -> list[dict]:
    query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    url = f"{OPENF1_BASE}/{endpoint}" + (f"?{query}" if query else "")
    try:
        data = fetch_json(url)
    except urllib.error.HTTPError as exc:
        # OpenF1 answers 404 when a session simply has no rows for an endpoint
        # (e.g. no pit stops recorded). That is "no data", not an error.
        if exc.code == 404:
            return []
        raise
    if not isinstance(data, list):
        raise ValueError(f"Unexpected payload from {url}: {type(data).__name__}")
    return data


def sessions(year: int | None = None, session_key: int | None = None) -> list[dict]:
    return _get("sessions", year=year, session_key=session_key)


def meetings(year: int | None = None) -> list[dict]:
    return _get("meetings", year=year)


def laps(session_key: int) -> list[dict]:
    return _get("laps", session_key=session_key)


def stints(session_key: int) -> list[dict]:
    return _get("stints", session_key=session_key)


def pits(session_key: int) -> list[dict]:
    return _get("pit", session_key=session_key)


def drivers(session_key: int) -> list[dict]:
    return _get("drivers", session_key=session_key)


def positions(session_key: int) -> list[dict]:
    return _get("position", session_key=session_key)


def weather(session_key: int) -> list[dict]:
    return _get("weather", session_key=session_key)


def race_control(session_key: int) -> list[dict]:
    return _get("race_control", session_key=session_key)


def car_data(session_key: int, driver_number: int, **filters: Any) -> list[dict]:
    """Telemetry at ~3.7 Hz. Large — request per driver and downsample locally."""
    return _get("car_data", session_key=session_key, driver_number=driver_number, **filters)
