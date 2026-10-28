"""Jolpica (Ergast-compatible) client — results, grid, qualifying, standings."""

from __future__ import annotations

import re
from typing import Any

from .config import JOLPICA_BASE
from .http import fetch_json

_LAP_RE = re.compile(r"^(?:(\d+):)?(\d{1,2})\.(\d+)$")


def _mrdata(url: str) -> dict:
    payload = fetch_json(url)
    return payload.get("MRData", {})


def races(year: int) -> list[dict]:
    data = _mrdata(f"{JOLPICA_BASE}/{year}/races.json?limit=100")
    return data.get("RaceTable", {}).get("Races", [])


def results(year: int, round_number: int) -> list[dict]:
    data = _mrdata(f"{JOLPICA_BASE}/{year}/{round_number}/results.json?limit=100")
    races = data.get("RaceTable", {}).get("Races", [])
    return races[0].get("Results", []) if races else []


def qualifying(year: int, round_number: int) -> list[dict]:
    data = _mrdata(f"{JOLPICA_BASE}/{year}/{round_number}/qualifying.json?limit=100")
    races = data.get("RaceTable", {}).get("Races", [])
    return races[0].get("QualifyingResults", []) if races else []


def pit_stops(year: int, round_number: int) -> list[dict]:
    data = _mrdata(f"{JOLPICA_BASE}/{year}/{round_number}/pitstops.json?limit=100")
    races = data.get("RaceTable", {}).get("Races", [])
    return races[0].get("PitStops", []) if races else []


def driver_standings(year: int) -> list[dict]:
    data = _mrdata(f"{JOLPICA_BASE}/{year}/driverstandings.json?limit=100")
    lists = data.get("StandingsTable", {}).get("StandingsLists", [])
    return lists[0].get("DriverStandings", []) if lists else []


def constructor_standings(year: int) -> list[dict]:
    data = _mrdata(f"{JOLPICA_BASE}/{year}/constructorstandings.json?limit=100")
    lists = data.get("StandingsTable", {}).get("StandingsLists", [])
    return lists[0].get("ConstructorStandings", []) if lists else []


def parse_lap_time(value: str | None) -> float | None:
    """Convert ``1:29.179`` / ``29.179`` to seconds. Returns None if unparseable."""
    if not value:
        return None
    value = value.strip()
    match = _LAP_RE.match(value)
    if not match:
        return None
    minutes = int(match.group(1) or 0)
    seconds = int(match.group(2))
    millis = int(match.group(3).ljust(3, "0")[:3])
    return minutes * 60 + seconds + millis / 1000.0
