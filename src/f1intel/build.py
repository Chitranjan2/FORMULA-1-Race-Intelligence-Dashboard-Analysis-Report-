"""Build the real Formula 1 dataset from OpenF1 + Jolpica.

Produces, in ``data/processed/``:
    sessions      one row per ingested OpenF1 session
    race_laps     one row per driver-lap of every ingested race, enriched with
                  stint compound, tyre life, pit-stop duration and result context
    quali_best    each driver's best qualifying lap (and sector times) per round
    results       race classification, grid slot, points, status
    qualifying    qualifying classification
    standings     final driver and constructor standings per season
    weather       session-average conditions
    manifest      provenance: row counts, seasons, source URLs, build time

All data is downloaded from public APIs and cached on disk. Re-runs are cheap
and work offline once a season has been ingested.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

from . import __version__, jolpica, openf1
from .config import CACHE_DIR, DEFAULT_SEASONS, PIT_OUT_TOLERANCE, PROCESSED_DIR
from .storage import save_table

RACE_SESSION = "Race"
QUALI_SESSION = "Qualifying"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _match_round(session_date: date | None, races: list[dict], used: set[int]) -> dict | None:
    """Match an OpenF1 race session to a Jolpica round by nearest race date."""
    if session_date is None:
        return None
    best, best_gap = None, 10_000
    for race in races:
        if race["round"] in used:
            continue
        race_date = _as_date(f"{race['date']}T00:00:00+00:00")
        if race_date is None:
            continue
        gap = abs((race_date - session_date).days)
        if gap < best_gap:
            best, best_gap = race, gap
    if best is not None and best_gap <= 3:
        used.add(best["round"])
        return best
    return None


def _safety_car_laps(messages: list[dict]) -> set[int]:
    """Approximate neutralised lap numbers from race-control messages.

    OpenF1 does not publish a single flag timeline, so deployment / withdrawal
    messages are parsed into a lap set. Laps flagged here are excluded from pace
    analysis because they are not representative of green-flag performance.
    """
    laps: set[int] = set()
    active = False
    ordered = sorted(messages, key=lambda m: m.get("date") or "")
    for message in ordered:
        text = (message.get("message") or "").upper()
        lap = message.get("lap_number")
        if lap is None:
            continue
        is_sc = "SAFETY CAR" in text or "VSC" in text
        if not is_sc:
            continue
        ends = any(token in text for token in ("IN THIS LAP", "ENDING", "WITHDRAWN", "CLEAR"))
        if ends:
            active = False
            continue
        active = True
        laps.add(int(lap))
    if not active:
        pass
    return laps


def _stint_lookup(stints: list[dict], laps: list[dict]) -> pd.DataFrame:
    """Expand stint ranges into one row per driver-lap with compound and tyre life."""
    columns = ["driver_number", "lap_number", "compound", "stint_number", "tyre_life"]
    if not stints:
        return pd.DataFrame(columns=columns)
    rows: list[dict] = []
    for stint in stints:
        driver = stint.get("driver_number")
        start = stint.get("lap_start")
        end = stint.get("lap_end")
        if driver is None or start is None or end is None:
            continue
        age_at_start = stint.get("tyre_age_at_start") or 0
        for lap in range(int(start), int(end) + 1):
            rows.append(
                {
                    "driver_number": driver,
                    "lap_number": lap,
                    "compound": (stint.get("compound") or "UNKNOWN").upper(),
                    "stint_number": stint.get("stint_number"),
                    "tyre_life": age_at_start + (lap - int(start)),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _pit_lookup(pits: list[dict]) -> pd.DataFrame:
    if not pits:
        return pd.DataFrame(columns=["driver_number", "lap_number", "pit_duration", "stop_duration"])
    frame = pd.DataFrame(pits)
    frame = frame.rename(
        columns={"lane_duration": "lane_duration", "stop_duration": "stop_duration", "pit_duration": "pit_duration"}
    )
    keep = ["driver_number", "lap_number", "pit_duration", "stop_duration"]
    for column in keep:
        if column not in frame.columns:
            frame[column] = None
    return frame[keep]


def _driver_lookup(drivers: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "driver_number": d.get("driver_number"),
                "driver": d.get("name_acronym") or d.get("broadcast_name"),
                "driver_full_name": d.get("full_name"),
                "team": d.get("team_name"),
                "team_colour": d.get("team_colour"),
            }
            for d in drivers
        ]
    )
    return frame.drop_duplicates(subset=["driver_number"])


# --------------------------------------------------------------------------- #
# per-session extraction
# --------------------------------------------------------------------------- #
def _race_frame(session: dict, round_number: int, race: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = session["session_key"]
    laps_raw = openf1.laps(key)
    if not laps_raw:
        return pd.DataFrame(), pd.DataFrame()

    laps = pd.DataFrame(laps_raw).rename(
        columns={
            "duration_sector_1": "sector1",
            "duration_sector_2": "sector2",
            "duration_sector_3": "sector3",
            "lap_duration": "lap_time",
            "is_pit_out_lap": "is_pit_out_lap",
        }
    )
    laps = laps[["driver_number", "lap_number", "lap_time", "sector1", "sector2", "sector3", "is_pit_out_lap"]]
    laps = laps[laps["lap_number"].notna() & laps["lap_time"].notna()].copy()
    laps["lap_number"] = laps["lap_number"].astype(int)

    stints = _stint_lookup(openf1.stints(key), laps_raw)
    pits = _pit_lookup(openf1.pits(key))
    drivers = _driver_lookup(openf1.drivers(key))
    sc_laps = _safety_car_laps(openf1.race_control(key))

    merged = laps.merge(stints, on=["driver_number", "lap_number"], how="left")
    merged = merged.merge(pits, on=["driver_number", "lap_number"], how="left")
    merged = merged.merge(drivers, on="driver_number", how="left")

    # Normalise dtypes before the frames are concatenated: a race with no recorded
    # stops leaves these columns entirely NA, and mixing all-NA columns with float
    # columns makes pandas guess the result dtype (FutureWarning in pandas 2.x).
    for column in ("pit_duration", "stop_duration"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce").astype("float64")

    merged["year"] = session.get("year") or race.get("season")
    merged["round"] = round_number
    merged["race_name"] = race.get("raceName")
    merged["circuit"] = race.get("Circuit", {}).get("circuitName")
    merged["country"] = race.get("Circuit", {}).get("Location", {}).get("country")
    merged["session_key"] = key
    merged["is_neutralized"] = merged["lap_number"].astype(int).isin(sc_laps)
    merged["is_pit_in_lap"] = merged["pit_duration"].notna()
    merged["is_pit_out_lap"] = merged["is_pit_out_lap"].fillna(False).astype(bool)
    merged["has_clean_air"] = ~merged["is_pit_out_lap"] & ~merged["is_pit_in_lap"] & ~merged["is_neutralized"]

    weather_rows = openf1.weather(key)
    if weather_rows:
        weather = pd.DataFrame(weather_rows)
        weather_summary = pd.DataFrame(
            [
                {
                    "session_key": key,
                    "air_temp_mean": weather.get("air_temperature", pd.Series(dtype=float)).mean(),
                    "track_temp_mean": weather.get("track_temperature", pd.Series(dtype=float)).mean(),
                    "rainfall_any": bool(weather.get("rainfall", pd.Series(dtype=float)).fillna(0).astype(float).gt(0).any()),
                }
            ]
        )
    else:
        weather_summary = pd.DataFrame()

    return merged, weather_summary


def _quali_frame(session: dict, round_number: int, race: dict) -> pd.DataFrame:
    key = session["session_key"]
    laps_raw = openf1.laps(key)
    if not laps_raw:
        return pd.DataFrame()
    laps = pd.DataFrame(laps_raw).rename(
        columns={
            "duration_sector_1": "sector1",
            "duration_sector_2": "sector2",
            "duration_sector_3": "sector3",
            "lap_duration": "lap_time",
        }
    )
    laps = laps[["driver_number", "lap_number", "lap_time", "sector1", "sector2", "sector3"]]
    laps = laps[laps["lap_time"].notna()]
    if laps.empty:
        return pd.DataFrame()
    drivers = _driver_lookup(openf1.drivers(key))
    best = laps.loc[laps.groupby("driver_number")["lap_time"].idxmin()].merge(drivers, on="driver_number", how="left")
    best["year"] = session.get("year") or race.get("season")
    best["round"] = round_number
    best["session_key"] = key
    return best


# --------------------------------------------------------------------------- #
# main build
# --------------------------------------------------------------------------- #
def build(seasons: Iterable[int] | None = None, *, limit_per_season: int | None = None) -> dict[str, Any]:
    seasons = tuple(seasons or DEFAULT_SEASONS)
    started = time.time()

    race_frames: list[pd.DataFrame] = []
    quali_frames: list[pd.DataFrame] = []
    session_rows: list[dict] = []
    weather_frames: list[pd.DataFrame] = []
    results_rows: list[dict] = []
    qualifying_rows: list[dict] = []
    standings_rows: list[dict] = []
    race_index_rows: list[dict] = []

    for year in seasons:
        print(f"[build] {year}: fetching schedule from Jolpica")
        races = jolpica.races(year)
        if not races:
            print(f"[build] {year}: no races returned, skipping")
            continue
        for race in races:  # Jolpica returns round numbers as strings
            race["round"] = int(race["round"])
        if limit_per_season:
            races = races[:limit_per_season]

        # Race results / qualifying classification
        for race in races:
            round_number = race["round"]
            for row in jolpica.results(year, round_number):
                results_rows.append(
                    {
                        "year": year,
                        "round": round_number,
                        "race_name": race.get("raceName"),
                        "circuit": race.get("Circuit", {}).get("circuitName"),
                        "driver_code": row.get("Driver", {}).get("code"),
                        "driver_name": f"{row.get('Driver', {}).get('givenName', '')} {row.get('Driver', {}).get('familyName', '')}".strip(),
                        "team": row.get("Constructor", {}).get("name"),
                        "grid": pd.to_numeric(row.get("grid"), errors="coerce"),
                        "position": pd.to_numeric(row.get("position"), errors="coerce"),
                        "position_text": row.get("positionText"),
                        "points": pd.to_numeric(row.get("points"), errors="coerce"),
                        "laps_completed": pd.to_numeric(row.get("laps"), errors="coerce"),
                        "status": row.get("status"),
                    }
                )
            for row in jolpica.qualifying(year, round_number):
                qualifying_rows.append(
                    {
                        "year": year,
                        "round": round_number,
                        "driver_code": row.get("Driver", {}).get("code"),
                        "team": row.get("Constructor", {}).get("name"),
                        "qualifying_position": pd.to_numeric(row.get("position"), errors="coerce"),
                        "q1": jolpica.parse_lap_time(row.get("Q1")),
                        "q2": jolpica.parse_lap_time(row.get("Q2")),
                        "q3": jolpica.parse_lap_time(row.get("Q3")),
                    }
                )
        for row in jolpica.driver_standings(year):
            standings_rows.append(
                {
                    "year": year,
                    "kind": "driver",
                    "name": f"{row.get('Driver', {}).get('givenName', '')} {row.get('Driver', {}).get('familyName', '')}".strip(),
                    "team": (row.get("Constructors") or [{}])[0].get("name"),
                    "position": pd.to_numeric(row.get("position"), errors="coerce"),
                    "points": pd.to_numeric(row.get("points"), errors="coerce"),
                }
            )
        for row in jolpica.constructor_standings(year):
            standings_rows.append(
                {
                    "year": year,
                    "kind": "constructor",
                    "name": row.get("Constructor", {}).get("name"),
                    "team": row.get("Constructor", {}).get("name"),
                    "position": pd.to_numeric(row.get("position"), errors="coerce"),
                    "points": pd.to_numeric(row.get("points"), errors="coerce"),
                }
            )

        print(f"[build] {year}: fetching OpenF1 sessions")
        sessions = [
            s
            for s in openf1.sessions(year=year)
            if s.get("session_name") in (RACE_SESSION, QUALI_SESSION)
        ]
        sessions.sort(key=lambda s: s.get("date_start") or "")
        used_rounds: set[int] = set()

        for session in sessions:
            name = session.get("session_name")
            session_date = _as_date(session.get("date_start"))
            race = _match_round(session_date, races, used_rounds if name == RACE_SESSION else set())
            if race is None:
                print(f"[build]   skip {name} {session.get('date_start')} (no round match)")
                continue
            round_number = race["round"]
            session_rows.append(
                {
                    "session_key": session.get("session_key"),
                    "year": year,
                    "round": round_number,
                    "meeting_key": session.get("meeting_key"),
                    "circuit_short_name": session.get("circuit_short_name"),
                    "country_name": session.get("country_name"),
                    "date_start": session.get("date_start"),
                    "session_name": name,
                }
            )
            if name == RACE_SESSION:
                race_index_rows.append(
                    {
                        "year": year,
                        "round": round_number,
                        "race_name": race.get("raceName"),
                        "circuit": race.get("Circuit", {}).get("circuitName"),
                        "country": race.get("Circuit", {}).get("Location", {}).get("country"),
                        "date": race.get("date"),
                        "session_key": session.get("session_key"),
                    }
                )
                frame, weather = _race_frame(session, round_number, race)
                if not frame.empty:
                    race_frames.append(frame)
                if not weather.empty:
                    weather_frames.append(weather)
                print(f"[build]   {year} R{round_number:02d} {race.get('raceName')}: {len(frame)} driver-laps")
            else:
                frame = _quali_frame(session, round_number, race)
                if not frame.empty:
                    quali_frames.append(frame)
                print(f"[build]   {year} R{round_number:02d} qualifying: {len(frame)} drivers")

    tables = {
        "sessions": pd.DataFrame(session_rows),
        "races": pd.DataFrame(race_index_rows),
        "race_laps": pd.concat(race_frames, ignore_index=True) if race_frames else pd.DataFrame(),
        "quali_best": pd.concat(quali_frames, ignore_index=True) if quali_frames else pd.DataFrame(),
        "results": pd.DataFrame(results_rows),
        "qualifying": pd.DataFrame(qualifying_rows),
        "standings": pd.DataFrame(standings_rows),
        "weather": pd.concat(weather_frames, ignore_index=True) if weather_frames else pd.DataFrame(),
    }

    written: dict[str, int] = {}
    for name, frame in tables.items():
        path = save_table(frame, name)
        written[name] = len(frame)
        print(f"[build] wrote {path.name}: {len(frame)} rows")

    manifest = {
        "package_version": __version__,
        "built_at_utc": datetime.now().astimezone().isoformat(),
        "seasons": list(seasons),
        "limit_per_season": limit_per_season,
        "row_counts": written,
        "duration_seconds": round(time.time() - started, 1),
        "sources": {
            "laps_stints_pits_weather_racecontrol": "OpenF1 (https://api.openf1.org/v1)",
            "results_qualifying_standings_schedule": "Jolpica (https://api.jolpi.ca/ergast/f1)",
        },
        "notes": [
            "Lap times, sector times, tyre compounds and pit-stop durations are downloaded from OpenF1.",
            "Nothing in this dataset is simulated or randomly generated.",
            "Neutralised laps are derived from race-control messages and may be approximate.",
        ],
    }
    (PROCESSED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[build] manifest written -> {PROCESSED_DIR / 'manifest.json'}")
    return manifest


if __name__ == "__main__":
    build()
