"""Real race-pace, tyre-degradation and pit-loss analytics.

Method (all parameters estimated from the downloaded data, none assumed):

1. **Clean-lap filter** — drop pit-in/out laps, neutralised laps, wet running,
   and statistical outliers.
2. **Per-race pace model** — for each Grand Prix, fit
   ``lap_time ~ lap_number + tyre_life + compound + driver``.
   ``lap_number`` stands in for fuel load and track evolution (both linear in
   race distance), ``tyre_life`` captures degradation, and driver fixed effects
   absorb individual pace. Subtracting the fitted fuel and tyre components gives
   a *fuel- and tyre-corrected* lap time, i.e. car pace.
3. **Degradation** — refit per race and compound to report seconds lost per lap
   of tyre age.
4. **Pit loss** — measured empirically per stop: the extra time spent on the
   pit-in and pit-out laps relative to the driver's own clean-lap median.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import MAX_LAP_TIME, MIN_LAP_TIME, OUTPUTS_DIR
from .storage import load_table, save_table

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
STREET_CIRCUITS = ("Monaco", "Singapore", "Baku", "Las Vegas", "Jeddah", "Miami")


# --------------------------------------------------------------------------- #
# loading / filtering
# --------------------------------------------------------------------------- #
def load_laps() -> pd.DataFrame:
    laps = load_table("race_laps")
    for column in ("lap_time", "sector1", "sector2", "sector3", "tyre_life", "pit_duration", "stop_duration"):
        if column in laps.columns:
            laps[column] = pd.to_numeric(laps[column], errors="coerce")
    laps["round"] = pd.to_numeric(laps["round"], errors="coerce").astype("Int64")
    laps["year"] = pd.to_numeric(laps["year"], errors="coerce").astype("Int64")
    return laps


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    mask = (
        laps["has_clean_air"].fillna(False)
        & laps["lap_time"].between(MIN_LAP_TIME, MAX_LAP_TIME)
        & laps["compound"].isin(DRY_COMPOUNDS)
        & laps["tyre_life"].notna()
    )
    clean = laps.loc[mask].copy()

    # Per driver-race outlier removal (median absolute deviation).
    grouped = clean.groupby(["session_key", "driver_number"])["lap_time"]
    median = grouped.transform("median")
    mad = grouped.transform(lambda s: (s - s.median()).abs().median())
    clean = clean[(clean["lap_time"] - median).abs() <= 4.0 * mad.clip(lower=0.05)]
    return clean.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# pace model
# --------------------------------------------------------------------------- #
def _design(group: pd.DataFrame) -> np.ndarray | None:
    """Design matrix for one race: intercept, fuel proxy, tyre age, compound, driver."""
    if len(group) < 60 or group["driver"].nunique() < 4:
        return None
    n = len(group)
    columns = [
        np.ones(n),
        group["lap_number"].to_numpy(dtype=float),
        group["tyre_life"].to_numpy(dtype=float),
        (group["compound"] == "SOFT").to_numpy(dtype=float),
        (group["compound"] == "HARD").to_numpy(dtype=float),
    ]
    drivers = sorted(group["driver"].dropna().unique())
    for driver in drivers[1:]:
        columns.append((group["driver"] == driver).to_numpy(dtype=float))
    matrix = np.column_stack(columns)
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        return None
    return matrix


def fit_race_pace(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return fuel/tyre-corrected per-lap pace plus a coefficient summary."""
    corrected_frames: list[pd.DataFrame] = []
    coefficients: list[dict[str, Any]] = []

    for (year, round_number), group in clean.groupby(["year", "round"]):
        matrix = _design(group)
        if matrix is None:
            continue
        target = group["lap_time"].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        fuel_per_lap, deg_per_lap = float(coef[1]), float(coef[2])
        compound_soft, compound_hard = float(coef[3]), float(coef[4])

        frame = group.copy()
        compound_effect = np.select(
            [frame["compound"] == "SOFT", frame["compound"] == "HARD"],
            [compound_soft, compound_hard],
            default=0.0,
        )
        frame["corrected_pace"] = (
            frame["lap_time"]
            - fuel_per_lap * frame["lap_number"]
            - deg_per_lap * frame["tyre_life"]
            - compound_effect
        )
        corrected_frames.append(frame)
        coefficients.append(
            {
                "year": int(year),
                "round": int(round_number),
                "fuel_effect_s_per_lap": round(fuel_per_lap, 4),
                "deg_effect_s_per_lap": round(deg_per_lap, 4),
                "n_laps": int(len(group)),
            }
        )

    if not corrected_frames:
        return pd.DataFrame(), {"race_coefficients": []}
    corrected = pd.concat(corrected_frames, ignore_index=True)
    return corrected, {"race_coefficients": coefficients}


def driver_pace(corrected: pd.DataFrame) -> pd.DataFrame:
    """Median corrected pace per driver-race, and gap to the fastest car."""
    pace = (
        corrected.groupby(["year", "round", "race_name", "driver", "team"], dropna=False)["corrected_pace"]
        .median()
        .reset_index()
    )
    pace["gap_to_fastest_s"] = pace["corrected_pace"] - pace.groupby(["year", "round"])["corrected_pace"].transform("min")
    return pace


def team_pace_by_race(pace: pd.DataFrame) -> pd.DataFrame:
    """Fastest car per team per race (min of its two drivers) and gap to the quickest team."""
    team = (
        pace.groupby(["year", "round", "race_name", "team"], dropna=False)["gap_to_fastest_s"]
        .min()
        .reset_index()
        .rename(columns={"gap_to_fastest_s": "team_gap_s"})
    )
    team["team_gap_pct"] = team["team_gap_s"] / 90.0 * 100.0
    return team.sort_values(["year", "round", "team_gap_s"])


def pace_evolution(team_pace: pd.DataFrame) -> pd.DataFrame:
    """Rolling three-race gap, giving each team's development trajectory."""
    evolution = team_pace.sort_values(["team", "year", "round"]).copy()
    evolution["rolling_gap_s"] = (
        evolution.groupby("team")["team_gap_s"].transform(lambda s: s.rolling(3, min_periods=1).median())
    )
    return evolution


# --------------------------------------------------------------------------- #
# degradation & pit loss
# --------------------------------------------------------------------------- #
def degradation_table(clean: pd.DataFrame) -> pd.DataFrame:
    """Seconds lost per lap of tyre age, fitted per race and compound."""
    rows: list[dict[str, Any]] = []
    for (year, round_number, circuit, compound), group in clean.groupby(
        ["year", "round", "circuit", "compound"], dropna=False
    ):
        if len(group) < 40 or group["driver"].nunique() < 4:
            continue
        n = len(group)
        drivers = sorted(group["driver"].dropna().unique())
        columns = [
            np.ones(n),
            group["tyre_life"].to_numpy(dtype=float),
            group["lap_number"].to_numpy(dtype=float),
        ]
        for driver in drivers[1:]:
            columns.append((group["driver"] == driver).to_numpy(dtype=float))
        matrix = np.column_stack(columns)
        if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
            continue
        target = group["lap_time"].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        residual = target - matrix @ coef
        ss_tot = float(((target - target.mean()) ** 2).sum())
        r2 = 1.0 - float((residual ** 2).sum()) / ss_tot if ss_tot else 0.0
        rows.append(
            {
                "year": int(year),
                "round": int(round_number),
                "circuit": circuit,
                "compound": compound,
                "deg_s_per_lap": round(float(coef[1]), 4),
                "r2": round(r2, 3),
                "n_laps": int(len(group)),
            }
        )
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    # Aggregate across seasons; report the median and the sample size behind it.
    summary = (
        table.groupby(["circuit", "compound"])
        .agg(
            deg_s_per_lap=("deg_s_per_lap", "median"),
            deg_p10=("deg_s_per_lap", lambda s: float(np.percentile(s, 10))),
            deg_p90=("deg_s_per_lap", lambda s: float(np.percentile(s, 90))),
            races=("year", "nunique"),
            laps=("n_laps", "sum"),
        )
        .reset_index()
        .sort_values(["circuit", "compound"])
    )
    return summary


def pit_loss_table(laps: pd.DataFrame, pace: pd.DataFrame) -> pd.DataFrame:
    """Empirical pit loss: extra time on the pit-in + pit-out laps vs a clean lap."""
    clean_median = (
        laps[laps["has_clean_air"].fillna(False)]
        .groupby(["session_key", "driver_number"])["lap_time"]
        .median()
        .rename("clean_median")
        .reset_index()
    )
    ordered = laps.sort_values(["session_key", "driver_number", "lap_number"]).copy()
    ordered["next_lap_time"] = ordered.groupby(["session_key", "driver_number"])["lap_time"].shift(-1)

    stops = ordered[ordered["pit_duration"].notna() & ordered["next_lap_time"].notna()].copy()
    if stops.empty:
        return pd.DataFrame()
    stops = stops.merge(clean_median, on=["session_key", "driver_number"], how="left")
    stops = stops[stops["clean_median"].notna()]
    stops["pit_loss_s"] = (
        stops["lap_time"] + stops["next_lap_time"] - 2 * stops["clean_median"]
    )
    # Physically plausible window only; guards against red flags and SC periods.
    stops = stops[stops["pit_loss_s"].between(12.0, 40.0)]

    per_race = (
        stops.groupby(["year", "round", "circuit"])
        .agg(
            pit_loss_s=("pit_loss_s", "median"),
            pit_stops=("pit_loss_s", "size"),
            stop_duration_s=("stop_duration", "median"),
        )
        .reset_index()
    )
    return per_race.sort_values(["year", "round"])


# --------------------------------------------------------------------------- #
# head-to-head
# --------------------------------------------------------------------------- #
def teammate_head_to_head(pace: pd.DataFrame, quali: pd.DataFrame) -> pd.DataFrame:
    """Race-pace and qualifying head-to-head between team mates."""
    race = pace.dropna(subset=["driver"])
    race_wins: dict[tuple[str, str], dict[str, int]] = {}
    for (year, round_number, team), group in race.groupby(["year", "round", "team"], dropna=False):
        if len(group) != 2:
            continue
        pair = tuple(sorted(group["driver"].tolist()))
        faster = group.loc[group["corrected_pace"].idxmin(), "driver"]
        slower = group.loc[group["corrected_pace"].idxmax(), "driver"]
        entry = race_wins.setdefault(pair, {})
        entry[faster] = entry.get(faster, 0) + 1
        entry[f"__{slower}__"] = entry.get(f"__{slower}__", 0) + 1

    quali_wins: dict[tuple[str, str], dict[str, int]] = {}
    if not quali.empty:
        quasi = quali.dropna(subset=["driver", "lap_time"]).copy()
        for (year, round_number, team), group in quasi.groupby(["year", "round", "team"], dropna=False):
            if len(group) < 2:
                continue
            best = group.loc[group["lap_time"].idxmin(), "driver"]
            pair = tuple(sorted(group["driver"].unique().tolist()[:2]))
            if len(pair) != 2:
                continue
            entry = quali_wins.setdefault(pair, {})
            entry[best] = entry.get(best, 0) + 1

    rows: list[dict[str, Any]] = []
    for pair, wins in race_wins.items():
        a, b = pair
        rows.append(
            {
                "driver_a": a,
                "driver_b": b,
                "race_pace_a": wins.get(a, 0),
                "race_pace_b": wins.get(f"__{b}__", 0),
                "quali_a": quali_wins.get(pair, {}).get(a, 0),
                "quali_b": quali_wins.get(pair, {}).get(b, 0),
            }
        )
    return pd.DataFrame(rows).sort_values("race_pace_a", ascending=False) if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    """Compute every analytic table, write CSVs/JSON, and return a summary."""
    laps = load_laps()
    if laps.empty:
        raise RuntimeError("race_laps is empty — run `python -m f1intel build` first")

    clean = clean_laps(laps)
    corrected, pace_meta = fit_race_pace(clean)
    pace = driver_pace(corrected)
    team_pace = team_pace_by_race(pace)
    evolution = pace_evolution(team_pace)
    degradation = degradation_table(clean)
    pit_loss = pit_loss_table(laps, pace)
    quali = load_table("quali_best")
    h2h = teammate_head_to_head(pace, quali)

    tables = {
        "driver_pace": pace,
        "team_pace": team_pace,
        "pace_evolution": evolution,
        "degradation": degradation,
        "pit_loss": pit_loss,
        "teammate_h2h": h2h,
    }
    for name, frame in tables.items():
        if not frame.empty:
            save_table(frame, name)
            print(f"[analysis] {name}: {len(frame)} rows")

    summary: dict[str, Any] = {
        "seasons": sorted(int(y) for y in laps["year"].dropna().unique()),
        "races": int(laps[["year", "round"]].drop_duplicates().shape[0]),
        "driver_laps_total": int(len(laps)),
        "clean_laps_used": int(len(clean)),
        "fuel_effect_median_s_per_lap": round(
            float(np.median([r["fuel_effect_s_per_lap"] for r in pace_meta["race_coefficients"]])), 4
        ) if pace_meta["race_coefficients"] else None,
        "deg_median_s_per_lap": round(float(degradation["deg_s_per_lap"].median()), 4) if not degradation.empty else None,
        "pit_loss_median_s": round(float(pit_loss["pit_loss_s"].median()), 2) if not pit_loss.empty else None,
        "pit_stops_measured": int(pit_loss["pit_stops"].sum()) if not pit_loss.empty else 0,
        "fastest_team_per_season": {},
    }
    for year, group in team_pace.groupby("year"):
        best = group.groupby("team")["team_gap_s"].median().sort_values().head(1)
        if not best.empty:
            summary["fastest_team_per_season"][str(int(year))] = {
                "team": str(best.index[0]),
                "median_gap_s": round(float(best.iloc[0]), 3),
            }
    (OUTPUTS_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[analysis] summary -> {OUTPUTS_DIR / 'analysis_summary.json'}")
    return summary


if __name__ == "__main__":
    run()
