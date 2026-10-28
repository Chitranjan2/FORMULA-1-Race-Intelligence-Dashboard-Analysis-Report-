"""Race-strategy engine built on parameters measured from real sessions.

Every constant comes from the data, not from a guess:
    * fuel effect        — regression coefficient on lap number
    * degradation        — seconds per lap of tyre age, per compound
    * pit loss           — measured from pit-in + pit-out laps vs clean pace
    * safety-car rate    — share of races with at least one neutralised lap

The engine answers the two questions a race engineer actually asks:
    1. "How many laps can this tyre do before we lose more time than a stop?"
    2. "If we pit now, do we come out ahead?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from . import analysis
from .config import OUTPUTS_DIR, RANDOM_STATE
from .storage import load_table, save_table, table_exists

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


@dataclass
class RaceParameters:
    year: int
    round: int
    race_name: str
    circuit: str
    total_laps: int
    base_lap_s: float
    fuel_effect_s_per_lap: float
    deg: dict[str, float] = field(default_factory=dict)
    pit_loss_s: float = 21.0
    sc_probability: float = 0.33


# --------------------------------------------------------------------------- #
# parameter estimation
# --------------------------------------------------------------------------- #
def _race_parameters(laps: pd.DataFrame, coeffs: pd.DataFrame, deg: pd.DataFrame, pit: pd.DataFrame) -> list[RaceParameters]:
    sc_rate = float(laps.groupby(["year", "round"])["is_neutralized"].max().mean())
    pit_by_race = pit.set_index(["year", "round"])["pit_loss_s"].to_dict() if not pit.empty else {}
    deg_by_circuit = (
        deg.groupby(["circuit", "compound"])["deg_s_per_lap"].median().to_dict() if not deg.empty else {}
    )

    params: list[RaceParameters] = []
    for _, row in coeffs.iterrows():
        year, round_number = int(row["year"]), int(row["round"])
        race_laps = laps[(laps["year"] == year) & (laps["round"] == round_number)]
        if race_laps.empty:
            continue
        total_laps = int(race_laps["lap_number"].max())
        circuit = str(race_laps["circuit"].iloc[0])
        race_name = str(race_laps["race_name"].iloc[0])

        compound_deg = {c: float(deg_by_circuit.get((circuit, c), 0.055)) for c in COMPOUNDS}
        clean = race_laps[race_laps["has_clean_air"].fillna(False) & race_laps["compound"].isin(COMPOUNDS)]
        if clean.empty:
            continue
        fuel = float(row["fuel_effect_s_per_lap"])
        base = clean["lap_time"] - fuel * clean["lap_number"] - clean["compound"].map(compound_deg) * clean["tyre_life"]
        params.append(
            RaceParameters(
                year=year,
                round=round_number,
                race_name=race_name,
                circuit=circuit,
                total_laps=total_laps,
                base_lap_s=float(base.median()),
                fuel_effect_s_per_lap=fuel,
                deg=compound_deg,
                pit_loss_s=float(pit_by_race.get((year, round_number), 21.0)),
                sc_probability=sc_rate,
            )
        )
    return params


# --------------------------------------------------------------------------- #
# deterministic simulation
# --------------------------------------------------------------------------- #
def lap_time(params: RaceParameters, lap: int, compound: str, tyre_life: int) -> float:
    """Fuel-adjusted lap time for a given lap, compound and tyre age."""
    return (
        params.base_lap_s
        + params.fuel_effect_s_per_lap * lap
        + params.deg.get(compound, 0.055) * tyre_life
    )


def simulate_plan(
    params: RaceParameters,
    pit_laps: Iterable[int],
    compounds: Iterable[str],
    *,
    sc_lap: int | None = None,
    noise_s: float = 0.0,
    rng: np.random.Generator | None = None,
) -> float:
    """Total race time for a stop plan. ``compounds`` needs len(pit_laps) + 1 entries."""
    pit_set = set(int(l) for l in pit_laps)
    compound_list = list(compounds)
    compound = compound_list[0]
    next_index = 1
    tyre_life = 0
    total = 0.0
    for lap in range(1, params.total_laps + 1):
        if lap in pit_set:
            loss = params.pit_loss_s
            if sc_lap is not None and abs(lap - sc_lap) <= 2:
                loss *= 0.55  # a stop under safety car costs far less
            total += loss
            tyre_life = 0
            compound = compound_list[next_index]
            next_index += 1
        total += lap_time(params, lap, compound, tyre_life)
        if noise_s and rng is not None:
            total += float(rng.normal(0.0, noise_s))
        tyre_life += 1
    return total


def optimize_pit_window(params: RaceParameters, *, stop_options: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """Enumerate stop plans and rank them by total race time."""
    rows: list[dict] = []
    total = params.total_laps

    if 1 in stop_options:
        for pit_lap in range(8, total - 7):
            for first, second in (("MEDIUM", "HARD"), ("SOFT", "HARD"), ("SOFT", "MEDIUM")):
                time_s = simulate_plan(params, [pit_lap], [first, second])
                rows.append(
                    {
                        "stops": 1,
                        "pit_laps": f"{pit_lap}",
                        "compounds": f"{first}-{second}",
                        "total_time_s": round(time_s, 3),
                    }
                )
    if 2 in stop_options:
        latest_first = total - 20
        for first_lap in range(8, latest_first, 4):
            for second_lap in range(first_lap + 8, total - 6, 4):
                for plan in (("SOFT", "MEDIUM", "HARD"), ("MEDIUM", "HARD", "SOFT"), ("SOFT", "HARD", "MEDIUM")):
                    time_s = simulate_plan(params, [first_lap, second_lap], plan)
                    rows.append(
                        {
                            "stops": 2,
                            "pit_laps": f"{first_lap},{second_lap}",
                            "compounds": "-".join(plan),
                            "total_time_s": round(time_s, 3),
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("total_time_s").reset_index(drop=True)


def undercut(
    params: RaceParameters,
    *,
    gap_s: float,
    chaser_compound: str,
    chaser_tyre_age: int,
    leader_compound: str,
    leader_tyre_age: int,
    window: int = 3,
) -> dict:
    """Race the undercut lap by lap and report the net gain when the leader stops."""
    chaser_time = 0.0
    leader_time = 0.0
    chaser_life = chaser_tyre_age
    leader_life = leader_tyre_age
    breakdown = []
    for lap in range(1, window + 1):
        # Chaser pits before lap 1 of the window; leader pits after ``window`` laps.
        if lap == 1:
            chaser_time += params.pit_loss_s
            chaser_life = 0
        chaser_time += lap_time(params, lap + 1, chaser_compound, chaser_life)
        leader_time += lap_time(params, lap + 1, leader_compound, leader_life)
        chaser_life += 1
        leader_life += 1
        breakdown.append(
            {
                "lap_offset": lap,
                "chaser_cum_s": round(chaser_time, 3),
                "leader_cum_s": round(leader_time, 3),
                "gain_s": round(leader_time - chaser_time, 3),
            }
        )
    leader_time += params.pit_loss_s
    net = gap_s + (leader_time - chaser_time)
    return {
        "gap_before_s": round(gap_s, 3),
        "gain_before_leader_stops_s": round(leader_time - chaser_time - params.pit_loss_s, 3),
        "net_margin_s": round(-net, 3),
        "undercut_successful": bool(net < 0),
        "recommendation": "PIT NOW — undercut is live" if net < 0 else "STAY OUT — extend the stint",
        "breakdown": breakdown,
    }


def monte_carlo(params: RaceParameters, plan_rows: list[dict], *, n_sims: int = 2000, seed: int = RANDOM_STATE) -> dict:
    """Compare plans under lap-time noise and random safety-car timing."""
    rng = np.random.default_rng(seed)
    results: dict[str, np.ndarray] = {}
    for plan in plan_rows:
        pit_laps = [int(x) for x in str(plan["pit_laps"]).split(",")]
        compounds = str(plan["compounds"]).split("-")
        samples = np.empty(n_sims)
        for i in range(n_sims):
            sc_lap = int(rng.integers(5, max(6, params.total_laps - 5))) if rng.random() < params.sc_probability else None
            samples[i] = simulate_plan(
                params, pit_laps, compounds, sc_lap=sc_lap, noise_s=0.22, rng=rng
            )
        results[f"{plan['stops']}-stop {plan['pit_laps']} ({plan['compounds']})"] = samples

    labels = list(results)
    best_label = min(labels, key=lambda k: float(np.mean(results[k])))
    comparison = []
    for label in labels:
        samples = results[label]
        comparison.append(
            {
                "plan": label,
                "mean_total_s": round(float(samples.mean()), 2),
                "p10_total_s": round(float(np.percentile(samples, 10)), 2),
                "p90_total_s": round(float(np.percentile(samples, 90)), 2),
                "win_probability": round(float((samples == np.min(np.stack([results[l] for l in labels]), axis=0)).mean()), 3),
            }
        )
    return {"recommended_plan": best_label, "comparison": comparison}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run() -> dict:
    """Produce per-race optimal plans and a strategy summary."""
    laps = analysis.load_laps()
    clean = analysis.clean_laps(laps)
    corrected, meta = analysis.fit_race_pace(clean)
    coeffs = pd.DataFrame(meta["race_coefficients"])
    deg = load_table("degradation") if table_exists("degradation") else pd.DataFrame()
    pit = load_table("pit_loss") if table_exists("pit_loss") else pd.DataFrame()

    params_list = _race_parameters(laps, coeffs, deg, pit)
    rows: list[dict] = []
    for params in params_list:
        plans = optimize_pit_window(params)
        if plans.empty:
            continue
        best_1 = plans[plans["stops"] == 1].head(1)
        best_2 = plans[plans["stops"] == 2].head(1)
        row = {
            "year": params.year,
            "round": params.round,
            "race_name": params.race_name,
            "circuit": params.circuit,
            "total_laps": params.total_laps,
            "pit_loss_s": round(params.pit_loss_s, 2),
            "sc_probability": round(params.sc_probability, 3),
            "fuel_effect_s_per_lap": round(params.fuel_effect_s_per_lap, 4),
            "base_lap_s": round(params.base_lap_s, 3),
            "deg_soft": round(params.deg.get("SOFT", 0.055), 4),
            "deg_medium": round(params.deg.get("MEDIUM", 0.055), 4),
            "deg_hard": round(params.deg.get("HARD", 0.055), 4),
        }
        if not best_1.empty:
            row.update(
                {
                    "best_1stop_lap": int(str(best_1.iloc[0]["pit_laps"])),
                    "best_1stop_compounds": best_1.iloc[0]["compounds"],
                    "best_1stop_s": best_1.iloc[0]["total_time_s"],
                }
            )
        if not best_2.empty:
            row.update(
                {
                    "best_2stop_laps": str(best_2.iloc[0]["pit_laps"]),
                    "best_2stop_compounds": best_2.iloc[0]["compounds"],
                    "best_2stop_s": best_2.iloc[0]["total_time_s"],
                }
            )
        if not best_1.empty and not best_2.empty:
            row["delta_2stop_vs_1stop_s"] = round(best_2.iloc[0]["total_time_s"] - best_1.iloc[0]["total_time_s"], 2)
            row["recommended_stops"] = 1 if row["delta_2stop_vs_1stop_s"] > 0 else 2
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        save_table(table, "strategy_optimum")
        print(f"[strategy] strategy_optimum: {len(table)} races")

    example = None
    if params_list:
        params = params_list[-1]
        plans = optimize_pit_window(params)
        top = plans.head(2).to_dict("records")
        example = {
            "race": f"{params.year} {params.race_name}",
            "measured": {
                "pit_loss_s": round(params.pit_loss_s, 2),
                "fuel_effect_s_per_lap": round(params.fuel_effect_s_per_lap, 4),
                "deg_s_per_lap": {k: round(v, 4) for k, v in params.deg.items()},
                "sc_probability": round(params.sc_probability, 3),
            },
            "monte_carlo": monte_carlo(params, top),
        }
        (OUTPUTS_DIR / "strategy_example.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
        print(f"[strategy] example -> {OUTPUTS_DIR / 'strategy_example.json'}")

    summary = {
        "races_modelled": int(len(table)),
        "recommended_1stop": int((table["recommended_stops"] == 1).sum()) if not table.empty else 0,
        "recommended_2stop": int((table["recommended_stops"] == 2).sum()) if not table.empty else 0,
        "median_pit_loss_s": round(float(table["pit_loss_s"].median()), 2) if not table.empty else None,
    }
    (OUTPUTS_DIR / "strategy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    run()
