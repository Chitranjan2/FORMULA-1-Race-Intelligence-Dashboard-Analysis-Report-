"""F1 Race Intelligence — Streamlit dashboard over the real processed dataset.

Run with:
    streamlit run app/app.py

Everything shown here is computed from OpenF1 + Jolpica downloads recorded in
``data/processed/manifest.json``. Read that tab before quoting any number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from f1intel import strategy as strat  # noqa: E402
from f1intel.config import OUTPUTS_DIR  # noqa: E402
from f1intel.storage import load_table  # noqa: E402

st.set_page_config(page_title="F1 Race Intelligence", page_icon="🏎️", layout="wide")


# --------------------------------------------------------------------------- #
# cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    try:
        return load_table(name)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def document(name: str) -> dict:
    path = OUTPUTS_DIR / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def manifest() -> dict:
    path = REPO_ROOT / "data" / "processed" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


laps = table("race_laps")
results = table("results")
standings = table("standings")
races = table("races")
driver_pace = table("driver_pace")
team_pace = table("team_pace")
evolution = table("pace_evolution")
degradation = table("degradation")
pit_loss = table("pit_loss")
h2h = table("teammate_h2h")
strategy_optimum = table("strategy_optimum")
predictions = table("model_predictions")
analysis_summary = document("analysis_summary.json")
model_metrics = document("model_metrics.json")
manifest_data = manifest()

st.title("🏎️ F1 Race Intelligence")
st.caption(
    "Real Formula 1 data from **OpenF1** (timing, stints, pit stops) and **Jolpica** "
    "(results, qualifying, standings). Nothing here is simulated."
)

if laps.empty:
    st.error(
        "No processed data found. Run:\n\n"
        "```\npython -m f1intel build\npython -m f1intel analyze\npython -m f1intel train\n```"
    )
    st.stop()

season_options = sorted(int(y) for y in driver_pace["year"].dropna().unique()) if not driver_pace.empty else []
season = st.sidebar.selectbox("Season", season_options, index=len(season_options) - 1)
st.sidebar.metric("Driver-laps ingested", f"{len(laps):,}")
if analysis_summary:
    st.sidebar.metric("Clean laps used", f"{analysis_summary.get('clean_laps_used', 0):,}")
    st.sidebar.metric("Pit stops measured", f"{analysis_summary.get('pit_stops_measured', 0):,}")

tab_season, tab_pace, tab_tyres, tab_strategy, tab_model, tab_data = st.tabs(
    ["🏁 Season", "⏱️ Pace", "🛞 Tyres", "🧠 Strategy", "🤖 Model", "📦 Data"]
)

# --------------------------------------------------------------------------- #
# season overview
# --------------------------------------------------------------------------- #
with tab_season:
    season_results = results[results["year"] == season] if not results.empty else pd.DataFrame()
    season_races = races[races["year"] == season] if not races.empty else pd.DataFrame()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Races ingested", len(season_races))
    col_b.metric("Drivers scored", season_results["driver_code"].nunique() if not season_results.empty else 0)
    winners = (
        season_results.sort_values(["round", "position"]).groupby("round").first().reset_index()
        if not season_results.empty
        else pd.DataFrame()
    )
    if not winners.empty:
        col_c.metric("Different winners", winners["driver_code"].nunique())

    if not season_results.empty:
        driver_table = (
            season_results.groupby(["driver_code", "team"], dropna=False)
            .agg(
                Races=("round", "nunique"),
                Wins=("position", lambda s: int((s == 1).sum())),
                Podiums=("position", lambda s: int((s <= 3).sum())),
                Points=("points", "sum"),
                AvgFinish=("position", "mean"),
            )
            .reset_index()
            .sort_values("Points", ascending=False)
        )
        driver_table["AvgFinish"] = driver_table["AvgFinish"].round(2)
        st.subheader(f"{season} drivers")
        st.dataframe(driver_table, use_container_width=True, hide_index=True)

        st.subheader("Points progression")
        progress = season_results.sort_values(["round", "driver_code"]).copy()
        progress["cum_points"] = progress.groupby("driver_code")["points"].cumsum()
        pivot = progress.pivot_table(index="round", columns="driver_code", values="cum_points", aggfunc="last")
        st.line_chart(pivot.ffill(), height=380)

        st.subheader("Race winners")
        winners_small = winners[["round", "race_name", "driver_code", "team", "grid"]].rename(
            columns={"driver_code": "Winner", "team": "Team", "grid": "Grid"}
        )
        st.dataframe(winners_small, use_container_width=True, hide_index=True)

    if not standings.empty:
        st.subheader("Final standings")
        kind = st.radio("Table", ["driver", "constructor"], horizontal=True, label_visibility="collapsed")
        table_rows = standings[(standings["year"] == season) & (standings["kind"] == kind)]
        st.dataframe(table_rows[["position", "name", "team", "points"]], use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# pace
# --------------------------------------------------------------------------- #
with tab_pace:
    if evolution is None or evolution.empty:
        st.info("Pace tables missing — run: python -m f1intel analyze")
    else:
        season_evolution = evolution[evolution["year"] == season]
        st.subheader("Car performance trajectory")
        st.caption(
            "Fuel- and tyre-corrected pace gap to the fastest car, 3-race rolling median. "
            "Downward = development working."
        )
        top_teams = season_evolution.groupby("team")["team_gap_s"].median().sort_values().head(6).index.tolist()
        pivot = season_evolution[season_evolution["team"].isin(top_teams)].pivot_table(
            index="round", columns="team", values="rolling_gap_s"
        )
        st.line_chart(pivot, height=400)

        st.subheader("Teammate head-to-head")
        if not h2h.empty:
            st.caption("Race-pace wins and qualifying wins over the ingested seasons.")
            st.dataframe(h2h, use_container_width=True, hide_index=True)

        st.subheader("Driver pace ranking")
        ranking = (
            driver_pace[driver_pace["year"] == season]
            .groupby(["driver", "team"], dropna=False)["gap_to_fastest_s"]
            .median()
            .reset_index()
            .sort_values("gap_to_fastest_s")
        )
        ranking["gap_to_fastest_s"] = ranking["gap_to_fastest_s"].round(3)
        st.dataframe(ranking, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# tyres
# --------------------------------------------------------------------------- #
with tab_tyres:
    st.subheader("Tyre degradation by circuit")
    st.caption(
        "Seconds lost per lap of tyre age, fitted per race with fuel burn and driver pace "
        "controlled for, then median-aggregated per circuit."
    )
    if degradation.empty:
        st.info("Degradation table missing — run: python -m f1intel analyze")
    else:
        compound = st.multiselect("Compound", sorted(degradation["compound"].unique()), default=["MEDIUM", "HARD"])
        view = degradation[degradation["compound"].isin(compound)].sort_values("deg_s_per_lap", ascending=False)
        st.dataframe(
            view[["circuit", "compound", "deg_s_per_lap", "deg_p10", "deg_p90", "races", "laps"]],
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(view.pivot_table(index="circuit", columns="compound", values="deg_s_per_lap"), height=420)

    if not pit_loss.empty:
        st.subheader("Measured pit loss")
        st.caption("Extra time on the pit-in and pit-out laps versus the driver's own clean-lap median.")
        st.line_chart(pit_loss[pit_loss["year"] == season].set_index("round")["pit_loss_s"], height=300)

# --------------------------------------------------------------------------- #
# strategy lab
# --------------------------------------------------------------------------- #
with tab_strategy:
    st.subheader("Strategy lab")
    st.caption("Parameters below are measured from the selected race, not assumed.")
    if strategy_optimum.empty:
        st.info("Strategy table missing — run: python -m f1intel strategy")
    else:
        season_strategy = strategy_optimum[strategy_optimum["year"] == season]
        if season_strategy.empty:
            season_strategy = strategy_optimum
        labels = (season_strategy["round"].astype(str) + " · " + season_strategy["race_name"]).tolist()
        choice = st.selectbox("Race", labels)
        selected = season_strategy.iloc[labels.index(choice)]
        row = selected.to_dict()

        cols = st.columns(5)
        cols[0].metric("Measured pit loss", f"{row['pit_loss_s']:.1f} s")
        cols[1].metric("Fuel effect", f"{row['fuel_effect_s_per_lap']:+.3f} s/lap")
        cols[2].metric("Soft deg", f"{row.get('deg_soft', 0):.3f} s/lap")
        cols[3].metric("Hard deg", f"{row.get('deg_hard', 0):.3f} s/lap")
        cols[4].metric("Safety-car rate", f"{row['sc_probability'] * 100:.0f}% of races")

        st.write("**What the model recommends**")
        best = pd.DataFrame(
            [
                {
                    "Plan": "1-stop",
                    "Pit laps": row.get("best_1stop_lap"),
                    "Compounds": row.get("best_1stop_compounds"),
                    "Projected time": f"{row.get('best_1stop_s', float('nan')):.1f} s",
                },
                {
                    "Plan": "2-stop",
                    "Pit laps": row.get("best_2stop_laps"),
                    "Compounds": row.get("best_2stop_compounds"),
                    "Projected time": f"{row.get('best_2stop_s', float('nan')):.1f} s",
                },
            ]
        )
        st.dataframe(best, hide_index=True, use_container_width=True)
        delta = row.get("delta_2stop_vs_1stop_s")
        if delta is not None and pd.notna(delta):
            st.success(
                f"2-stop projects {abs(delta):.1f} s {'slower' if delta > 0 else 'faster'} than the best "
                f"1-stop → **{int(row['recommended_stops'])}-stop recommended**."
            )

        st.divider()
        st.write("**Sensitivity: what if the pit loss or degradation is different?**")
        pit_loss_override = st.slider("Pit loss (s)", 15.0, 30.0, float(row["pit_loss_s"]), 0.1)
        deg_scale = st.slider("Degradation multiplier", 0.5, 1.6, 1.0, 0.05)
        if st.button("Re-run pit-window optimisation"):
            params = strat.RaceParameters(
                year=int(row["year"]),
                round=int(row["round"]),
                race_name=str(row["race_name"]),
                circuit=str(row["circuit"]),
                total_laps=int(row["total_laps"]),
                base_lap_s=float(row["base_lap_s"]),
                fuel_effect_s_per_lap=float(row["fuel_effect_s_per_lap"]),
                deg={
                    "SOFT": float(row.get("deg_soft", 0.055)) * deg_scale,
                    "MEDIUM": float(row.get("deg_medium", 0.055)) * deg_scale,
                    "HARD": float(row.get("deg_hard", 0.055)) * deg_scale,
                },
                pit_loss_s=pit_loss_override,
                sc_probability=float(row["sc_probability"]),
            )
            plans = strat.optimize_pit_window(params)
            st.dataframe(plans.head(10), hide_index=True, use_container_width=True)
            if len(plans) >= 2:
                sim = strat.monte_carlo(params, plans.head(2).to_dict("records"), n_sims=500)
                st.write(f"Monte Carlo over 500 simulated races: **{sim['recommended_plan']}**")
                st.dataframe(pd.DataFrame(sim["comparison"]), hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader("Podium model")
    if not model_metrics:
        st.info("Model metrics missing — run: python -m f1intel train")
    else:
        st.caption(f"Target: {model_metrics.get('target')} · Validation: {model_metrics.get('validation')}")
        by_season = pd.DataFrame(model_metrics.get("by_season", []))
        if not by_season.empty:
            baseline = pd.json_normalize(by_season["grid_baseline"]).add_prefix("baseline_")
            view = pd.concat([by_season.drop(columns=["grid_baseline"]), baseline], axis=1)
            st.write("**Walk-forward results (train on earlier seasons only)**")
            st.dataframe(
                view[["season", "train_rows", "accuracy", "roc_auc", "brier", "baseline_accuracy", "baseline_roc_auc"]].round(3),
                hide_index=True,
                use_container_width=True,
            )

        pooled = model_metrics.get("pooled", {})
        cols = st.columns(6)
        cols[0].metric("ROC-AUC", pooled.get("roc_auc"))
        cols[1].metric("Accuracy", pooled.get("accuracy"))
        cols[2].metric("Brier", pooled.get("brier"))
        cols[3].metric("Log loss", pooled.get("log_loss"))
        cols[4].metric("Precision", pooled.get("precision"))
        cols[5].metric("Recall", pooled.get("recall"))

        calibration = pd.DataFrame(model_metrics.get("calibration", []))
        if not calibration.empty:
            st.write("**Calibration** — predicted probability vs observed frequency")
            st.dataframe(
                calibration.rename(columns={"mean_predicted": "Predicted", "observed": "Observed"}),
                hide_index=True,
                use_container_width=True,
            )
            st.line_chart(calibration.set_index("mean_predicted")["observed"], height=260)

        if not predictions.empty:
            st.write("**Holdout predictions**")
            view = predictions.sort_values(["year", "round", "podium_proba"], ascending=[False, True, False])
            st.dataframe(view, use_container_width=True, hide_index=True, height=360)

# --------------------------------------------------------------------------- #
# data provenance
# --------------------------------------------------------------------------- #
with tab_data:
    st.subheader("Provenance")
    st.json(manifest_data or {"error": "manifest missing — run python -m f1intel build"})
    st.subheader("What this dataset is not")
    st.markdown(
        "- It is **not** official Formula 1 telemetry. OpenF1 is an independent, unofficial project.\n"
        "- It contains **no simulated rows**. Every lap, stint, pit stop and result was downloaded.\n"
        "- Neutralised laps are approximated from race-control messages and may miss edge cases.\n"
        "- Strategy projections are a model, not advice; real races are decided by weather, "
        "traffic and human judgement.\n"
        "- This project is not affiliated with Formula 1, the FIA, or any team."
    )
