"""Static figures for the README and portfolio.

Deliberately dependency-light: if matplotlib is unavailable the module prints a
notice instead of failing the pipeline.
"""

from __future__ import annotations

import json

import pandas as pd

from .config import OUTPUTS_DIR
from .storage import load_table, table_exists

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False

F1_RED = "#E10600"
PALETTE = ["#E10600", "#1E41FF", "#00D2BE", "#FF8700", "#006F62", "#DC0000", "#900000", "#FFFFFF", "#B6BABD", "#F596C8"]


def _team_colours(teams: list[str]) -> dict[str, str]:
    return {team: PALETTE[i % len(PALETTE)] for i, team in enumerate(sorted(teams))}


def _save(fig, name: str) -> None:
    path = OUTPUTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="#0E1117")
    plt.close(fig)
    print(f"[report] {path.name}")


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor("#0E1117")
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color="#B6BABD")
    ax.set_ylabel(ylabel, color="#B6BABD")
    ax.tick_params(colors="#B6BABD")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.grid(True, alpha=0.15)


def plot_pace_evolution() -> None:
    if not table_exists("pace_evolution"):
        return
    evolution = load_table("pace_evolution")
    if evolution.empty:
        return
    season = int(evolution["year"].max())
    subset = evolution[evolution["year"] == season]
    top = subset.groupby("team")["team_gap_s"].median().sort_values().head(6).index.tolist()
    subset = subset[subset["team"].isin(top)]
    colours = _team_colours(top)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#0E1117")
    for team, group in subset.groupby("team"):
        group = group.sort_values("round")
        ax.plot(group["round"], group["rolling_gap_s"], label=team, color=colours[team], lw=2.2, marker="o", ms=3)
    _style(ax, f"{season} car performance trajectory (3-race rolling)", "Round", "Gap to fastest car (s)")
    ax.invert_yaxis()  # smaller gap = better, so plot it upward
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("#B6BABD")
    _save(fig, "pace_evolution_teams.png")


def plot_degradation() -> None:
    if not table_exists("degradation"):
        return
    degradation = load_table("degradation")
    if degradation.empty:
        return
    summary = degradation.groupby("compound")["deg_s_per_lap"].agg(["median", "count"]).reindex(["SOFT", "MEDIUM", "HARD"])
    summary = summary.dropna()
    if summary.empty:
        return
    colours = {"SOFT": F1_RED, "MEDIUM": "#FFD700", "HARD": "#B6BABD"}

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#0E1117")
    ax.bar(summary.index, summary["median"], color=[colours.get(c, "#888888") for c in summary.index], alpha=0.9)
    for i, (value, count) in enumerate(zip(summary["median"], summary["count"])):
        ax.text(i, value, f"{value:.3f} s/lap\n{int(count)} circuits", ha="center", va="bottom", color="white", fontsize=9)
    _style(ax, "Measured tyre degradation by compound", "Compound", "Seconds lost per lap of tyre age")
    _save(fig, "degradation_by_compound.png")


def plot_pit_loss() -> None:
    if not table_exists("pit_loss"):
        return
    pit = load_table("pit_loss")
    if pit.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0E1117")
    ax.hist(pit["pit_loss_s"], bins=20, color=F1_RED, alpha=0.85, edgecolor="#3a0000")
    median = float(pit["pit_loss_s"].median())
    ax.axvline(median, color="white", ls="--", lw=1.5)
    ax.text(median + 0.2, ax.get_ylim()[1] * 0.9, f"median {median:.1f}s", color="white", fontsize=10)
    _style(ax, "Pit loss measured per race", "Pit loss (s)", "Races")
    _save(fig, "pit_loss_distribution.png")


def plot_teammate_h2h() -> None:
    if not table_exists("teammate_h2h"):
        return
    h2h = load_table("teammate_h2h")
    if h2h.empty:
        return
    h2h = h2h.sort_values("race_pace_a", ascending=True).tail(12)
    labels = [f"{row.driver_a} vs {row.driver_b}" for row in h2h.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#0E1117")
    ax.barh(labels, h2h["race_pace_a"], color=F1_RED, alpha=0.9, label="wins of driver A")
    ax.barh(labels, h2h["race_pace_b"], left=h2h["race_pace_a"], color="#1E41FF", alpha=0.75, label="wins of driver B")
    _style(ax, "Team-mate race-pace head-to-head", "Races won on corrected pace", "")
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color("#B6BABD")
    _save(fig, "teammate_head_to_head.png")


def plot_calibration() -> None:
    path = OUTPUTS_DIR / "model_metrics.json"
    if not path.exists():
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    calibration = pd.DataFrame(metrics.get("calibration", []))
    if calibration.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor="#0E1117")
    ax.plot([0, 1], [0, 1], ls="--", color="#666666", label="perfect calibration")
    ax.plot(calibration["mean_predicted"], calibration["observed"], color=F1_RED, lw=2.5, marker="o", label="model")
    _style(ax, "Calibration — walk-forward holdout", "Predicted podium probability", "Observed podium rate")
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("#B6BABD")
    _save(fig, "model_calibration.png")


def run() -> None:
    if not HAVE_MPL:
        print("[report] matplotlib not installed — skipping figures")
        return
    plot_pace_evolution()
    plot_degradation()
    plot_pit_loss()
    plot_teammate_h2h()
    plot_calibration()


if __name__ == "__main__":
    run()
