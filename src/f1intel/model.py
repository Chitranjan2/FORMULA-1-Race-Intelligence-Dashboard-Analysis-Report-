"""Podium-prediction model with honest temporal validation.

The previous version of this project reported a single holdout score. This
module instead:

* uses **walk-forward validation** — for every holdout season, the model is
  trained only on races that happened *before* it (no future information), and
  each season is scored separately;
* always compares against a **grid-only baseline**, so it is clear how much the
  engineered features actually add;
* reports **probability quality** (Brier score, log loss) and a calibration
  table, not just accuracy — a model that says "80% podium" should be right
  about 80% of the time.

Use ``python -m f1intel.model`` to rebuild the artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import HOLDOUT_SEASONS, OUTPUTS_DIR, RANDOM_STATE
from .storage import load_table, save_table, table_exists

STREET_CIRCUITS = ("Monaco", "Singapore", "Baku", "Las Vegas", "Jeddah", "Miami")

FEATURE_COLUMNS = [
    "grid",
    "quali_gap_s",
    "quali_position",
    "rolling_pace_gap_s",
    "rolling_podium_rate",
    "rolling_avg_finish",
    "round",
    "is_street",
    "race_laps",
]


@dataclass
class FeatureFrame:
    data: pd.DataFrame
    features: list[str]


def build_features() -> FeatureFrame:
    """Assemble one row per driver-race using only pre-race information."""
    results = load_table("results").copy()
    quali = load_table("quali_best")
    driver_pace = load_table("driver_pace") if table_exists("driver_pace") else pd.DataFrame()

    results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
    results["position"] = pd.to_numeric(results["position"], errors="coerce")
    results["grid"] = results["grid"].replace(0, 20).fillna(20)  # pit-lane start
    results = results[results["position"].notna()].copy()
    results["podium_finish"] = (results["position"] <= 3).astype(int)
    results["round"] = pd.to_numeric(results["round"], errors="coerce").astype(int)
    results["year"] = pd.to_numeric(results["year"], errors="coerce").astype(int)

    # Qualifying: best lap per driver and gap to pole.
    quasi = quali.dropna(subset=["driver", "lap_time"]).copy()
    quasi["year"] = pd.to_numeric(quasi["year"], errors="coerce").astype(int)
    quasi["round"] = pd.to_numeric(quasi["round"], errors="coerce").astype(int)
    quasi["lap_time"] = pd.to_numeric(quasi["lap_time"], errors="coerce")
    quasi = quasi.sort_values(["year", "round", "lap_time"])
    quasi["quali_position"] = quasi.groupby(["year", "round"]).cumcount() + 1
    quasi["pole_time"] = quasi.groupby(["year", "round"])["lap_time"].transform("min")
    quasi["quali_gap_s"] = quasi["lap_time"] - quasi["pole_time"]
    quasi = quasi.rename(columns={"driver": "driver_code"})[
        ["year", "round", "driver_code", "quali_gap_s", "quali_position"]
    ]
    frame = results.merge(quasi, on=["year", "round", "driver_code"], how="left")

    # Rolling car pace from the corrected-pace analysis, shifted to avoid leakage.
    if not driver_pace.empty:
        pace = driver_pace[["year", "round", "driver", "gap_to_fastest_s"]].rename(
            columns={"driver": "driver_code", "gap_to_fastest_s": "race_pace_gap_s"}
        )
        pace["year"] = pd.to_numeric(pace["year"], errors="coerce").astype(int)
        pace["round"] = pd.to_numeric(pace["round"], errors="coerce").astype(int)
        frame = frame.merge(pace, on=["year", "round", "driver_code"], how="left")
        frame = frame.sort_values(["driver_code", "year", "round"])
        frame["rolling_pace_gap_s"] = (
            frame.groupby("driver_code")["race_pace_gap_s"].transform(lambda s: s.shift(1).expanding().median())
        )
    else:
        frame["race_pace_gap_s"] = np.nan
        frame["rolling_pace_gap_s"] = np.nan

    # Rolling results form — shifted one race back, so the current race never leaks.
    frame = frame.sort_values(["driver_code", "year", "round"])
    grouped = frame.groupby("driver_code")
    frame["rolling_podium_rate"] = grouped["podium_finish"].transform(lambda s: s.shift(1).expanding().mean())
    frame["rolling_avg_finish"] = grouped["position"].transform(lambda s: s.shift(1).expanding().mean())

    frame["is_street"] = frame["circuit"].fillna("").apply(
        lambda c: int(any(name.lower() in str(c).lower() for name in STREET_CIRCUITS))
    )
    laps_per_race = results.groupby(["year", "round"])["laps_completed"].max().rename("race_laps").reset_index()
    frame = frame.merge(laps_per_race, on=["year", "round"], how="left")
    frame["race_laps"] = pd.to_numeric(frame["race_laps"], errors="coerce")

    return FeatureFrame(frame, FEATURE_COLUMNS)


def _fit(train: pd.DataFrame, features: list[str]):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=3,
                    learning_rate=0.06,
                    max_iter=300,
                    min_samples_leaf=20,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(train[features], train["podium_finish"])
    return model


def _grid_baseline(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict[str, float]:
    """Logistic regression on grid position alone — the honest floor to beat."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    model = Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(train[["grid"]], train["podium_finish"])
    proba = model.predict_proba(test[["grid"]])[:, 1]
    return _scores(test["podium_finish"], proba)


def _scores(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (proba >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4) if y_true.nunique() > 1 else float("nan"),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "log_loss": round(float(log_loss(y_true, proba, labels=[0, 1])), 4),
    }


def _calibration_table(y_true: pd.Series, proba: np.ndarray) -> list[dict[str, Any]]:
    from sklearn.calibration import calibration_curve

    try:
        fraction, mean_predicted = calibration_curve(y_true, proba, n_bins=8, strategy="quantile")
    except ValueError:
        return []
    return [
        {"mean_predicted": round(float(m), 4), "observed": round(float(f), 4)}
        for m, f in zip(mean_predicted, fraction)
    ]


def run() -> dict[str, Any]:
    frame = build_features()
    data = frame.data
    features = frame.features

    seasons = sorted(data["year"].unique())
    holdouts = [s for s in seasons if s in HOLDOUT_SEASONS]
    if not holdouts:
        holdouts = seasons[-1:]

    per_season: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    pooled_true: list[pd.Series] = []
    pooled_proba: list[np.ndarray] = []

    for season in holdouts:
        train = data[data["year"] < season]
        test = data[data["year"] == season]
        if train.empty or test.empty:
            continue
        model = _fit(train, features)
        proba = model.predict_proba(test[features])[:, 1]
        score = _scores(test["podium_finish"], proba)
        score["season"] = int(season)
        score["train_rows"] = int(len(train))
        score["train_seasons"] = sorted(int(y) for y in train["year"].unique())
        score["grid_baseline"] = _grid_baseline(train, test, features)
        per_season.append(score)

        preds = test[["year", "round", "race_name", "driver_code", "team", "grid", "position"]].copy()
        preds["podium_true"] = test["podium_finish"].to_numpy()
        preds["podium_proba"] = proba
        predictions.append(preds)
        pooled_true.append(test["podium_finish"])
        pooled_proba.append(proba)
        print(f"[model] {season}: AUC={score['roc_auc']:.3f} (grid baseline {score['grid_baseline']['roc_auc']:.3f})")

    if not per_season:
        raise RuntimeError("No walk-forward seasons could be evaluated — check the ingested date range")

    pooled = _scores(pd.concat(pooled_true), np.concatenate(pooled_proba))
    pooled["calibration"] = _calibration_table(pd.concat(pooled_true), np.concatenate(pooled_proba))
    pooled["per_season"] = per_season

    # Final artifact: refit on every completed season for live inference.
    full_train = data.copy()
    final_model = _fit(full_train, features)

    try:
        import joblib

        joblib.dump(
            {
                "pipeline": final_model,
                "features": features,
                "trained_through": int(data["year"].max()),
                "pooled_metrics": {k: v for k, v in pooled.items() if k not in ("calibration", "per_season")},
            },
            OUTPUTS_DIR / "podium_model.joblib",
        )
        print(f"[model] artifact -> {OUTPUTS_DIR / 'podium_model.joblib'}")
    except ImportError:
        print("[model] joblib unavailable; skipping artifact")

    if predictions:
        all_predictions = pd.concat(predictions, ignore_index=True)
        save_table(all_predictions, "model_predictions")

    metrics = {
        "target": "podium_finish (finish position <= 3)",
        "model": "SimpleImputer(median) + HistGradientBoostingClassifier(max_depth=3)",
        "validation": "walk-forward by season (train only on earlier seasons)",
        "holdout_seasons": sorted(int(s) for s in holdouts),
        "feature_columns": features,
        "pooled": {k: v for k, v in pooled.items() if k not in ("calibration", "per_season")},
        "by_season": per_season,
        "calibration": pooled["calibration"],
        "notes": [
            "Baseline is a grid-only logistic regression, included so the lift from engineered features is explicit.",
            "Rolling features are shifted by one race, so no current-race information leaks into the features.",
            "Probability quality (Brier / log loss / calibration) is reported because accuracy alone is misleading for rare events.",
        ],
    }
    (OUTPUTS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[model] metrics -> {OUTPUTS_DIR / 'model_metrics.json'}")
    return metrics


if __name__ == "__main__":
    run()
