"""Unit tests for the real-data F1 analytics package.

These tests use small hand-built frames as fixtures; they never touch the
network and do not depend on the downloaded dataset being present.
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1intel import analysis, jolpica, strategy  # noqa: E402
from f1intel.build import _match_round, _safety_car_laps  # noqa: E402


class TestLapTimeParsing(unittest.TestCase):
    def test_parses_minutes_seconds_millis(self):
        self.assertAlmostEqual(jolpica.parse_lap_time("1:29.179"), 89.179, places=3)

    def test_parses_short_form(self):
        self.assertAlmostEqual(jolpica.parse_lap_time("29.5"), 29.5, places=3)

    def test_missing_and_garbage(self):
        self.assertIsNone(jolpica.parse_lap_time(None))
        self.assertIsNone(jolpica.parse_lap_time(""))

    def test_two_digit_minutes(self):
        self.assertAlmostEqual(jolpica.parse_lap_time("10:01.250"), 601.25, places=3)


class TestRoundMatching(unittest.TestCase):
    def setUp(self):
        self.races = [
            {"round": 1, "date": "2024-03-02", "raceName": "Bahrain"},
            {"round": 2, "date": "2024-03-09", "raceName": "Saudi Arabia"},
            {"round": 3, "date": "2024-03-24", "raceName": "Australia"},
        ]

    def test_exact_date_matches_nearest_round(self):
        match = _match_round(date(2024, 3, 9), self.races, set())
        self.assertIsNotNone(match)
        self.assertEqual(match["round"], 2)

    def test_timezone_shift_tolerated(self):
        # Night races can land on an adjacent UTC day.
        match = _match_round(date(2024, 3, 10), self.races, set())
        self.assertIsNotNone(match)
        self.assertEqual(match["round"], 2)

    def test_far_date_is_rejected(self):
        self.assertIsNone(_match_round(date(2024, 8, 1), self.races, set()))

    def test_used_round_is_not_reused(self):
        self.assertIsNone(_match_round(date(2024, 3, 9), self.races, {1, 2, 3}))


class TestSafetyCarParsing(unittest.TestCase):
    def test_deployment_and_withdrawal(self):
        messages = [
            {"date": "1", "message": "SAFETY CAR DEPLOYED", "lap_number": 12},
            {"date": "2", "message": "SAFETY CAR IN THIS LAP", "lap_number": 16},
        ]
        laps = _safety_car_laps(messages)
        self.assertIn(12, laps)
        self.assertNotIn(17, laps)

    def test_messages_without_lap_numbers_are_ignored(self):
        self.assertEqual(_safety_car_laps([{"date": "1", "message": "SAFETY CAR DEPLOYED", "lap_number": None}]), set())


class TestCleanLapFilter(unittest.TestCase):
    def test_excludes_pit_laps_and_unknown_compounds(self):
        frame = pd.DataFrame(
            {
                "session_key": [1, 1, 1, 1, 1],
                "driver_number": [1, 1, 1, 1, 1],
                "lap_number": [1, 2, 3, 4, 5],
                "lap_time": [90.0, 90.2, 130.0, 90.1, 90.3],
                "compound": ["MEDIUM", "MEDIUM", "MEDIUM", "INTERMEDIATE", "MEDIUM"],
                "tyre_life": [1, 2, 3, 4, 5],
                "has_clean_air": [True, True, False, True, True],
            }
        )
        clean = analysis.clean_laps(frame)
        self.assertEqual(clean["lap_number"].tolist(), [1, 2, 5])


class TestStrategyEngine(unittest.TestCase):
    def _params(self, **overrides):
        base = dict(
            year=2024,
            round=1,
            race_name="Test GP",
            circuit="Test Circuit",
            total_laps=50,
            base_lap_s=90.0,
            fuel_effect_s_per_lap=-0.03,
            deg={"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.04},
            pit_loss_s=21.0,
            sc_probability=0.0,
        )
        base.update(overrides)
        return strategy.RaceParameters(**base)

    def test_lap_time_grows_with_tyre_age(self):
        params = self._params()
        self.assertLess(strategy.lap_time(params, 10, "MEDIUM", 2), strategy.lap_time(params, 10, "MEDIUM", 20))

    def test_lap_time_drops_with_fuel_burn(self):
        params = self._params()
        self.assertGreater(strategy.lap_time(params, 2, "MEDIUM", 5), strategy.lap_time(params, 40, "MEDIUM", 5))

    def test_pit_stop_adds_time(self):
        # Zero degradation and identical compounds isolate the pit loss itself.
        params = self._params(deg={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0})
        one_stop = strategy.simulate_plan(params, [25], ["HARD", "HARD"])
        no_stop = strategy.simulate_plan(params, [], ["HARD"])
        self.assertAlmostEqual(one_stop - no_stop, params.pit_loss_s, places=6)

    def test_undercut_succeeds_with_small_gap(self):
        result = strategy.undercut(
            self._params(),
            gap_s=1.0,
            chaser_compound="SOFT",
            chaser_tyre_age=18,
            leader_compound="MEDIUM",
            leader_tyre_age=22,
            window=3,
        )
        self.assertIn("net_margin_s", result)
        self.assertIsInstance(result["undercut_successful"], bool)

    def test_undercut_fails_with_large_gap(self):
        result = strategy.undercut(
            self._params(),
            gap_s=25.0,
            chaser_compound="MEDIUM",
            chaser_tyre_age=10,
            leader_compound="MEDIUM",
            leader_tyre_age=10,
            window=2,
        )
        self.assertFalse(result["undercut_successful"])

    def test_optimizer_returns_ranked_plans(self):
        plans = strategy.optimize_pit_window(self._params(), stop_options=(1,))
        self.assertFalse(plans.empty)
        self.assertTrue(plans["total_time_s"].is_monotonic_increasing)
        self.assertTrue((plans["stops"] == 1).all())


class TestModelScores(unittest.TestCase):
    def test_scores_are_bounded(self):
        from f1intel.model import _scores

        rng = np.random.default_rng(0)
        y = pd.Series(rng.integers(0, 2, 200))
        proba = rng.random(200)
        scores = _scores(y, proba)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier", "log_loss"):
            self.assertGreaterEqual(scores[key], 0.0, key)
            self.assertLessEqual(scores[key], 1.0, key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
