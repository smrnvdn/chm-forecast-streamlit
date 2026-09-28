"""Regression checks for the shared two-region dashboard."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.forecast_data import REGIONS, daily_summary, snapshot_path, validate_snapshot
from src.presentation import forecast_matrix


class RegionSnapshotTests(unittest.TestCase):
    def test_both_snapshots_have_one_target_and_valid_attempts(self) -> None:
        for code in REGIONS:
            with self.subTest(region=code):
                hourly = validate_snapshot(pd.read_csv(snapshot_path(code), parse_dates=["date"]), code)
                daily = daily_summary(hourly)
                self.assertEqual(len(daily), 161)
                self.assertEqual(len(hourly), 1968)
                matrix = forecast_matrix(hourly, REGIONS[code].score_kind)
                self.assertEqual(matrix.shape, (161, 14))

    def test_volgograd_frozen_metrics_and_market_hour_mapping(self) -> None:
        hourly = validate_snapshot(pd.read_csv(snapshot_path("volgograd"), parse_dates=["date"]), "volgograd")
        daily = daily_summary(hourly)
        self.assertEqual((int(daily.hit_at_2.sum()), int(daily.hit_segment_2plus2.sum())), (82, 94))
        self.assertEqual(int(hourly.market_hour.min()), 8)
        self.assertEqual(int(hourly.market_hour.max()), 21)
        self.assertIn("probability", hourly.columns)
        self.assertIn("score", hourly.columns)
        self.assertTrue((hourly.groupby("date").probability.sum().sub(1).abs() < 1e-8).all())

        # Calibration and display formatting must never change m0399's picks.
        matrix = forecast_matrix(hourly, "probability")
        displayed_picks = matrix.attrs["predicted_cells"]
        self.assertEqual(len(displayed_picks), int(hourly.is_predicted.sum()))
        for (date, segment), part in hourly.groupby(["date", "segment"]):
            raw_top = set(part.sort_values(["score", "market_hour"], ascending=[False, True]).market_hour.head(2))
            calibrated_top = set(part.sort_values(["probability", "market_hour"], ascending=[False, True]).market_hour.head(2))
            marked = set(part.loc[part.is_predicted, "market_hour"])
            self.assertEqual(raw_top, calibrated_top, (date, segment))
            self.assertEqual(raw_top, marked, (date, segment))

    def test_unlabeled_future_style_day_has_no_false_accuracy(self) -> None:
        hourly = pd.read_csv(snapshot_path("volgograd"), parse_dates=["date"])
        one_day = hourly.loc[hourly.date.eq(hourly.date.max())].copy()
        one_day["date"] = pd.Timestamp("2026-10-01")
        one_day["is_actual"] = 0
        one_day["hit_at_2"] = float("nan")
        one_day["hit_segment_2plus2"] = float("nan")
        checked = validate_snapshot(one_day, "volgograd")
        self.assertFalse(checked.fact_available.any())
        self.assertFalse(checked.is_actual.any())
        matrix = forecast_matrix(checked, "probability")
        self.assertEqual(len(matrix.attrs["actual_cells"]), 0)
        self.assertEqual(len(matrix.attrs["predicted_cells"]), 2)


class DashboardTests(unittest.TestCase):
    def test_switching_regions_preserves_each_snapshot(self) -> None:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric], ["161", "59.6%", "64.6%", "61.5%"])

        app.selectbox(key="forecast_region").select("volgograd").run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric], ["161", "50.9%", "58.4%", "59.6%"])

        app.multiselect(key="forecast_months_volgograd").set_value([pd.Timestamp("2026-08-01")]).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "21")

        app.selectbox(key="forecast_region").select("sverdlovsk").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "161")


if __name__ == "__main__":
    unittest.main()
