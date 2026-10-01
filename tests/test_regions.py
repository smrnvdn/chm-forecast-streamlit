"""Checks for regional snapshots, monthly filters and direct regional URLs."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.forecast_data import REGIONS, daily_summary, snapshot_path, validate_snapshot
from src.presentation import forecast_matrix


class RegionSnapshotTests(unittest.TestCase):
    def test_all_snapshots_have_one_target_and_valid_attempts(self) -> None:
        for code in REGIONS:
            with self.subTest(region=code):
                hourly = validate_snapshot(pd.read_csv(snapshot_path(code), parse_dates=["date"]), code)
                daily = daily_summary(hourly)
                self.assertEqual(len(daily), 183)
                self.assertEqual(len(hourly), 2232)
                matrix = forecast_matrix(hourly, REGIONS[code].score_kind)
                self.assertEqual(matrix.shape, (183, 14))
                september = hourly.loc[hourly.date.dt.month.eq(9)]
                self.assertEqual(september.date.nunique(), 22)
                self.assertFalse(september.fact_available.any())
                self.assertTrue(september[["hit_at_2", "hit_segment_2plus2"]].isna().all().all())
                self.assertEqual(set(september.market_hour), set(range(8, 16)) | set(range(18, 22)))
                self.assertTrue(september.groupby("date").is_predicted.sum().eq(4).all())
                self.assertEqual(len(forecast_matrix(september, "probability").attrs["actual_cells"]), 0)

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

    def test_perm_frozen_metrics_and_calibration_preserves_picks(self) -> None:
        hourly = validate_snapshot(pd.read_csv(snapshot_path("perm"), parse_dates=["date"]), "perm")
        daily = daily_summary(hourly)
        self.assertEqual((int(daily.hit_at_2.sum()), int(daily.hit_segment_2plus2.sum())), (99, 113))
        self.assertEqual(sorted(hourly.month.unique()), [f"2026-{m:02}" for m in range(1, 10)])
        self.assertTrue((hourly.load_history_last_month == (hourly.date.dt.to_period("M") - 2).astype(str)).all())
        for (_, _), part in hourly.groupby(["date", "segment"]):
            raw = set(part.sort_values(["score", "market_hour"], ascending=[False, True]).market_hour.head(2))
            self.assertEqual(raw, set(part.loc[part.is_predicted, "market_hour"]))

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
        self.assertEqual(len(matrix.attrs["predicted_cells"]), 4)


class DashboardTests(unittest.TestCase):
    def make_app(self) -> AppTest:
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=20)

    def test_direct_region_links_and_invalid_region(self) -> None:
        for code in REGIONS:
            with self.subTest(region=code):
                app = self.make_app()
                app.query_params["region"] = code
                app.query_params["other"] = "preserved"
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(app.selectbox(key="forecast_region").value, code)
                self.assertIn(REGIONS[code].genitive, app.title[0].value)
                self.assertEqual(app.query_params["region"], [code])
                self.assertEqual(app.query_params["other"], ["preserved"])
                self.assertEqual([m.value for m in app.metric], ["22", "—", "—", "—"])
                self.assertEqual(app.multiselect(key=f"forecast_months_{code}").value, [pd.Timestamp("2026-09-01")])
                self.assertTrue(any("неизвестными фактическими" in notice.value for notice in app.info))
        app = self.make_app()
        app.query_params["region"] = "unknown"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="forecast_region").value, "sverdlovsk")
        self.assertEqual(app.query_params["region"], ["sverdlovsk"])

    def test_url_change_in_existing_session(self) -> None:
        app = self.make_app().run()
        app.query_params["region"] = "perm"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="forecast_region").value, "perm")
        self.assertEqual([m.value for m in app.metric[:3]], ["22", "—", "—"])

    def test_switching_regions_preserves_each_snapshot(self) -> None:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=20).run()
        self.assertFalse(app.exception)
        past = [pd.Timestamp(f"2026-{m:02}-01") for m in range(1, 9)]
        app.multiselect(key="forecast_months_sverdlovsk").set_value(past).run()
        self.assertEqual([m.value for m in app.metric], ["161", "59.6%", "64.6%", "61.5%"])

        app.selectbox(key="forecast_region").select("volgograd").run()
        app.multiselect(key="forecast_months_volgograd").set_value(past).run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric], ["161", "50.9%", "58.4%", "59.6%"])
        self.assertEqual(app.query_params["region"], ["volgograd"])

        app.multiselect(key="forecast_months_volgograd").set_value([pd.Timestamp("2026-08-01")]).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "21")

        app.selectbox(key="forecast_region").select("sverdlovsk").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "22")
        self.assertEqual(app.query_params["region"], ["sverdlovsk"])

        app.selectbox(key="forecast_region").select("perm").run()
        app.multiselect(key="forecast_months_perm").set_value(past).run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric[:3]], ["161", "61.5%", "70.2%"])
        self.assertEqual(app.query_params["region"], ["perm"])
        app.multiselect(key="forecast_months_perm").set_value([pd.Timestamp("2026-08-01")]).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "21")

    def test_mixed_months_do_not_count_unpublished_facts(self) -> None:
        app = self.make_app().run()
        app.multiselect(key="forecast_months_sverdlovsk").set_value([pd.Timestamp(f"2026-{m:02}-01") for m in range(1, 10)]).run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric[:3]], ["183", "59.6%", "64.6%"])
        self.assertTrue(any("161 из 183" in caption.value for caption in app.caption))


if __name__ == "__main__":
    unittest.main()
