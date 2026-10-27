"""Business copy, probability order and actual Kemerovo release in the shared UI."""
import unittest
from pathlib import Path
import numpy as np,pandas as pd
from streamlit.testing.v1 import AppTest
from src.forecast_data import REGIONS,validate_snapshot,snapshot_path,daily_summary

class KemerovoReleaseTests(unittest.TestCase):
    def test_release_probabilities_keep_raw_hour_order_and_metrics(self):
        frame=validate_snapshot(pd.read_csv(snapshot_path('kemerovo'),parse_dates=['date']),'kemerovo')
        self.assertTrue(np.allclose(frame.groupby('date').probability.sum(),1))
        for _,day in frame.groupby('date'):
            score=day.sort_values(['score','market_hour'],ascending=[False,True]).market_hour.tolist()
            probability=day.sort_values(['probability','market_hour'],ascending=[False,True]).market_hour.tolist()
            self.assertEqual(score,probability)
        daily=daily_summary(frame);latest=daily[daily.date.dt.year.eq(2026)]
        self.assertEqual((len(latest),int(latest.hit_at_2.sum()),int(latest.hit_segment_2plus2.sum())),(183,149,163))
        self.assertTrue(frame.load_history_last_month.eq((frame.date.dt.to_period('M')-2).astype(str)).all())

    def test_business_summary_only(self):
        text=REGIONS['kemerovo'].about_markdown.lower()
        for forbidden in ['proxy','ретроспектив','утеч','фактической погод','наблюдённ','эксперимент','калибров','неточност']:
            self.assertNotIn(forbidden,text)
        self.assertIn('исторический профиль',text);self.assertIn('100%',text)

    def test_month_filters_and_single_segment_in_real_app(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'streamlit_app.py'),default_timeout=20)
        app.query_params['region']='kemerovo';app.run();self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key='forecast_region').value,'kemerovo')
        app.multiselect(key='forecast_months_kemerovo').set_value([pd.Timestamp(f'2026-{m:02}-01') for m in range(1,10)]).run()
        self.assertFalse(app.exception)
        self.assertEqual([v.value for v in app.metric[:3]],['183','149/183 (81.4%)','163/183 (89.1%)'])
        app.multiselect(key='forecast_months_kemerovo').set_value([pd.Timestamp('2026-07-01')]).run()
        self.assertFalse(app.exception)
        self.assertEqual([v.value for v in app.metric],['23','18/23 (78.3%)','18/23 (78.3%)','—'])
        self.assertEqual(len(app.dataframe[0].value),23)

if __name__=='__main__':unittest.main()
