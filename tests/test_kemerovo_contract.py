"""Raw ranker scores, M-2 and unlabelled forecasts; no permanent region registration."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from streamlit.testing.v1 import AppTest
from src.forecast_data import REGIONS,RegionConfig,validate_snapshot
from src.presentation import forecast_matrix

CONFIG=RegionConfig(name='Кемеровская область',genitive='Кемеровской области',snapshot_file='audit_fixture.csv',
 model_label='CatBoostRanker QueryRMSE · 22 признака',score_kind='ranking_score',about_markdown='Проверка контракта ранкера.',
 matrix_caption='В ячейках — оценка ранжирования часов. Синим отмечены выбранные часы.')

def fixture(date='2026-09-02',labelled=True,one_segment=False):
    hours=list(range(5,18)) if one_segment else [5,6,7,8,13,14,15,16,17]
    scores=np.linspace(-2,3,len(hours));segments=[1 if one_segment or h<=8 else 2 for h in hours]
    selected={16,17} if one_segment else {7,8,16,17}
    return pd.DataFrame(dict(date=pd.Timestamp(date),market_hour=hours,segment=segments,segment_count=1 if one_segment else 2,
      score=scores,is_predicted=[h in selected for h in hours],is_actual=[labelled and h==17 for h in hours],
      hit_at_2=1 if labelled else np.nan,hit_segment_2plus2=1 if labelled else np.nan,
      load_history_last_month=str(pd.Period(date,freq='M')-2),model_version='Kemerovo-QueryRMSE22-M-2'))

class KemerovoContractTests(unittest.TestCase):
    def setUp(self):
        self.regions=patch.dict(REGIONS,{'kemerovo':CONFIG});self.regions.start();self.addCleanup(self.regions.stop)

    def test_raw_scores_render_without_probability_or_percent(self):
        checked=validate_snapshot(fixture(),'kemerovo');matrix=forecast_matrix(checked,'ranking_score')
        self.assertTrue(any('-2.000' in value for value in matrix.to_numpy().ravel()))
        self.assertFalse(any('%' in value for value in matrix.to_numpy().ravel()))
        self.assertEqual(len(matrix.attrs['predicted_cells']),4)

    def test_kemerovo_rejects_M_minus_1_history(self):
        bad=fixture();bad['load_history_last_month']='2026-08'
        with self.assertRaisesRegex(ValueError,'M-2'):validate_snapshot(bad,'kemerovo')

    def test_one_segment_has_only_two_attempts(self):
        checked=validate_snapshot(fixture('2026-07-01',one_segment=True),'kemerovo')
        self.assertEqual(int(checked.is_predicted.sum()),2)

    def run_fixture_app(self,unlabelled):
        frame=fixture('2026-10-01' if unlabelled else '2026-09-02',labelled=not unlabelled)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'fixture.csv';frame.to_csv(source,index=False)
            with patch('src.forecast_data.snapshot_path',return_value=source):
                app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'streamlit_app.py'),default_timeout=20)
                app.query_params['region']='kemerovo';app.run()
                self.assertFalse(app.exception);self.assertFalse(app.error)
                self.assertTrue(any('оценка ранжирования' in c.value for c in app.caption))
                self.assertFalse(any('оценочная вероятность' in c.value for c in app.caption))
                self.assertFalse(any('%' in str(v) for v in app.dataframe[0].value.to_numpy().ravel()))
                self.assertEqual(app.metric[1].value,'—' if unlabelled else '1/1 (100.0%)')

    def test_app_with_raw_ranker_scores(self):self.run_fixture_app(False)
    def test_app_without_actual_labels_has_no_false_accuracy(self):self.run_fixture_app(True)

if __name__=='__main__':unittest.main()
