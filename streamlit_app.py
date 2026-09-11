"""Maximum-hour forecast dashboard for Sverdlovsk Oblast."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.forecast_data import available_months, daily_summary, month_label, snapshot_path
from src.presentation import forecast_matrix, style_matrix


st.set_page_config(
    page_title="Прогноз ЧМ · Свердловская область",
    page_icon=":material/bolt:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_forecast_snapshot() -> pd.DataFrame:
    """Load the checked monthly walk-forward output prepared for this dashboard."""
    hourly = pd.read_csv(snapshot_path(), parse_dates=["date"])
    hourly["month_start"] = hourly["date"].dt.to_period("M").dt.to_timestamp()
    for column in ("is_predicted", "is_actual"):
        hourly[column] = hourly[column].astype(bool)
    return hourly


hourly_all = load_forecast_snapshot()
months = available_months(hourly_all)

st.title("Прогноз часа максимума для Свердловской области")

with st.expander("О модели и данных", expanded=False, icon=":material/info:"):
    st.markdown(
        """Это прогноз часов максимума для Свердловской области. Его формирует модель градиентного бустинга CatBoost, обученная на фактических данных с 2018 года. При расчёте учитываются календарь, часы замеров, плановый РСВ, погодные условия и статистика завершённых периодов.

Каждый месяц оценивается честно: например, для прогноза на август модель обучается только на фактах до 31 июля. Данные августа по плановому РСВ, погоде и графику замерных часов используются для расчёта, а фактический час максимума становится известен модели только после построения прогноза — для проверки результата. Такой порядок исключает использование будущего результата при обучении. Ранее применявшаяся статистика частично оценивалась на тех же периодах, по которым была рассчитана, поэтому могла давать завышенную оценку точности.

**Почему в таблице есть прочерки и пропуски дат.** В таблицу включены только рабочие дни, для которых определяется час максимума; выходные, праздники и иные нерабочие дни не показываются. Прочерк означает, что этот час не входит в утверждённые часы замеров (ЗЧ): модель не рассматривает его как возможный час максимума и поэтому не выводит для него вероятность."""
    )

with st.container(border=True):
    selected_months = st.multiselect(
        "Месяцы прогноза для 2026 года",
        options=months,
        default=months,
        format_func=month_label,
        select_all=True,
        placeholder="Выберите один или несколько месяцев",
        wrap=False,
        key="forecast_months",
    )
    st.caption("Выберите один или несколько месяцев. Для дня с одним сегментом выдаются два часа, с двумя сегментами — по два часа в каждом.")

if not selected_months:
    st.warning("Выберите хотя бы один месяц, чтобы увидеть прогноз.", icon=":material/calendar_month:")
    st.stop()

hourly = hourly_all.loc[hourly_all["month_start"].isin(selected_months)].copy()
daily = daily_summary(hourly)
matrix = forecast_matrix(hourly)

with st.container(horizontal=True):
    st.metric("Рабочих дней", f"{len(daily)}", border=True)
    st.metric("Попадание модели при двух попытках", f"{daily['hit_at_2'].mean():.1%}", border=True)
    st.metric("Попадание при двух сегментах (2+2)", f"{daily['hit_segment_2plus2'].mean():.1%}", border=True)

st.subheader("Итоговый прогноз")
st.caption("В ячейках — вероятность часа максимума по модели. Синим отмечен прогноз; зелёным — фактический час максимума. Галочка означает попадание прогноза в факт.")
matrix_height = min(820, max(420, 74 + len(matrix) * 35))
st.dataframe(style_matrix(matrix), height=matrix_height, width="stretch")
