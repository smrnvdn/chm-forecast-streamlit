"""Shared maximum-hour forecast dashboard for Russian regions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.forecast_data import REGIONS, available_months, daily_summary, month_label, snapshot_path, validate_snapshot
from src.presentation import forecast_matrix, style_matrix


st.set_page_config(
    page_title="Прогноз часа максимума · регионы",
    page_icon=":material/bolt:",
    layout="wide",
)


@st.cache_data(show_spinner=False, max_entries=8)
def load_forecast_snapshot(region_code: str, file_mtime_ns: int) -> pd.DataFrame:
    """Load one checked regional snapshot; mtime invalidates the file cache."""
    del file_mtime_ns
    hourly = pd.read_csv(snapshot_path(region_code), parse_dates=["date"])
    hourly = validate_snapshot(hourly, region_code)
    hourly["month_start"] = hourly["date"].dt.to_period("M").dt.to_timestamp()
    return hourly


region_code = st.selectbox(
    "Регион",
    options=tuple(REGIONS),
    format_func=lambda code: REGIONS[code].name,
    key="forecast_region",
)
region = REGIONS[region_code]
st.title(f"Прогноз часа максимума для {region.genitive}")
st.caption(f"Модель: {region.model_label}")

try:
    source_mtime_ns = snapshot_path(region_code).stat().st_mtime_ns
    hourly_all = load_forecast_snapshot(region_code, source_mtime_ns)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Не удалось загрузить проверенный прогноз региона: {exc}")
    st.stop()
months = available_months(hourly_all)

with st.expander("О модели и данных", expanded=False, icon=":material/info:"):
    st.markdown(region.about_markdown)

with st.container(border=True):
    selected_months = st.multiselect(
        "Месяцы прогноза",
        options=months,
        default=months,
        format_func=month_label,
        select_all=True,
        placeholder="Выберите один или несколько месяцев",
        wrap=False,
        key=f"forecast_months_{region_code}",
    )
    st.caption("Выберите один или несколько месяцев. Для дня с одним сегментом выдаются два часа, с двумя сегментами — по два часа в каждом.")

if not selected_months:
    st.warning("Выберите хотя бы один месяц, чтобы увидеть прогноз.", icon=":material/calendar_month:")
    st.stop()

hourly = hourly_all.loc[hourly_all["month_start"].isin(selected_months)].copy()
daily = daily_summary(hourly)
matrix = forecast_matrix(hourly, region.score_kind)
verified = daily.loc[daily.fact_available]


def rate_text(values: pd.Series) -> str:
    return f"{values.mean():.1%}" if len(values) else "—"

with st.container(horizontal=True):
    two_segments = verified.loc[verified.segment_count.eq(2)]
    st.metric("Замерных дней", f"{len(daily)}", border=True)
    st.metric("Попадание модели при двух попытках", rate_text(verified["hit_at_2"]), border=True)
    st.metric("2+2 на всех днях", rate_text(verified["hit_segment_2plus2"]), border=True)
    st.metric("2+2 при двух участках", rate_text(two_segments["hit_segment_2plus2"]), border=True)

if len(verified) != len(daily):
    st.caption(f"Фактический час максимума опубликован для {len(verified)} из {len(daily)} выбранных дней; точность рассчитана только по этим дням.")

st.subheader("Итоговый прогноз")
st.caption(region.matrix_caption)
matrix_height = min(820, max(420, 74 + len(matrix) * 35))
st.dataframe(style_matrix(matrix), height=matrix_height, width="stretch")
