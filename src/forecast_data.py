"""Region registry and validation for the shared forecast dashboard."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd


MONTH_NAMES = (
    "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
    "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь",
)


@dataclass(frozen=True)
class RegionConfig:
    name: str
    genitive: str
    snapshot_file: str
    model_label: str
    score_kind: str
    about_markdown: str
    matrix_caption: str


# Adding a region requires only a checked snapshot and one registry entry;
# the selection, metrics and matrix stay shared across regions.
REGIONS = {
    "sverdlovsk": RegionConfig(
        name="Свердловская область",
        genitive="Свердловской области",
        snapshot_file="forecast_2026.csv",
        model_label="CatBoost-79",
        score_kind="probability",
        about_markdown="""Это прогноз часов максимума для Свердловской области. Его формирует модель градиентного бустинга CatBoost, обученная на фактических данных с 2018 года. При расчёте учитываются календарь, часы замеров, плановый РСВ, погодные условия и статистика завершённых периодов.

Для прогноза каждого месяца используются только фактические данные, которые были известны до его начала. Например, прогноз на август строится на фактах до 31 июля. Плановый РСВ, погода и график замерных часов за август учитываются, потому что они доступны заранее. Фактический час максимума августа используется только после построения прогноза — чтобы оценить его точность. Поэтому будущие результаты не влияют на прогноз. Ранее применявшаяся статистика частично оценивалась на тех же периодах, по которым была рассчитана, поэтому могла давать завышенную оценку точности.

**Почему в таблице есть прочерки и пропуски дат.** В таблицу включены только рабочие дни, для которых определяется час максимума; выходные, праздники и иные нерабочие дни не показываются. Прочерк означает, что этот час не входит в утверждённые часы замеров: модель не рассматривает его как возможный час максимума и поэтому не выводит для него вероятность.""",
        matrix_caption="В ячейках — вероятность часа максимума по модели. Синим отмечен прогноз; зелёным — фактический час максимума. Галочка означает попадание прогноза в факт.",
    ),
    "volgograd": RegionConfig(
        name="Волгоградская область",
        genitive="Волгоградской области",
        snapshot_file="forecast_volgograd_2026.csv",
        model_label="CatBoost m0399 · 45 признаков",
        score_kind="probability",
        about_markdown="""Это сохранённый помесячный проверочный прогноз для Волгоградской области модели **CatBoost m0399** (45 признаков). Для каждого месяца модель обучалась на доступном факте потребления только до конца позапрошлого месяца. На целевой месяц ей передавались график замерных часов, погода и цены РСВ. В этом исследовательском прогоне фактические погода и РСВ использованы **как допущение об идеальном прогнозе/плане**; реальные архивные прогнозы не проверялись. Фактический час максимума применён только для оценки результата.

В ячейках показано распределение **оценочных вероятностей часа максимума** по допустимым часам дня; до округления в сумме они дают 100%. Изначально модель предсказывает форму почасовой нагрузки, поэтому её оценки преобразованы в вероятности методом softmax. Параметр преобразования подобран только по ранее доступным фактам 2023–ноября 2024 года; на 2025–2026 годах он не подгонялся. Преобразование не меняет порядок часов и выбранные моделью часы. Это оценка, а не гарантия точной вероятности для каждого отдельного дня. Снимок охватывает январь–август 2026 года; новых суток приложение пока не прогнозирует.

Прочерки — часы вне утверждённых замерных интервалов. Даты без официального часа максимума не показываются. Целевой минимум 64,2% относится к метрике 2+2 на **всех замерных днях полного года**; по имеющимся восьми месяцам его выполнение за весь 2026 год не подтверждено.""",
        matrix_caption="В ячейках — оценочная вероятность часа максимума; за каждый день сумма до округления равна 100%. Синим отмечен прогноз; зелёным — фактический час максимума. Галочка означает попадание прогноза в факт. Часы указаны по рыночной нумерации МСК.",
    ),
}


def month_label(month: pd.Timestamp) -> str:
    """Return a readable Russian month name without a technical period code."""
    return f"{MONTH_NAMES[month.month - 1]} {month.year}"


def available_months(hourly: pd.DataFrame) -> list[pd.Timestamp]:
    """List only months for which the audited forecast snapshot has rows."""
    return sorted(hourly["month_start"].drop_duplicates().tolist())


def daily_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    """Reduce repeated hourly audit fields to one record per working day."""
    return hourly.sort_values("date").drop_duplicates("date").copy()


def snapshot_path(region_code: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / REGIONS[region_code].snapshot_file


def validate_snapshot(hourly: pd.DataFrame, region_code: str) -> pd.DataFrame:
    """Reject incomplete/misaligned snapshots before showing any metric."""
    config = REGIONS[region_code]
    required = {"date", "market_hour", "segment", "segment_count", "is_predicted",
                "is_actual", "hit_at_2", "hit_segment_2plus2"}
    required.add("probability" if config.score_kind == "probability" else "score")
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"Нет обязательных полей прогноза: {', '.join(sorted(missing))}")
    if hourly.empty or hourly.duplicated(["date", "market_hour"]).any():
        raise ValueError("Пустой прогноз или повторяющиеся дата/час")
    if not hourly.market_hour.between(1, 24).all():
        raise ValueError("Неверный рыночный час")
    for column in ("is_predicted", "is_actual"):
        if not hourly[column].isin([True, False, 0, 1]).all():
            raise ValueError(f"Неверный флаг {column}")
        hourly[column] = hourly[column].astype(bool)
    for column in ("hit_at_2", "hit_segment_2plus2"):
        if not hourly[column].dropna().isin([0, 1]).all():
            raise ValueError(f"Неверная метрика {column}")
    group = hourly.groupby("date")
    actual_count = group.is_actual.sum()
    if (actual_count.gt(1).any()
        or group.segment_count.nunique().ne(1).any()
        or group.segment.nunique().ne(group.segment_count.first()).any()
        or group.is_predicted.sum().ne(group.segment_count.first().mul(2)).any()
        or hourly.groupby(["date", "segment"]).is_predicted.sum().ne(2).any()
        or group.hit_at_2.nunique(dropna=False).ne(1).any()
        or group.hit_segment_2plus2.nunique(dropna=False).ne(1).any()):
        raise ValueError("Несогласованные часы, участки или дневные результаты")
    has_fact = actual_count.eq(1)
    metric = group[["hit_at_2", "hit_segment_2plus2"]].first()
    if metric.loc[has_fact].isna().any().any() or metric.loc[~has_fact].notna().any().any():
        raise ValueError("Дневные метрики должны быть заполнены только после публикации факта")
    hourly["fact_available"] = hourly.date.map(has_fact).astype(bool)
    value_col = "probability" if config.score_kind == "probability" else "score"
    if not np.isfinite(hourly[value_col].to_numpy(float)).all():
        raise ValueError("Прогноз содержит пустые/бесконечные оценки")
    if config.score_kind == "probability" and not hourly.probability.between(0, 1).all():
        raise ValueError("Вероятности вне диапазона 0–1")
    if config.score_kind == "probability" and not np.allclose(group.probability.sum(), 1, atol=1e-8):
        raise ValueError("Вероятности допустимых часов должны давать 100% за каждый день")
    # Display rounding may create visual ties, but selected hours must always
    # remain the top two *unrounded* model scores inside each eligible segment.
    for _, segment in hourly.groupby(["date", "segment"]):
        ranked = segment.sort_values([value_col, "market_hour"], ascending=[False, True], kind="stable")
        expected = set(ranked.market_hour.head(2))
        selected = set(segment.loc[segment.is_predicted, "market_hour"])
        if selected != expected:
            raise ValueError("Выбранные часы не совпадают с рейтингом модели")
    for _, day in hourly.loc[hourly.fact_available].groupby("date"):
        actual_hour = int(day.loc[day.is_actual, "market_hour"].iloc[0])
        ranked = day.sort_values([value_col, "market_hour"], ascending=[False, True], kind="stable")
        expected_hit2 = int(actual_hour in set(ranked.market_hour.head(2)))
        expected_hybrid = int(day.loc[day.is_predicted, "market_hour"].eq(actual_hour).any())
        if expected_hit2 != int(day.hit_at_2.iloc[0]) or expected_hybrid != int(day.hit_segment_2plus2.iloc[0]):
            raise ValueError("Дневные метрики не совпадают с выбранными часами")
    return hourly
