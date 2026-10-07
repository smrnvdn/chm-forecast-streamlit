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
        about_markdown="""**CatBoost-79** оценивает вероятность часа максимума для Свердловской области. Модель учитывает календарь и сезонность, график замерных часов, план РСВ, погоду и статистику завершённых периодов. CatBoost позволяет учитывать совместное влияние этих факторов и нелинейные зависимости между ними.

Для каждого прогнозируемого месяца используется история по месяц M−2 включительно: например, для сентября 2026 — по июль. Фактический ЧМ прогнозируемого месяца не участвует в расчёте.

Модель ранжирует допустимые часы каждого рабочего дня. При одном замерном интервале выбираются два часа; при двух — по два часа в каждом. Вероятности помогают оценить, на каких часах сосредоточен прогноз.

Показываются рабочие замерные дни. Прочерк означает час вне утверждённых замерных интервалов. Рыночный час h обозначает интервал [h−1, h) по МСК. Фактические ЧМ и показатели точности появляются после публикации результатов месяца.""",
        matrix_caption="В ячейках — вероятность часа максимума по модели. Синим отмечен прогноз; зелёным — фактический час максимума. Галочка означает попадание прогноза в факт.",
    ),
    "volgograd": RegionConfig(
        name="Волгоградская область",
        genitive="Волгоградской области",
        snapshot_file="forecast_volgograd_2026.csv",
        model_label="CatBoost m0399 · 45 признаков",
        score_kind="probability",
        about_markdown="""**CatBoost m0399** использует 45 признаков для прогноза формы почасовой нагрузки Волгоградской области. Модель учитывает календарь и сезонность, структуру замерных часов, план РСВ и погодные условия. Это позволяет сравнивать часы с учётом условий конкретного дня и выбирать наиболее вероятное время максимума.

История для прогноза месяца M заканчивается месяцем M−2. Для сентября 2026 используется история по июль включительно. Фактический ЧМ прогнозируемого месяца не участвует в расчёте.

Оценки нагрузки преобразуются в оценочные вероятности по допустимым часам дня; их сумма до округления равна 100%. Порядок часов при этом сохраняется. При одном замерном интервале выбираются два часа; при двух — по два часа в каждом.

Показываются рабочие замерные дни. Прочерк означает час вне утверждённых замерных интервалов. Рыночный час h обозначает интервал [h−1, h) по МСК. Фактические ЧМ и показатели точности появляются после публикации результатов месяца.""",
        matrix_caption="В ячейках — оценочная вероятность часа максимума; за каждый день сумма до округления равна 100%. Синим отмечен прогноз; зелёным — фактический час максимума. Галочка означает попадание прогноза в факт. Часы указаны по рыночной нумерации МСК.",
    ),
    "perm": RegionConfig(
        name="Пермский край",
        genitive="Пермского края",
        snapshot_file="forecast_perm_2026.csv",
        model_label="CatBoost E25 · 18 признаков",
        score_kind="probability",
        about_markdown="""**CatBoost E25** использует 18 признаков для прогноза формы почасовой нагрузки Пермского края. Модель учитывает календарь и сезонность, структуру замерных часов, температуру, влажность, ветер и исторический профиль потребления. Эти факторы помогают учитывать как привычный суточный ритм, так и погодные условия конкретного дня.

История потребления и зависимые от неё признаки для месяца M заканчиваются месяцем M−2. Для сентября 2026 используется история по июль включительно.

Оценки нагрузки преобразуются в оценочные вероятности по допустимым часам дня; их сумма до округления равна 100%. Порядок часов при этом сохраняется. При одном замерном интервале выбираются два часа; при двух — по два часа в каждом.

Показываются рабочие замерные дни. Прочерк означает час вне утверждённых замерных интервалов. Рыночный час h обозначает интервал [h−1, h) по МСК. Фактические ЧМ и показатели точности появляются после публикации результатов месяца.""",
        matrix_caption="В ячейках — оценочная вероятность часа максимума; за каждый день сумма до округления равна 100%. Синим отмечен прогноз; зелёным — фактический час максимума. Галочка означает попадание прогноза в факт. Часы указаны по рыночной нумерации МСК.",
    ),
    "vologda": RegionConfig(
        name="Вологодская область",
        genitive="Вологодской области",
        snapshot_file="forecast_vologda_2026.csv",
        model_label="CatBoost p000 · 420 признаков · без ветра",
        score_kind="probability",
        about_markdown="""**CatBoost p000** использует 420 признаков для прогноза формы почасовой нагрузки Вологодской области. Модель учитывает календарь и сезонность, замерные интервалы, почасовую цену РСВ, температуру, влажность, солнечные признаки и исторический профиль потребления. Признаки ветра исключены.

Для каждого месяца модель обучается на предшествующих 36 месяцах с историей по M−2 включительно. Все признаки, зависящие от потребления, также рассчитываются с собственной отсечкой M−2. Для сентября 2026 используется факт только по июль.

Оценки нагрузки преобразуются в оценочные вероятности по всем допустимым часам дня; сумма до округления равна 100%. Преобразование настроено по прошлым периодам, без использования 2025–2026 годов, и сохраняет порядок часов модели. При одном замерном интервале выбираются два часа, при двух — по два в каждом. Проценты являются оценками, а не гарантией попадания.

Это фиксированный демонстрационный расчёт. Погода представлена наблюдениями станции Вологда (аэропорт), используемыми как ретроспективная замена прогноза погоды. Сентябрь рассчитан после окончания месяца без использования его фактического потребления и ЧМ. Цена РСВ в обучении и расчёте берётся по одному правилу: прежний региональный источник до ноября 2023, затем ежедневная цена покупки Вологодской области.

Показываются рабочие замерные дни. Прочерк означает час вне замерных интервалов. Рыночный час h обозначает интервал [h−1, h) по МСК. Для сентября фактические ЧМ не загружены, поэтому точность не рассчитывается.""",
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
                "is_actual", "hit_at_2", "hit_segment_2plus2", "load_history_last_month", "model_version"}
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
    expected_history_month = (hourly["date"].dt.to_period("M") - 2).astype(str)
    # The Sverdlovsk archive was rebuilt under this contract and keeps a
    # row-level audit field. Older regional archives may not expose that
    # historical metadata for completed months, so do not reject them here.
    if region_code in {"sverdlovsk", "vologda"} and not hourly["load_history_last_month"].astype(str).eq(expected_history_month).all():
        raise ValueError("Снимок нарушает правило M-2 для факта и зависимой истории")
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
