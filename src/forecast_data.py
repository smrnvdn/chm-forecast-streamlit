"""Load the audited CatBoost-79 dashboard snapshot."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MONTH_NAMES = (
    "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
    "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь",
)


def month_label(month: pd.Timestamp) -> str:
    """Return a readable Russian month name without a technical period code."""
    return MONTH_NAMES[month.month - 1]


def available_months(hourly: pd.DataFrame) -> list[pd.Timestamp]:
    """List only months for which the audited forecast snapshot has rows."""
    return sorted(hourly["month_start"].drop_duplicates().tolist())


def daily_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    """Reduce repeated hourly audit fields to one record per working day."""
    return hourly.sort_values("date").drop_duplicates("date").copy()


def snapshot_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "forecast_2026.csv"
