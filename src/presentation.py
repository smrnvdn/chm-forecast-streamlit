"""Presentation helpers for the shared regional forecast matrix."""

from __future__ import annotations

import pandas as pd


def forecast_matrix(hourly: pd.DataFrame, score_kind: str) -> pd.DataFrame:
    """Build a date × market-hour matrix and retain cell status for styling."""
    display = hourly.copy()
    if score_kind == "probability":
        display["cell"] = display.probability.map(lambda value: f"{value:.0%}")
    elif score_kind == "load_shape_score":
        # The Volgograd model predicts a relative hourly load shape, not an
        # event probability. Keep its signed percentage-point scale visible.
        display["cell"] = display.score.map(lambda value: f"{value:+.1f}%")
    else:
        raise ValueError(f"Unknown score kind: {score_kind}")
    display.loc[display.is_predicted & display.is_actual, "cell"] = (
        "✓ " + display.loc[display.is_predicted & display.is_actual, "cell"]
    )
    matrix = display.pivot(index="date", columns="market_hour", values="cell")
    # Preserve the original 08:00–21:00 display while allowing future
    # regions whose approved candidate hours extend beyond that range.
    first_hour = min(8, int(display.market_hour.min()))
    last_hour = max(21, int(display.market_hour.max()))
    matrix = matrix.reindex(columns=range(first_hour, last_hour + 1), fill_value="—").fillna("—")
    matrix.columns = [f"{hour:02d}:00" for hour in matrix.columns]
    weekdays = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")
    labels = {
        timestamp: f"{timestamp:%d.%m} · {weekdays[timestamp.weekday()]}"
        for timestamp in pd.to_datetime(matrix.index)
    }
    matrix.index = matrix.index.map(labels)
    matrix.index.name = "Дата"
    matrix.attrs["actual_cells"] = {
        (labels[pd.Timestamp(row.date)], f"{int(row.market_hour):02d}:00")
        for row in display.loc[display["is_actual"]].itertuples(index=False)
    }
    matrix.attrs["predicted_cells"] = {
        (labels[pd.Timestamp(row.date)], f"{int(row.market_hour):02d}:00")
        for row in display.loc[display["is_predicted"]].itertuples(index=False)
    }
    return matrix


def style_matrix(matrix: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Highlight the operational forecast and the observed actual maximum hour."""
    actual_cells = matrix.attrs.get("actual_cells", set())
    predicted_cells = matrix.attrs.get("predicted_cells", set())

    def cell_styles(frame: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("color: #CBD5E1", index=frame.index, columns=frame.columns)
        for date, hour in predicted_cells:
            # Streamlit's dataframe renderer can discard the alpha channel of
            # rgba() styles. This is the pre-composed equivalent of a 24%
            # blue veil over the dashboard's dark table background.
            styles.loc[date, hour] = "background-color: #2B5C9A; color: #E8F1FF; font-weight: 600"
        for date, hour in actual_cells:
            styles.loc[date, hour] = "background-color: #166534; color: #F0FDF4; font-weight: 700"
        styles = styles.mask(frame.eq("—"), "color: #64748B")
        return styles

    # Sets in ``DataFrame.attrs`` are useful while preparing styles but are not
    # serialisable by Streamlit's dataframe transport.
    presentation = matrix.copy()
    presentation.attrs = {}
    return presentation.style.apply(cell_styles, axis=None)
