"""Trading-calendar primitives used by point-in-time and target logic."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


class TradingCalendar:
    """Immutable wrapper around an explicitly supplied exchange calendar."""

    def __init__(self, dates: Iterable[object]) -> None:
        normalized = (
            pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique().sort_values()
        )
        if normalized.empty:
            raise ValueError("A trading calendar must contain at least one date")
        self._dates = normalized

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._dates

    def add(self, date: object, offset: int) -> pd.Timestamp:
        """Return a date offset by trading sessions, including zero offset."""
        value = pd.Timestamp(date).normalize()
        position = self._dates.searchsorted(value)
        if position >= len(self._dates) or self._dates[position] != value:
            raise ValueError(f"Date is not in the trading calendar: {value.date()}")
        target = position + offset
        if target < 0 or target >= len(self._dates):
            raise IndexError("Trading-day offset is outside the calendar")
        return self._dates[target]

    def contains(self, date: object) -> bool:
        value = pd.Timestamp(date).normalize()
        return bool(value in self._dates)
