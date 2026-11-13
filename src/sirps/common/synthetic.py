"""Deterministic synthetic data for the two-company initialization pilot."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_pilot_prices(*, seed: int = 2026, periods: int = 80) -> pd.DataFrame:
    """Return clearly synthetic daily prices for two companies and a benchmark.

    The fixture is for pipeline contracts and tests only; it is not NSE data.
    """
    if periods < 65:
        raise ValueError("The pilot needs at least 65 trading observations")

    dates = pd.bdate_range("2020-01-02", periods=periods)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for company_id, base_price in (("SYN001", 100.0), ("SYN002", 75.0)):
        returns = rng.normal(0.0004, 0.012, size=periods)
        close = base_price * np.exp(np.cumsum(returns))
        for date, price in zip(dates, close, strict=True):
            rows.append(
                {
                    "company_id": company_id,
                    "date": date.date().isoformat(),
                    "open": round(float(price * 0.995), 6),
                    "high": round(float(price * 1.01), 6),
                    "low": round(float(price * 0.99), 6),
                    "close": round(float(price), 6),
                    "adj_close": round(float(price), 6),
                    "volume": 100000,
                    "source": "synthetic_pilot",
                    "ingest_date": "2026-01-01T00:00:00",
                }
            )

    benchmark_returns = rng.normal(0.0003, 0.008, size=periods)
    benchmark = 10000.0 * np.exp(np.cumsum(benchmark_returns))
    for date, price in zip(dates, benchmark, strict=True):
        rows.append(
            {
                "company_id": "NIFTY50_TRI_SYNTHETIC",
                "date": date.date().isoformat(),
                "open": round(float(price), 6),
                "high": round(float(price), 6),
                "low": round(float(price), 6),
                "close": round(float(price), 6),
                "adj_close": round(float(price), 6),
                "volume": 0,
                "source": "synthetic_pilot",
                "ingest_date": "2026-01-01T00:00:00",
            }
        )
    return pd.DataFrame(rows)
