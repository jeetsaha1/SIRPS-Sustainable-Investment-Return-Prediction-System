"""Generate one deterministic mock CSV for each core SIRPS schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sirps.common.io import read_table, write_table
from sirps.common.synthetic import make_pilot_prices
from sirps.features.pilot import build_pilot_dataset
from sirps.latent import generate_latent_profiles

from .config import GenerationConfig


def _fundamentals(company_ids: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for company_id in company_ids:
        for item, value in (("revenue", 1000.0), ("net_margin", 0.12)):
            rows.append(
                {
                    "company_id": company_id,
                    "period_end": "2019-12-31",
                    "period_type": "annual",
                    "item": item,
                    "value": value,
                    "basis": "consolidated",
                    "filed_timestamp": "2020-02-15T10:00:00",
                    "available_date": "2020-02-18",
                }
            )
    return pd.DataFrame(rows)


def _esg(company_ids: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for company_id in company_ids:
        for number in range(1, 9):
            rows.append(
                {
                    "company_id": company_id,
                    "fiscal_year": 2019,
                    "indicator_id": f"ESG{number:02d}",
                    "value": float(10 + number),
                    "unit_raw": "synthetic_unit",
                    "unit_std": "percentage",
                    "status": "value",
                    "source_doc": "synthetic_mock_brsr",
                    "page": 1,
                    "filed_timestamp": "2020-02-15T10:00:00",
                    "available_date": "2020-02-18",
                    "restated_flag": False,
                    "entered_by": "M1",
                    "verified_by": "M1",
                }
            )
    return pd.DataFrame(rows)


def _regimes(prices: pd.DataFrame) -> pd.DataFrame:
    benchmark = prices[prices["company_id"] == "NIFTY50_TRI_SYNTHETIC"].copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark.sort_values("date")
    close = benchmark["adj_close"]
    ma50 = close.rolling(50, min_periods=1).mean()
    ma200 = close.rolling(200, min_periods=1).mean()
    drawdown = close / close.cummax() - 1
    raw = np.where(
        drawdown <= -0.15, "Bear", np.where((close > ma200) & (ma50 > ma200), "Bull", "Neutral")
    )
    return pd.DataFrame(
        {
            "date": benchmark["date"].dt.date.astype(str),
            "close": close,
            "ma50": ma50,
            "ma200": ma200,
            "drawdown_252": drawdown,
            "raw_regime": raw,
            "confirmed_regime": raw,
        }
    )


def generate_mock_data(
    output_dir: str | Path, config: GenerationConfig | None = None
) -> dict[str, Path]:
    """Write mock CSVs for every core table plus latent profiles."""
    settings = config or GenerationConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prices = make_pilot_prices(seed=settings.seed, periods=settings.periods)
    dataset = build_pilot_dataset(
        prices, horizon=settings.forecast_horizon, step=settings.sample_step
    )
    from sirps.modeling.pilot import make_pilot_predictions

    company_master = read_table(
        Path(__file__).parents[1] / "data" / "processed" / "company_master.csv"
    )
    tables = {
        "company_master": company_master,
        "prices_daily": prices,
        "fundamentals_pit": _fundamentals(settings.company_ids),
        "esg_facts": _esg(settings.company_ids),
        "regime_daily": _regimes(prices),
        "model_dataset": dataset,
        "predictions": make_pilot_predictions(dataset),
        "latent_profiles": generate_latent_profiles(settings.company_ids, seed=settings.seed),
    }
    paths = {}
    for schema, frame in tables.items():
        paths[schema] = write_table(frame, output / f"{schema}.csv", schema=schema)
    return paths
