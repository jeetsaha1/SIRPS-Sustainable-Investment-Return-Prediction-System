"""Small synthetic feature/target builder for the initialization pilot."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def build_pilot_dataset(prices: pd.DataFrame, *, horizon: int = 63, step: int = 5) -> pd.DataFrame:
    """Build a causal, non-ESG pilot model dataset from synthetic prices."""
    required = {"company_id", "date", "adj_close", "volume"}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"Pilot prices are missing columns: {', '.join(missing)}")

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    benchmark_id = "NIFTY50_TRI_SYNTHETIC"
    benchmark = frame[frame["company_id"] == benchmark_id].set_index("date")["adj_close"]
    rows: list[dict[str, object]] = []
    for company_id, company in frame[frame["company_id"] != benchmark_id].groupby("company_id"):
        company = company.sort_values("date").reset_index(drop=True)
        for index in range(21, len(company) - horizon, step):
            prediction = company.iloc[index]
            entry = company.iloc[index + 1]
            exit_row = company.iloc[index + horizon]
            entry_benchmark = benchmark.loc[entry["date"]]
            exit_benchmark = benchmark.loc[exit_row["date"]]
            features = {
                "return_5d": float(
                    company.iloc[index]["adj_close"] / company.iloc[index - 5]["adj_close"] - 1
                ),
                "return_21d": float(
                    company.iloc[index]["adj_close"] / company.iloc[index - 21]["adj_close"] - 1
                ),
                "price_to_ma20": float(
                    company.iloc[index]["adj_close"]
                    / company.iloc[index - 19 : index + 1]["adj_close"].mean()
                ),
                "log_volume": float(np.log1p(company.iloc[index]["volume"])),
            }
            y_abs = float(np.log(exit_row["adj_close"] / entry["adj_close"]))
            y_excess = y_abs - float(np.log(exit_benchmark / entry_benchmark))
            rows.append(
                {
                    "company_id": company_id,
                    "date": prediction["date"].date().isoformat(),
                    "entry_date": entry["date"].date().isoformat(),
                    "target_end_date": exit_row["date"].date().isoformat(),
                    "features": json.dumps(features, sort_keys=True),
                    "y_excess": y_excess,
                    "y_abs": y_abs,
                    "regime": "Neutral",
                    "has_esg": False,
                    "esg_available_date": None,
                }
            )
    return pd.DataFrame(rows)
