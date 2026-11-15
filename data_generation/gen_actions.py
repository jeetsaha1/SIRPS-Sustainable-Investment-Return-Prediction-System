"""Corporate actions generator for SIRPS Member 2.

Generates data/corporate_actions.csv containing:
1. Stock splits for C03 (2022-08-11, 2:1) and C06 (2023-04-13, 2:1).
2. Annual dividend distributions for all 10 companies across 2021-2025 (50 total dividends).

Strictly adheres to central schemas, deterministic random seeding, and referential integrity.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

# Add repository root to path if running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import data_generation.config as config
from common.io import load, save
from common.schemas import validate_dataset_schema

logger = logging.getLogger(__name__)

# Canonical corporate action splits required by specification
REQUIRED_SPLITS = [
    {
        "company_id": "C03",
        "ex_date": "2022-08-11",
        "action_type": "split",
        "ratio_or_amount": 2.0,
        "description": "Stock Split 2:1 (Subdivision of shares from Face Value ₹10 to ₹5)",
    },
    {
        "company_id": "C06",
        "ex_date": "2023-04-13",
        "action_type": "split",
        "ratio_or_amount": 2.0,
        "description": "Stock Split 2:1 (Subdivision of shares from Face Value ₹10 to ₹5)",
    },
]

# Baseline dividend yield parameters by company archetype/profile
DIVIDEND_PROFILES = {
    "C01": {"base_dps": 4.50, "growth": 0.08},
    "C02": {"base_dps": 3.00, "growth": 0.07},
    "C03": {"base_dps": 7.50, "growth": 0.06},  # Post-split will adjust or reflect per-share
    "C04": {"base_dps": 2.50, "growth": 0.10},
    "C05": {"base_dps": 5.00, "growth": 0.09},
    "C06": {"base_dps": 6.50, "growth": 0.12},
    "C07": {"base_dps": 5.50, "growth": 0.05},
    "C08": {"base_dps": 5.00, "growth": 0.08},
    "C09": {"base_dps": 2.20, "growth": 0.07},
    "C10": {"base_dps": 2.80, "growth": 0.10},
}


def generate_corporate_actions(
    calendar_df: Optional[pd.DataFrame] = None,
    company_master_df: Optional[pd.DataFrame] = None,
    seed: int = 97,
) -> pd.DataFrame:
    """Generate corporate actions (splits and dividends) for all 10 companies.

    Args:
        calendar_df: Optional preloaded trading calendar DataFrame.
        company_master_df: Optional preloaded company master DataFrame.
        seed: Random seed for deterministic dividend date selection.

    Returns:
        DataFrame conforming to the 'corporate_actions' schema.
    """
    if calendar_df is None:
        calendar_df = load("trading_calendar")
    if company_master_df is None:
        company_master_df = load("company_master")

    valid_dates = set(calendar_df["date"].astype(str))
    all_trading_dates = sorted(list(valid_dates))

    # Verify that required split dates are valid trading days
    for split in REQUIRED_SPLITS:
        if split["ex_date"] not in valid_dates:
            raise ValueError(
                f"Required split date {split['ex_date']} is not a valid trading day in trading_calendar!"
            )

    rng = np.random.default_rng(seed)
    actions = []

    # 1. Add canonical splits
    for split in REQUIRED_SPLITS:
        actions.append(
            {
                "company_id": split["company_id"],
                "ex_date": split["ex_date"],
                "action_type": split["action_type"],
                "ratio_or_amount": float(split["ratio_or_amount"]),
                "description": split["description"],
                "data_source": config.DATA_SOURCE,
            }
        )

    # 2. Add realistic annual dividends for each company across 5 years (2021-2025)
    # Typically declared following annual general meetings (July - September)
    company_ids = sorted(company_master_df["company_id"].tolist())
    dividend_years = [2021, 2022, 2023, 2024, 2025]

    for comp_id in company_ids:
        prof = DIVIDEND_PROFILES.get(comp_id, {"base_dps": 3.00, "growth": 0.08})
        base_dps = prof["base_dps"]
        growth = prof["growth"]

        for idx, yr in enumerate(dividend_years):
            # Target window: August of each year
            window_dates = [
                d for d in all_trading_dates
                if d.startswith(f"{yr}-08-") or d.startswith(f"{yr}-09-")
            ]
            if not window_dates:
                # Fallback to any Q3 date in that year
                window_dates = [
                    d for d in all_trading_dates
                    if d.startswith(f"{yr}-07-") or d.startswith(f"{yr}-08-") or d.startswith(f"{yr}-09-")
                ]

            # Deterministically choose an ex-date within window
            chosen_date = window_dates[int(rng.integers(0, len(window_dates)))]

            # Compute DPS with mild growth and small random variation
            dps = base_dps * ((1.0 + growth) ** idx) * float(rng.uniform(0.95, 1.05))

            # Adjust DPS post-split so cash value remains consistent
            if comp_id == "C03" and chosen_date >= "2022-08-11":
                dps = dps / 2.0
            if comp_id == "C06" and chosen_date >= "2023-04-13":
                dps = dps / 2.0

            dps = round(dps, 2)

            actions.append(
                {
                    "company_id": comp_id,
                    "ex_date": chosen_date,
                    "action_type": "dividend",
                    "ratio_or_amount": float(dps),
                    "description": f"Annual Dividend FY{yr-1}-{str(yr)[-2:]} of ₹{dps:.2f} per share",
                    "data_source": config.DATA_SOURCE,
                }
            )

    df_actions = pd.DataFrame(actions)

    # Sort deterministically by ex_date and company_id
    df_actions = df_actions.sort_values(by=["ex_date", "company_id"]).reset_index(drop=True)

    # Assign sequential primary key CA001, CA002, ...
    df_actions["action_id"] = [f"CA{i+1:03d}" for i in range(len(df_actions))]

    # Order columns according to schema
    cols = [
        "action_id",
        "company_id",
        "ex_date",
        "action_type",
        "ratio_or_amount",
        "description",
        "data_source",
    ]
    df_actions = df_actions[cols]

    # Validate against central schema
    validate_dataset_schema(df_actions, "corporate_actions")

    return df_actions


def main() -> None:
    """Generate and persist corporate actions dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Starting corporate actions generation...")

    df_actions = generate_corporate_actions()
    output_path = save(df_actions, "corporate_actions")

    logger.info("Successfully generated %d corporate actions.", len(df_actions))
    logger.info("Saved dataset to: %s", output_path)

    # Diagnostic summary
    splits = df_actions[df_actions["action_type"] == "split"]
    dividends = df_actions[df_actions["action_type"] == "dividend"]
    print("\nCorporate Actions Summary:")
    print(f"Total actions: {len(df_actions)}")
    print(f"Total splits:  {len(splits)}")
    for _, s in splits.iterrows():
        print(f"  - {s['action_id']}: {s['company_id']} on {s['ex_date']} (Ratio {s['ratio_or_amount']})")
    print(f"Total dividends: {len(dividends)} (across {dividends['company_id'].nunique()} companies)")
    print(f"Dividend DPS range: ₹{dividends['ratio_or_amount'].min():.2f} - ₹{dividends['ratio_or_amount'].max():.2f}")


if __name__ == "__main__":
    main()
