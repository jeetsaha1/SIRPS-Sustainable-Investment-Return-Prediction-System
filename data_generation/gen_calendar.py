"""SIRPS Trading Calendar Generator (Member 2 — Market & Trading Data).

Generates the realistic, reproducible synthetic National Stock Exchange (NSE)
trading calendar for the 2020–2025 simulation window (approx. 1,500 trading days).

Excludes:
- Weekends (Saturdays and Sundays).
- Fixed Indian national holidays (Republic Day, Independence Day, Gandhi Jayanti, Christmas)
  when they fall on weekdays.
- Seeded random market holidays (Diwali, Holi, Good Friday, Eid, etc.) calibrated to exactly
  4 market holidays per calendar year.

Data Contract:
- Target File: data/trading_calendar.csv
- Primary Key: ['date']
- Columns: ['date', 'trading_day_index', 'year', 'is_month_end', 'data_source']
- Provenance: 'SYNTHETIC_DEMO'
- Row Count: Exactly 1,526 trading sessions across 2020–2025.
"""

import sys
from pathlib import Path
from typing import Set, Tuple, List, Optional
import numpy as np
import pandas as pd

# Add repository root to path if running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_generation.config as config
from common.schemas import SCHEMAS, validate_dataset_schema
from common.io import save

# Fixed Indian market national holidays: (month, day)
FIXED_HOLIDAYS: Set[Tuple[int, int]] = {
    (1, 26),   # Republic Day
    (8, 15),   # Independence Day
    (10, 2),   # Mahatma Gandhi Jayanti
    (12, 25),  # Christmas
}

RANDOM_HOLIDAYS_PER_YEAR: int = 4
EXPECTED_TOTAL_SESSIONS: int = 1526


def is_fixed_holiday(d: pd.Timestamp) -> bool:
    """Checks whether a given date is a statutory fixed national market holiday."""
    return (d.month, d.day) in FIXED_HOLIDAYS


def generate_trading_calendar(seed: int = config.SEED_STREAMS["calendar"]) -> pd.DataFrame:
    """Generates the official 2020-2025 synthetic trading calendar for SIRPS.
    
    Args:
        seed: Random seed for deterministic selection of annual market holidays.
        
    Returns:
        pd.DataFrame: Formatted and validated trading calendar adhering to
                      the SCHEMAS['trading_calendar'] contract.
    """
    rng = np.random.default_rng(seed)
    all_trading_dates: List[pd.Timestamp] = []
    
    for y in range(config.START_YEAR, config.END_YEAR + 1):
        # 1. Generate all business days (Monday-Friday) for the calendar year
        b_days = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="B")
        
        # 2. Filter out fixed statutory holidays falling on weekdays
        candidates = [d for d in b_days if not is_fixed_holiday(d)]
        
        # 3. Deterministically choose exactly 4 random market holidays
        chosen_indices = set(rng.choice(len(candidates), size=RANDOM_HOLIDAYS_PER_YEAR, replace=False))
        
        # 4. Retain non-holiday trading sessions
        year_trading_days = [d for idx, d in enumerate(candidates) if idx not in chosen_indices]
        all_trading_dates.extend(year_trading_days)
        
    # Sort chronologically to guarantee strict ascending order
    all_trading_dates.sort()
    
    if len(all_trading_dates) != EXPECTED_TOTAL_SESSIONS:
        raise ValueError(
            f"Calendar session count mismatch: expected {EXPECTED_TOTAL_SESSIONS}, "
            f"got {len(all_trading_dates)}."
        )
        
    # Assemble calendar dataframe records
    records = []
    for d in all_trading_dates:
        records.append({
            "date": d.strftime("%Y-%m-%d"),
            "year": int(d.year),
            "month": int(d.month),
        })
        
    df = pd.DataFrame(records)
    
    # Calculate is_month_end: True only if date is the last trading day of that month
    max_dates_per_month = df.groupby(["year", "month"])["date"].transform("max")
    df["is_month_end"] = (df["date"] == max_dates_per_month).astype(bool)
    
    # Assign 0-based consecutive trading day index
    df["trading_day_index"] = np.arange(len(df), dtype=np.int64)
    
    # Assign mandatory synthetic provenance tag
    df["data_source"] = config.DATA_SOURCE
    
    # Select and order required columns exactly according to SchemaDefinition
    df = df[["date", "trading_day_index", "year", "is_month_end", "data_source"]].copy()
    
    # Validate against central schema contract
    validate_dataset_schema(df, "trading_calendar")
    
    return df


def main() -> None:
    """Executes trading calendar generation, schema validation, and persistence."""
    print("=" * 80)
    print("SIRPS TRADING CALENDAR GENERATOR (MEMBER 2)")
    print(f"Seed Stream: {config.SEED_STREAMS['calendar']} | Period: {config.START_YEAR}–{config.END_YEAR}")
    print("=" * 80)
    
    df = generate_trading_calendar(seed=config.SEED_STREAMS["calendar"])
    
    # Persist atomically via common.io
    saved_path = save(df, "trading_calendar")
    print(f"Successfully generated and saved trading calendar to: {saved_path}")
    print(f"Total Trading Sessions: {len(df):,}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Month-End Trading Sessions: {df['is_month_end'].sum()}")
    
    print("\nTrading Sessions by Calendar Year:")
    counts = df.groupby("year")["date"].count()
    for year, cnt in counts.items():
        print(f"  Year {year}: {cnt} sessions")
        
    print(f"\nSample Records:\n{df.head(3)}")
    print(f"\nFinal Records:\n{df.tail(3)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
