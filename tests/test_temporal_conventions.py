"""
Tests for temporal conventions, date semantics, and calendar-year vs fiscal-year key consistency.
"""

import pandas as pd
import pytest
from common.io import load_dataset


def test_latent_profiles_temporal_keys():
    """Verify that latent_profiles.csv uses integer calendar year (2020..2025) and valid fiscal_year labels."""
    df = load_dataset("latent_profiles")

    # Check columns
    assert "year" in df.columns
    assert "fiscal_year" in df.columns

    # Check year data type and values
    assert pd.api.types.is_integer_dtype(df["year"])
    unique_years = sorted(df["year"].unique())
    assert unique_years == [2020, 2021, 2022, 2023, 2024, 2025]

    # Check fiscal_year labels format
    expected_fys = [f"FY{y}-{str(y+1)[2:]}" for y in unique_years]
    unique_fys = sorted(df["fiscal_year"].unique())
    assert unique_fys == sorted(expected_fys)

    # Check that each (company_id, year) pair is unique
    assert len(df) == 60
    assert df.duplicated(subset=["company_id", "year"]).sum() == 0


def test_market_calendar_join_compatibility():
    """Verify that simulated daily dates can join to latent_profiles via date.dt.year cleanly."""
    latent_df = load_dataset("latent_profiles")

    # Construct synthetic daily dates across 2020-2025
    daily_dates = pd.date_range("2020-01-01", "2025-12-31", freq="B")  # Business days
    prices_stub = pd.DataFrame({
        "company_id": "C01",
        "date": daily_dates,
    })
    prices_stub["year"] = prices_stub["date"].dt.year

    # Perform inner join on ['company_id', 'year']
    merged = pd.merge(prices_stub, latent_df, on=["company_id", "year"], how="inner")

    # All business days should successfully match a latent profile year
    assert len(merged) == len(prices_stub)
    assert set(merged["year"].unique()) == {2020, 2021, 2022, 2023, 2024, 2025}
    assert "esg_latent_score" in merged.columns
    assert "beta" in merged.columns
