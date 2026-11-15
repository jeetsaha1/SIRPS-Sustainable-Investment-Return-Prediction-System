"""Automated Unit Tests for Module 2: Company Explorer & Analytics.

Validates the data loading, company selection, date filtering, KPI metrics,
benchmark rebase methodology, technical overlays, corporate actions handling,
and data dictionary integrity.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from common.io import load
import modules.company_explorer as ce


@pytest.fixture(scope="module")
def company_master_df() -> pd.DataFrame:
    """Fixture providing loaded company master."""
    return load("company_master")


@pytest.fixture(scope="module")
def prices_daily_df() -> pd.DataFrame:
    """Fixture providing loaded prices daily dataset."""
    return load("prices_daily")


@pytest.fixture(scope="module")
def index_df() -> pd.DataFrame:
    """Fixture providing loaded index daily dataset."""
    return load("index_daily")


@pytest.fixture(scope="module")
def corporate_actions_df() -> pd.DataFrame:
    """Fixture providing loaded corporate actions dataset."""
    return load("corporate_actions")


# ==============================================================================
# 1. MODULE INTERFACE & DISPATCH TESTS
# ==============================================================================

def test_company_explorer_module_interface() -> None:
    """Verifies that modules.company_explorer can be imported and exposes callable render()."""
    assert hasattr(ce, "render"), "modules.company_explorer must have a 'render' attribute"
    assert callable(ce.render), "'render' in modules.company_explorer must be callable"


def test_app_routing_to_company_explorer() -> None:
    """Verifies app.py routes '2 Company Explorer' to 'modules.company_explorer'."""
    import app
    assert "2 Company Explorer" in app.PAGES
    assert app.PAGES["2 Company Explorer"] == "modules.company_explorer"


# ==============================================================================
# 2. DATA LOADING & COMPANY SELECTION TESTS
# ==============================================================================

def test_company_selection_and_isolation(
    prices_daily_df: pd.DataFrame,
    company_master_df: pd.DataFrame,
) -> None:
    """Verifies all 10 companies exist and filtering produces isolated datasets."""
    companies = company_master_df["company_id"].tolist()
    assert len(companies) == 10

    for c in companies:
        c_df = ce.filter_by_company(prices_daily_df, c)
        assert len(c_df) == 1526, f"Company {c} should have 1,526 rows"
        assert (c_df["company_id"] == c).all(), f"Data contamination detected for {c}"
        assert c_df["date"].is_monotonic_increasing, f"Dates for {c} must be sorted chronologically"


# ==============================================================================
# 3. DATE FILTERING TESTS
# ==============================================================================

def test_date_filtering_valid_and_chronological(prices_daily_df: pd.DataFrame) -> None:
    """Verifies date filtering adheres to requested bounds and chronological order."""
    c01_df = ce.filter_by_company(prices_daily_df, "C01")
    start = "2022-01-01"
    end = "2022-12-31"

    filtered = ce.filter_by_date_range(c01_df, start, end)
    assert len(filtered) > 0
    date_strs = pd.to_datetime(filtered["date"]).dt.strftime("%Y-%m-%d")
    assert (date_strs >= start).all()
    assert (date_strs <= end).all()
    assert filtered["date"].is_monotonic_increasing


def test_date_filtering_reversed_range_handled_safely(prices_daily_df: pd.DataFrame) -> None:
    """Verifies that passing reversed dates (start > end) is handled safely by swapping."""
    c01_df = ce.filter_by_company(prices_daily_df, "C01")
    # Reversed inputs
    filtered = ce.filter_by_date_range(c01_df, "2023-12-31", "2023-01-01")
    assert len(filtered) > 0
    date_strs = pd.to_datetime(filtered["date"]).dt.strftime("%Y-%m-%d")
    assert (date_strs >= "2023-01-01").all()
    assert (date_strs <= "2023-12-31").all()


def test_date_filtering_empty_range_handled(prices_daily_df: pd.DataFrame) -> None:
    """Verifies that an out-of-bounds date range returns an empty DataFrame gracefully."""
    c01_df = ce.filter_by_company(prices_daily_df, "C01")
    filtered = ce.filter_by_date_range(c01_df, "2018-01-01", "2018-12-31")
    assert len(filtered) == 0


# ==============================================================================
# 4. KPI CALCULATIONS & FINANCIAL CORRECTNESS
# ==============================================================================

def test_calculate_kpis_normal_series(
    prices_daily_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> None:
    """Verifies KPI calculations on realistic company price series."""
    c01_df = ce.filter_by_company(prices_daily_df, "C01")
    kpis = ce.calculate_kpis(c01_df, index_df)

    assert kpis["n_obs"] == 1526
    assert kpis["last_price"] > 0
    assert abs(kpis["last_price"] - float(c01_df["adj_close"].iloc[-1])) < 1e-4

    # Volatility should be realistic annualized percentage
    assert 0.15 <= kpis["ann_vol"] <= 0.60, f"Unusual volatility: {kpis['ann_vol']}"

    # Maximum drawdown should be negative and <= 0
    assert -0.80 <= kpis["mdd"] <= 0.0, f"Unusual MDD: {kpis['mdd']}"

    # Beta should be close to 1.0 (target is 1.04)
    assert 0.85 <= kpis["beta"] <= 1.20, f"Unusual beta: {kpis['beta']}"


def test_calculate_kpis_empty_and_insufficient_history() -> None:
    """Verifies that empty and single-row DataFrames do not cause division by zero or NaN."""
    empty_df = pd.DataFrame(columns=["date", "adj_close"])
    kpis_empty = ce.calculate_kpis(empty_df)
    assert kpis_empty["n_obs"] == 0
    assert kpis_empty["cum_return"] == 0.0
    assert kpis_empty["ann_vol"] == 0.0
    assert kpis_empty["mdd"] == 0.0

    single_row = pd.DataFrame({"date": ["2020-01-01"], "adj_close": [100.0]})
    kpis_single = ce.calculate_kpis(single_row)
    assert kpis_single["n_obs"] == 1
    assert kpis_single["cum_return"] == 0.0
    assert kpis_single["ann_vol"] == 0.0
    assert kpis_single["mdd"] == 0.0


def test_calculate_drawdown_series(prices_daily_df: pd.DataFrame) -> None:
    """Verifies drawdown time-series properties: DD_t <= 0 and DD_t = 0 at peaks."""
    c03_df = ce.filter_by_company(prices_daily_df, "C03")
    dd_df = ce.calculate_drawdown_series(c03_df)

    assert len(dd_df) == len(c03_df)
    assert (dd_df["drawdown"] <= 1e-8).all(), "Drawdown must never exceed 0"
    assert (dd_df["drawdown"] >= -1.0).all(), "Drawdown must never drop below -100%"
    # Initial point must be at drawdown 0.0 (first peak is first price)
    assert abs(dd_df["drawdown"].iloc[0]) < 1e-6


def test_calculate_moving_averages_exact_windows_and_causality(prices_daily_df: pd.DataFrame) -> None:
    """Verifies MA50 requires 50 observations, MA200 requires 200, and no future leakage occurs."""
    c01_df = ce.filter_by_company(prices_daily_df, "C01")
    ma_df = ce.calculate_moving_averages(c01_df, windows=(50, 200), price_col="adj_close")

    assert "ma_50" in ma_df.columns
    assert "ma_200" in ma_df.columns

    # MA50 must be NaN for first 49 observations
    assert ma_df["ma_50"].iloc[:49].isna().all(), "MA50 must be NaN before 50 observations"
    assert not np.isnan(ma_df["ma_50"].iloc[49]), "MA50 must be populated at index 49 (50th observation)"
    expected_ma50_at_50 = float(ma_df["adj_close"].iloc[:50].mean())
    assert abs(ma_df["ma_50"].iloc[49] - expected_ma50_at_50) < 1e-4

    # MA200 must be NaN for first 199 observations
    assert ma_df["ma_200"].iloc[:199].isna().all(), "MA200 must be NaN before 200 observations"
    assert not np.isnan(ma_df["ma_200"].iloc[199]), "MA200 must be populated at index 199 (200th observation)"
    expected_ma200_at_200 = float(ma_df["adj_close"].iloc[:200].mean())
    assert abs(ma_df["ma_200"].iloc[199] - expected_ma200_at_200) < 1e-4


def test_moving_averages_price_mode_selection(prices_daily_df: pd.DataFrame) -> None:
    """Verifies that moving averages evaluate from the chosen price column (close vs adj_close)."""
    # Use C03 which underwent a 2:1 split in 2022
    c03_df = ce.filter_by_company(prices_daily_df, "C03")
    ma_adj = ce.calculate_moving_averages(c03_df, windows=(50,), price_col="adj_close")
    ma_raw = ce.calculate_moving_averages(c03_df, windows=(50,), price_col="close")

    # In 2020 prior to split, raw close is 2x adjusted close, so raw MA50 must be ~2x adjusted MA50
    assert abs(ma_raw["ma_50"].iloc[100] / ma_adj["ma_50"].iloc[100] - 2.0) < 0.05


def test_moving_averages_full_history_preservation_under_date_filter(prices_daily_df: pd.DataFrame) -> None:
    """Verifies that calculating on full history retains valid MA200 values when filtered to later years."""
    c01_full = ce.filter_by_company(prices_daily_df, "C01")
    full_ma = ce.calculate_moving_averages(c01_full, windows=(50, 200), price_col="adj_close")

    # Filter to 2024 (sessions ~1000 onward)
    filtered = ce.filter_by_date_range(full_ma, "2024-01-01", "2024-12-31")

    # The very first session of 2024 must have fully populated MA50 and MA200
    assert not np.isnan(filtered["ma_50"].iloc[0]), "MA50 should be populated in 2024"
    assert not np.isnan(filtered["ma_200"].iloc[0]), "MA200 should be populated in 2024"
    # Matches the exact full-history observation
    match_row = full_ma[full_ma["date"] == filtered["date"].iloc[0]].iloc[0]
    assert abs(filtered["ma_200"].iloc[0] - match_row["ma_200"]) < 1e-6


# ==============================================================================
# 5. BENCHMARK COMPARISON & REBASING TESTS
# ==============================================================================

def test_rebase_to_100_starts_at_100(
    prices_daily_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> None:
    """Verifies both company and benchmark start exactly at 100.0 on common start date."""
    c05_df = ce.filter_by_company(prices_daily_df, "C05")
    rebased = ce.rebase_to_100(c05_df, index_df, comp_col="adj_close", bench_col="nifty_tri")

    assert len(rebased) == 1526
    assert abs(rebased["company_rebased"].iloc[0] - 100.0) < 1e-4
    assert abs(rebased["benchmark_rebased"].iloc[0] - 100.0) < 1e-4
    assert abs(rebased["excess_spread"].iloc[0] - 0.0) < 1e-4


def test_rebase_to_100_uses_nifty_tri(
    prices_daily_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> None:
    """Verifies benchmark comparison uses total return index (TRI), not price-only close."""
    c01_df = ce.filter_by_company(prices_daily_df, "C01")
    rebased = ce.rebase_to_100(c01_df, index_df, comp_col="adj_close", bench_col="nifty_tri")

    # Benchmark rebased must track nifty_tri, not nifty_close
    expected_tri_end = (index_df["nifty_tri"].iloc[-1] / index_df["nifty_tri"].iloc[0]) * 100.0
    assert abs(rebased["benchmark_rebased"].iloc[-1] - expected_tri_end) < 1e-4


# ==============================================================================
# 6. CORPORATE ACTIONS INTEGRATION TESTS
# ==============================================================================

def test_corporate_actions_company_filter(corporate_actions_df: pd.DataFrame) -> None:
    """Verifies corporate actions filtering isolates company specific events."""
    c03_actions = ce.filter_by_company(corporate_actions_df, "C03")
    assert (c03_actions["company_id"] == "C03").all()

    splits = c03_actions[c03_actions["action_type"] == "split"]
    assert len(splits) == 1
    assert splits.iloc[0]["ratio_or_amount"] == 2.0


# ==============================================================================
# 7. DATA DICTIONARY INTEGRITY TESTS
# ==============================================================================

def test_data_dictionary_exists_and_documents_all_six_datasets() -> None:
    """Verifies docs/member_2_data_dictionary.md exists and covers all required datasets."""
    dict_path = Path(__file__).resolve().parent.parent / "docs" / "member_2_data_dictionary.md"
    assert dict_path.exists(), "docs/member_2_data_dictionary.md must exist"

    content = dict_path.read_text(encoding="utf-8")

    required_datasets = [
        "company_master.csv",
        "latent_profiles.csv",
        "trading_calendar.csv",
        "index_daily.csv",
        "prices_daily.csv",
        "corporate_actions.csv",
    ]

    for ds in required_datasets:
        assert ds in content, f"Dataset '{ds}' must be documented in data dictionary"

    # Verify synthetic provenance disclaimer is explicit
    assert "SYNTHETIC_DEMO" in content
    assert "warm-up" in content.lower()
    assert "63" in content
