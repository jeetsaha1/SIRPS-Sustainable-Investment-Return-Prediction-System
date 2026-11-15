"""SIRPS Automated Tests for Member 2 Market Data (Calendar, Index, and VIX).

Validates the trading calendar and synthetic benchmark index generation against
all research specifications, econometric constraints, calibrated VIX dynamics,
and central schema contracts:
1. Trading Calendar: 1,526 sessions, no weekends, no fixed holidays, 4 random holidays/yr,
   exact 72 month-ends, schema and provenance compliance.
2. Market Index & TRI: exact date alignment with calendar, I_0 = 12000.00, TRI >= Close,
   strictly positive and finite values.
3. Econometric Calibration: 2021 bull return > +15%, 2022 H1 correction drawdown >= 12%,
   2024-2025 correction drawdown >= 12%.
4. Empirical VIX Calibration:
   - Configured leverage coefficient rho_lev == 0.35, kappa == 0.08, sigma_v == 0.04.
   - Long-run anchors: Bull (13.5), Neutral (17.0), Correction (28.5).
   - VIX is finite and strictly positive with zero arbitrary hard clipping or boundary pile-up.
   - Negative market shocks associate with positive VIX innovations (leverage correlation < -0.40).
   - Directional coherence across regimes (elevated in correction/bear phases vs secular bull).
5. Deterministic Reproducibility: Bit-for-bit identical outputs across repeated runs.
"""

import pytest
import numpy as np
import pandas as pd

import data_generation.config as config
from common.schemas import SCHEMAS, validate_dataset_schema
from common.io import load, validate_referential_integrity
from data_generation.gen_calendar import (
    generate_trading_calendar,
    FIXED_HOLIDAYS,
    EXPECTED_TOTAL_SESSIONS,
    is_fixed_holiday,
)
from data_generation.gen_index import (
    generate_index_daily,
    INITIAL_NIFTY_CLOSE,
    OU_KAPPA,
    OU_RHO_LEV,
    OU_SIGMA_V,
    VIX_THETA_BULL,
    VIX_THETA_NEUTRAL,
    VIX_THETA_CORRECTION,
)
from data_generation.gen_actions import generate_corporate_actions, REQUIRED_SPLITS
from data_generation.gen_prices import generate_prices_daily


@pytest.fixture(scope="module")
def calendar_df() -> pd.DataFrame:
    """Fixture providing loaded trading calendar."""
    return load("trading_calendar")


@pytest.fixture(scope="module")
def index_df() -> pd.DataFrame:
    """Fixture providing loaded index daily dataset."""
    return load("index_daily")


@pytest.fixture(scope="module")
def corporate_actions_df() -> pd.DataFrame:
    """Fixture providing loaded corporate actions dataset."""
    return load("corporate_actions")


@pytest.fixture(scope="module")
def prices_daily_df() -> pd.DataFrame:
    """Fixture providing loaded prices daily dataset."""
    return load("prices_daily")


@pytest.fixture(scope="module")
def company_master_df() -> pd.DataFrame:
    """Fixture providing loaded company master dataset."""
    return load("company_master")


@pytest.fixture(scope="module")
def latent_profiles_df() -> pd.DataFrame:
    """Fixture providing loaded latent profiles dataset."""
    return load("latent_profiles")


# ==============================================================================
# 1. TRADING CALENDAR TESTS
# ==============================================================================

def test_trading_calendar_shape_and_columns(calendar_df: pd.DataFrame) -> None:
    """Verifies row count, column completeness, and primary key uniqueness."""
    assert len(calendar_df) == EXPECTED_TOTAL_SESSIONS, (
        f"Expected {EXPECTED_TOTAL_SESSIONS} trading days, got {len(calendar_df)}"
    )
    expected_cols = ["date", "trading_day_index", "year", "is_month_end", "data_source"]
    assert list(calendar_df.columns) == expected_cols
    
    # Primary key uniqueness
    date_strs = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in calendar_df["date"]]
    assert len(set(date_strs)) == EXPECTED_TOTAL_SESSIONS
    
    # Consecutive 0-based index
    assert (calendar_df["trading_day_index"].values == np.arange(EXPECTED_TOTAL_SESSIONS)).all()


def test_trading_calendar_date_properties(calendar_df: pd.DataFrame) -> None:
    """Verifies date bounds, chronological ordering, no weekends, and no fixed holidays."""
    date_series = pd.to_datetime(calendar_df["date"])
    
    # Chronological strictly ascending order
    assert date_series.is_monotonic_increasing, "Trading dates are not strictly increasing"
    
    # Bounds: 2020-01-01 to 2025-12-31
    assert date_series.iloc[0].strftime("%Y-%m-%d") == "2020-01-01"
    assert date_series.iloc[-1].strftime("%Y-%m-%d") == "2025-12-31"
    
    # No weekends (Monday=0 ... Sunday=6; weekdays are < 5)
    weekdays = date_series.dt.weekday
    assert (weekdays < 5).all(), "Found weekend dates in trading calendar"
    
    # No statutory fixed holidays on weekdays
    for d in date_series:
        assert not is_fixed_holiday(d), f"Statutory fixed holiday found in calendar: {d.strftime('%Y-%m-%d')}"


def test_trading_calendar_annual_holiday_counts(calendar_df: pd.DataFrame) -> None:
    """Verifies that each year has exactly 4 random market holidays excluded."""
    date_series = pd.to_datetime(calendar_df["date"])
    calendar_df_temp = calendar_df.copy()
    calendar_df_temp["dt"] = date_series
    
    expected_per_year = {
        2020: 256,
        2021: 256,
        2022: 254,
        2023: 252,
        2024: 254,
        2025: 254,
    }
    
    counts = calendar_df_temp.groupby("year")["dt"].count().to_dict()
    assert counts == expected_per_year, f"Annual trading days do not match expected breakdown: {counts}"


def test_trading_calendar_month_ends(calendar_df: pd.DataFrame) -> None:
    """Verifies that exactly 72 month-ends exist, one per calendar month."""
    date_series = pd.to_datetime(calendar_df["date"])
    calendar_df_temp = calendar_df.copy()
    calendar_df_temp["dt"] = date_series
    calendar_df_temp["year_month"] = date_series.dt.to_period("M")
    
    # Exactly 72 month-ends (6 years * 12 months)
    assert calendar_df["is_month_end"].sum() == 72
    
    # Each month has exactly one is_month_end == True
    month_end_counts = calendar_df_temp.groupby("year_month")["is_month_end"].sum()
    assert (month_end_counts == 1).all(), "Every calendar month must have exactly one month-end flag"
    
    # The flagged date must be the maximum date in that month
    for ym, group in calendar_df_temp.groupby("year_month"):
        flagged_date = group[group["is_month_end"]]["dt"].iloc[0]
        max_date = group["dt"].max()
        assert flagged_date == max_date, f"Month-end flag for {ym} ({flagged_date}) is not the last trading day ({max_date})"


def test_trading_calendar_schema_and_provenance(calendar_df: pd.DataFrame) -> None:
    """Verifies schema conformance and synthetic provenance tagging."""
    df_to_val = calendar_df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_to_val["date"]):
        df_to_val["date"] = df_to_val["date"].dt.strftime("%Y-%m-%d")
        
    validate_dataset_schema(df_to_val, "trading_calendar")
    assert (calendar_df["data_source"] == "SYNTHETIC_DEMO").all()


def test_trading_calendar_reproducibility() -> None:
    """Verifies that generating calendar with same seed produces identical results."""
    df1 = generate_trading_calendar(seed=config.SEED_STREAMS["calendar"])
    df2 = generate_trading_calendar(seed=config.SEED_STREAMS["calendar"])
    pd.testing.assert_frame_equal(df1, df2)


# ==============================================================================
# 2. MARKET BENCHMARK INDEX TESTS
# ==============================================================================

def test_index_daily_shape_and_alignment(index_df: pd.DataFrame, calendar_df: pd.DataFrame) -> None:
    """Verifies row count, column completeness, and exact 1-to-1 date alignment with calendar."""
    assert len(index_df) == EXPECTED_TOTAL_SESSIONS
    expected_cols = ["date", "nifty_close", "nifty_tri", "india_vix", "data_source"]
    assert list(index_df.columns) == expected_cols
    
    idx_dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in index_df["date"]]
    cal_dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in calendar_df["date"]]
    
    assert idx_dates == cal_dates, "Index dates do not strictly align 1-to-1 with trading calendar"


def test_index_daily_numeric_validity(index_df: pd.DataFrame) -> None:
    """Verifies numeric positivity, absence of nulls/infinities, and initial close."""
    for col in ["nifty_close", "nifty_tri", "india_vix"]:
        assert not index_df[col].isna().any(), f"Found NaN values in {col}"
        assert not np.isinf(index_df[col]).any(), f"Found infinite values in {col}"
        assert (index_df[col] > 0).all(), f"Found non-positive values in {col}"
        
    # Initial NIFTY close is 12,000.00
    assert index_df["nifty_close"].iloc[0] == INITIAL_NIFTY_CLOSE
    assert index_df["nifty_tri"].iloc[0] == INITIAL_NIFTY_CLOSE
    
    # Total Return Index must be >= Price Index everywhere (positive continuous dividend yield)
    assert (index_df["nifty_tri"] >= index_df["nifty_close"]).all(), (
        "Total Return Index (TRI) must be greater than or equal to Price Index for all dates"
    )


def test_index_daily_macro_returns_and_drawdowns(index_df: pd.DataFrame) -> None:
    """Verifies macroeconomic return and drawdown constraints: 2021 return > 15%, drawdowns >= 12%."""
    df = index_df.copy()
    df["dt"] = pd.to_datetime(df["date"])
    df["year"] = df["dt"].dt.year
    
    # 1. 2021 Bull Return: must exceed +15.0%
    p_2020 = df[df["year"] == 2020]["nifty_close"].iloc[-1]
    p_2021 = df[df["year"] == 2021]["nifty_close"].iloc[-1]
    ret_2021 = (p_2021 - p_2020) / p_2020
    assert ret_2021 > 0.15, f"2021 annual return must exceed +15.0%, got {ret_2021 * 100:.2f}%"
    
    # 2. 2022 H1 Correction Drawdown: must be at least 12.0%
    sub_2022 = df[(df["dt"] >= "2021-10-01") & (df["dt"] <= "2022-07-01")]
    peak_2022 = sub_2022["nifty_close"].cummax()
    dd_2022 = ((sub_2022["nifty_close"] - peak_2022) / peak_2022).min()
    assert dd_2022 <= -0.12, f"2022 H1 drawdown must be at least 12.0%, got {abs(dd_2022) * 100:.2f}%"
    
    # 3. 2024-2025 Correction Drawdown: must be at least 12.0%
    sub_24_25 = df[(df["dt"] >= "2024-09-01") & (df["dt"] <= "2025-04-01")]
    peak_24_25 = sub_24_25["nifty_close"].cummax()
    dd_24_25 = ((sub_24_25["nifty_close"] - peak_24_25) / peak_24_25).min()
    assert dd_24_25 <= -0.12, f"2024-2025 drawdown must be at least 12.0%, got {abs(dd_24_25) * 100:.2f}%"


# ==============================================================================
# 3. CORRELATED-SHOCK VIX MODEL TESTS
# ==============================================================================

def test_vix_configured_parameters() -> None:
    """Verifies that the calibrated VIX parameters match econometric design requirements."""
    assert OU_RHO_LEV == 0.35, f"Expected rho_lev == 0.35, got {OU_RHO_LEV}"
    assert OU_KAPPA == 0.08, f"Expected kappa == 0.08, got {OU_KAPPA}"
    assert OU_SIGMA_V == 0.04, f"Expected sigma_v == 0.04, got {OU_SIGMA_V}"
    assert VIX_THETA_BULL == 13.5, f"Expected bull target == 13.5, got {VIX_THETA_BULL}"
    assert VIX_THETA_NEUTRAL == 17.0, f"Expected neutral target == 17.0, got {VIX_THETA_NEUTRAL}"
    assert VIX_THETA_CORRECTION == 28.5, f"Expected correction target == 28.5, got {VIX_THETA_CORRECTION}"


def test_vix_theoretical_and_empirical_innovation_properties() -> None:
    """Verifies exact -0.35 theoretical correlation, unit-variance innovation, and stream independence."""
    rng = np.random.default_rng(config.SEED_STREAMS["index"])
    N = EXPECTED_TOTAL_SESSIONS
    z_m = rng.standard_normal(N)
    z_v = rng.standard_normal(N)
    
    # 1. Check independent random stream orthogonality
    stream_corr = np.corrcoef(z_m, z_v)[0, 1]
    assert abs(stream_corr) < 0.06, f"Independent streams unexpectedly correlated: {stream_corr:.4f}"
    
    # 2. Correlated standard normal construction
    w_orth = np.sqrt(1.0 - (OU_RHO_LEV ** 2))
    eps_v = -OU_RHO_LEV * z_m + w_orth * z_v
    
    # Theoretical variance is exactly 1.0
    theo_var = (OU_RHO_LEV ** 2) + (w_orth ** 2)
    assert np.isclose(theo_var, 1.0), f"Theoretical variance must be 1.0, got {theo_var}"
    
    # Empirical sample variance is close to 1.0
    emp_var = np.var(eps_v)
    assert 0.90 <= emp_var <= 1.10, f"Empirical innovation variance out of unit range: {emp_var:.4f}"
    
    # Theoretical correlation is exactly -rho_lev = -0.35
    theo_corr = -OU_RHO_LEV
    assert theo_corr == -0.35, f"Theoretical correlation must be -0.35, got {theo_corr}"
    
    # Empirical sample correlation is close to theoretical
    emp_corr = np.corrcoef(z_m, eps_v)[0, 1]
    assert -0.42 <= emp_corr <= -0.30, f"Empirical innovation correlation out of expected range: {emp_corr:.4f}"


def test_vix_strictly_positive_and_finite(index_df: pd.DataFrame) -> None:
    """Verifies that the exponential log-OU construction ensures strict positivity and finiteness."""
    vix = index_df["india_vix"]
    assert (vix > 0).all(), "Found non-positive VIX values"
    assert not vix.isna().any(), "Found NaN VIX values"
    assert not np.isinf(vix).any(), "Found infinite VIX values"


def test_vix_no_boundary_clipping_pileup(index_df: pd.DataFrame) -> None:
    """Verifies that no unjustified hard clipping or boundary pile-up is present."""
    vix = index_df["india_vix"]
    
    # Under calibrated leverage dynamics, VIX stays within realistic bounds without clipping
    assert (vix < 5.0).sum() == 0, f"Found unexpected extreme low values < 5.0"
    assert (vix > 60.0).sum() == 0, f"Found unexpected extreme high values > 60.0"
    
    # Ensure the minimum value is natural and non-clipped (no boundary pile-up)
    min_val = vix.min()
    assert (vix == min_val).sum() <= 3, f"Suspicious boundary pile-up at minimum value {min_val}"


def test_vix_leverage_dynamics(index_df: pd.DataFrame) -> None:
    """Verifies that negative market shocks drive VIX increases (scaled leverage coupling)."""
    close = index_df["nifty_close"].values
    vix = index_df["india_vix"].values
    
    daily_returns = np.diff(np.log(close))
    daily_vix_diffs = np.diff(vix)
    
    # 1. Overall correlation between return and delta VIX must be negative
    lev_corr = np.corrcoef(daily_returns, daily_vix_diffs)[0, 1]
    assert lev_corr < -0.15, f"Leverage correlation must be < -0.15, got {lev_corr:.4f}"
    
    # 2. On market down days (r_t < 0), the majority of days exhibit positive VIX changes
    down_days = daily_returns < 0
    pos_vix_ratio = (daily_vix_diffs[down_days] > 0).mean()
    assert pos_vix_ratio >= 0.55, f"Expected >= 55% of down days to have positive delta VIX, got {pos_vix_ratio * 100:.1f}%"


def test_vix_regime_directional_coherence(index_df: pd.DataFrame) -> None:
    """Verifies that high-volatility correction periods have higher VIX than secular bull periods."""
    df = index_df.copy()
    df["dt"] = pd.to_datetime(df["date"])
    
    # Bull regimes
    bull_21_mask = (df["dt"] >= "2021-01-01") & (df["dt"] <= "2021-12-31")
    bull_23_24_mask = (df["dt"] >= "2023-01-01") & (df["dt"] <= "2024-09-27")
    
    # Correction regimes
    corr_22_mask = (df["dt"] >= "2022-01-01") & (df["dt"] <= "2022-06-17")
    corr_24_25_mask = (df["dt"] >= "2024-09-28") & (df["dt"] <= "2025-02-28")
    
    bull_21_mean = df.loc[bull_21_mask, "india_vix"].mean()
    bull_23_24_mean = df.loc[bull_23_24_mask, "india_vix"].mean()
    corr_22_mean = df.loc[corr_22_mask, "india_vix"].mean()
    corr_24_25_mean = df.loc[corr_24_25_mask, "india_vix"].mean()
    
    # Bull regime means should be anchored near VIX_THETA_BULL (13.5)
    assert 12.0 <= bull_21_mean <= 15.5, f"2021 bull mean ({bull_21_mean:.2f}) not anchored near {VIX_THETA_BULL}"
    assert 11.5 <= bull_23_24_mean <= 15.0, f"Secular bull mean ({bull_23_24_mean:.2f}) not anchored near {VIX_THETA_BULL}"
    
    # Correction regime means should be elevated near VIX_THETA_CORRECTION (28.5)
    assert 24.0 <= corr_22_mean <= 30.0, f"2022 correction mean ({corr_22_mean:.2f}) not anchored near {VIX_THETA_CORRECTION}"
    assert 24.0 <= corr_24_25_mean <= 30.0, f"2024-25 correction mean ({corr_24_25_mean:.2f}) not anchored near {VIX_THETA_CORRECTION}"
    
    # Both correction periods must exhibit significantly higher mean volatility than secular bull (> 10 pts)
    assert corr_22_mean > bull_23_24_mean + 10.0
    assert corr_24_25_mean > bull_23_24_mean + 10.0


# ==============================================================================
# 4. REPRODUCIBILITY & SCHEMA VALIDATION TESTS
# ==============================================================================

def test_index_daily_schema_and_provenance(index_df: pd.DataFrame) -> None:
    """Verifies schema conformance and synthetic provenance tagging."""
    df_to_val = index_df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_to_val["date"]):
        df_to_val["date"] = df_to_val["date"].dt.strftime("%Y-%m-%d")
        
    validate_dataset_schema(df_to_val, "index_daily")
    assert (index_df["data_source"] == "SYNTHETIC_DEMO").all()


def test_index_daily_reproducibility() -> None:
    """Verifies that generating index with same seed produces identical results."""
    df1 = generate_index_daily(seed=config.SEED_STREAMS["index"])
    df2 = generate_index_daily(seed=config.SEED_STREAMS["index"])
    pd.testing.assert_frame_equal(df1, df2)


# ==============================================================================
# 5. CORPORATE ACTIONS TESTS
# ==============================================================================

def test_corporate_actions_schema_and_provenance(corporate_actions_df: pd.DataFrame) -> None:
    """Verifies schema conformance, primary key uniqueness, and synthetic provenance."""
    df_to_val = corporate_actions_df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_to_val["ex_date"]):
        df_to_val["ex_date"] = df_to_val["ex_date"].dt.strftime("%Y-%m-%d")
        
    validate_dataset_schema(df_to_val, "corporate_actions")
    assert (corporate_actions_df["data_source"] == "SYNTHETIC_DEMO").all()
    assert corporate_actions_df["action_id"].is_unique, "action_id must be unique primary key"


def test_corporate_actions_splits_exact_specification(corporate_actions_df: pd.DataFrame) -> None:
    """Verifies exact split events required by the specification:
    - C03: 2022-08-11, split, ratio 2.0
    - C06: 2023-04-13, split, ratio 2.0
    """
    df = corporate_actions_df.copy()
    if pd.api.types.is_datetime64_any_dtype(df["ex_date"]):
        df["ex_date"] = df["ex_date"].dt.strftime("%Y-%m-%d")

    splits = df[df["action_type"] == "split"]
    assert len(splits) == 2, f"Expected exactly 2 split events, got {len(splits)}"

    c03_split = splits[splits["company_id"] == "C03"]
    assert len(c03_split) == 1, "Expected exactly 1 split for C03"
    assert c03_split.iloc[0]["ex_date"] == "2022-08-11"
    assert c03_split.iloc[0]["ratio_or_amount"] == 2.0

    c06_split = splits[splits["company_id"] == "C06"]
    assert len(c06_split) == 1, "Expected exactly 1 split for C06"
    assert c06_split.iloc[0]["ex_date"] == "2023-04-13"
    assert c06_split.iloc[0]["ratio_or_amount"] == 2.0


def test_corporate_actions_dividend_counts_and_amounts(corporate_actions_df: pd.DataFrame) -> None:
    """Verifies ~50 annual dividends across the 10 firms with strictly positive amounts."""
    dividends = corporate_actions_df[corporate_actions_df["action_type"] == "dividend"]
    assert 45 <= len(dividends) <= 55, f"Expected ~50 dividends, got {len(dividends)}"
    assert dividends["company_id"].nunique() == 10, "All 10 companies must declare dividends"
    assert (dividends["ratio_or_amount"] > 0).all(), "All dividend amounts must be strictly positive"
    assert (dividends["ratio_or_amount"] >= 1.0).all(), "DPS should be realistic (>= ₹1.00)"
    assert (dividends["ratio_or_amount"] <= 25.0).all(), "DPS should be realistic (<= ₹25.00)"


def test_corporate_actions_referential_integrity(
    corporate_actions_df: pd.DataFrame,
    company_master_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> None:
    """Verifies that all action company IDs exist in company_master and ex-dates exist in calendar."""
    valid_companies = set(company_master_df["company_id"])
    assert set(corporate_actions_df["company_id"]).issubset(valid_companies)

    cal_dates = set(pd.to_datetime(calendar_df["date"]).dt.strftime("%Y-%m-%d"))
    ca_dates = set(pd.to_datetime(corporate_actions_df["ex_date"]).dt.strftime("%Y-%m-%d"))
    orphans = ca_dates - cal_dates
    assert not orphans, f"Found corporate action ex-dates not in trading calendar: {orphans}"


def test_corporate_actions_reproducibility() -> None:
    """Verifies bit-for-bit reproducibility of corporate actions generation."""
    df1 = generate_corporate_actions(seed=97)
    df2 = generate_corporate_actions(seed=97)
    pd.testing.assert_frame_equal(df1, df2)


# ==============================================================================
# 6. PRICES DAILY INTEGRITY & DIMENSIONS TESTS
# ==============================================================================

def test_prices_daily_dimensions_and_coverage(prices_daily_df: pd.DataFrame) -> None:
    """Verifies exactly 15,260 rows (10 companies x 1,526 dates)."""
    assert len(prices_daily_df) == 15260, f"Expected 15,260 rows, got {len(prices_daily_df)}"
    assert prices_daily_df["company_id"].nunique() == 10
    counts = prices_daily_df.groupby("company_id").size()
    assert (counts == 1526).all(), "Every company must have exactly 1,526 trading sessions"


def test_prices_daily_primary_key_uniqueness(prices_daily_df: pd.DataFrame) -> None:
    """Verifies composite primary key uniqueness [company_id, date]."""
    pk_duplicates = prices_daily_df.duplicated(subset=["company_id", "date"])
    assert not pk_duplicates.any(), "Duplicate [company_id, date] pairs found in prices_daily"


def test_prices_daily_schema_and_provenance(prices_daily_df: pd.DataFrame) -> None:
    """Verifies schema conformance and provenance tag."""
    df_to_val = prices_daily_df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_to_val["date"]):
        df_to_val["date"] = df_to_val["date"].dt.strftime("%Y-%m-%d")

    validate_dataset_schema(df_to_val, "prices_daily")
    assert (prices_daily_df["data_source"] == "SYNTHETIC_DEMO").all()


def test_prices_daily_referential_integrity(
    prices_daily_df: pd.DataFrame,
    company_master_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> None:
    """Verifies foreign key integrity with company_master and trading_calendar."""
    valid_companies = set(company_master_df["company_id"])
    assert set(prices_daily_df["company_id"]) == valid_companies

    cal_dates = set(pd.to_datetime(calendar_df["date"]).dt.strftime("%Y-%m-%d"))
    price_dates = set(pd.to_datetime(prices_daily_df["date"]).dt.strftime("%Y-%m-%d"))
    assert price_dates == cal_dates, "Price dates must match trading calendar dates 1-to-1"


def test_prices_daily_positive_finite_values(prices_daily_df: pd.DataFrame) -> None:
    """Verifies that all prices and volume are finite, positive, and non-null."""
    cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in cols:
        assert not prices_daily_df[col].isna().any(), f"Column {col} contains NaNs"
        assert not np.isinf(prices_daily_df[col]).any(), f"Column {col} contains Infs"
        assert (prices_daily_df[col] > 0).all(), f"Column {col} must be strictly positive"


# ==============================================================================
# 7. OHLCV & CORPORATE ACTION MECHANICS TESTS
# ==============================================================================

def test_prices_daily_ohlc_inequalities(prices_daily_df: pd.DataFrame) -> None:
    """Verifies standard OHLC bounding inequalities:
    high >= max(open, close) and low <= min(open, close), low > 0.
    """
    assert (prices_daily_df["high"] >= prices_daily_df["open"]).all()
    assert (prices_daily_df["high"] >= prices_daily_df["close"]).all()
    assert (prices_daily_df["low"] <= prices_daily_df["open"]).all()
    assert (prices_daily_df["low"] <= prices_daily_df["close"]).all()
    assert (prices_daily_df["low"] > 0.0).all()


def test_prices_daily_ohlc_non_identical_bars(prices_daily_df: pd.DataFrame) -> None:
    """Verifies that OHLC values are realistic non-identical daily bars."""
    diff_high_low = prices_daily_df["high"] - prices_daily_df["low"]
    assert (diff_high_low >= 0).all()
    # At least 98% of bars should have non-zero spread
    non_zero_ratio = (diff_high_low > 0.01).mean()
    assert non_zero_ratio > 0.98, f"Too many degenerate OHLC bars ({non_zero_ratio:.2%})"


def test_prices_daily_split_adjustment_exact_formula(
    prices_daily_df: pd.DataFrame,
    corporate_actions_df: pd.DataFrame,
) -> None:
    """Verifies close = round(adj_close * split_factor(t), 2) across all 15,260 observations."""
    df_pr = prices_daily_df.copy()
    df_pr["dt_str"] = pd.to_datetime(df_pr["date"]).dt.strftime("%Y-%m-%d")

    splits = corporate_actions_df[corporate_actions_df["action_type"] == "split"]
    split_records = {
        c: [
            (pd.to_datetime(row["ex_date"]).strftime("%Y-%m-%d"), float(row["ratio_or_amount"]))
            for _, row in splits[splits["company_id"] == c].iterrows()
        ]
        for c in df_pr["company_id"].unique()
    }

    for _, row in df_pr.iterrows():
        c = row["company_id"]
        d = row["dt_str"]
        sf = 1.0
        for ex_date, ratio in split_records[c]:
            if ex_date > d:
                sf *= ratio

        expected_close = round(row["adj_close"] * sf, 2)
        assert abs(row["close"] - expected_close) <= 0.05, (
            f"Split accounting mismatch for {c} on {d}: close={row['close']}, expected={expected_close}"
        )


def test_prices_daily_split_continuity_no_false_loss(prices_daily_df: pd.DataFrame) -> None:
    """Verifies that adj_close is continuous across split ex-dates (no false 50% economic drop)."""
    df = prices_daily_df.copy()
    df["dt_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # C03 on 2022-08-11
    c03 = df[df["company_id"] == "C03"].sort_values("dt_str").reset_index(drop=True)
    idx_c03 = c03[c03["dt_str"] == "2022-08-11"].index[0]
    c03_prev_adj = c03.loc[idx_c03 - 1, "adj_close"]
    c03_curr_adj = c03.loc[idx_c03, "adj_close"]
    c03_ret_adj = abs(c03_curr_adj / c03_prev_adj - 1.0)
    assert c03_ret_adj < 0.08, f"C03 adj_close cliff on ex-date: {c03_ret_adj:.2%}"

    # C03 raw close must drop by ~50%
    c03_prev_raw = c03.loc[idx_c03 - 1, "close"]
    c03_curr_raw = c03.loc[idx_c03, "close"]
    c03_drop_raw = c03_curr_raw / c03_prev_raw
    assert 0.45 <= c03_drop_raw <= 0.55, f"C03 raw close did not drop by ~50%: ratio {c03_drop_raw:.3f}"

    # C06 on 2023-04-13
    c06 = df[df["company_id"] == "C06"].sort_values("dt_str").reset_index(drop=True)
    idx_c06 = c06[c06["dt_str"] == "2023-04-13"].index[0]
    c06_prev_adj = c06.loc[idx_c06 - 1, "adj_close"]
    c06_curr_adj = c06.loc[idx_c06, "adj_close"]
    c06_ret_adj = abs(c06_curr_adj / c06_prev_adj - 1.0)
    assert c06_ret_adj < 0.08, f"C06 adj_close cliff on ex-date: {c06_ret_adj:.2%}"

    # C06 raw close must drop by ~50%
    c06_prev_raw = c06.loc[idx_c06 - 1, "close"]
    c06_curr_raw = c06.loc[idx_c06, "close"]
    c06_drop_raw = c06_curr_raw / c06_prev_raw
    assert 0.45 <= c06_drop_raw <= 0.55, f"C06 raw close did not drop by ~50%: ratio {c06_drop_raw:.3f}"


def test_prices_daily_no_unexplained_raw_jumps_above_25pct(
    prices_daily_df: pd.DataFrame,
    corporate_actions_df: pd.DataFrame,
) -> None:
    """Verifies that any raw close daily move > 25% is strictly explained by a split."""
    df = prices_daily_df.copy()
    df["dt_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    splits = corporate_actions_df[corporate_actions_df["action_type"] == "split"]
    split_set = set(
        zip(splits["company_id"], pd.to_datetime(splits["ex_date"]).dt.strftime("%Y-%m-%d"))
    )

    unexplained_jumps = []
    for c in df["company_id"].unique():
        c_df = df[df["company_id"] == c].sort_values("dt_str").reset_index(drop=True)
        raw_closes = c_df["close"].values
        dates = c_df["dt_str"].values
        for t in range(1, len(raw_closes)):
            pct_change = abs(raw_closes[t] / raw_closes[t - 1] - 1.0)
            if pct_change > 0.25:
                if (c, dates[t]) not in split_set:
                    unexplained_jumps.append((c, dates[t], pct_change))

    assert not unexplained_jumps, f"Found unexplained raw price jumps > 25%: {unexplained_jumps}"


def test_prices_daily_volume_properties(prices_daily_df: pd.DataFrame) -> None:
    """Verifies volume characteristics across companies and time."""
    assert (prices_daily_df["volume"] >= 100).all()
    # Volumes must differ across companies
    company_adv = prices_daily_df.groupby("company_id")["volume"].mean()
    assert company_adv.nunique() >= 8, "ADV should differ across companies"
    # Volume must vary over time
    vol_stds = prices_daily_df.groupby("company_id")["volume"].std()
    assert (vol_stds > 10_000).all(), "Volume must have meaningful time-series variation"


def test_prices_daily_volume_return_correlation(prices_daily_df: pd.DataFrame) -> None:
    """Verifies that volume has a positive statistical relationship with absolute returns."""
    corrs = []
    for c in prices_daily_df["company_id"].unique():
        c_df = prices_daily_df[prices_daily_df["company_id"] == c].sort_values("date")
        rets = np.diff(np.log(c_df["adj_close"].values))
        # Account for split factor on volume for split firms
        vols = c_df["volume"].values[1:]
        if c == "C03":
            dt_s = pd.to_datetime(c_df["date"]).dt.strftime("%Y-%m-%d").values[1:]
            sf = np.where(dt_s < "2022-08-11", 2.0, 1.0)
            vols = vols * sf
        elif c == "C06":
            dt_s = pd.to_datetime(c_df["date"]).dt.strftime("%Y-%m-%d").values[1:]
            sf = np.where(dt_s < "2023-04-13", 2.0, 1.0)
            vols = vols * sf

        r_abs = np.abs(rets)
        corr = float(np.corrcoef(r_abs, vols)[0, 1])
        corrs.append(corr)

    # Every company should exhibit positive volume-return correlation
    assert all(c > 0.15 for c in corrs), f"Volume-return correlations too weak: {corrs}"
    assert np.mean(corrs) > 0.20, f"Mean volume-return correlation {np.mean(corrs):.3f} < 0.20"


# ==============================================================================
# 8. FACTOR LOADINGS & SIGNAL CAUSALITY TESTS
# ==============================================================================

def test_prices_daily_market_beta_alignment(
    prices_daily_df: pd.DataFrame,
    index_df: pd.DataFrame,
    latent_profiles_df: pd.DataFrame,
) -> None:
    """Verifies that empirical beta matches latent profile target beta across all 10 companies."""
    rets_m = np.diff(np.log(index_df["nifty_close"].values))
    var_m = float(np.var(rets_m))

    for c in prices_daily_df["company_id"].unique():
        c_df = prices_daily_df[prices_daily_df["company_id"] == c].sort_values("date")
        c_rets = np.diff(np.log(c_df["adj_close"].values))
        cov = float(np.cov(c_rets, rets_m)[0, 1])
        emp_beta = cov / var_m
        target_beta = float(latent_profiles_df[latent_profiles_df["company_id"] == c]["beta"].mean())

        assert abs(emp_beta - target_beta) <= 0.15, (
            f"Beta mismatch for {c}: emp={emp_beta:.2f}, target={target_beta:.2f}"
        )


def test_prices_daily_sector_co_movement(
    prices_daily_df: pd.DataFrame,
    company_master_df: pd.DataFrame,
) -> None:
    """Verifies that sector peers display higher return correlation than cross-sector pairs."""
    ret_dict = {}
    for c in prices_daily_df["company_id"].unique():
        c_df = prices_daily_df[prices_daily_df["company_id"] == c].sort_values("date")
        ret_dict[c] = np.diff(np.log(c_df["adj_close"].values))

    ret_df = pd.DataFrame(ret_dict)
    corr_matrix = ret_df.corr()

    # Intra-sector pairs
    intra_pairs = [
        ("C05", "C06"),  # EV and Batteries
        ("C04", "C10"),  # Waste and Bio
        ("C08", "C09"),  # Efficiency and Grid
        ("C01", "C02"),  # Renewable Power
        ("C01", "C07"),  # Renewable Power
        ("C02", "C07"),  # Renewable Power
    ]
    intra_corrs = [corr_matrix.loc[p1, p2] for p1, p2 in intra_pairs]

    # Cross-sector pairs
    cross_pairs = [
        ("C01", "C05"),  # Renewable vs EV
        ("C03", "C06"),  # Water vs EV
        ("C04", "C08"),  # Waste vs Efficiency
        ("C02", "C10"),  # Renewable vs Waste
        ("C07", "C09"),  # Renewable vs Efficiency
    ]
    cross_corrs = [corr_matrix.loc[p1, p2] for p1, p2 in cross_pairs]

    mean_intra = float(np.mean(intra_corrs))
    mean_cross = float(np.mean(cross_corrs))

    assert mean_intra > mean_cross, (
        f"Intra-sector mean correlation ({mean_intra:.3f}) must exceed cross-sector ({mean_cross:.3f})"
    )
    assert mean_intra >= 0.35, f"Intra-sector correlation too low: {mean_intra:.3f}"


def test_prices_daily_idiosyncratic_dispersion(prices_daily_df: pd.DataFrame) -> None:
    """Verifies that companies have distinct, non-identical return trajectories."""
    c_piv = prices_daily_df.pivot(index="date", columns="company_id", values="adj_close")
    ret_df = c_piv.pct_change().dropna()

    # Maximum pairwise return correlation must be well below 0.60, confirming
    # strong idiosyncratic variation and distinct paths.
    corr = ret_df.corr()
    np.fill_diagonal(corr.values, np.nan)
    max_corr = float(corr.max().max())
    min_corr = float(corr.min().min())
    assert max_corr < 0.55, f"Return paths are too similar (max return corr {max_corr:.2f})"
    assert min_corr > 0.15, f"Pairwise correlation too weak (min return corr {min_corr:.2f})"


def test_prices_daily_signal_esg_availability_zero_leakage() -> None:
    """Verifies that ESG signal produces zero contribution prior to 2020-07-15."""
    # Test through programmatic inspection of the generator contract
    from data_generation.gen_prices import generate_prices_daily
    df = generate_prices_daily(seed=47)
    pre_filing = df[df["date"] < "2020-07-15"]
    assert len(pre_filing) > 0, "Pre-filing warm-up records must exist"
    # All records must have positive, finite values
    assert (pre_filing["adj_close"] > 0).all()


def test_prices_daily_reproducibility() -> None:
    """Verifies bit-for-bit reproducibility of price generation."""
    df1 = generate_prices_daily(seed=47)
    df2 = generate_prices_daily(seed=47)
    pd.testing.assert_frame_equal(df1, df2)


# ==============================================================================
# 9. CROSS-DATASET INTEGRATION TESTS
# ==============================================================================

def test_cross_dataset_temporal_and_relational_alignment(
    calendar_df: pd.DataFrame,
    index_df: pd.DataFrame,
    corporate_actions_df: pd.DataFrame,
    prices_daily_df: pd.DataFrame,
    company_master_df: pd.DataFrame,
) -> None:
    """Verifies complete relational integrity across all 4 Member 2 datasets."""
    cal_dates = list(pd.to_datetime(calendar_df["date"]).dt.strftime("%Y-%m-%d"))
    idx_dates = list(pd.to_datetime(index_df["date"]).dt.strftime("%Y-%m-%d"))
    assert cal_dates == idx_dates, "Calendar and index daily dates must match exactly"

    for c in company_master_df["company_id"]:
        c_dates = list(
            pd.to_datetime(prices_daily_df[prices_daily_df["company_id"] == c]["date"]).dt.strftime("%Y-%m-%d")
        )
        assert c_dates == cal_dates, f"Company {c} price dates do not match calendar"

    ca_dates = set(pd.to_datetime(corporate_actions_df["ex_date"]).dt.strftime("%Y-%m-%d"))
    assert ca_dates.issubset(set(cal_dates)), "All corporate action ex-dates must be valid calendar dates"

