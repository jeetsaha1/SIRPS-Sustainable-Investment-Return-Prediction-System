"""SIRPS Synthetic Market Benchmark & Volatility Index Generator (Member 2 — Market & Trading Data).

Generates the daily synthetic NIFTY 50 Benchmark Index, NIFTY Total Return Index (TRI),
and India VIX volatility benchmark across the 2020–2025 simulation window.

Econometric & Research Specifications:
- Benchmark Base: Initial close I_0 = 12,000.00 on 2020-01-01.
- Total Return Index (TRI): Implements continuous dividend compounding:
      I_t^TRI = I_t * exp(0.013 * d_t / 365.25)
  with a realistic 1.30% annual dividend reinvestment yield, ensuring I_t^TRI >= I_t for all t.
- Geometric Brownian Motion with Regime Switching:
      log(I_t / I_{t-1}) = (mu_t - 0.5 * sigma_t^2) * dt + sigma_t * sqrt(dt) * z_{m,t}
  using 5-day EWMA smoothed drift (mu_t) and volatility (sigma_t) parameters across realistic
  historical-plus market phases:
    * 2020: Pre-COVID normal, COVID crash to ~9,167 trough, rapid recovery to ~13,440.
    * 2021: Post-COVID Bull expansion (return > +15.0%, closing ~16,190).
    * 2022 H1: Global tightening & inflation shock correction (drawdown >= 12.0%).
    * 2022 H2: Consolidation and recovery to ~16,400.
    * 2023–Q3 2024: Secular Bull rally peaking around ~25,000.
    * Q4 2024–Q1 2025: Foreign capital outflow correction (drawdown >= 12.0%).
    * Mid–Late 2025: Structural recovery closing ~23,670.
- Synthetic India VIX Model (Correlated-Shock Specification):
  Continuous log-Ornstein-Uhlenbeck process with exact correlated standard-normal innovations:
      eps_v,t = -rho_lev * z_{m,t} + sqrt(1 - rho_lev^2) * z_{v,t}
      x_t = x_{t-1} + kappa * (log(theta_{R_t}) - x_{t-1}) + sigma_{log V} * eps_v,t
      V_t = exp(x_t)
  where:
    * kappa = 0.08 (Daily mean reversion speed)
    * rho_lev = 0.35 (Theoretical correlation between market and volatility innovations: -0.35)
    * sigma_{log V} = 0.04 (Volatility of log-volatility innovation)
    * Unit-variance standardized innovation: Var(eps_v,t) = rho_lev^2 + (1 - rho_lev^2) = 1.0.
    * Exact theoretical correlation: Corr(z_{m,t}, eps_v,t) = -rho_lev = -0.35.
    * Bull long-run target: theta_Bull = 13.5
    * Neutral long-run target: theta_Neutral = 17.0
    * Correction/Bear long-run target: theta_Correction = 28.5
    * Naturally strictly positive by construction via exp(x_t) with zero arbitrary hard clipping.
    * Smoothed using a 5-day EWMA to reflect published market indices.

Data Contract:
- Target File: data/index_daily.csv
- Primary Key: ['date']
- Foreign Key: date -> trading_calendar.date
- Columns: ['date', 'nifty_close', 'nifty_tri', 'india_vix', 'data_source']
- Provenance: 'SYNTHETIC_DEMO'
- Row Count: Exactly 1,526 rows matching trading calendar dates 1-to-1.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

# Add repository root to path if running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_generation.config as config
from common.schemas import SCHEMAS, validate_dataset_schema
from common.io import load, save

INITIAL_NIFTY_CLOSE: float = 12000.00
ANNUAL_DIVIDEND_YIELD: float = 0.013  # 1.30% continuous reinvested yield
INITIAL_VIX: float = 13.80

# Daily time step in years (252 trading sessions per annum)
DT: float = 1.0 / 252.0
SQRT_DT: float = np.sqrt(DT)

# Ornstein-Uhlenbeck VIX calibrated parameters (Empirically Calibrated Leverage Specification)
OU_KAPPA: float = 0.08       # Daily mean reversion speed
OU_SIGMA_V: float = 0.04     # Volatility of log-volatility innovation
OU_RHO_LEV: float = 0.35     # Dimensionless equity market leverage parameter

# Regime-dependent long-run VIX anchors (theta)
VIX_THETA_BULL: float = 13.5
VIX_THETA_NEUTRAL: float = 17.0
VIX_THETA_CORRECTION: float = 28.5


def get_regime_targets(d_str: str) -> Tuple[float, float, float]:
    """Returns regime target parameters (mu, sigma, vix_theta) for a given calendar date."""
    if d_str < "2020-02-20":
        # Pre-COVID normal market (Neutral)
        return 0.05, 0.14, VIX_THETA_NEUTRAL
    elif d_str <= "2020-03-24":
        # COVID crash phase (Bear / Correction)
        return -2.60, 0.55, VIX_THETA_CORRECTION
    elif d_str <= "2020-12-31":
        # COVID recovery phase (Neutral)
        return 0.25, 0.22, VIX_THETA_NEUTRAL
    elif d_str <= "2021-12-31":
        # 2021 Post-COVID strong bull expansion (Bull)
        return 0.52, 0.15, VIX_THETA_BULL
    elif d_str <= "2022-06-17":
        # 2022 H1 Inflation & interest rate tightening correction (Correction)
        return -0.22, 0.22, VIX_THETA_CORRECTION
    elif d_str <= "2022-12-31":
        # 2022 H2 Consolidation & rebound (Neutral)
        return 0.15, 0.16, VIX_THETA_NEUTRAL
    elif d_str <= "2024-09-27":
        # 2023 to Q3 2024 Secular bull rally (Bull)
        return 0.08, 0.13, VIX_THETA_BULL
    elif d_str <= "2025-02-28":
        # Q4 2024 to Q1 2025 Correction & consolidation (Correction)
        return -0.20, 0.20, VIX_THETA_CORRECTION
    else:
        # Mid-Late 2025 Structural recovery (Neutral)
        return 0.28, 0.15, VIX_THETA_NEUTRAL


def generate_index_daily(
    calendar_df: Optional[pd.DataFrame] = None,
    seed: int = config.SEED_STREAMS["index"]
) -> pd.DataFrame:
    """Generates the daily synthetic NIFTY benchmark index and India VIX dataset.
    
    Args:
        calendar_df: Optional pre-loaded trading calendar dataframe. If None,
                     loads from data/trading_calendar.csv.
        seed: Random seed for deterministic simulation. Default is SEED_STREAMS['index'] (46).
        
    Returns:
        pd.DataFrame: Formatted and validated index daily dataset adhering to
                      the SCHEMAS['index_daily'] contract.
    """
    if calendar_df is None:
        calendar_df = load("trading_calendar")
        
    # Guarantee consistent date string format and chronological ordering
    cal_dates_series = pd.to_datetime(calendar_df["date"])
    date_strings = [d.strftime("%Y-%m-%d") for d in cal_dates_series]
    N = len(date_strings)
    
    if N != 1526:
        raise ValueError(f"Expected 1,526 trading calendar dates, received {N}.")
        
    # 1. Establish regime target series and smooth transitions
    regimes = [get_regime_targets(d_str) for d_str in date_strings]
    df_reg = pd.DataFrame(regimes, columns=["mu", "sigma", "vix_theta"])
    smoothed_regimes = df_reg.ewm(span=5, adjust=False).mean()
    
    # 2. Seeded random innovations
    rng = np.random.default_rng(seed)
    z_m = rng.standard_normal(N)  # Market log-return shock
    z_v = rng.standard_normal(N)  # VIX idiosyncratic shock
    
    # 3. Simulate NIFTY Price Index (Geometric Brownian Motion)
    close = np.zeros(N, dtype=np.float64)
    close[0] = INITIAL_NIFTY_CLOSE
    
    for t in range(1, N):
        mu_t = smoothed_regimes.loc[t, "mu"]
        sigma_t = smoothed_regimes.loc[t, "sigma"]
        drift_comp = (mu_t - 0.5 * (sigma_t ** 2)) * DT
        shock_comp = sigma_t * SQRT_DT * z_m[t]
        close[t] = close[t - 1] * np.exp(drift_comp + shock_comp)
        
    # 4. Compute NIFTY Total Return Index (TRI)
    base_timestamp = cal_dates_series.iloc[0]
    elapsed_calendar_days = np.array(
        [(d - base_timestamp).days for d in cal_dates_series],
        dtype=np.float64
    )
    tri = close * np.exp(ANNUAL_DIVIDEND_YIELD * elapsed_calendar_days / 365.25)
    
    # 5. Simulate India VIX (Log-OU process with correlated standard normal innovation)
    # Standardized innovation: eps_v = -rho_lev * z_m[t] + sqrt(1 - rho_lev^2) * z_v[t]
    # Unit variance: Var(eps_v) = 1.0; Theoretical correlation: Corr(z_m, eps_v) = -rho_lev = -0.35
    log_vix = np.zeros(N, dtype=np.float64)
    log_vix[0] = np.log(INITIAL_VIX)
    w_orth = np.sqrt(1.0 - (OU_RHO_LEV ** 2))
    
    for t in range(1, N):
        theta_t = smoothed_regimes.loc[t, "vix_theta"]
        eps_v = -OU_RHO_LEV * z_m[t] + w_orth * z_v[t]
        d_log_v = (
            OU_KAPPA * (np.log(theta_t) - log_vix[t - 1])
            + OU_SIGMA_V * eps_v
        )
        log_vix[t] = log_vix[t - 1] + d_log_v
        
    # Strictly positive exponential construction V_t = exp(x_t)
    # Smoothed with a 5-day EWMA without arbitrary hard clipping
    raw_vix = np.exp(log_vix)
    smoothed_vix = pd.Series(raw_vix).ewm(span=5, adjust=False).mean().values
    
    # 6. Format final DataFrame adhering to central schema
    df = pd.DataFrame({
        "date": date_strings,
        "nifty_close": np.round(close, 2),
        "nifty_tri": np.round(tri, 2),
        "india_vix": np.round(smoothed_vix, 2),
        "data_source": config.DATA_SOURCE,
    })
    
    # Enforce strictly registered column order
    df = df[["date", "nifty_close", "nifty_tri", "india_vix", "data_source"]].copy()
    
    # Validate against central schema
    validate_dataset_schema(df, "index_daily")
    
    return df


def main() -> None:
    """Executes index generation, statistical sanity checks, and persistence."""
    print("=" * 80)
    print("SIRPS BENCHMARK INDEX & VOLATILITY GENERATOR (MEMBER 2)")
    print(f"Seed Stream: {config.SEED_STREAMS['index']} | Benchmark Initial Close: {INITIAL_NIFTY_CLOSE:,.2f}")
    print(f"VIX Parameters: kappa={OU_KAPPA}, rho_lev={OU_RHO_LEV} (scaled leverage), sigma_v={OU_SIGMA_V}")
    print(f"VIX Long-Run Targets: Bull={VIX_THETA_BULL}, Neutral={VIX_THETA_NEUTRAL}, Correction={VIX_THETA_CORRECTION}")
    print("=" * 80)
    
    calendar_df = load("trading_calendar")
    df = generate_index_daily(calendar_df=calendar_df, seed=config.SEED_STREAMS["index"])
    
    saved_path = save(df, "index_daily")
    print(f"Successfully generated and saved index daily dataset to: {saved_path}")
    print(f"Total Observations: {len(df):,}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    
    # Statistical Diagnostics
    df_diag = df.copy()
    df_diag["year"] = pd.to_datetime(df_diag["date"]).dt.year
    df_diag["dt"] = pd.to_datetime(df_diag["date"])
    
    print("\n--- Year-End Closes and Annual Returns ---")
    year_ends = df_diag.groupby("year").last()
    prev_close = INITIAL_NIFTY_CLOSE
    for y, row in year_ends.iterrows():
        y_close = row["nifty_close"]
        y_tri = row["nifty_tri"]
        ann_ret = (y_close - prev_close) / prev_close
        print(f"  FY End {y}: Close = {y_close:,.2f} | TRI = {y_tri:,.2f} | Price Return = {ann_ret * 100:+.2f}%")
        prev_close = y_close
        
    # Check 2021 return target (> 15%)
    p_2020 = df_diag[df_diag["year"] == 2020]["nifty_close"].iloc[-1]
    p_2021 = df_diag[df_diag["year"] == 2021]["nifty_close"].iloc[-1]
    ret_2021 = (p_2021 - p_2020) / p_2020
    print(f"\n2021 Return Verification: {ret_2021 * 100:+.2f}% (Target: > +15.00%) -> {'PASSED' if ret_2021 > 0.15 else 'FAILED'}")
    
    # Check 2022 Drawdown
    sub_2022 = df_diag[(df_diag["date"] >= "2021-10-01") & (df_diag["date"] <= "2022-07-01")]
    peak_2022 = sub_2022["nifty_close"].cummax()
    dd_2022 = ((sub_2022["nifty_close"] - peak_2022) / peak_2022).min()
    print(f"2022 H1 Max Drawdown: {dd_2022 * 100:.2f}% (Target: <= -12.00%) -> {'PASSED' if dd_2022 <= -0.12 else 'FAILED'}")
    
    # Check 2024-2025 Drawdown
    sub_24_25 = df_diag[(df_diag["date"] >= "2024-09-01") & (df_diag["date"] <= "2025-04-01")]
    peak_24_25 = sub_24_25["nifty_close"].cummax()
    dd_24_25 = ((sub_24_25["nifty_close"] - peak_24_25) / peak_24_25).min()
    print(f"2024–2025 Max Drawdown: {dd_24_25 * 100:.2f}% (Target: <= -12.00%) -> {'PASSED' if dd_24_25 <= -0.12 else 'FAILED'}")
    
    # VIX Diagnostics
    vix = df["india_vix"]
    print(f"\nVIX Summary Statistics (Calibrated Leverage Specification):")
    print(f"  Full Sample: Min = {vix.min():.2f}, Median = {vix.median():.2f}, Mean = {vix.mean():.2f}, Max = {vix.max():.2f}, Std = {vix.std():.2f}")
    
    # Regime Masks
    bull_2021_mask = (df_diag["dt"] >= "2021-01-01") & (df_diag["dt"] <= "2021-12-31")
    bull_23_24_mask = (df_diag["dt"] >= "2023-01-01") & (df_diag["dt"] <= "2024-09-27")
    corr_22_mask = (df_diag["dt"] >= "2022-01-01") & (df_diag["dt"] <= "2022-06-17")
    corr_24_25_mask = (df_diag["dt"] >= "2024-09-28") & (df_diag["dt"] <= "2025-02-28")
    
    print("\n--- Regime-Specific VIX Breakdown ---")
    print(f"  2021 Bull (theta={VIX_THETA_BULL}): Mean = {df_diag.loc[bull_2021_mask, 'india_vix'].mean():.2f}, Median = {df_diag.loc[bull_2021_mask, 'india_vix'].median():.2f}")
    print(f"  2023–Q3 2024 Secular Bull (theta={VIX_THETA_BULL}): Mean = {df_diag.loc[bull_23_24_mask, 'india_vix'].mean():.2f}, Median = {df_diag.loc[bull_23_24_mask, 'india_vix'].median():.2f}")
    print(f"  2022 H1 Correction (theta={VIX_THETA_CORRECTION}): Mean = {df_diag.loc[corr_22_mask, 'india_vix'].mean():.2f}, Median = {df_diag.loc[corr_22_mask, 'india_vix'].median():.2f}")
    print(f"  Q4 2024–Q1 2025 Correction (theta={VIX_THETA_CORRECTION}): Mean = {df_diag.loc[corr_24_25_mask, 'india_vix'].mean():.2f}, Median = {df_diag.loc[corr_24_25_mask, 'india_vix'].median():.2f}")
    
    print("\n--- Annual VIX Breakdown ---")
    for y in range(config.START_YEAR, config.END_YEAR + 1):
        sub_y = df_diag[df_diag["year"] == y]["india_vix"]
        print(f"  Year {y}: Min = {sub_y.min():.2f}, Median = {sub_y.median():.2f}, Mean = {sub_y.mean():.2f}, Max = {sub_y.max():.2f}")
        
    # Check Leverage Effect (Negative return/VIX delta correlation)
    rng_diag = np.random.default_rng(config.SEED_STREAMS["index"])
    z_m_diag = rng_diag.standard_normal(len(df))
    z_v_diag = rng_diag.standard_normal(len(df))
    w_orth_diag = np.sqrt(1.0 - (OU_RHO_LEV ** 2))
    eps_v_diag = -OU_RHO_LEV * z_m_diag + w_orth_diag * z_v_diag
    emp_innov_corr = np.corrcoef(z_m_diag, eps_v_diag)[0, 1]
    emp_innov_var = np.var(eps_v_diag)

    daily_returns = np.diff(np.log(df["nifty_close"].values))
    daily_vix_diffs = np.diff(df["india_vix"].values)
    lev_corr = np.corrcoef(daily_returns, daily_vix_diffs)[0, 1]
    
    neg_days_mask = daily_returns < 0
    neg_day_corr = np.corrcoef(daily_returns[neg_days_mask], daily_vix_diffs[neg_days_mask])[0, 1]
    pos_vix_on_down_days = (daily_vix_diffs[neg_days_mask] > 0).mean() * 100
    
    print(f"\n--- Innovation & Leverage Diagnostics (Correlated-Shock Specification) ---")
    print(f"  Theoretical Innovation Corr(z_m, eps_v): {-OU_RHO_LEV:.2f}")
    print(f"  Empirical Standardized Innovation Corr(z_m, eps_v): {emp_innov_corr:.4f}")
    print(f"  Empirical Standardized Innovation Variance: {emp_innov_var:.4f} (Theoretical: 1.0000)")
    print(f"  Post-Smoothing Corr(daily_return, delta_VIX): {lev_corr:.4f}")
    print(f"  Corr on Negative Return Days: {neg_day_corr:.4f}")
    print(f"  Probability of Positive VIX Shock on Market Down-Day: {pos_vix_on_down_days:.1f}%")
    print(f"  Extreme Outliers: < 5.0: {(vix < 5.0).sum()}, > 60.0: {(vix > 60.0).sum()}")
    print(f"  Artificial Boundary Activations: 0 (No hard clipping imposed)")
    
    print("\nSample Records:\n", df.head(3))
    print("\nFinal Records:\n", df.tail(3))
    print("=" * 80)


if __name__ == "__main__":
    main()
