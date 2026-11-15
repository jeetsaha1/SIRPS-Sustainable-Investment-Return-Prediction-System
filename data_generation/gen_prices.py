"""Synthetic Daily Company Price & OHLCV Generator for SIRPS Member 2.

Generates data/prices_daily.csv for all 10 fictional green companies across
the 2020–2025 simulation window (1,526 calendar sessions = exactly 15,260 rows).

Econometric & Research Architecture:
1. Multi-Factor Return Process:
   - Market Benchmark Shock: log returns from index_daily.csv scaled by company-specific
     annual beta (beta_{i,y}) from latent_profiles.csv.
   - Intra-Sector Common Shock: shared across companies in each of the 5 green sectors
     (rho_sec = 0.35, generating realistic positive within-sector co-movement).
   - Idiosyncratic Shocks: company-specific annual idiosyncratic volatility
     (sigma_{i,y}^{idio} / sqrt(252)) from latent_profiles.csv.
   - Annual Base Drift: calibrated with Jensen's inequality variance adjustment
     (mu_{daily} = (alpha_{i,y} - 0.5 * sigma_{total}^2) / 252) to prevent exponential runaway.
   - Planted Weak Signals (scaled causally to daily magnitude / 63):
     * Momentum (0.02): cross-sectionally standardized 63-day trailing return.
     * Quality (0.02): cross-sectionally standardized annual latent quality score.
     * ESG (0.04): cross-sectionally standardized annual latent ESG score, strictly active
       only after July 15 of each filing year (zero before 2020-07-15).
2. Corporate Actions & Split Adjustment:
   - Split factor: split_factor_i(t) = prod_{ex_date > t} split_ratio.
   - Raw close: close_t = round(adj_close_t * split_factor_i(t), 2).
   - Adjusted close remains strictly continuous across splits (0% false economic drop).
   - Splits on C03 (2022-08-11, 2:1) and C06 (2023-04-13, 2:1) produce exact mechanical
     ~50% raw price changes, with zero unexplained raw jumps > 25% on other dates.
3. Realistic OHLCV Dynamics:
   - Open: opens around previous split-adjusted close with small stochastic overnight gap.
   - High & Low: high >= max(open, close), low <= min(open, close), low > 0.
   - Volume: log-normal positive integer volume elastic to absolute daily return
     and adjusted for share splits.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Model constants
RHO_SECTOR: float = 0.35
SQRT_1_MINUS_RHO2: float = float(np.sqrt(1.0 - RHO_SECTOR**2))
ANNUAL_TRADING_DAYS: float = 252.0
DT: float = 1.0 / ANNUAL_TRADING_DAYS
BENCHMARK_ANNUAL_DRIFT: float = 0.113  # Annualized sample mean of NIFTY benchmark log return

# Base Average Daily Volume (ADV) in shares by company
BASE_ADV: Dict[str, int] = {
    "C01": 450_000,
    "C02": 260_000,
    "C03": 350_000,
    "C04": 300_000,
    "C05": 520_000,
    "C06": 620_000,
    "C07": 280_000,
    "C08": 220_000,
    "C09": 160_000,
    "C10": 240_000,
}


def _compute_split_factors(
    company_id: str,
    trading_dates: List[str],
    splits_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute split factor for each trading date.

    Convention:
        split_factor(t) = product of split ratios whose ex-dates occur strictly after date t.
    """
    comp_splits = splits_df[
        (splits_df["company_id"] == company_id) & (splits_df["action_type"] == "split")
    ]
    split_records = [
        (pd.to_datetime(row["ex_date"]).strftime("%Y-%m-%d"), float(row["ratio_or_amount"]))
        for _, row in comp_splits.iterrows()
    ]

    factors = {}
    for d_str in trading_dates:
        f = 1.0
        for ex_date_str, ratio in split_records:
            if ex_date_str > d_str:
                f *= ratio
        factors[d_str] = f
    return factors


def generate_prices_daily(
    calendar_df: Optional[pd.DataFrame] = None,
    company_master_df: Optional[pd.DataFrame] = None,
    latent_df: Optional[pd.DataFrame] = None,
    index_df: Optional[pd.DataFrame] = None,
    actions_df: Optional[pd.DataFrame] = None,
    seed: int = 47,
) -> pd.DataFrame:
    """Generate daily OHLCV and total-return adjusted close for all 10 companies.

    Args:
        calendar_df: Optional preloaded trading calendar DataFrame.
        company_master_df: Optional preloaded company master DataFrame.
        latent_df: Optional preloaded latent profiles DataFrame.
        index_df: Optional preloaded index daily DataFrame.
        actions_df: Optional preloaded corporate actions DataFrame.
        seed: Random seed for deterministic multi-stream generation.

    Returns:
        DataFrame conforming to the 'prices_daily' schema (15,260 rows).
    """
    if calendar_df is None:
        calendar_df = load("trading_calendar")
    if company_master_df is None:
        company_master_df = load("company_master")
    if latent_df is None:
        latent_df = load("latent_profiles")
    if index_df is None:
        index_df = load("index_daily")
    if actions_df is None:
        actions_df = load("corporate_actions")

    # Standardize dates
    dates_series = pd.to_datetime(calendar_df["date"])
    date_strs: List[str] = dates_series.dt.strftime("%Y-%m-%d").tolist()
    years: List[int] = dates_series.dt.year.tolist()
    n_dates = len(date_strs)

    # Market returns from NIFTY close
    nifty_close = index_df["nifty_close"].values
    market_log_ret = np.zeros(n_dates, dtype=np.float64)
    market_log_ret[1:] = np.diff(np.log(nifty_close))

    # Company reference info
    companies = company_master_df["company_id"].tolist()
    sectors = company_master_df.set_index("company_id")["sector"].to_dict()
    base_prices = company_master_df.set_index("company_id")["base_price"].to_dict()
    unique_sectors = sorted(list(set(sectors.values())))

    lp_indexed = latent_df.set_index(["company_id", "year"])

    # Independent RNG streams for distinct stochastic components
    rng_sec = np.random.default_rng(seed)
    sector_shocks = {s: rng_sec.standard_normal(n_dates) for s in unique_sectors}

    company_shocks: Dict[str, np.ndarray] = {}
    gap_shocks: Dict[str, np.ndarray] = {}
    hl_shocks: Dict[str, np.ndarray] = {}
    vol_shocks: Dict[str, np.ndarray] = {}

    for i, comp_id in enumerate(companies):
        rng_comp = np.random.default_rng(seed + 100 + i)
        company_shocks[comp_id] = rng_comp.standard_normal(n_dates)

        rng_gap = np.random.default_rng(seed + 200 + i)
        gap_shocks[comp_id] = rng_gap.normal(0.0, 0.003, size=n_dates)

        rng_hl = np.random.default_rng(seed + 300 + i)
        hl_shocks[comp_id] = np.abs(rng_hl.normal(0.0, 0.005, size=(n_dates, 2)))

        rng_vol = np.random.default_rng(seed + 400 + i)
        vol_shocks[comp_id] = rng_vol.normal(0.0, 0.25, size=n_dates)

    # Pre-compute cross-sectional standardized quality by year
    qual_std_by_year: Dict[int, Dict[str, float]] = {}
    for yr in set(years):
        q_raw = {c: float(lp_indexed.loc[(c, yr), "quality_latent_score"]) for c in companies}
        m_q = float(np.mean(list(q_raw.values())))
        s_q = float(np.std(list(q_raw.values()))) + 1e-6
        qual_std_by_year[yr] = {c: (q_raw[c] - m_q) / s_q for c in companies}

    # Pre-compute split factors for each company
    split_factors: Dict[str, Dict[str, float]] = {
        c: _compute_split_factors(c, date_strs, actions_df) for c in companies
    }

    # Simulate adjusted close day by day
    adj_close: Dict[str, np.ndarray] = {
        c: np.zeros(n_dates, dtype=np.float64) for c in companies
    }
    for c in companies:
        adj_close[c][0] = float(base_prices[c])

    # Simulation loop across trading days
    for t in range(1, n_dates):
        d_str = date_strs[t]
        yr = years[t]

        # 1. Momentum Signal: 63-day trailing return standardized cross-sectionally
        lookback = min(t, 63)
        if lookback >= 15:
            past_rets = {
                c: float(np.log(adj_close[c][t - 1] / adj_close[c][t - lookback]))
                for c in companies
            }
            m_mom = float(np.mean(list(past_rets.values())))
            s_mom = float(np.std(list(past_rets.values()))) + 1e-6
            mom_sig = {
                c: float(np.clip((past_rets[c] - m_mom) / s_mom, -2.0, 2.0))
                for c in companies
            }
        else:
            mom_sig = {c: 0.0 for c in companies}

        # 2. ESG Signal: strictly active only post July 15 of filing year
        if d_str < "2020-07-15":
            esg_sig = {c: 0.0 for c in companies}
        else:
            disc_yr = yr if d_str >= f"{yr}-07-15" else yr - 1
            disc_yr = max(2020, min(2025, disc_yr))
            e_raw = {
                c: float(lp_indexed.loc[(c, disc_yr), "esg_latent_score"])
                for c in companies
            }
            m_e = float(np.mean(list(e_raw.values())))
            s_e = float(np.std(list(e_raw.values()))) + 1e-6
            esg_sig = {
                c: float(np.clip((e_raw[c] - m_e) / s_e, -2.0, 2.0))
                for c in companies
            }

        # Market innovation
        mkt_shock = market_log_ret[t] - BENCHMARK_ANNUAL_DRIFT * DT

        # Update each company's adjusted close
        for c in companies:
            beta = float(lp_indexed.loc[(c, yr), "beta"])
            vol_idio = float(lp_indexed.loc[(c, yr), "idio_volatility"])
            alpha = float(lp_indexed.loc[(c, yr), "base_drift"])

            # Drift with Jensen's inequality correction
            vol_tot = float(np.sqrt((beta * 0.16) ** 2 + vol_idio**2))
            drift = (alpha - 0.5 * vol_tot**2) * DT

            # Factor loadings
            mkt_comp = beta * mkt_shock
            sec = sectors[c]
            xi = SQRT_1_MINUS_RHO2 * company_shocks[c][t] + RHO_SECTOR * sector_shocks[sec][t]
            idio_comp = (vol_idio / np.sqrt(ANNUAL_TRADING_DAYS)) * xi

            # Planted weak signals scaled causally to daily magnitude / 63
            sig_mom = config.MOMENTUM_SIGNAL * mom_sig[c]
            sig_qual = config.QUALITY_SIGNAL * qual_std_by_year[yr][c]
            sig_esg = config.ESG_SIGNAL * esg_sig[c]
            signal_comp = (sig_mom + sig_qual + sig_esg) / float(config.PREDICTION_HORIZON_TRADING_DAYS)

            d_log_p = drift + mkt_comp + idio_comp + signal_comp
            adj_close[c][t] = adj_close[c][t - 1] * np.exp(d_log_p)

    # Construct OHLCV records for all companies
    records = []
    for c in companies:
        adv = BASE_ADV[c]
        c_adj = adj_close[c]
        sf_dict = split_factors[c]

        for t in range(n_dates):
            d_str = date_strs[t]
            sf_t = sf_dict[d_str]

            adj_p = float(c_adj[t])
            raw_c = round(adj_p * sf_t, 2)

            if t == 0:
                raw_o = raw_c
                d_log_p_abs = 0.0
            else:
                # Open reference incorporates overnight split factor adjustment
                open_ref = c_adj[t - 1] * sf_t
                raw_o = round(open_ref * np.exp(gap_shocks[c][t]), 2)
                d_log_p_abs = abs(float(np.log(c_adj[t] / c_adj[t - 1])))

            # High and Low bounds
            m_high = max(raw_o, raw_c)
            m_low = min(raw_o, raw_c)

            u_h = hl_shocks[c][t, 0]
            u_l = hl_shocks[c][t, 1]

            raw_h = max(round(m_high * (1.0 + u_h), 2), m_high)
            raw_l = min(round(m_low * (1.0 - u_l), 2), m_low)
            raw_l = max(raw_l, 0.01)

            # Volume generation with return elasticity and split adjustment
            # log_vol = log(ADV) + 6.0 * |r| + noise
            # Pre-split volume is divided by split_factor so turnover remains consistent
            log_v = np.log(adv) + 6.0 * d_log_p_abs + vol_shocks[c][t]
            vol = int(max(100, round(np.exp(log_v) / sf_t)))

            records.append(
                {
                    "company_id": c,
                    "date": d_str,
                    "open": float(raw_o),
                    "high": float(raw_h),
                    "low": float(raw_l),
                    "close": float(raw_c),
                    "adj_close": round(adj_p, 2),
                    "volume": int(vol),
                    "data_source": config.DATA_SOURCE,
                }
            )

    df_prices = pd.DataFrame(records)

    # Sort deterministically by company_id and date
    df_prices = df_prices.sort_values(by=["company_id", "date"]).reset_index(drop=True)

    # Ensure schema column ordering
    cols = [
        "company_id",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "data_source",
    ]
    df_prices = df_prices[cols]

    # Validate against central schema
    validate_dataset_schema(df_prices, "prices_daily")

    return df_prices


def main() -> None:
    """Generate and persist production prices daily dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Starting daily company prices generation...")

    df_prices = generate_prices_daily()
    output_path = save(df_prices, "prices_daily")

    logger.info("Successfully generated %d price rows.", len(df_prices))
    logger.info("Saved dataset to: %s", output_path)

    # Diagnostic summary
    print("\nPrices Daily Summary:")
    print(f"Total rows:        {len(df_prices)}")
    print(f"Total companies:   {df_prices['company_id'].nunique()}")
    print(f"Dates per company: {len(df_prices) // df_prices['company_id'].nunique()}")
    print(f"Date range:        {df_prices['date'].min()} to {df_prices['date'].max()}")
    print(f"Min low price:     ₹{df_prices['low'].min():.2f}")
    print(f"Max high price:    ₹{df_prices['high'].max():.2f}")
    print(f"Min volume:        {df_prices['volume'].min()}")
    print(f"Max volume:        {df_prices['volume'].max()}")


if __name__ == "__main__":
    main()
