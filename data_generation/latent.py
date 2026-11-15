"""SIRPS Latent Factor Generator (Member 1).

Generates data/latent_profiles.csv containing annual latent factor scores and derived
market parameters across 10 fictional companies × 6 years (2020–2025) = 60 observations.

Features simulated:
- esg_latent_score: ESG performance factor, standardized in [-2, +2]
- quality_latent_score: Fundamental balance sheet and earnings quality, standardized in [-2, +2]
- risk_latent_score: Operational and business uncertainty factor, standardized in [-2, +2]
- growth_latent_score: Market expansion and revenue trajectory, standardized in [-2, +2]
- beta: Systematic market exposure, bounded in [0.40, 1.80]
- idio_volatility: Annualised idiosyncratic volatility, bounded in [0.10, 0.50]
- base_drift: Annualised base drift, bounded in [-0.02, 0.18]

Mathematical Transition Model:
State vector X_t = [ESG_t, Quality_t, Risk_t, Growth_t]^T
Target correlation matrix Sigma is factored via Cholesky decomposition L * L^T = Sigma.
At year t=0 (2020):
  X_0 = mu_sector + sigma_init * (L @ Z_0), where Z_0 ~ N(0, I_4)
For years t=1..5 (2021..2025):
  Innovation eta_t = sigma_innov * (L @ Z_t), where Z_t ~ N(0, I_4)
  X_t = mu_sector + Phi * (X_{t-1} - mu_sector) + eta_t
Scores are clamped to [-2.0, +2.0].

Risk, Beta, and Idiosyncratic Volatility Coherence:
- Beta captures systematic sensitivity. High latent risk increases systematic vulnerability,
  while high quality dampens it.
- Idiosyncratic volatility captures firm-specific dispersion (technology and operational risks).
  High latent risk and high growth uncertainty increase idiosyncratic volatility, while quality dampens it.
- This structural separation prevents double-counting risk across systematic and unsystematic channels.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_generation.config as config
from common.io import load, save

def generate_latent_profiles(
    seed: Optional[int] = None,
    output_mock: bool = True
) -> pd.DataFrame:
    """Generates the 60-row annual latent profiles dataset.
    
    Args:
        seed: Random seed for reproducibility (defaults to config.SEED_STREAMS["latent"]).
        output_mock: Whether to also write a copy to data/mock/latent_profiles.csv.
        
    Returns:
        pd.DataFrame: Validated latent profiles table.
    """
    if seed is None:
        seed = config.SEED_STREAMS["latent"]
        
    rng = np.random.default_rng(seed)
    
    # Load company master to get company IDs and sectors
    cm_df = load("company_master", use_mock=False)
    
    # Verify Cholesky factor of target correlation matrix
    L = config.CHOLESKY_L  # 4x4 lower triangular matrix
    
    # Persistence vector Phi = [esg, qual, risk, growth]
    phi = np.array([
        config.ANNUAL_PERSISTENCE["esg"],
        config.ANNUAL_PERSISTENCE["qual"],
        config.ANNUAL_PERSISTENCE["risk"],
        config.ANNUAL_PERSISTENCE["growth"]
    ], dtype=np.float64)
    
    # Initial standard deviations
    sigma_init = np.array([
        config.INITIAL_STD["esg"],
        config.INITIAL_STD["qual"],
        config.INITIAL_STD["risk"],
        config.INITIAL_STD["growth"]
    ], dtype=np.float64)
    
    # Innovation standard deviations
    sigma_innov = np.array([
        config.INNOVATION_STD["esg"],
        config.INNOVATION_STD["qual"],
        config.INNOVATION_STD["risk"],
        config.INNOVATION_STD["growth"]
    ], dtype=np.float64)
    
    years = config.HISTORICAL_YEARS  # (2020, 2021, 2022, 2023, 2024, 2025)
    rows: List[Dict[str, Any]] = []
    
    for _, company in cm_df.iterrows():
        cid = company["company_id"]
        sector = company["sector"]
        base = config.SECTOR_BASELINES[sector]
        
        # Compute company-specific archetype baseline adjustments
        adj = config.compute_archetype_baseline_adjustments(company.to_dict())
        
        # Company-specific baseline means: sector baseline + archetype offsets
        mu = np.array([
            base["esg"] + adj["delta_esg"],
            base["qual"] + adj["delta_qual"],
            base["risk"] + adj["delta_risk"],
            base["growth"] + adj["delta_growth"]
        ], dtype=np.float64)
        
        # t=0 (2020): Initial state with correlated draw
        z0 = rng.standard_normal(4)
        correlated_draw_0 = L @ z0
        x = mu + sigma_init * correlated_draw_0
        x = np.clip(x, config.LATENT_SCORE_BOUNDS[0], config.LATENT_SCORE_BOUNDS[1])
        
        for t_idx, yr in enumerate(years):
            if t_idx > 0:
                # Correlated annual innovation
                zt = rng.standard_normal(4)
                correlated_draw_t = L @ zt
                innov = sigma_innov * correlated_draw_t
                x = mu + phi * (x - mu) + innov
                x = np.clip(x, config.LATENT_SCORE_BOUNDS[0], config.LATENT_SCORE_BOUNDS[1])
                
            esg_score, qual_score, risk_score, growth_score = x
            
            # Coherent Beta calculation:
            # Baseline + systematic risk loading - quality stabilization + idiosyncratic shock
            eps_beta = rng.normal(0.0, 0.04)
            beta_raw = base["beta"] + 0.12 * risk_score - 0.08 * qual_score + eps_beta
            beta = float(np.clip(beta_raw, config.BETA_BOUNDS[0], config.BETA_BOUNDS[1]))
            
            # Coherent Idiosyncratic Volatility calculation:
            # Baseline + operational risk + growth uncertainty - quality buffer + noise
            eps_idio = rng.normal(0.0, 0.015)
            idio_raw = base["idio_vol"] + 0.04 * risk_score + 0.02 * growth_score - 0.02 * qual_score + eps_idio
            idio_vol = float(np.clip(idio_raw, config.IDIO_VOLATILITY_BOUNDS[0], config.IDIO_VOLATILITY_BOUNDS[1]))
            
            # Coherent Base Drift calculation:
            # Baseline + growth trend + quality premium + small shock
            eps_drift = rng.normal(0.0, 0.005)
            drift_raw = base["base_drift"] + 0.015 * growth_score + 0.010 * qual_score + eps_drift
            base_drift = float(np.clip(drift_raw, config.BASE_DRIFT_BOUNDS[0], config.BASE_DRIFT_BOUNDS[1]))
            
            fy_str = f"FY{yr}-{str(yr + 1)[-2:]}"
            
            rows.append({
                "company_id": cid,
                "fiscal_year": fy_str,
                "year": yr,
                "esg_latent_score": round(float(esg_score), 4),
                "quality_latent_score": round(float(qual_score), 4),
                "risk_latent_score": round(float(risk_score), 4),
                "growth_latent_score": round(float(growth_score), 4),
                "beta": round(beta, 4),
                "idio_volatility": round(idio_vol, 4),
                "base_drift": round(base_drift, 4),
                "profile_version": config.PROFILE_VERSION,
                "data_source": config.DATA_SOURCE,
            })
            
    df = pd.DataFrame(rows)
    
    # Validation checks
    assert len(df) == 60, f"Expected 60 observations (10 companies × 6 years), got {len(df)}"
    assert df["company_id"].nunique() == 10, "Expected 10 unique companies"
    assert (df["esg_latent_score"] >= -2.0).all() and (df["esg_latent_score"] <= 2.0).all(), "ESG score out of [-2, 2]"
    assert (df["quality_latent_score"] >= -2.0).all() and (df["quality_latent_score"] <= 2.0).all(), "Quality score out of [-2, 2]"
    assert (df["risk_latent_score"] >= -2.0).all() and (df["risk_latent_score"] <= 2.0).all(), "Risk score out of [-2, 2]"
    assert (df["growth_latent_score"] >= -2.0).all() and (df["growth_latent_score"] <= 2.0).all(), "Growth score out of [-2, 2]"
    assert (df["beta"] >= config.BETA_BOUNDS[0]).all() and (df["beta"] <= config.BETA_BOUNDS[1]).all(), "Beta out of bounds"
    assert (df["idio_volatility"] >= config.IDIO_VOLATILITY_BOUNDS[0]).all() and (df["idio_volatility"] <= config.IDIO_VOLATILITY_BOUNDS[1]).all(), "Idio vol out of bounds"
    assert (df["base_drift"] >= config.BASE_DRIFT_BOUNDS[0]).all() and (df["base_drift"] <= config.BASE_DRIFT_BOUNDS[1]).all(), "Base drift out of bounds"
    
    # Save dataset
    saved_path = save(df, "latent_profiles", is_mock=False)
    print(f"✓ Successfully generated: {saved_path} ({len(df)} company-year records)")
    
    if output_mock:
        mock_path = save(df, "latent_profiles", is_mock=True)
        print(f"✓ Synchronized mock latent profiles: {mock_path}")
        
    return df

if __name__ == "__main__":
    generate_latent_profiles()
