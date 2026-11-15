"""SIRPS Latent Factor Statistical Diagnostics (Member 1).

Computes comprehensive statistical and econometric diagnostics for the generated
latent profiles:
1. Theoretical target correlation matrix vs Empirical sample correlation matrix
2. Absolute difference matrix and maximum sampling error
3. Empirical sample covariance matrix
4. First and second moments (mean, std) for all latent scores and derived parameters
5. Min/max range and bounds compliance verification
6. Economic coherence metrics (corr(Risk, Beta), corr(Risk, IdioVol), corr(Quality, Beta))
7. Persists structured JSON to results/member_1/latent_factor_diagnostics.json
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_generation.config as config
from common.io import load

RESULTS_DIR: Path = PROJECT_ROOT / "results" / "member_1"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

def compute_latent_diagnostics(save_output: bool = True) -> Dict[str, Any]:
    """Computes all mathematical and statistical metrics for data/latent_profiles.csv."""
    df = load("latent_profiles", use_mock=False)
    
    score_cols = ["esg_latent_score", "quality_latent_score", "risk_latent_score", "growth_latent_score"]
    param_cols = ["beta", "idio_volatility", "base_drift"]
    
    target_corr = config.TARGET_CORRELATION_MATRIX
    emp_corr_df = df[score_cols].corr()
    emp_corr = emp_corr_df.values
    diff_matrix = emp_corr - target_corr
    emp_cov_df = df[score_cols].cov()
    
    # Moments & Bounds
    summary: Dict[str, Dict[str, Any]] = {}
    for col in score_cols:
        min_v = float(df[col].min())
        max_v = float(df[col].max())
        summary[col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(min_v, 4),
            "max": round(max_v, 4),
            "bounds": list(config.LATENT_SCORE_BOUNDS),
            "in_bounds": bool(min_v >= config.LATENT_SCORE_BOUNDS[0] and max_v <= config.LATENT_SCORE_BOUNDS[1])
        }
        
    for col, bounds in [
        ("beta", config.BETA_BOUNDS),
        ("idio_volatility", config.IDIO_VOLATILITY_BOUNDS),
        ("base_drift", config.BASE_DRIFT_BOUNDS)
    ]:
        min_v = float(df[col].min())
        max_v = float(df[col].max())
        summary[col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(min_v, 4),
            "max": round(max_v, 4),
            "bounds": list(bounds),
            "in_bounds": bool(min_v >= bounds[0] and max_v <= bounds[1])
        }
        
    # Economic Coherence Correlations
    corr_risk_beta = float(df["risk_latent_score"].corr(df["beta"]))
    corr_risk_idio = float(df["risk_latent_score"].corr(df["idio_volatility"]))
    corr_qual_beta = float(df["quality_latent_score"].corr(df["beta"]))
    
    diagnostics = {
        "metadata": {
            "dataset": "data/latent_profiles.csv",
            "n_observations": len(df),
            "n_companies": int(df["company_id"].nunique()),
            "years": sorted(df["year"].unique().tolist()),
            "master_seed": config.MASTER_SEED,
            "latent_seed": config.SEED_STREAMS["latent"],
            "provenance": config.DATA_SOURCE
        },
        "correlation_analysis": {
            "theoretical_target_matrix": target_corr.tolist(),
            "empirical_sample_matrix": [[round(float(v), 4) for v in row] for row in emp_corr],
            "absolute_difference_matrix": [[round(float(abs(v)), 4) for v in row] for row in diff_matrix],
            "max_absolute_error": round(float(np.max(np.abs(diff_matrix))), 4),
            "sample_size": len(df),
            "cross_sectional_units": int(df["company_id"].nunique()),
            "temporal_periods": int(df["year"].nunique()),
            "statistical_interpretation": {
                "heuristic_sampling_scale": round(float(1.0 / np.sqrt(len(df))), 4),
                "dependence_limitations": (
                    "Observations are not independent and identically distributed (i.i.d.). "
                    "The panel exhibits AR(1) temporal persistence (phi=0.85-0.90) and repeated "
                    "observations across 10 firms. Therefore, 1/sqrt(N) is merely an order-of-magnitude "
                    "sampling heuristic, not an asymptotic standard error for hypothesis testing. "
                    "Empirical differences reflect finite-sample realisation rather than model misspecification."
                )
            }
        },
        "theoretical_target_correlation_matrix": {
            "factors": score_cols,
            "matrix": target_corr.tolist(),
            "eigenvalues": np.linalg.eigvalsh(target_corr).tolist(),
            "is_positive_definite": bool(np.all(np.linalg.eigvalsh(target_corr) > 0))
        },
        "empirical_sample_correlation_matrix": {
            "factors": score_cols,
            "matrix": [[round(float(v), 4) for v in row] for row in emp_corr]
        },
        "correlation_difference_matrix_empirical_minus_target": {
            "matrix": [[round(float(v), 4) for v in row] for row in diff_matrix],
            "max_absolute_error": round(float(np.max(np.abs(diff_matrix))), 4),
            "heuristic_sampling_scale": round(float(1.0 / np.sqrt(len(df))), 4)
        },
        "empirical_sample_covariance_matrix": {
            "factors": score_cols,
            "matrix": [[round(float(v), 4) for v in row] for row in emp_cov_df.values]
        },
        "factor_summary_and_bounds": summary,
        "economic_coherence": {
            "corr_risk_latent_score_vs_beta": round(corr_risk_beta, 4),
            "corr_risk_latent_score_vs_idio_volatility": round(corr_risk_idio, 4),
            "corr_quality_latent_score_vs_beta": round(corr_qual_beta, 4),
            "coherence_check_passed": bool(corr_risk_beta > 0.50 and corr_risk_idio > 0.50 and corr_qual_beta < -0.20)
        }
    }
    
    if save_output:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = RESULTS_DIR / "latent_factor_diagnostics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        print(f"✓ Saved latent diagnostics JSON: {json_path}")
        
    return diagnostics

if __name__ == "__main__":
    compute_latent_diagnostics()
