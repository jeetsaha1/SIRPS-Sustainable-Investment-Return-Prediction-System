"""SIRPS (Sustainable Investment Return Prediction System)
Foundational Configuration & Simulation Parameters.

This module defines all fixed research decisions, random seeds, sector baselines,
distribution parameters, target correlation structures, and planted predictive signals.
"""

from typing import Dict, Tuple, Any
import numpy as np

# ==============================================================================
# 1. FIXED RESEARCH DECISIONS (FROZEN PARAMETERS)
# ==============================================================================

PROJECT_NAME: str = "SIRPS"
PROJECT_LONG_NAME: str = "Sustainable Investment Return Prediction System"
PROJECT_SCOPE: str = "prototype_plus"
MARKET_SCENARIO: str = "historical_plus"
FACTOR_DESIGN: str = "correlated"
DATA_SOURCE: str = "SYNTHETIC_DEMO"
IS_SYNTHETIC: bool = True
UNIVERSE_GROUP: str = "core"
PROFILE_VERSION: str = "v1.0"

# Master Seed and Deterministic Sub-Stream Offsets
MASTER_SEED: int = 42

SEED_STREAMS: Dict[str, int] = {
    "company_master": MASTER_SEED + 1,    # 43
    "latent": MASTER_SEED + 2,            # 44
    "calendar": MASTER_SEED + 3,          # 45
    "index": MASTER_SEED + 4,             # 46
    "prices": MASTER_SEED + 5,            # 47
    "esg": MASTER_SEED + 6,               # 48
    "fundamentals": MASTER_SEED + 7,      # 49
    "regime": MASTER_SEED + 8,            # 50
    "model_dataset": MASTER_SEED + 9,     # 51
    "ml_models": MASTER_SEED + 10,        # 52
}

# Timeline and Sampling Specifications
START_YEAR: int = 2020
END_YEAR: int = 2025
HISTORICAL_YEARS: Tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)
WARMUP_YEARS: Tuple[int, ...] = (2020,)
MODELLING_YEARS: Tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)

APPROX_TRADING_DATES: int = 1500
APPROX_DAILY_OBSERVATIONS: int = 15000
PREDICTION_HORIZON_TRADING_DAYS: int = 63
SAMPLING_INTERVAL_TRADING_DAYS: int = 5

# Centralized Information Availability Lags (Synthetic Research Prototype Assumptions)
# NOTE: These availability parameters are research prototype modelling assumptions designed
# to prevent look-ahead bias in simulation, and are NOT statutory regulatory filing deadlines.
ANNUAL_ESG_DISCLOSURE_AVAILABILITY_MONTH: int = 7       # Approximate mid-July availability
ANNUAL_ESG_DISCLOSURE_AVAILABILITY_DAY: int = 15
QUARTERLY_FUNDAMENTALS_AVAILABILITY_LAG_DAYS: int = 45  # Approximate 45-day reporting lag


# ==============================================================================
# 2. PLANTED RESEARCH SIGNALS
# ==============================================================================
# Expected 63-trading-day excess return contribution per one standard deviation:
MOMENTUM_SIGNAL: float = 0.02
QUALITY_SIGNAL: float = 0.02
ESG_SIGNAL: float = 0.04  # Only active after filing available_date

# ==============================================================================
# 3. SECTOR BASELINES & ARCHETYPE MAPPING (SYNTHETIC GENERATOR PARAMETERS)
# ==============================================================================
# Starting sector baselines as specified in research design:
# ESG, Quality, Beta, Annualised Idiosyncratic Volatility, Annualised Base Drift
SECTOR_BASELINES: Dict[str, Dict[str, float]] = {
    "Renewable Power": {
        "esg": 0.45,
        "qual": 0.10,
        "risk": 0.05,
        "growth": 0.20,
        "beta": 1.05,
        "idio_vol": 0.27,
        "base_drift": 0.085,
    },
    "Water": {
        "esg": 0.55,
        "qual": 0.30,
        "risk": -0.30,
        "growth": 0.00,
        "beta": 0.75,
        "idio_vol": 0.19,
        "base_drift": 0.065,
    },
    "Waste and Bio": {
        "esg": 0.25,
        "qual": -0.05,
        "risk": 0.15,
        "growth": 0.15,
        "beta": 1.10,
        "idio_vol": 0.30,
        "base_drift": 0.090,
    },
    "EV and Batteries": {
        "esg": 0.30,
        "qual": 0.05,
        "risk": 0.35,
        "growth": 0.40,
        "beta": 1.30,
        "idio_vol": 0.34,
        "base_drift": 0.105,
    },
    "Efficiency and Grid": {
        "esg": 0.40,
        "qual": 0.20,
        "risk": -0.10,
        "growth": 0.15,
        "beta": 0.90,
        "idio_vol": 0.23,
        "base_drift": 0.075,
    },
}

# Archetype Characteristic Mappings to Latent Baseline Offsets
ARCHETYPE_GROWTH_PROFILE_MAP: Dict[str, float] = {
    "Very High": 0.10,
    "High": 0.05,
    "Medium-High": 0.02,
    "Medium": -0.05,
}

ARCHETYPE_REVENUE_STABILITY_MAP: Dict[str, Dict[str, float]] = {
    "High": {"qual": 0.08, "risk": -0.08},
    "Medium-High": {"qual": 0.04, "risk": -0.04},
    "Medium": {"qual": 0.00, "risk": 0.00},
}

ARCHETYPE_TECH_DEPENDENCE_MAP: Dict[str, Dict[str, float]] = {
    "Very High": {"risk": 0.08, "growth": 0.04},
    "High": {"risk": 0.04, "growth": 0.02},
    "Medium": {"risk": 0.00, "growth": 0.00},
}

ARCHETYPE_CAPITAL_INTENSITY_MAP: Dict[str, Dict[str, float]] = {
    "Very High": {"risk": 0.06, "qual": -0.04},
    "High": {"risk": 0.03, "qual": -0.02},
    "Medium-High": {"risk": 0.01, "qual": -0.01},
    "Medium": {"risk": 0.00, "qual": 0.00},
    "Low-Medium": {"risk": -0.03, "qual": 0.03},
}

def compute_archetype_baseline_adjustments(company: Dict[str, Any]) -> Dict[str, float]:
    """Computes company-specific baseline offsets from business archetype characteristics.
    
    Returns:
        Dict[str, float]: Adjustments {'delta_qual', 'delta_risk', 'delta_growth', 'delta_esg'}
    """
    g_prof = company.get("growth_profile", "Medium")
    r_stab = company.get("revenue_stability", "Medium")
    t_dep = company.get("technology_dependence", "Medium")
    c_int = company.get("capital_intensity", "Medium")
    
    d_growth = (
        ARCHETYPE_GROWTH_PROFILE_MAP.get(g_prof, 0.0)
        + ARCHETYPE_TECH_DEPENDENCE_MAP.get(t_dep, {}).get("growth", 0.0)
    )
    d_qual = (
        ARCHETYPE_REVENUE_STABILITY_MAP.get(r_stab, {}).get("qual", 0.0)
        + ARCHETYPE_CAPITAL_INTENSITY_MAP.get(c_int, {}).get("qual", 0.0)
    )
    d_risk = (
        ARCHETYPE_REVENUE_STABILITY_MAP.get(r_stab, {}).get("risk", 0.0)
        + ARCHETYPE_TECH_DEPENDENCE_MAP.get(t_dep, {}).get("risk", 0.0)
        + ARCHETYPE_CAPITAL_INTENSITY_MAP.get(c_int, {}).get("risk", 0.0)
    )
    d_esg = 0.0  # Sector baseline remains the core anchor for ESG
    
    return {
        "delta_esg": round(d_esg, 4),
        "delta_qual": round(d_qual, 4),
        "delta_risk": round(d_risk, 4),
        "delta_growth": round(d_growth, 4),
    }

# Sector universe counts (Total 10 companies)
REQUIRED_SECTOR_COUNTS: Dict[str, int] = {
    "Renewable Power": 3,
    "Water": 1,
    "Waste and Bio": 2,
    "EV and Batteries": 2,
    "Efficiency and Grid": 2,
}

# ==============================================================================
# 4. LATENT FACTOR EVOLUTION & CORRELATION STRUCTURE
# ==============================================================================

# 4-factor order: [ESG, Quality, Risk, Growth]
FACTOR_NAMES: Tuple[str, ...] = ("esg", "qual", "risk", "growth")

TARGET_CORRELATION_MATRIX: np.ndarray = np.array([
    [ 1.00,  0.30, -0.15,  0.10],  # ESG
    [ 0.30,  1.00, -0.55,  0.20],  # Quality
    [-0.15, -0.55,  1.00,  0.25],  # Risk
    [ 0.10,  0.20,  0.25,  1.00],  # Growth
], dtype=np.float64)

# Annual Persistence (Autoregressive AR(1) coefficients)
ANNUAL_PERSISTENCE: Dict[str, float] = {
    "esg": 0.90,
    "qual": 0.85,
    "risk": 0.85,
    "growth": 0.85,
}

# Initial Standard Deviations around Sector Baselines
INITIAL_STD: Dict[str, float] = {
    "esg": 0.70,
    "qual": 0.65,
    "risk": 0.60,
    "growth": 0.60,
}

# Annual Innovation Standard Deviations
INNOVATION_STD: Dict[str, float] = {
    "esg": 0.20,
    "qual": 0.25,
    "risk": 0.25,
    "growth": 0.25,
}

# Value Constraints and Bounds
LATENT_SCORE_BOUNDS: Tuple[float, float] = (-2.0, 2.0)
BETA_BOUNDS: Tuple[float, float] = (0.40, 1.80)
BETA_STD: float = 0.18
IDIO_VOLATILITY_BOUNDS: Tuple[float, float] = (0.10, 0.50)
IDIO_VOLATILITY_STD: float = 0.05
BASE_DRIFT_BOUNDS: Tuple[float, float] = (-0.02, 0.18)
BASE_DRIFT_STD: float = 0.015

BASE_PRICE_BOUNDS: Tuple[float, float] = (150.0, 900.0)
MAX_LISTING_DATE: str = "2020-01-01"

# ==============================================================================
# 5. MATRIX VERIFICATION HELPER
# ==============================================================================

def verify_correlation_matrix(corr: np.ndarray = TARGET_CORRELATION_MATRIX) -> np.ndarray:
    """Verifies that the target correlation matrix is symmetric and strictly positive definite.
    
    Returns:
        np.ndarray: Lower-triangular Cholesky factor L such that L @ L.T == corr.
        
    Raises:
        ValueError: If matrix is non-symmetric or not positive definite.
    """
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("Target correlation matrix is not symmetric.")
    
    eigvals = np.linalg.eigvalsh(corr)
    if np.any(eigvals <= 0):
        raise ValueError(f"Target correlation matrix is not positive definite. Eigenvalues: {eigvals}")
    
    return np.linalg.cholesky(corr)

# Pre-compute and validate Cholesky factor at module load time
CHOLESKY_L: np.ndarray = verify_correlation_matrix(TARGET_CORRELATION_MATRIX)
