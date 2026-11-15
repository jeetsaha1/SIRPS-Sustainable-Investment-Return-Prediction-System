"""SIRPS Integration & Mathematical Properties Test Suite (Member 1).

Tests end-to-end integration across components:
1. Common I/O loading, caching, and provenance rejection
2. Generator deterministic reproducibility (identical hash across repeated runs)
3. Positive definiteness and Cholesky decomposition of the 4-factor correlation matrix
4. Statistical and economic coherence of the simulated latent factors
   (Risk-Beta-IdioVol coherence, Quality resilience, bounded score domains)
"""

import sys
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data_generation.config as config
from common.schemas import SCHEMAS
from common.io import load, save, get_file_path
from data_generation.make_company_master import generate_company_master
from data_generation.latent import generate_latent_profiles

def test_provenance_rejection():
    """Verifies that common.io strictly rejects files without proper SYNTHETIC_DEMO tag."""
    # Attempt to load a fabricated file or inspect provenance check
    test_df = pd.DataFrame({
        "company_id": ["C99"],
        "data_source": ["REAL_NSE_DATA"]
    })
    temp_path = PROJECT_ROOT / "data" / "temp_invalid_provenance.csv"
    test_df.to_csv(temp_path, index=False)
    
    try:
        from common.io import _read_and_validate_csv
        with pytest.raises(ValueError, match="Provenance.*violation"):
            _read_and_validate_csv(temp_path, "company_master")
    finally:
        if temp_path.exists():
            temp_path.unlink()

def test_deterministic_reproducibility():
    """Verifies that identical seed produces identical hash outputs."""
    df1 = generate_latent_profiles(seed=config.SEED_STREAMS["latent"], output_mock=False)
    csv1 = df1.to_csv(index=False)
    hash1 = hashlib.sha256(csv1.encode("utf-8")).hexdigest()
    
    df2 = generate_latent_profiles(seed=config.SEED_STREAMS["latent"], output_mock=False)
    csv2 = df2.to_csv(index=False)
    hash2 = hashlib.sha256(csv2.encode("utf-8")).hexdigest()
    
    assert hash1 == hash2, "Latent generation is not deterministically reproducible!"

def test_correlation_matrix_properties():
    """Verifies mathematical properties of the target 4-factor correlation matrix."""
    corr = config.TARGET_CORRELATION_MATRIX
    
    # 1. Symmetry
    assert np.allclose(corr, corr.T, atol=1e-10), "Target correlation matrix is not symmetric"
    
    # 2. Diagonal entries are 1.0
    assert np.allclose(np.diag(corr), 1.0, atol=1e-10), "Diagonal entries must be 1.0"
    
    # 3. Positive definiteness (all eigenvalues > 0)
    eigvals = np.linalg.eigvalsh(corr)
    assert np.all(eigvals > 0), f"Correlation matrix is not positive definite. Min eigenvalue: {min(eigvals)}"
    assert min(eigvals) > 0.1, "Matrix is near-singular"
    
    # 4. Cholesky factorization identity: L @ L.T == corr
    L = config.CHOLESKY_L
    reconstructed = L @ L.T
    assert np.allclose(reconstructed, corr, atol=1e-10), "Cholesky decomposition identity failed"

def test_latent_factor_economic_coherence():
    """Verifies that latent factors exhibit coherent statistical and economic relationships."""
    df = load("latent_profiles", use_mock=False)
    
    # 1. Risk positively influences Beta (systematic exposure)
    corr_risk_beta = df["risk_latent_score"].corr(df["beta"])
    assert corr_risk_beta > 0.50, f"Expected positive correlation between Risk and Beta, got {corr_risk_beta:.3f}"
    
    # 2. Risk positively influences Idiosyncratic Volatility
    corr_risk_idio = df["risk_latent_score"].corr(df["idio_volatility"])
    assert corr_risk_idio > 0.50, f"Expected positive correlation between Risk and Idio Vol, got {corr_risk_idio:.3f}"
    
    # 3. Quality negatively influences Beta (stabilizing factor)
    corr_qual_beta = df["quality_latent_score"].corr(df["beta"])
    assert corr_qual_beta < -0.20, f"Expected negative correlation between Quality and Beta, got {corr_qual_beta:.3f}"
    
    # 4. Score constraints [-2.0, 2.0]
    for factor in ["esg_latent_score", "quality_latent_score", "risk_latent_score", "growth_latent_score"]:
        assert df[factor].min() >= -2.0, f"Factor {factor} minimum below -2.0"
        assert df[factor].max() <= 2.0, f"Factor {factor} maximum above 2.0"
        
    # 5. Beta bounds [0.40, 1.80]
    assert df["beta"].min() >= 0.40 and df["beta"].max() <= 1.80, "Beta outside bounds [0.40, 1.80]"
    
    # 6. Idiosyncratic Volatility bounds [0.10, 0.50]
    assert df["idio_volatility"].min() >= 0.10 and df["idio_volatility"].max() <= 0.50, "Idio vol outside bounds"
    
    # 7. Base Drift bounds [-0.02, 0.18]
    assert df["base_drift"].min() >= -0.02 and df["base_drift"].max() <= 0.18, "Base drift outside bounds"
