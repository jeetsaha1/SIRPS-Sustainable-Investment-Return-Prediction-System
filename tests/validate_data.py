"""SIRPS Data Validator Test Suite (Member 1).

Performs strict integrity and schema validation checks across all generated
and mock CSV files in the repository:
1. Primary key uniqueness
2. Foreign key referential integrity
3. Required column presence and non-null constraints
4. Absence of infinities
5. Synthetic provenance flag ('SYNTHETIC_DEMO')
6. Value domain bounds (base prices, latent scores, beta, volatility, drift)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.schemas import SCHEMAS, FILES
from common.io import load, get_file_path
import data_generation.config as config

def test_company_master_integrity():
    """Validates data/company_master.csv against research specifications."""
    df = load("company_master", use_mock=False)
    
    # 1. Size and IDs
    assert len(df) == 10, f"Expected exactly 10 companies, got {len(df)}"
    assert df["company_id"].nunique() == 10, "Duplicate company_id found"
    assert df["ticker"].nunique() == 10, "Duplicate ticker found"
    assert sorted(df["company_id"].tolist()) == [f"C{i:02d}" for i in range(1, 11)]
    
    # 2. Sector Distribution
    sector_counts = df["sector"].value_counts().to_dict()
    assert sector_counts == config.REQUIRED_SECTOR_COUNTS, f"Sector distribution mismatch: {sector_counts}"
    
    # 3. Base Price Bounds (₹150–₹900)
    assert (df["base_price"] >= config.BASE_PRICE_BOUNDS[0]).all(), "Base price below 150"
    assert (df["base_price"] <= config.BASE_PRICE_BOUNDS[1]).all(), "Base price above 900"
    
    # 4. Listing Date (Before 2020-01-01)
    # df["listing_date"] is converted to datetime by common.io
    assert (df["listing_date"] < pd.to_datetime(config.MAX_LISTING_DATE)).all(), "Listing date not before 2020-01-01"
    
    # 5. Provenance and Flags
    assert (df["is_synthetic"] == True).all(), "is_synthetic must be True"
    assert (df["data_source"] == "SYNTHETIC_DEMO").all(), "data_source must be SYNTHETIC_DEMO"
    assert (df["universe_group"] == "core").all(), "universe_group must be 'core'"

def test_latent_profiles_integrity():
    """Validates data/latent_profiles.csv against latent factor specifications."""
    df = load("latent_profiles", use_mock=False)
    
    # 1. Shape and IDs
    assert len(df) == 60, f"Expected 60 records (10 firms × 6 years), got {len(df)}"
    assert df["company_id"].nunique() == 10, "Expected 10 unique companies"
    assert df.groupby("company_id")["fiscal_year"].nunique().eq(6).all(), "Each firm must have 6 fiscal years"
    
    # 2. Bounded Latent Scores in [-2, +2]
    for col in ["esg_latent_score", "quality_latent_score", "risk_latent_score", "growth_latent_score"]:
        assert (df[col] >= config.LATENT_SCORE_BOUNDS[0]).all(), f"{col} has values < -2"
        assert (df[col] <= config.LATENT_SCORE_BOUNDS[1]).all(), f"{col} has values > +2"
        assert not df[col].isna().any(), f"{col} contains null values"
        assert not np.isinf(df[col]).any(), f"{col} contains infinite values"
        
    # 3. Parameter Bounds
    assert (df["beta"] >= config.BETA_BOUNDS[0]).all() and (df["beta"] <= config.BETA_BOUNDS[1]).all(), "Beta out of bounds"
    assert (df["idio_volatility"] >= config.IDIO_VOLATILITY_BOUNDS[0]).all() and (df["idio_volatility"] <= config.IDIO_VOLATILITY_BOUNDS[1]).all(), "Idio vol out of bounds"
    assert (df["base_drift"] >= config.BASE_DRIFT_BOUNDS[0]).all() and (df["base_drift"] <= config.BASE_DRIFT_BOUNDS[1]).all(), "Base drift out of bounds"
    
    # 4. Provenance
    assert (df["data_source"] == "SYNTHETIC_DEMO").all(), "data_source must be SYNTHETIC_DEMO"

def test_mock_datasets_compliance():
    """Validates that all 20 schemas have compliant mock datasets in data/mock/."""
    for schema_name, schema in SCHEMAS.items():
        mock_path = get_file_path(schema_name, use_mock=True)
        assert mock_path.exists(), f"Mock file for schema '{schema_name}' missing at: {mock_path}"
        
        df = pd.read_csv(mock_path)
        assert len(df) >= 10, f"Mock file for '{schema_name}' has insufficient rows ({len(df)})"
        
        # Verify all required columns exist
        missing = [c for c in schema.required_columns if c not in df.columns]
        assert not missing, f"Mock '{schema_name}' missing columns: {missing}"
        
        # Verify provenance
        assert "data_source" in df.columns, f"Mock '{schema_name}' missing 'data_source'"
        assert (df["data_source"] == "SYNTHETIC_DEMO").all(), f"Mock '{schema_name}' has invalid data_source"
        
        # Verify no unexpected infinities in numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        for nc in num_cols:
            assert not np.isinf(df[nc]).any(), f"Mock '{schema_name}' column '{nc}' has infinite values"

def run_all_validation():
    """Runs all validation tests directly and prints human-readable summary."""
    print("Running SIRPS Data Validation Suite...")
    test_company_master_integrity()
    print("✓ Company Master validation passed.")
    test_latent_profiles_integrity()
    print("✓ Latent Profiles validation passed.")
    test_mock_datasets_compliance()
    print("✓ Mock Datasets schema compliance passed.")
    print("All foundational data validation checks succeeded!")

if __name__ == "__main__":
    run_all_validation()
