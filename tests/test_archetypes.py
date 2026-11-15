"""
Tests for business archetypes, archetype-to-factor mapping, and strengthened schema/IO contracts.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from common.schemas import (
    validate_dataset_schema,
    get_schema,
    DATASET_SCHEMAS,
    PROVENANCE_TAG,
)
from common.io import (
    load_dataset,
    save_dataset,
    validate_referential_integrity,
    DATA_DIR,
    MOCK_DIR,
)
from data_generation.config import compute_archetype_baseline_adjustments


def test_company_archetypes_json_completeness():
    """Verify company_archetypes.json contains all 10 companies with required fields."""
    archetypes_path = Path("data_generation/company_archetypes.json")
    assert archetypes_path.exists(), "company_archetypes.json must exist"

    with open(archetypes_path, "r", encoding="utf-8") as f:
        archetypes = json.load(f)

    assert len(archetypes) == 10
    arch_by_id = {c["company_id"]: c for c in archetypes}
    assert sorted(arch_by_id.keys()) == [f"C{i:02d}" for i in range(1, 11)]

    required_fields = [
        "company_id",
        "company_name",
        "ticker",
        "sector",
        "sub_sector",
        "business_model",
        "capital_intensity",
        "growth_profile",
        "revenue_stability",
        "technology_dependence",
        "business_risk_type",
        "listing_date",
        "base_price",
        "profile_version",
    ]

    for cid, arch in arch_by_id.items():
        for field in required_fields:
            assert field in arch, f"Field {field} missing for {cid}"
            assert arch[field] is not None and str(arch[field]).strip() != ""


def test_archetype_baseline_adjustments_intra_sector_differentiation():
    """Verify that firms in the same sector receive distinct baseline adjustments."""
    archetypes_path = Path("data_generation/company_archetypes.json")
    with open(archetypes_path, "r", encoding="utf-8") as f:
        archetypes = json.load(f)

    renewables = [c for c in archetypes if c["sector"] == "Renewable Power"]
    assert len(renewables) == 3  # C01, C02, C07

    offsets = {c["company_id"]: compute_archetype_baseline_adjustments(c) for c in renewables}

    # Verify each firm has non-zero offsets
    for cid, delta in offsets.items():
        assert len(delta) == 4
        assert not all(v == 0.0 for v in delta.values())

    # Verify intra-sector differentiation (e.g. C01 Solar vs C07 Hydro)
    assert offsets["C01"] != offsets["C07"], "C01 Solar and C07 Hydro must have distinct adjustments"
    assert offsets["C01"] != offsets["C02"], "C01 Solar and C02 Wind must have distinct adjustments"
    # C01 has higher growth than C07
    assert offsets["C01"]["delta_growth"] > offsets["C07"]["delta_growth"]
    # C07 has higher revenue stability -> higher quality
    assert offsets["C07"]["delta_qual"] > offsets["C01"]["delta_qual"]


def test_validate_dataset_schema_rejects_duplicate_primary_keys():
    """Verify that validate_dataset_schema raises ValueError on duplicate primary keys."""
    # Create a DataFrame with all required columns but duplicate primary keys for company_master
    df_dup = pd.DataFrame({
        "company_id": ["C01", "C01"],  # Duplicate PK
        "company_name": ["Alpha Ltd", "Beta Ltd"],
        "ticker": ["ALPHALTD", "BETALTD"],
        "sector": ["Renewable Power", "Renewable Power"],
        "sub_sector": ["Solar", "Wind"],
        "universe_group": ["core", "core"],
        "listing_date": ["2015-01-01", "2015-01-01"],
        "base_price": [100.0, 200.0],
        "is_synthetic": [True, True],
        "data_source": [PROVENANCE_TAG, PROVENANCE_TAG],
        "business_model": ["Solar", "Wind"],
        "capital_intensity": ["High", "High"],
        "growth_profile": ["High", "High"],
        "revenue_stability": ["Medium", "Medium"],
        "technology_dependence": ["High", "High"],
        "business_risk_type": ["Resource", "Resource"],
        "profile_version": ["v1.0", "v1.0"],
        "provenance": [PROVENANCE_TAG, PROVENANCE_TAG],
    })

    with pytest.raises(ValueError, match="Primary key integrity violation"):
        validate_dataset_schema(df_dup, "company_master")


def test_production_mode_raises_file_not_found_on_missing_dataset(monkeypatch, tmp_path):
    """Verify that load_dataset(name, use_mock=False) strictly raises FileNotFoundError without fallback."""
    fake_missing_path = tmp_path / "trading_calendar.csv"
    monkeypatch.setattr("common.io.get_file_path", lambda name, use_mock=False: fake_missing_path)
    with pytest.raises(FileNotFoundError, match="Silent fallback to mock data is disabled in production mode"):
        load_dataset("trading_calendar", use_mock=False)


def test_validate_referential_integrity():
    """Verify referential integrity validation utility."""
    parent_df = pd.DataFrame({"company_id": ["C01", "C02", "C03"]})
    child_valid = pd.DataFrame({"company_id": ["C01", "C02"]})
    child_invalid = pd.DataFrame({"company_id": ["C01", "C99"]})

    # Should pass without exception
    validate_referential_integrity(
        child_valid, "child_test", ["company_id"],
        "parent_test", ["company_id"], parent_df
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="Referential integrity violation"):
        validate_referential_integrity(
            child_invalid, "child_test", ["company_id"],
            "parent_test", ["company_id"], parent_df
        )
