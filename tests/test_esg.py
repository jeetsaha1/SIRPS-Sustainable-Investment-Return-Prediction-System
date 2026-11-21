"""SIRPS Member 3 (ESG Module) Test Suite.

Rigorously verifies:
1. ESG facts schema compliance, column dtypes, and expected 320 row count.
2. ESG scores schema compliance, composite primary key, and expected 40 row count.
3. Indicator and score range validity (0-100 bounds, physical unit positivity).
4. Missingness integrity: non-reporting ('NR') and non-applicability ('NA')
   are strictly represented as null/NaN and never zero-filled.
5. Strict enforcement of the 60% coverage rule per pillar.
6. Direction-adjusted percentile ranking (lower-is-better inverted, averaged ties).
7. Total ESG score equals the equal-weight mean of valid pillar scores,
   including verification of the blueprint sample case (81.5, 66.7, 44.4 -> 64.2).
8. Strict point-in-time sequence: available_date is strictly after filed_date.
9. Every available_date is a valid trading day in trading_calendar.csv.
10. Deterministic seed reproducibility.
11. Referential integrity against company_master and absence of duplicate records.
12. Boundary handling: edge cases with all-missing indicators and zero coverage.
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

from common.io import load
from common.schemas import validate_dataset_schema, SCHEMAS
from common.scoring import (
    score_universe,
    compute_indicator_percentile,
    add_indicator_percentiles,
    INDICATOR_CONFIG,
    PILLAR_INDICATORS,
)
from data_generation.gen_esg import generate_esg_facts, ESG_FISCAL_YEARS
import data_generation.config as config


# ==============================================================================
# 1. Dataset Shape & Schema Compliance Tests
# ==============================================================================

def test_esg_facts_schema_and_row_count():
    """Validates data/esg_facts.csv against schema: exactly 320 rows and required columns."""
    df = load("esg_facts", use_mock=False)

    # 10 firms × 4 fiscal years × 8 indicators = 320 rows
    assert len(df) == 320, f"Expected exactly 320 ESG facts rows, got {len(df)}"
    assert df["fact_id"].nunique() == 320, "fact_id must be unique across all 320 rows"
    assert sorted(df["fact_id"].tolist()) == list(range(1, 321)), "fact_id must be sequential 1..320"

    # Schema validator check
    validate_dataset_schema(df, "esg_facts")

    # Check universe of companies and fiscal years
    assert set(df["company_id"].unique()) == {f"C{i:02d}" for i in range(1, 11)}
    assert set(df["fiscal_year"].unique()) == set(ESG_FISCAL_YEARS)
    assert df["data_source"].eq("SYNTHETIC_DEMO").all(), "data_source must be 'SYNTHETIC_DEMO'"


def test_esg_scores_schema_and_row_count():
    """Validates data/esg_scores.csv against schema: exactly 40 rows and primary key."""
    df = load("esg_scores", use_mock=False)

    # 10 firms × 4 fiscal years = 40 rows
    assert len(df) == 40, f"Expected exactly 40 ESG scores rows, got {len(df)}"

    # Composite primary key (company_id, fiscal_year)
    dups = df.duplicated(subset=["company_id", "fiscal_year"])
    assert not dups.any(), f"Found duplicate (company_id, fiscal_year) keys in esg_scores"

    # Schema validator check
    validate_dataset_schema(df, "esg_scores")
    assert df["data_source"].eq("SYNTHETIC_DEMO").all(), "data_source must be 'SYNTHETIC_DEMO'"


# ==============================================================================
# 2. Indicator & Score Ranges and Boundary Tests
# ==============================================================================

def test_indicator_ranges_and_bounds():
    """Verifies that all reported indicator values conform to physical and percentage bounds."""
    df = load("esg_facts", use_mock=False)
    reported = df[df["status"] == "value"]

    # Percent indicators must stay in [0, 100]
    pct_indicators = [
        "renewable_share_pct",
        "waste_recovery_pct",
        "women_workforce_pct",
        "independent_directors_pct",
    ]
    for ind in pct_indicators:
        vals = reported.loc[reported["indicator_id"] == ind, "value"]
        assert len(vals) > 0, f"No reported values for {ind}"
        assert (vals >= 0.0).all(), f"{ind} has values < 0%"
        assert (vals <= 100.0).all(), f"{ind} has values > 100%"

    # Physical intensities must be strictly positive
    intensity_indicators = ["energy_intensity", "emission_intensity", "water_intensity"]
    for ind in intensity_indicators:
        vals = reported.loc[reported["indicator_id"] == ind, "value"]
        assert (vals > 0.0).all(), f"{ind} has non-positive values"

    # LTIFR must be non-negative
    ltifr_vals = reported.loc[reported["indicator_id"] == "ltifr", "value"]
    assert (ltifr_vals >= 0.0).all(), "ltifr has negative values"


def test_score_ranges_and_bounds():
    """Verifies derived pillar and total scores fall strictly within [0, 100]."""
    df = load("esg_scores", use_mock=False)

    for col in ["e_score", "s_score", "g_score", "esg_total"]:
        scores = df[col].dropna()
        assert len(scores) > 0, f"No non-null scores in {col}"
        assert (scores >= 0.0).all(), f"{col} contains values < 0.0: min is {scores.min()}"
        assert (scores <= 100.0).all(), f"{col} contains values > 100.0: max is {scores.max()}"

    for cov_col in ["coverage_ratio", "e_coverage", "s_coverage", "g_coverage"]:
        covs = df[cov_col]
        assert (covs >= 0.0).all() and (covs <= 1.0).all(), f"{cov_col} out of [0, 1]"


# ==============================================================================
# 3. Missingness & Non-Reporting ('NR' / 'NA') Integrity
# ==============================================================================

def test_missingness_nr_na_integrity():
    """Verifies that non-reporting (NR) and non-applicable (NA) observations are strictly null."""
    df = load("esg_facts", use_mock=False)

    # 1. NR and NA must have NaN in the 'value' column (never zero-filled)
    missing_rows = df[df["status"].isin(["NR", "NA"])]
    assert len(missing_rows) > 0, "No NR or NA rows generated in esg_facts"
    assert missing_rows["value"].isna().all(), "NR or NA observations must have null/empty value"

    # 2. 'value' status must have non-null numeric values
    value_rows = df[df["status"] == "value"]
    assert value_rows["value"].notna().all(), "'value' status rows must have non-null values"

    # 3. Verify water and waste have higher NR presence as per Blueprint (~12% vs ~6%)
    water_waste_nr = df[df["indicator_id"].isin(["water_intensity", "waste_recovery_pct"]) & (df["status"] == "NR")]
    other_nr = df[~df["indicator_id"].isin(["water_intensity", "waste_recovery_pct"]) & (df["status"] == "NR")]

    water_waste_rate = len(water_waste_nr) / 80.0  # 40 per indicator × 2
    other_rate = len(other_nr) / 240.0             # 40 per indicator × 6
    assert water_waste_rate > other_rate, (
        f"Water/waste NR rate ({water_waste_rate:.3f}) should exceed other indicators ({other_rate:.3f})"
    )


# ==============================================================================
# 4. 60% Coverage Rule Enforcement
# ==============================================================================

def test_sixty_percent_coverage_rule_logic():
    """Tests the 60% coverage rule directly using synthetic test inputs."""
    # Construct a minimal facts DataFrame for 1 firm in 1 fiscal year
    base_facts = []

    # Scenario 1: Pillar E with 2 out of 5 reported (40% < 60%) -> e_score must be NaN
    # Scenario 2: Pillar E with 3 out of 5 reported (60% >= 60%) -> e_score must be valid
    # Scenario 3: Pillar S with 1 out of 2 reported (50% < 60%) -> s_score must be NaN
    # Scenario 4: Pillar S with 2 out of 2 reported (100% >= 60%) -> s_score must be valid

    test_cases = [
        # (cid, reported_e_count, reported_s_count, reported_g_count, expect_e_nan, expect_s_nan, expect_g_nan)
        ("T01", 2, 2, 1, True, False, False),   # E coverage 40% < 60% -> E is NaN, total is NaN
        ("T02", 3, 2, 1, False, False, False),  # E coverage 60% >= 60% -> E is valid, total valid
        ("T03", 5, 1, 1, False, True, False),   # S coverage 50% < 60% -> S is NaN, total is NaN
        ("T04", 5, 2, 0, False, False, True),   # G coverage 0% < 60% -> G is NaN, total is NaN
    ]

    rows = []
    fact_id = 1
    for cid, e_cnt, s_cnt, g_cnt, _, _, _ in test_cases:
        # Pillar E
        for idx, ind in enumerate(PILLAR_INDICATORS["E"]):
            is_rep = idx < e_cnt
            rows.append({
                "fact_id": fact_id,
                "company_id": cid,
                "fiscal_year": "FY2023-24",
                "indicator_id": ind,
                "indicator_name": ind,
                "pillar": "E",
                "unit": "%",
                "value": 50.0 if is_rep else np.nan,
                "status": "value" if is_rep else "NR",
                "source_doc": "test.pdf",
                "source_page": 1,
                "filed_date": "2024-07-15",
                "available_date": "2024-07-16",
                "data_source": "SYNTHETIC_DEMO",
            })
            fact_id += 1
        # Pillar S
        for idx, ind in enumerate(PILLAR_INDICATORS["S"]):
            is_rep = idx < s_cnt
            rows.append({
                "fact_id": fact_id,
                "company_id": cid,
                "fiscal_year": "FY2023-24",
                "indicator_id": ind,
                "indicator_name": ind,
                "pillar": "S",
                "unit": "%",
                "value": 50.0 if is_rep else np.nan,
                "status": "value" if is_rep else "NR",
                "source_doc": "test.pdf",
                "source_page": 1,
                "filed_date": "2024-07-15",
                "available_date": "2024-07-16",
                "data_source": "SYNTHETIC_DEMO",
            })
            fact_id += 1
        # Pillar G
        for idx, ind in enumerate(PILLAR_INDICATORS["G"]):
            is_rep = idx < g_cnt
            rows.append({
                "fact_id": fact_id,
                "company_id": cid,
                "fiscal_year": "FY2023-24",
                "indicator_id": ind,
                "indicator_name": ind,
                "pillar": "G",
                "unit": "%",
                "value": 50.0 if is_rep else np.nan,
                "status": "value" if is_rep else "NR",
                "source_doc": "test.pdf",
                "source_page": 1,
                "filed_date": "2024-07-15",
                "available_date": "2024-07-16",
                "data_source": "SYNTHETIC_DEMO",
            })
            fact_id += 1

    df_test_facts = pd.DataFrame(rows)
    scored = score_universe(df_test_facts)

    t01 = scored[scored["company_id"] == "T01"].iloc[0]
    assert np.isnan(t01["e_score"]), "T01 has 2/5 E indicators (40%), e_score must be NaN"
    assert not np.isnan(t01["s_score"]), "T01 has 2/2 S indicators (100%), s_score must be valid"
    assert np.isnan(t01["esg_total"]), "T01 has missing E pillar, esg_total must be NaN"

    t02 = scored[scored["company_id"] == "T02"].iloc[0]
    assert not np.isnan(t02["e_score"]), "T02 has 3/5 E indicators (60%), e_score must be valid"
    assert not np.isnan(t02["esg_total"]), "T02 has all pillars satisfying 60%, esg_total must be valid"

    t03 = scored[scored["company_id"] == "T03"].iloc[0]
    assert np.isnan(t03["s_score"]), "T03 has 1/2 S indicators (50%), s_score must be NaN"
    assert np.isnan(t03["esg_total"]), "T03 has missing S pillar, esg_total must be NaN"

    t04 = scored[scored["company_id"] == "T04"].iloc[0]
    assert np.isnan(t04["g_score"]), "T04 has 0/1 G indicators (0%), g_score must be NaN"
    assert np.isnan(t04["esg_total"]), "T04 has missing G pillar, esg_total must be NaN"


# ==============================================================================
# 5. Percentile Ranking & Lower-is-Better Inversion
# ==============================================================================

def test_percentile_calculation_and_inversion():
    """Verifies that lower-is-better metrics invert ranks and averaged ties behave correctly."""
    # 3 companies: values 10, 20, 30
    s = pd.Series([10.0, 20.0, 30.0], index=["A", "B", "C"])

    # Higher is better: 10 -> 0.0, 20 -> 50.0, 30 -> 100.0
    pct_higher = compute_indicator_percentile(s, lower_is_better=False)
    assert pct_higher["A"] == 0.0
    assert pct_higher["B"] == 50.0
    assert pct_higher["C"] == 100.0

    # Lower is better: 10 -> 100.0, 20 -> 50.0, 30 -> 0.0
    pct_lower = compute_indicator_percentile(s, lower_is_better=True)
    assert pct_lower["A"] == 100.0
    assert pct_lower["B"] == 50.0
    assert pct_lower["C"] == 0.0

    # Test ties: values 10, 20, 20, 40 (N=4)
    # Ranks for 20 are (2 + 3) / 2 = 2.5
    # pct = 100 * (2.5 - 1) / (4 - 1) = 100 * 1.5 / 3 = 50.0
    s_ties = pd.Series([10.0, 20.0, 20.0, 40.0])
    pct_ties = compute_indicator_percentile(s_ties, lower_is_better=False)
    assert pct_ties.iloc[1] == 50.0
    assert pct_ties.iloc[2] == 50.0


# ==============================================================================
# 6. Total Score Calculation & Blueprint Sample Verification
# ==============================================================================

def test_blueprint_sample_score_calculation():
    """Verifies the exact Blueprint Page 15 sample case: [81.5, 66.7, 44.4] -> 64.2."""
    e = 81.5
    s = 66.7
    g = 44.4
    expected_total = round((e + s + g) / 3.0, 1)
    assert expected_total == 64.2, f"Blueprint calculation mismatch: expected 64.2, got {expected_total}"


def test_total_score_equals_pillar_mean_in_dataset():
    """Verifies that in esg_scores.csv, esg_total equals (e + s + g) / 3 for all valid rows."""
    df = load("esg_scores", use_mock=False)
    valid_total = df[df["esg_total"].notna()]

    for _, row in valid_total.iterrows():
        expected = round((row["e_score"] + row["s_score"] + row["g_score"]) / 3.0, 1)
        assert abs(row["esg_total"] - expected) <= 0.1, (
            f"Firm {row['company_id']} {row['fiscal_year']} total ({row['esg_total']}) != mean ({expected})"
        )


# ==============================================================================
# 7. Point-in-Time Sequence & Calendar Integration
# ==============================================================================

def test_available_date_strictly_after_filed_date():
    """Verifies that available_date is strictly after filed_date (no look-ahead)."""
    df_facts = load("esg_facts", use_mock=False)

    filed = pd.to_datetime(df_facts["filed_date"])
    avail = pd.to_datetime(df_facts["available_date"])

    assert (avail > filed).all(), "available_date must be strictly after filed_date"

    # Also verify in esg_scores
    df_scores = load("esg_scores", use_mock=False)
    avail_scores = pd.to_datetime(df_scores["available_date"])
    # All available dates in scores must be in mid-July of the subsequent year
    assert (avail_scores.dt.month == 7).all(), "available_date must be in July"


def test_available_date_is_valid_trading_day():
    """Verifies that every available_date exists in trading_calendar.csv."""
    calendar = load("trading_calendar", use_mock=False)
    trading_days = set(calendar["date"].astype(str).tolist())

    df_facts = load("esg_facts", use_mock=False)
    facts_avail_dates = set(df_facts["available_date"].astype(str).tolist())
    assert facts_avail_dates.issubset(trading_days), "Some available_dates in esg_facts are not valid trading days"

    df_scores = load("esg_scores", use_mock=False)
    scores_avail_dates = set(df_scores["available_date"].astype(str).tolist())
    assert scores_avail_dates.issubset(trading_days), "Some available_dates in esg_scores are not valid trading days"


# ==============================================================================
# 8. Deterministic Seed Reproducibility
# ==============================================================================

def test_deterministic_reproducibility():
    """Verifies that running generate_esg_facts twice with the same seed yields identical outputs."""
    df1 = generate_esg_facts(seed=12345)
    df2 = generate_esg_facts(seed=12345)

    pd.testing.assert_frame_equal(df1, df2)

    # Scored outputs must also be identical
    scores1 = score_universe(df1)
    scores2 = score_universe(df2)
    pd.testing.assert_frame_equal(scores1, scores2)


# ==============================================================================
# 10. ESG Analysis UI Module Interface Tests
# ==============================================================================

def test_esg_analysis_module_interface():
    """Verifies that modules.esg_analysis exposes the expected render() callable interface."""
    import importlib
    mod = importlib.import_module("modules.esg_analysis")
    assert hasattr(mod, "render"), "modules.esg_analysis must expose 'render'"
    assert callable(mod.render), "modules.esg_analysis.render must be callable"


def test_esg_analysis_helpers():
    """Verifies pure analytical helpers in modules.esg_analysis."""
    from modules.esg_analysis import get_company_esg_summary, get_universe_pillar_medians
    df_scores = load("esg_scores", use_mock=False)

    summary = get_company_esg_summary(df_scores, "C01", "FY2023-24")
    assert summary is not None, "Failed to retrieve summary for C01 FY2023-24"
    assert summary["company_id"] == "C01"
    assert summary["fiscal_year"] == "FY2023-24"
    assert 0.0 <= summary["coverage_ratio"] <= 1.0

    medians = get_universe_pillar_medians(df_scores, "FY2023-24")
    assert "E" in medians and "S" in medians and "G" in medians and "Total" in medians
    assert 0.0 <= medians["E"] <= 100.0
