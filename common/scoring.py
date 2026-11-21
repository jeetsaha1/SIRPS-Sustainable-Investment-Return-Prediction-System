"""SIRPS ESG Scoring and Coverage Engine (Member 3).

Implements the authoritative ESG scoring methodology specified in the
SIRPS Prototype Blueprint (Sections 4.3, 5.4, and 5.5).

Methodology:
1. Cross-Sectional Percentile Ranking per Fiscal Year:
   - For each (fiscal_year, indicator_id), only reported observations
     (status == 'value' and notnull) are ranked.
   - Percentile = 100 * (rank - 1) / (valid_firms - 1), with ties averaged.
   - If valid_firms == 1, percentile is 50.0.
   - Direction-adjusted:
     * Higher-is-better: score = percentile
     * Lower-is-better (energy, emissions, water, LTIFR): score = 100.0 - percentile
2. 60% Coverage Rule:
   - Pillar E (5 indicators): requires >= 3 reported (coverage >= 0.60).
     e_score is NaN if coverage < 60%; otherwise mean of reported E percentiles.
   - Pillar S (2 indicators): requires 2 reported (coverage = 1.0 >= 0.60).
     s_score is NaN if 1 or 0 reported; otherwise mean of reported S percentiles.
   - Pillar G (1 indicator): requires 1 reported (coverage = 1.0 >= 0.60).
     g_score is NaN if 0 reported; otherwise the single G percentile.
3. Total ESG Score (esg_total):
   - Equal-weight mean of the 3 pillars: (e_score + s_score + g_score) / 3.0.
   - Strictly empty (NaN) if ANY pillar score is missing/insufficient coverage.
   - Bounded in [0, 100], rounded to 1 decimal place.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Indicator Metadata and Directionality Mapping
INDICATOR_CONFIG: Dict[str, Dict[str, Any]] = {
    "renewable_share_pct": {
        "name": "Renewable energy share",
        "pillar": "E",
        "unit": "%",
        "direction": "higher",
        "lower_is_better": False,
    },
    "energy_intensity": {
        "name": "Energy intensity",
        "pillar": "E",
        "unit": "GJ per Rs cr",
        "direction": "lower",
        "lower_is_better": True,
    },
    "emission_intensity": {
        "name": "Emission intensity",
        "pillar": "E",
        "unit": "tCO2e per Rs cr",
        "direction": "lower",
        "lower_is_better": True,
    },
    "water_intensity": {
        "name": "Water intensity",
        "pillar": "E",
        "unit": "kL per Rs cr",
        "direction": "lower",
        "lower_is_better": True,
    },
    "waste_recovery_pct": {
        "name": "Waste recovery rate",
        "pillar": "E",
        "unit": "%",
        "direction": "higher",
        "lower_is_better": False,
    },
    "ltifr": {
        "name": "Lost time injury frequency rate",
        "pillar": "S",
        "unit": "per million hours",
        "direction": "lower",
        "lower_is_better": True,
    },
    "women_workforce_pct": {
        "name": "Women in workforce",
        "pillar": "S",
        "unit": "%",
        "direction": "higher",
        "lower_is_better": False,
    },
    "independent_directors_pct": {
        "name": "Independent directors share",
        "pillar": "G",
        "unit": "%",
        "direction": "higher",
        "lower_is_better": False,
    },
}

PILLAR_INDICATORS: Dict[str, List[str]] = {
    "E": [
        "renewable_share_pct",
        "energy_intensity",
        "emission_intensity",
        "water_intensity",
        "waste_recovery_pct",
    ],
    "S": [
        "ltifr",
        "women_workforce_pct",
    ],
    "G": [
        "independent_directors_pct",
    ],
}

TOTAL_INDICATORS_COUNT: int = 8
COVERAGE_THRESHOLD: float = 0.60


def compute_indicator_percentile(
    series: pd.Series, lower_is_better: bool = False
) -> pd.Series:
    """Computes cross-sectional direction-adjusted percentile ranks with averaged ties.

    Formula from Blueprint Page 15:
        percentile = 100 * (rank - 1) / (valid_firms - 1)
        flipped for lower-is-better indicators: 100 - percentile

    Args:
        series: Pandas Series of reported numerical values for a single indicator & FY.
        lower_is_better: True if lower values represent better sustainability performance.

    Returns:
        pd.Series: Direction-adjusted percentile ranks in [0, 100].
    """
    valid = series.dropna()
    n_valid = len(valid)

    if n_valid == 0:
        return pd.Series(index=series.index, dtype=float)
    if n_valid == 1:
        # Neutral median for single observation
        return pd.Series(50.0, index=series.index)

    # Average tie breaking per blueprint
    ranks = valid.rank(method="average", ascending=True)
    pct = 100.0 * (ranks - 1.0) / (n_valid - 1.0)

    if lower_is_better:
        pct = 100.0 - pct

    result = pd.Series(index=series.index, dtype=float)
    result.loc[valid.index] = pct.round(1)
    return result


def add_indicator_percentiles(facts_df: pd.DataFrame) -> pd.DataFrame:
    """Attaches a 'percentile' column (0-100) to the facts dataframe.

    Computes direction-adjusted percentile rank within each (fiscal_year, indicator_id)
    cohort for all valid reported observations (status == 'value').
    """
    df = facts_df.copy()
    df["percentile"] = np.nan

    for (fy, ind_id), group in df.groupby(["fiscal_year", "indicator_id"]):
        lower_better = INDICATOR_CONFIG.get(ind_id, {}).get("lower_is_better", False)

        # Only rank observations that are reported ('value') and non-null
        is_reported = (group["status"] == "value") & group["value"].notna()
        if is_reported.any():
            valid_vals = group.loc[is_reported, "value"]
            pcts = compute_indicator_percentile(valid_vals, lower_is_better=lower_better)
            df.loc[pcts.index, "percentile"] = pcts

    return df


def score_universe(facts: pd.DataFrame) -> pd.DataFrame:
    """Aggregates raw ESG facts into standardized company-level pillar and total scores.

    Args:
        facts: Raw facts DataFrame matching the esg_facts schema (320 rows).

    Returns:
        pd.DataFrame: Aggregated scores DataFrame matching the esg_scores schema (40 rows).
    """
    # 1. Attach individual indicator percentiles
    facts_with_pct = add_indicator_percentiles(facts)

    # 2. Extract available date per (company_id, fiscal_year)
    date_map = facts.groupby(["company_id", "fiscal_year"])["available_date"].first()

    scores_list: List[Dict[str, Any]] = []

    # Group by company and fiscal year
    grouped = facts_with_pct.groupby(["company_id", "fiscal_year"])

    for (cid, fy), group in grouped:
        avail_date = date_map.get((cid, fy))

        # Track reported count per pillar
        pillar_scores: Dict[str, Optional[float]] = {}
        pillar_coverage: Dict[str, float] = {}

        total_reported_count = 0

        for pillar, ind_list in PILLAR_INDICATORS.items():
            total_pillar_inds = len(ind_list)
            pillar_rows = group[group["indicator_id"].isin(ind_list)]

            # Reported means status == 'value' with valid numeric value and percentile
            reported_rows = pillar_rows[(pillar_rows["status"] == "value") & pillar_rows["percentile"].notna()]
            k_reported = len(reported_rows)
            total_reported_count += k_reported

            cov = float(k_reported) / float(total_pillar_inds)
            pillar_coverage[pillar] = round(cov, 4)

            # 60% Coverage Rule:
            # Pillar E (5): needs >= 3 (60.0%)
            # Pillar S (2): needs 2 (100.0%, since 1/2 = 50% < 60%)
            # Pillar G (1): needs 1 (100.0% >= 60%)
            if cov >= COVERAGE_THRESHOLD and k_reported > 0:
                p_score = float(reported_rows["percentile"].mean())
                pillar_scores[pillar] = round(p_score, 1)
            else:
                pillar_scores[pillar] = np.nan

        # Total coverage ratio across all 8 indicators
        overall_coverage = round(float(total_reported_count) / float(TOTAL_INDICATORS_COUNT), 4)

        # Total ESG Score: equal-weight mean of all 3 pillars
        # "0–100, equal-weight mean of pillars; empty if any pillar missing"
        e_sc = pillar_scores["E"]
        s_sc = pillar_scores["S"]
        g_sc = pillar_scores["G"]

        if pd.notna(e_sc) and pd.notna(s_sc) and pd.notna(g_sc):
            esg_tot = round(float((e_sc + s_sc + g_sc) / 3.0), 1)
        else:
            esg_tot = np.nan

        data_src = group["data_source"].iloc[0] if "data_source" in group.columns else "SYNTHETIC_DEMO"

        scores_list.append({
            "company_id": cid,
            "fiscal_year": fy,
            "available_date": str(avail_date),
            "e_score": e_sc,
            "s_score": s_sc,
            "g_score": g_sc,
            "esg_total": esg_tot,
            "coverage_ratio": overall_coverage,
            "e_coverage": pillar_coverage["E"],
            "s_coverage": pillar_coverage["S"],
            "g_coverage": pillar_coverage["G"],
            "data_source": data_src,
        })

    scores_df = pd.DataFrame(scores_list)
    # Sort deterministically by company_id, fiscal_year
    scores_df = scores_df.sort_values(["fiscal_year", "company_id"]).reset_index(drop=True)
    return scores_df
