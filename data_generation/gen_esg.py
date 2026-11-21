"""Synthetic ESG Data Generator for SIRPS Member 3.

Generates data/esg_facts.csv for all 10 fictional green companies across
the 4 reporting fiscal years (FY2021-22, FY2022-23, FY2023-24, FY2024-25).
10 companies × 4 fiscal years × 8 indicators = exactly 320 long-format rows.

Econometric & Research Architecture:
1. Grounded in Latent Profiles:
   - Uses `esg_latent_score` (z_{i,t} in [-2, +2]) from data/latent_profiles.csv.
   - Higher latent ESG score modulates higher renewable share, waste recovery,
     women in workforce, and independent directors, and reduces energy,
     emission, water intensities, and LTIFR.
2. Exact 8 BRSR Indicators:
   - Environmental (E):
     * renewable_share_pct (%, higher better, [0, 100])
     * energy_intensity (GJ per Rs cr, lower better, > 0)
     * emission_intensity (tCO2e per Rs cr, lower better, > 0)
     * water_intensity (kL per Rs cr, lower better, > 0)
     * waste_recovery_pct (%, higher better, [0, 100])
   - Social (S):
     * ltifr (Lost Time Injury Frequency Rate, per million hours, lower better, >= 0)
     * women_workforce_pct (%, higher better, [0, 100])
   - Governance (G):
     * independent_directors_pct (%, higher better, [0, 100])
3. Missingness & Non-Reporting (NR / NA):
   - Realistic missingness without zero-filling (~6% overall, ~12% for water & waste).
   - Missing values are strictly NaN (empty), never zero-filled.
4. Point-in-Time Availability:
   - Filed mid-July following fiscal year end (e.g., FY2021-22 filed around 2022-07-15).
   - available_date is strictly the next valid trading day from trading_calendar.csv.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import data_generation.config as config
from common.io import load, save
from common.schemas import validate_dataset_schema
from common.scoring import score_universe

logger = logging.getLogger(__name__)

# Target fiscal years for ESG reporting (4 FYs = 320 fact rows)
ESG_FISCAL_YEARS: Tuple[str, ...] = (
    "FY2021-22",
    "FY2022-23",
    "FY2023-24",
    "FY2024-25",
)

# Indicator definitions: (indicator_id, indicator_name, pillar, unit, optimal_direction)
INDICATOR_METADATA: List[Dict[str, str]] = [
    {
        "indicator_id": "renewable_share_pct",
        "indicator_name": "Renewable energy share",
        "pillar": "E",
        "unit": "%",
        "direction": "higher",
    },
    {
        "indicator_id": "energy_intensity",
        "indicator_name": "Energy intensity",
        "pillar": "E",
        "unit": "GJ per Rs cr",
        "direction": "lower",
    },
    {
        "indicator_id": "emission_intensity",
        "indicator_name": "Emission intensity",
        "pillar": "E",
        "unit": "tCO2e per Rs cr",
        "direction": "lower",
    },
    {
        "indicator_id": "water_intensity",
        "indicator_name": "Water intensity",
        "pillar": "E",
        "unit": "kL per Rs cr",
        "direction": "lower",
    },
    {
        "indicator_id": "waste_recovery_pct",
        "indicator_name": "Waste recovery rate",
        "pillar": "E",
        "unit": "%",
        "direction": "higher",
    },
    {
        "indicator_id": "ltifr",
        "indicator_name": "Lost time injury frequency rate",
        "pillar": "S",
        "unit": "per million hours",
        "direction": "lower",
    },
    {
        "indicator_id": "women_workforce_pct",
        "indicator_name": "Women in workforce",
        "pillar": "S",
        "unit": "%",
        "direction": "higher",
    },
    {
        "indicator_id": "independent_directors_pct",
        "indicator_name": "Independent directors share",
        "pillar": "G",
        "unit": "%",
        "direction": "higher",
    },
]

# Baseline sector parameters for indicator distributions
SECTOR_INDICATOR_BASELINES: Dict[str, Dict[str, float]] = {
    "Renewable Power": {
        "renewable_share_pct": 78.0,
        "energy_intensity": 28.0,
        "emission_intensity": 14.0,
        "water_intensity": 8.5,
        "waste_recovery_pct": 72.0,
        "ltifr": 0.55,
        "women_workforce_pct": 21.0,
        "independent_directors_pct": 55.0,
    },
    "Water": {
        "renewable_share_pct": 62.0,
        "energy_intensity": 35.0,
        "emission_intensity": 18.0,
        "water_intensity": 12.0,
        "waste_recovery_pct": 80.0,
        "ltifr": 0.45,
        "women_workforce_pct": 24.0,
        "independent_directors_pct": 52.0,
    },
    "Waste and Bio": {
        "renewable_share_pct": 55.0,
        "energy_intensity": 52.0,
        "emission_intensity": 26.0,
        "water_intensity": 18.0,
        "waste_recovery_pct": 86.0,
        "ltifr": 0.85,
        "women_workforce_pct": 22.0,
        "independent_directors_pct": 50.0,
    },
    "EV and Batteries": {
        "renewable_share_pct": 48.0,
        "energy_intensity": 62.0,
        "emission_intensity": 32.0,
        "water_intensity": 16.0,
        "waste_recovery_pct": 68.0,
        "ltifr": 0.90,
        "women_workforce_pct": 26.0,
        "independent_directors_pct": 54.0,
    },
    "Efficiency and Grid": {
        "renewable_share_pct": 58.0,
        "energy_intensity": 34.0,
        "emission_intensity": 19.0,
        "water_intensity": 10.0,
        "waste_recovery_pct": 70.0,
        "ltifr": 0.50,
        "women_workforce_pct": 25.0,
        "independent_directors_pct": 53.0,
    },
}


def find_next_trading_day(filed_date_str: str, calendar_dates: List[str]) -> str:
    """Finds the earliest trading day strictly after the filed_date."""
    filed_dt = pd.to_datetime(filed_date_str)
    for c_date in calendar_dates:
        if pd.to_datetime(c_date) > filed_dt:
            return c_date
    raise ValueError(f"No trading day found after {filed_date_str} in calendar.")


def generate_esg_facts(seed: Optional[int] = None) -> pd.DataFrame:
    """Generates the 320 synthetic ESG observation facts (data/esg_facts.csv).

    Args:
        seed: Random seed for deterministic generation. If None, uses SEED_STREAMS['esg'].

    Returns:
        pd.DataFrame: Validated esg_facts DataFrame with 320 rows.
    """
    if seed is None:
        seed = config.SEED_STREAMS.get("esg", config.MASTER_SEED + 6)

    rng = np.random.default_rng(seed)

    # 1. Load dependencies
    company_master = load("company_master", use_mock=False)
    latent_profiles = load("latent_profiles", use_mock=False)
    calendar = load("trading_calendar", use_mock=False)

    calendar_dates = calendar["date"].astype(str).tolist()

    # Pre-map company sector and latent profiles
    company_sectors = dict(zip(company_master["company_id"], company_master["sector"]))

    # Create lookup map for latent ESG scores: (company_id, fiscal_year) -> score
    latent_map = {}
    for _, row in latent_profiles.iterrows():
        latent_map[(row["company_id"], row["fiscal_year"])] = float(row["esg_latent_score"])

    rows: List[Dict[str, Any]] = []
    fact_id = 1

    # Standard filing date map: mid-July of the following calendar year
    # FY2021-22 -> filed in July 2022
    filing_years = {
        "FY2021-22": 2022,
        "FY2022-23": 2023,
        "FY2023-24": 2024,
        "FY2024-25": 2025,
    }

    companies = sorted(company_master["company_id"].tolist())

    for fy in ESG_FISCAL_YEARS:
        f_year = filing_years[fy]
        filed_date_str = f"{f_year}-07-15"
        available_date_str = find_next_trading_day(filed_date_str, calendar_dates)

        for cid in companies:
            sector = company_sectors.get(cid, "Renewable Power")
            baselines = SECTOR_INDICATOR_BASELINES.get(sector, SECTOR_INDICATOR_BASELINES["Renewable Power"])
            z_esg = latent_map.get((cid, fy), 0.0)

            source_doc = f"BRSR_SYNTH_{cid}_{fy}.pdf"

            for ind in INDICATOR_METADATA:
                ind_id = ind["indicator_id"]
                ind_name = ind["indicator_name"]
                pillar = ind["pillar"]
                unit = ind["unit"]
                direction = ind["direction"]

                # Base page range depending on pillar
                if pillar == "E":
                    source_page = int(rng.integers(32, 49))
                elif pillar == "S":
                    source_page = int(rng.integers(50, 63))
                else:
                    source_page = int(rng.integers(64, 78))

                # Determine Missingness Status (NR / NA / value)
                # Blueprint: "about 6% NR (12% for water and waste)"
                is_water_or_waste = ind_id in ("water_intensity", "waste_recovery_pct")
                nr_prob = 0.125 if is_water_or_waste else 0.05
                na_prob = 0.015  # Small realistic non-applicability

                draw = rng.uniform(0.0, 1.0)
                if draw < na_prob:
                    status = "NA"
                    val = np.nan
                elif draw < (na_prob + nr_prob):
                    status = "NR"
                    val = np.nan
                else:
                    status = "value"
                    base_val = baselines[ind_id]
                    noise = rng.normal(0.0, 1.0)

                    # Econometric modulation using z_esg in [-2, +2]
                    if ind_id == "renewable_share_pct":
                        raw = base_val + 14.0 * z_esg + 4.5 * noise
                        val = round(float(np.clip(raw, 5.0, 99.0)), 1)
                    elif ind_id == "energy_intensity":
                        raw = base_val * np.exp(-0.22 * z_esg + 0.10 * noise)
                        val = round(float(np.clip(raw, 8.0, 130.0)), 2)
                    elif ind_id == "emission_intensity":
                        raw = base_val * np.exp(-0.25 * z_esg + 0.12 * noise)
                        val = round(float(np.clip(raw, 2.5, 90.0)), 2)
                    elif ind_id == "water_intensity":
                        raw = base_val * np.exp(-0.20 * z_esg + 0.15 * noise)
                        val = round(float(np.clip(raw, 1.5, 65.0)), 2)
                    elif ind_id == "waste_recovery_pct":
                        raw = base_val + 11.0 * z_esg + 4.0 * noise
                        val = round(float(np.clip(raw, 10.0, 98.0)), 1)
                    elif ind_id == "ltifr":
                        raw = max(0.0, base_val - 0.22 * z_esg + 0.12 * noise)
                        val = round(float(np.clip(raw, 0.0, 4.5)), 2)
                    elif ind_id == "women_workforce_pct":
                        raw = base_val + 5.5 * z_esg + 2.5 * noise
                        val = round(float(np.clip(raw, 8.0, 50.0)), 1)
                    elif ind_id == "independent_directors_pct":
                        raw = base_val + 7.0 * z_esg + 3.0 * noise
                        val = round(float(np.clip(raw, 33.3, 80.0)), 1)
                    else:
                        val = round(float(base_val + noise), 2)

                rows.append({
                    "fact_id": fact_id,
                    "company_id": cid,
                    "fiscal_year": fy,
                    "indicator_id": ind_id,
                    "indicator_name": ind_name,
                    "pillar": pillar,
                    "unit": unit,
                    "value": val,
                    "status": status,
                    "source_doc": source_doc,
                    "source_page": source_page,
                    "filed_date": filed_date_str,
                    "available_date": available_date_str,
                    "data_source": config.DATA_SOURCE,
                })
                fact_id += 1

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    """Execution entrypoint: generates, validates, and saves esg_facts.csv and esg_scores.csv."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Generating synthetic ESG facts (data/esg_facts.csv)...")

    df_facts = generate_esg_facts()
    validate_dataset_schema(df_facts, "esg_facts")
    facts_path = save(df_facts, "esg_facts", is_mock=False)
    logger.info(f"Successfully generated and validated {len(df_facts)} ESG facts at: {facts_path}")

    logger.info("Scoring universe and generating ESG scores (data/esg_scores.csv)...")
    df_scores = score_universe(df_facts)
    validate_dataset_schema(df_scores, "esg_scores")
    scores_path = save(df_scores, "esg_scores", is_mock=False)
    logger.info(f"Successfully generated and validated {len(df_scores)} ESG scores at: {scores_path}")


if __name__ == "__main__":
    main()
