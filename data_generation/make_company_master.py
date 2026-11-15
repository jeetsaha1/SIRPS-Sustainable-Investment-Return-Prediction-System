"""SIRPS Company Master Generator (Member 1).

Generates data/company_master.csv containing exactly 10 fictional sustainable companies
with their business archetypes, sector assignments, listing dates, base prices,
and synthetic provenance flags.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from common.schemas import SCHEMAS
from common.io import save
import data_generation.config as config

ARCHETYPES_PATH: Path = Path(__file__).resolve().parent / "company_archetypes.json"

def generate_company_master(output_mock: bool = True) -> pd.DataFrame:
    """Loads company archetypes, applies rigorous validation, and saves company_master.csv."""
    if not ARCHETYPES_PATH.exists():
        raise FileNotFoundError(f"Archetype specification not found at {ARCHETYPES_PATH}")
        
    with open(ARCHETYPES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # 1. Row count validation
    if len(df) != 10:
        raise ValueError(f"Company master must contain exactly 10 companies, found {len(df)}")
        
    # 2. Company ID validation (C01 to C10)
    expected_ids = [f"C{i:02d}" for i in range(1, 11)]
    actual_ids = sorted(df["company_id"].tolist())
    if actual_ids != expected_ids:
        raise ValueError(f"Company IDs must be C01-C10. Found: {actual_ids}")
        
    # 3. Ticker uniqueness
    if df["ticker"].nunique() != 10:
        raise ValueError("Duplicate tickers detected in company master.")
        
    # 4. Sector distribution verification
    sector_counts = df["sector"].value_counts().to_dict()
    for sector, expected_count in config.REQUIRED_SECTOR_COUNTS.items():
        actual_count = sector_counts.get(sector, 0)
        if actual_count != expected_count:
            raise ValueError(
                f"Sector '{sector}' expected {expected_count} companies, found {actual_count}"
            )
            
    # 5. Base price bounds validation (₹150 to ₹900)
    min_bp, max_bp = config.BASE_PRICE_BOUNDS
    invalid_prices = df[(df["base_price"] < min_bp) | (df["base_price"] > max_bp)]
    if len(invalid_prices) > 0:
        raise ValueError(f"Base price out of bounds [{min_bp}, {max_bp}]:\n{invalid_prices[['company_id', 'base_price']]}")
        
    # 6. Listing date validation (Before 2020-01-01)
    invalid_listings = df[df["listing_date"] >= config.MAX_LISTING_DATE]
    if len(invalid_listings) > 0:
        raise ValueError(f"Listing date must be before {config.MAX_LISTING_DATE}:\n{invalid_listings[['company_id', 'listing_date']]}")
        
    # 7. Flags
    if not (df["universe_group"] == config.UNIVERSE_GROUP).all():
        raise ValueError(f"All universe_group values must be '{config.UNIVERSE_GROUP}'")
    if not (df["is_synthetic"] == config.IS_SYNTHETIC).all():
        raise ValueError(f"All is_synthetic values must be True")
    if not (df["data_source"] == config.DATA_SOURCE).all():
        raise ValueError(f"All data_source values must be '{config.DATA_SOURCE}'")
        
    # Save primary dataset
    saved_path = save(df, "company_master", is_mock=False)
    print(f"✓ Successfully generated: {saved_path} ({len(df)} companies)")
    
    if output_mock:
        mock_path = save(df, "company_master", is_mock=True)
        print(f"✓ Synchronized mock master: {mock_path}")
        
    return df

if __name__ == "__main__":
    generate_company_master()
