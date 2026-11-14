"""Deterministic latent company profiles for synthetic data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_latent_profiles(
    company_ids: tuple[str, ...] = ("SYN001", "SYN002"), *, seed: int = 2026
) -> pd.DataFrame:
    """Generate hidden factors used only to make mock data internally coherent."""
    rng = np.random.default_rng(seed)
    rows = []
    for index, company_id in enumerate(company_ids, start=1):
        factors = rng.normal(0.0, 1.0, size=4)
        rows.append(
            {
                "latent_id": f"LAT{index:03d}",
                "company_id": company_id,
                "market_factor": round(float(factors[0]), 6),
                "financial_factor": round(float(factors[1]), 6),
                "esg_factor": round(float(factors[2]), 6),
                "risk_factor": round(float(factors[3]), 6),
                "seed": seed,
                "source_label": "synthetic_latent_profile",
            }
        )
    return pd.DataFrame(rows)
