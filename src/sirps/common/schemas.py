"""Versioned table contracts shared by all SIRPS modules."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

SCHEMA_VERSION = "1.0"

SCHEMAS: dict[str, tuple[str, ...]] = {
    "company_master": (
        "company_id",
        "name",
        "isin",
        "nse_symbol",
        "sector",
        "sub_sector",
        "universe_group",
        "eligibility_evidence",
        "entry_date",
        "exit_date",
    ),
    "prices_daily": (
        "company_id",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "source",
        "ingest_date",
    ),
    "fundamentals_pit": (
        "company_id",
        "period_end",
        "period_type",
        "item",
        "value",
        "basis",
        "filed_timestamp",
        "available_date",
    ),
    "esg_facts": (
        "company_id",
        "fiscal_year",
        "indicator_id",
        "value",
        "unit_raw",
        "unit_std",
        "status",
        "source_doc",
        "page",
        "filed_timestamp",
        "available_date",
        "restated_flag",
        "entered_by",
        "verified_by",
    ),
    "regime_daily": (
        "date",
        "close",
        "ma50",
        "ma200",
        "drawdown_252",
        "raw_regime",
        "confirmed_regime",
    ),
    "model_dataset": (
        "company_id",
        "date",
        "entry_date",
        "target_end_date",
        "features",
        "y_excess",
        "y_abs",
        "regime",
        "has_esg",
        "esg_available_date",
    ),
    "predictions": (
        "run_id",
        "experiment",
        "model",
        "company_id",
        "date",
        "y_hat_excess",
        "y_hat_abs",
        "lower80",
        "upper80",
        "model_version",
        "data_cutoff",
    ),
    "latent_profiles": (
        "latent_id",
        "company_id",
        "market_factor",
        "financial_factor",
        "esg_factor",
        "risk_factor",
        "seed",
        "source_label",
    ),
}

ALLOWED_ESG_STATUSES = frozenset({"value", "NA", "NR", "NC"})
ALLOWED_UNIVERSE_GROUPS = frozenset({"core", "transition", "control"})


def validate_columns(frame: pd.DataFrame, schema: str) -> None:
    """Raise ValueError when a table is missing required columns."""
    try:
        required = SCHEMAS[schema]
    except KeyError as error:
        raise ValueError(f"Unknown SIRPS schema: {schema}") from error

    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{schema} is missing required columns: {', '.join(missing)}")


def validate_allowed_values(frame: pd.DataFrame, schema: str) -> None:
    """Validate controlled vocabulary fields without changing the input frame."""
    validate_columns(frame, schema)
    if schema == "esg_facts":
        values = set(frame["status"].dropna().astype(str))
        invalid = sorted(values - ALLOWED_ESG_STATUSES)
        if invalid:
            raise ValueError(f"Invalid ESG status values: {', '.join(invalid)}")
    if schema == "company_master":
        values = set(frame["universe_group"].dropna().astype(str))
        invalid = sorted(values - ALLOWED_UNIVERSE_GROUPS)
        if invalid:
            raise ValueError(f"Invalid universe_group values: {', '.join(invalid)}")


def validate_schema(
    frame: pd.DataFrame, schema: str, *, metadata: Mapping[str, str] | None = None
) -> None:
    """Validate columns, controlled values, and optional schema metadata."""
    validate_allowed_values(frame, schema)
    if metadata is not None and metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema version {SCHEMA_VERSION}")
