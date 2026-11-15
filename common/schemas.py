"""SIRPS Common Schemas and Data Contract Definitions.

This module provides the central schema definitions for all datasets in the SIRPS
prototype, specifying column names, expected data types, primary keys, foreign keys,
and integrity constraints.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

class DatasetStatus(str, Enum):
    """Lifecycle status classification for datasets within SIRPS architecture."""
    FOUNDATION_GENERATED = "FOUNDATION_GENERATED"
    MOCK_FIXTURE = "MOCK_FIXTURE"
    FUTURE_MEMBER_GENERATED = "FUTURE_MEMBER_GENERATED"
    NOT_YET_AVAILABLE = "NOT_YET_AVAILABLE"


@dataclass(frozen=True)
class SchemaDefinition:
    """Specification of a dataset schema."""
    name: str
    relative_path: str
    primary_key: List[str]
    required_columns: List[str]
    column_dtypes: Dict[str, str]
    date_columns: List[str] = field(default_factory=list)
    nullable_columns: List[str] = field(default_factory=list)
    foreign_keys: Dict[str, str] = field(default_factory=dict)
    owner: str = "M1"
    description: str = ""

# Mandatory synthetic provenance identifier
PROVENANCE_TAG: str = "SYNTHETIC_DEMO"

# Logical name to relative file path mapping
FILES: Dict[str, str] = {
    # M1 Foundational Datasets
    "company_master": "data/company_master.csv",
    "latent_profiles": "data/latent_profiles.csv",
    
    # M2 Market Datasets
    "trading_calendar": "data/trading_calendar.csv",
    "index_daily": "data/index_daily.csv",
    "prices_daily": "data/prices_daily.csv",
    "corporate_actions": "data/corporate_actions.csv",
    
    # M3 ESG Datasets
    "esg_facts": "data/esg_facts.csv",
    "esg_scores": "data/esg_scores.csv",
    
    # M4 Financial & Regime Datasets
    "fundamentals_quarterly": "data/fundamentals_quarterly.csv",
    "regime_daily": "data/regime_daily.csv",
    "portfolio_default": "results/portfolio_default.csv",
    
    # M5 Dataset & Feature Engineering
    "model_dataset": "data/model_dataset.csv",
    "latest_features": "data/latest_features.csv",
    
    # M6 ML & Model Results
    "fold_definitions": "results/fold_definitions.csv",
    "predictions": "results/predictions.csv",
    "latest_predictions": "results/latest_predictions.csv",
    "metrics_summary": "results/metrics_summary.csv",
    "experiment_results": "results/experiment_results.csv",
    "placebo_distribution": "results/placebo_distribution.csv",
    "feature_importance": "results/feature_importance.csv",
    "local_contributions": "results/local_contributions.csv",
}

# Detailed Schema Catalog
SCHEMAS: Dict[str, SchemaDefinition] = {
    "company_master": SchemaDefinition(
        name="company_master",
        relative_path="data/company_master.csv",
        primary_key=["company_id"],
        required_columns=[
            "company_id", "company_name", "ticker", "sector", "sub_sector",
            "universe_group", "listing_date", "base_price", "is_synthetic", "data_source",
            # Research characteristics
            "business_model", "capital_intensity", "growth_profile", "revenue_stability",
            "technology_dependence", "business_risk_type", "profile_version"
        ],
        column_dtypes={
            "company_id": "string",
            "company_name": "string",
            "ticker": "string",
            "sector": "string",
            "sub_sector": "string",
            "universe_group": "string",
            "listing_date": "string",
            "base_price": "float64",
            "is_synthetic": "bool",
            "business_model": "string",
            "capital_intensity": "string",
            "growth_profile": "string",
            "revenue_stability": "string",
            "technology_dependence": "string",
            "business_risk_type": "string",
            "profile_version": "string",
            "data_source": "string"
        },
        date_columns=["listing_date"],
        owner="M1",
        description="Universe master of 10 fictional sustainable companies."
    ),
    
    "latent_profiles": SchemaDefinition(
        name="latent_profiles",
        relative_path="data/latent_profiles.csv",
        primary_key=["company_id", "fiscal_year"],
        required_columns=[
            "company_id", "fiscal_year", "year", "esg_latent_score", "quality_latent_score",
            "risk_latent_score", "growth_latent_score", "beta", "idio_volatility",
            "base_drift", "profile_version", "data_source"
        ],
        column_dtypes={
            "company_id": "string",
            "fiscal_year": "string",
            "year": "int64",
            "esg_latent_score": "float64",
            "quality_latent_score": "float64",
            "risk_latent_score": "float64",
            "growth_latent_score": "float64",
            "beta": "float64",
            "idio_volatility": "float64",
            "base_drift": "float64",
            "profile_version": "string",
            "data_source": "string"
        },
        foreign_keys={"company_id": "company_master.company_id"},
        owner="M1",
        description="Annual latent profiles (60 rows = 10 firms × 6 years) driving cross-module consistency."
    ),
    
    "trading_calendar": SchemaDefinition(
        name="trading_calendar",
        relative_path="data/trading_calendar.csv",
        primary_key=["date"],
        required_columns=["date", "trading_day_index", "year", "is_month_end", "data_source"],
        column_dtypes={
            "date": "string",
            "trading_day_index": "int64",
            "year": "int64",
            "is_month_end": "bool",
            "data_source": "string"
        },
        date_columns=["date"],
        owner="M2",
        description="NSE synthetic trading day calendar 2020-2025 (~1500 days)."
    ),
    
    "index_daily": SchemaDefinition(
        name="index_daily",
        relative_path="data/index_daily.csv",
        primary_key=["date"],
        required_columns=["date", "nifty_close", "nifty_tri", "india_vix", "data_source"],
        column_dtypes={
            "date": "string",
            "nifty_close": "float64",
            "nifty_tri": "float64",
            "india_vix": "float64",
            "data_source": "string"
        },
        date_columns=["date"],
        foreign_keys={"date": "trading_calendar.date"},
        owner="M2",
        description="Daily NIFTY 50 price index, TRI, and India VIX."
    ),
    
    "prices_daily": SchemaDefinition(
        name="prices_daily",
        relative_path="data/prices_daily.csv",
        primary_key=["company_id", "date"],
        required_columns=["company_id", "date", "open", "high", "low", "close", "adj_close", "volume", "data_source"],
        column_dtypes={
            "company_id": "string",
            "date": "string",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "adj_close": "float64",
            "volume": "int64",
            "data_source": "string"
        },
        date_columns=["date"],
        foreign_keys={"company_id": "company_master.company_id", "date": "trading_calendar.date"},
        owner="M2",
        description="Daily OHLCV and total-return adjusted close for all 10 companies."
    ),
    
    "corporate_actions": SchemaDefinition(
        name="corporate_actions",
        relative_path="data/corporate_actions.csv",
        primary_key=["action_id"],
        required_columns=["action_id", "company_id", "ex_date", "action_type", "ratio_or_amount", "description", "data_source"],
        column_dtypes={
            "action_id": "string",
            "company_id": "string",
            "ex_date": "string",
            "action_type": "string",
            "ratio_or_amount": "float64",
            "description": "string",
            "data_source": "string"
        },
        date_columns=["ex_date"],
        foreign_keys={"company_id": "company_master.company_id"},
        owner="M2",
        description="Stock splits, bonus issues, and cash dividends."
    ),
    
    "esg_facts": SchemaDefinition(
        name="esg_facts",
        relative_path="data/esg_facts.csv",
        primary_key=["fact_id"],
        required_columns=[
            "fact_id", "company_id", "fiscal_year", "indicator_id", "indicator_name",
            "pillar", "unit", "value", "status", "source_doc", "source_page",
            "filed_date", "available_date", "data_source"
        ],
        column_dtypes={
            "fact_id": "int64",
            "company_id": "string",
            "fiscal_year": "string",
            "indicator_id": "string",
            "indicator_name": "string",
            "pillar": "string",
            "unit": "string",
            "value": "float64",
            "status": "string",
            "source_doc": "string",
            "source_page": "int64",
            "filed_date": "string",
            "available_date": "string",
            "data_source": "string"
        },
        date_columns=["filed_date", "available_date"],
        nullable_columns=["value"],
        foreign_keys={"company_id": "company_master.company_id"},
        owner="M3",
        description="BRSR long-format raw ESG indicator reporting with missingness status."
    ),
    
    "esg_scores": SchemaDefinition(
        name="esg_scores",
        relative_path="data/esg_scores.csv",
        primary_key=["company_id", "fiscal_year"],
        required_columns=[
            "company_id", "fiscal_year", "available_date", "e_score", "s_score",
            "g_score", "esg_total", "coverage_ratio", "e_coverage", "s_coverage",
            "g_coverage", "data_source"
        ],
        column_dtypes={
            "company_id": "string",
            "fiscal_year": "string",
            "available_date": "string",
            "e_score": "float64",
            "s_score": "float64",
            "g_score": "float64",
            "esg_total": "float64",
            "coverage_ratio": "float64",
            "e_coverage": "float64",
            "s_coverage": "float64",
            "g_coverage": "float64",
            "data_source": "string"
        },
        date_columns=["available_date"],
        nullable_columns=["e_score", "s_score", "g_score", "esg_total"],
        foreign_keys={"company_id": "company_master.company_id"},
        owner="M3",
        description="Derived ESG pillar and total percentile rank scores."
    ),
    
    "fundamentals_quarterly": SchemaDefinition(
        name="fundamentals_quarterly",
        relative_path="data/fundamentals_quarterly.csv",
        primary_key=["company_id", "period_end"],
        required_columns=[
            "company_id", "period_end", "fiscal_year", "fiscal_quarter", "filed_date",
            "available_date", "revenue", "ebit", "net_profit", "interest_expense",
            "operating_cash_flow", "total_assets", "total_debt", "total_equity",
            "eps", "shares_outstanding", "data_source"
        ],
        column_dtypes={
            "company_id": "string",
            "period_end": "string",
            "fiscal_year": "string",
            "fiscal_quarter": "int64",
            "filed_date": "string",
            "available_date": "string",
            "revenue": "float64",
            "ebit": "float64",
            "net_profit": "float64",
            "interest_expense": "float64",
            "operating_cash_flow": "float64",
            "total_assets": "float64",
            "total_debt": "float64",
            "total_equity": "float64",
            "eps": "float64",
            "shares_outstanding": "float64",
            "data_source": "string"
        },
        date_columns=["period_end", "filed_date", "available_date"],
        foreign_keys={"company_id": "company_master.company_id"},
        owner="M4",
        description="Quarterly financial statements and balance sheet metrics."
    ),
    
    "regime_daily": SchemaDefinition(
        name="regime_daily",
        relative_path="data/regime_daily.csv",
        primary_key=["date"],
        required_columns=[
            "date", "nifty_close", "ma50", "ma200", "high_252", "drawdown_252",
            "raw_regime", "confirmed_regime", "regime_code", "data_source"
        ],
        column_dtypes={
            "date": "string",
            "nifty_close": "float64",
            "ma50": "float64",
            "ma200": "float64",
            "high_252": "float64",
            "drawdown_252": "float64",
            "raw_regime": "string",
            "confirmed_regime": "string",
            "regime_code": "int64",
            "data_source": "string"
        },
        date_columns=["date"],
        nullable_columns=["ma50", "ma200"],
        foreign_keys={"date": "trading_calendar.date"},
        owner="M4",
        description="Daily rule-based causal market regime classification."
    ),
    
    "model_dataset": SchemaDefinition(
        name="model_dataset",
        relative_path="data/model_dataset.csv",
        primary_key=["company_id", "date"],
        required_columns=[
            "company_id", "date", "entry_date", "target_end_date",
            "y_excess", "y_abs", "regime", "regime_code", "data_source"
        ],
        column_dtypes={
            "company_id": "string",
            "date": "string",
            "entry_date": "string",
            "target_end_date": "string",
            "y_excess": "float64",
            "y_abs": "float64",
            "regime": "string",
            "regime_code": "int64",
            "data_source": "string"
        },
        date_columns=["date", "entry_date", "target_end_date"],
        foreign_keys={"company_id": "company_master.company_id"},
        owner="M5",
        description="Unified ML training dataset with point-in-time features and 63-day forward targets."
    ),

    "latest_features": SchemaDefinition(
        name="latest_features",
        relative_path="data/latest_features.csv",
        primary_key=["company_id", "date"],
        required_columns=["company_id", "date", "regime", "regime_code", "data_source"],
        column_dtypes={
            "company_id": "string",
            "date": "string",
            "regime": "string",
            "regime_code": "int64",
            "data_source": "string"
        },
        date_columns=["date"],
        foreign_keys={"company_id": "company_master.company_id"},
        owner="M5",
        description="Latest feature vector per company as of 2025-12-31 for live inference."
    ),

    "fold_definitions": SchemaDefinition(
        name="fold_definitions",
        relative_path="results/fold_definitions.csv",
        primary_key=["fold_id"],
        required_columns=["fold_id", "train_start", "train_end", "test_start", "test_end", "data_source"],
        column_dtypes={
            "fold_id": "int64",
            "train_start": "string",
            "train_end": "string",
            "test_start": "string",
            "test_end": "string",
            "data_source": "string"
        },
        date_columns=["train_start", "train_end", "test_start", "test_end"],
        owner="M6",
        description="Walk-forward purge/embargo validation folds."
    ),

    "predictions": SchemaDefinition(
        name="predictions",
        relative_path="results/predictions.csv",
        primary_key=["experiment", "model", "company_id", "date"],
        required_columns=["experiment", "model", "company_id", "date", "y_true", "y_hat", "fold_id", "data_source"],
        column_dtypes={
            "experiment": "string",
            "model": "string",
            "company_id": "string",
            "date": "string",
            "y_true": "float64",
            "y_hat": "float64",
            "fold_id": "int64",
            "data_source": "string"
        },
        date_columns=["date"],
        owner="M6",
        description="Out-of-sample predictions across models, experiments, and folds."
    ),

    "latest_predictions": SchemaDefinition(
        name="latest_predictions",
        relative_path="results/latest_predictions.csv",
        primary_key=["experiment", "model", "company_id", "date"],
        required_columns=["experiment", "model", "company_id", "date", "y_hat_excess", "y_hat_abs", "lower80", "upper80", "data_source"],
        column_dtypes={
            "experiment": "string",
            "model": "string",
            "company_id": "string",
            "date": "string",
            "y_hat_excess": "float64",
            "y_hat_abs": "float64",
            "lower80": "float64",
            "upper80": "float64",
            "data_source": "string"
        },
        date_columns=["date"],
        owner="M6",
        description="Latest model predictions as of 2025-12-31 with 80% prediction intervals."
    ),

    "metrics_summary": SchemaDefinition(
        name="metrics_summary",
        relative_path="results/metrics_summary.csv",
        primary_key=["experiment", "model", "scope"],
        required_columns=[
            "experiment", "model", "scope", "n_dates", "n_obs", "msfe", "rmse", "mae",
            "oos_r2_zero", "oos_r2_expmean", "rank_ic_mean", "rank_ic_ir",
            "directional_accuracy", "topk_spread", "data_source"
        ],
        column_dtypes={
            "experiment": "string",
            "model": "string",
            "scope": "string",
            "n_dates": "int64",
            "n_obs": "int64",
            "msfe": "float64",
            "rmse": "float64",
            "mae": "float64",
            "oos_r2_zero": "float64",
            "oos_r2_expmean": "float64",
            "rank_ic_mean": "float64",
            "rank_ic_ir": "float64",
            "directional_accuracy": "float64",
            "topk_spread": "float64",
            "data_source": "string"
        },
        owner="M6",
        description="Comprehensive evaluation metrics by experiment and market scope."
    ),

    "experiment_results": SchemaDefinition(
        name="experiment_results",
        relative_path="results/experiment_results.csv",
        primary_key=["hypothesis", "model", "scope"],
        required_columns=[
            "hypothesis", "model", "scope", "delta_msfe", "ci_low", "ci_high",
            "cw_stat", "cw_pvalue", "n_dates", "inferential_flag", "note", "data_source"
        ],
        column_dtypes={
            "hypothesis": "string",
            "model": "string",
            "scope": "string",
            "delta_msfe": "float64",
            "ci_low": "float64",
            "ci_high": "float64",
            "cw_stat": "float64",
            "cw_pvalue": "float64",
            "n_dates": "int64",
            "inferential_flag": "bool",
            "note": "string",
            "data_source": "string"
        },
        owner="M6",
        description="Hypothesis test results (H1, H2, H3) with Clark-West test statistics."
    ),

    "placebo_distribution": SchemaDefinition(
        name="placebo_distribution",
        relative_path="results/placebo_distribution.csv",
        primary_key=["model", "permutation_id"],
        required_columns=["model", "permutation_id", "delta_msfe_vs_aprime", "data_source"],
        column_dtypes={
            "model": "string",
            "permutation_id": "int64",
            "delta_msfe_vs_aprime": "float64",
            "data_source": "string"
        },
        owner="M6",
        description="Empirical placebo distribution from shuffled ESG permuted runs."
    ),

    "feature_importance": SchemaDefinition(
        name="feature_importance",
        relative_path="results/feature_importance.csv",
        primary_key=["experiment", "model", "feature"],
        required_columns=["experiment", "model", "feature", "feature_group", "importance_perm", "importance_rank", "data_source"],
        column_dtypes={
            "experiment": "string",
            "model": "string",
            "feature": "string",
            "feature_group": "string",
            "importance_perm": "float64",
            "importance_rank": "int64",
            "data_source": "string"
        },
        owner="M6",
        description="Permutation feature importance rankings across models."
    ),

    "local_contributions": SchemaDefinition(
        name="local_contributions",
        relative_path="results/local_contributions.csv",
        primary_key=["model", "experiment", "company_id", "date", "feature"],
        required_columns=["model", "experiment", "company_id", "date", "feature", "feature_value", "contribution", "base_value", "data_source"],
        column_dtypes={
            "model": "string",
            "experiment": "string",
            "company_id": "string",
            "date": "string",
            "feature": "string",
            "feature_value": "float64",
            "contribution": "float64",
            "base_value": "float64",
            "data_source": "string"
        },
        date_columns=["date"],
        owner="M6",
        description="Exact additive feature contributions satisfying sum(contribution) + base_value = y_hat."
    ),

    "portfolio_default": SchemaDefinition(
        name="portfolio_default",
        relative_path="results/portfolio_default.csv",
        primary_key=["date", "strategy"],
        required_columns=["date", "strategy", "portfolio_value", "daily_return", "cumulative_return", "drawdown", "turnover", "data_source"],
        column_dtypes={
            "date": "string",
            "strategy": "string",
            "portfolio_value": "float64",
            "daily_return": "float64",
            "cumulative_return": "float64",
            "drawdown": "float64",
            "turnover": "float64",
            "data_source": "string"
        },
        date_columns=["date"],
        owner="M4",
        description="Backtest simulation results comparing top-k long, benchmark, and equal-weight strategies."
    )
}

def validate_dataset_schema(df: Any, schema_name: str) -> None:
    """Validates a DataFrame against its registered SchemaDefinition.
    
    Performs comprehensive checks:
    1. Provenance flag ('SYNTHETIC_DEMO')
    2. Column presence for all required schema fields
    3. Primary key uniqueness (no duplicate key records)
    4. Non-null constraints for non-nullable required columns
    5. Numeric finiteness (no positive or negative infinity)
    
    Raises:
        KeyError: If schema_name is not registered in SCHEMAS.
        ValueError: If any integrity or schema constraint fails.
    """
    import numpy as np
    import pandas as pd
    
    if schema_name not in SCHEMAS:
        raise KeyError(f"Unknown schema: '{schema_name}'. Registered: {list(SCHEMAS.keys())}")
        
    schema = SCHEMAS[schema_name]
    
    # 1. Provenance Check
    if "data_source" not in df.columns:
        raise ValueError(
            f"Provenance violation in '{schema_name}': missing mandatory 'data_source' column."
        )
    invalid_sources = df[df["data_source"] != "SYNTHETIC_DEMO"]
    if len(invalid_sources) > 0:
        raise ValueError(
            f"Provenance violation in '{schema_name}': found {len(invalid_sources)} rows "
            f"with data_source != 'SYNTHETIC_DEMO'."
        )
        
    # 2. Required Columns Presence
    missing_cols = [c for c in schema.required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Schema mismatch in '{schema_name}': missing required columns: {missing_cols}"
        )
        
    # 3. Primary Key Uniqueness
    if schema.primary_key:
        pk_cols = [c for c in schema.primary_key if c in df.columns]
        if len(pk_cols) == len(schema.primary_key):
            dup_mask = df.duplicated(subset=pk_cols, keep=False)
            n_dups = int(dup_mask.sum())
            if n_dups > 0:
                sample_dups = df[dup_mask][pk_cols].head(4).to_dict(orient="records")
                raise ValueError(
                    f"Primary key integrity violation in '{schema_name}': found {n_dups} duplicate "
                    f"records on key {schema.primary_key}. Sample: {sample_dups}"
                )
                
    # 4. Nullability Constraints
    for col in schema.required_columns:
        if col not in schema.nullable_columns:
            n_nulls = int(df[col].isna().sum())
            if n_nulls > 0:
                raise ValueError(
                    f"Null constraint violation in '{schema_name}': required non-nullable column "
                    f"'{col}' contains {n_nulls} null values."
                )
                
    # 5. Numeric Infinity Checks
    num_cols = df.select_dtypes(include=[np.number]).columns
    for nc in num_cols:
        n_infs = int(np.isinf(df[nc]).sum())
        if n_infs > 0:
            raise ValueError(
                f"Numeric anomaly in '{schema_name}': column '{nc}' contains {n_infs} infinite values."
            )

def get_schema(name: str) -> Optional[SchemaDefinition]:
    """Retrieves a registered SchemaDefinition by name."""
    return SCHEMAS.get(name)

def get_dataset_status(name: str, project_root: Optional[Path] = None) -> DatasetStatus:
    """Returns the current deployment status of a registered dataset.
    
    Classifies datasets as:
    - FOUNDATION_GENERATED: Production dataset generated by Member 1 foundation.
    - FUTURE_MEMBER_GENERATED: Production dataset generated by downstream member (M2–M7).
    - MOCK_FIXTURE: Sample mock dataset available for development/testing.
    - NOT_YET_AVAILABLE: Neither production nor mock dataset exists.
    """
    if name not in SCHEMAS:
        return DatasetStatus.NOT_YET_AVAILABLE
        
    schema = SCHEMAS[name]
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
        
    prod_path = project_root / schema.relative_path
    if prod_path.exists():
        if schema.owner == "M1":
            return DatasetStatus.FOUNDATION_GENERATED
        else:
            return DatasetStatus.FUTURE_MEMBER_GENERATED
            
    mock_path = project_root / "data" / "mock" / Path(schema.relative_path).name
    if mock_path.exists():
        return DatasetStatus.MOCK_FIXTURE
        
    return DatasetStatus.NOT_YET_AVAILABLE

# Aliases for convenience and backward compatibility
DATASET_SCHEMAS = SCHEMAS



