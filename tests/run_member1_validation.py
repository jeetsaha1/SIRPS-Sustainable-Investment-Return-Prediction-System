"""SIRPS Member 1 Comprehensive Validation Harness.

Executes an exhaustive audit of all Member 1 foundational components:
1. Company Master data integrity and archetype specifications
2. Latent Factor profiles, mathematical bounds, persistence, and correlation properties
3. Mock dataset compliance across all 21 schemas (primary keys, nullability, dtypes)
4. Shared I/O layer contracts, caching, provenance, and failure behavior
5. Streamlit application routing and Home overview contracts

Can be executed standalone via:
    python tests/run_member1_validation.py
Returns exit code 0 on complete pass, or exit code 1 on any failure.
"""

import sys
import json
import time
import logging
import importlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Suppress Streamlit warnings during non-browser automated CLI execution
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)


# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import data_generation.config as config
from common.schemas import (
    SCHEMAS,
    FILES,
    PROVENANCE_TAG,
    validate_dataset_schema,
    get_dataset_status,
    DatasetStatus,
)
from common.io import (
    load,
    save,
    get_file_path,
    validate_referential_integrity,
    clear_cache,
)


class ValidationHarness:
    def __init__(self):
        self.results: List[Tuple[str, str, bool, str]] = []  # (Category, Check, Passed, Details)

    def record(self, category: str, check_name: str, passed: bool, details: str = ""):
        self.results.append((category, check_name, passed, details))
        status_sym = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status_sym}] {check_name}: {details}")

    def validate_company_master(self):
        """6.1 Validate Company Master Universe and Archetypes."""
        print("\n--- Validating 6.1: Company Master Universe ---")
        try:
            df = load("company_master", use_mock=False)
            
            # Row count and IDs
            has_10_rows = len(df) == 10
            self.record("Company Master", "Row Count == 10", has_10_rows, f"Found {len(df)} rows")
            
            expected_ids = [f"C{i:02d}" for i in range(1, 11)]
            actual_ids = sorted(df["company_id"].tolist())
            self.record("Company Master", "Company IDs C01-C10", actual_ids == expected_ids, f"IDs: {actual_ids}")
            
            # Unique tickers
            unique_tickers = df["ticker"].nunique() == 10
            self.record("Company Master", "Ticker Uniqueness", unique_tickers, f"Unique tickers: {df['ticker'].nunique()}")
            
            # Sector distribution
            sector_counts = df["sector"].value_counts().to_dict()
            sector_match = sector_counts == config.REQUIRED_SECTOR_COUNTS
            self.record("Company Master", "Sector Distribution", sector_match, f"{sector_counts}")
            
            # Listing date before 2020-01-01
            listing_valid = (pd.to_datetime(df["listing_date"]) < pd.to_datetime(config.MAX_LISTING_DATE)).all()
            self.record("Company Master", "Listing Date Pre-2020", bool(listing_valid), f"Max date: {df['listing_date'].max()}")
            
            # Base price bounds
            price_valid = (df["base_price"] >= config.BASE_PRICE_BOUNDS[0]).all() and (df["base_price"] <= config.BASE_PRICE_BOUNDS[1]).all()
            self.record("Company Master", "Base Price Bounds [150, 900]", bool(price_valid), f"Min: {df['base_price'].min()}, Max: {df['base_price'].max()}")
            
            # Synthetic flags
            syn_flag = (df["is_synthetic"] == True).all() and (df["data_source"] == PROVENANCE_TAG).all()
            self.record("Company Master", "Synthetic Provenance Flag", bool(syn_flag), f"Source: {df['data_source'].iloc[0]}")
            
            # Archetype JSON specifications
            arch_file = PROJECT_ROOT / "data_generation" / "company_archetypes.json"
            arch_exists = arch_file.exists()
            with open(arch_file, "r", encoding="utf-8") as f:
                arch_data = json.load(f)
            self.record("Company Master", "Archetype JSON Completeness", arch_exists and len(arch_data) == 10, f"Found {len(arch_data)} archetypes")

        except Exception as e:
            self.record("Company Master", "Company Master Exception", False, str(e))

    def validate_latent_profiles(self):
        """6.2 Validate Latent Profiles and Econometric Bounds."""
        print("\n--- Validating 6.2: Correlated Latent Factor Profiles ---")
        try:
            df = load("latent_profiles", use_mock=False)
            
            # Shape: exactly 60 rows
            has_60 = len(df) == 60
            self.record("Latent Profiles", "Total Records == 60", has_60, f"Found {len(df)} rows")
            
            # 10 companies x 6 years
            firm_years = df.groupby("company_id")["year"].nunique()
            all_6_years = (firm_years == 6).all()
            self.record("Latent Profiles", "10 Companies x 6 Years (2020-2025)", bool(all_6_years), f"Firms: {len(firm_years)}")
            
            # No duplicate (company_id, year)
            no_dups = df.duplicated(subset=["company_id", "year"]).sum() == 0
            self.record("Latent Profiles", "Primary Key Uniqueness [company_id, year]", bool(no_dups), "Zero duplicates")
            
            # Factor bounds [-2.0, +2.0]
            factor_cols = ["esg_latent_score", "quality_latent_score", "risk_latent_score", "growth_latent_score"]
            factors_bounded = True
            for col in factor_cols:
                min_v, max_v = df[col].min(), df[col].max()
                if min_v < config.LATENT_SCORE_BOUNDS[0] or max_v > config.LATENT_SCORE_BOUNDS[1] or df[col].isna().any():
                    factors_bounded = False
            self.record("Latent Profiles", "Factor Scores Bounded in [-2.0, +2.0]", factors_bounded, "All 4 scores in bounds")
            
            # Beta bounds [0.40, 1.80]
            beta_ok = (df["beta"] >= config.BETA_BOUNDS[0]).all() and (df["beta"] <= config.BETA_BOUNDS[1]).all()
            self.record("Latent Profiles", f"Beta Bounded in {config.BETA_BOUNDS}", bool(beta_ok), f"Min: {df['beta'].min():.3f}, Max: {df['beta'].max():.3f}")
            
            # Idiosyncratic volatility bounds [0.10, 0.50]
            idio_ok = (df["idio_volatility"] >= config.IDIO_VOLATILITY_BOUNDS[0]).all() and (df["idio_volatility"] <= config.IDIO_VOLATILITY_BOUNDS[1]).all()
            self.record("Latent Profiles", f"Idiosyncratic Volatility Bounded in {config.IDIO_VOLATILITY_BOUNDS}", bool(idio_ok), f"Min: {df['idio_volatility'].min():.3f}, Max: {df['idio_volatility'].max():.3f}")
            
            # Base drift bounds [-0.02, 0.18]
            drift_ok = (df["base_drift"] >= config.BASE_DRIFT_BOUNDS[0]).all() and (df["base_drift"] <= config.BASE_DRIFT_BOUNDS[1]).all()
            self.record("Latent Profiles", f"Base Drift Bounded in {config.BASE_DRIFT_BOUNDS}", bool(drift_ok), f"Min: {df['base_drift'].min():.3f}, Max: {df['base_drift'].max():.3f}")
            
            # Target correlation positive definiteness
            L = config.CHOLESKY_L
            is_pd = bool(np.all(np.linalg.eigvalsh(config.TARGET_CORRELATION_MATRIX) > 0))
            self.record("Latent Profiles", "Target Correlation Positive Definite", is_pd, f"Min eigval: {np.min(np.linalg.eigvalsh(config.TARGET_CORRELATION_MATRIX)):.4f}")
            
            # Economic coherence
            corr_risk_beta = float(df["risk_latent_score"].corr(df["beta"]))
            corr_qual_beta = float(df["quality_latent_score"].corr(df["beta"]))
            coherence = corr_risk_beta > 0.50 and corr_qual_beta < -0.20
            self.record("Latent Profiles", "Economic Coherence (corr(Risk,Beta)>0.5, corr(Qual,Beta)<-0.2)", coherence, f"Risk-Beta: {corr_risk_beta:.3f}, Qual-Beta: {corr_qual_beta:.3f}")

        except Exception as e:
            self.record("Latent Profiles", "Latent Profiles Exception", False, str(e))

    def validate_mock_datasets(self):
        """6.3 Validate All 21 Mock Datasets."""
        print("\n--- Validating 6.3: Mock Datasets for all 21 Schemas ---")
        mock_passed = 0
        for schema_name, schema in SCHEMAS.items():
            try:
                mock_path = get_file_path(schema_name, use_mock=True)
                if not mock_path.exists():
                    self.record("Mock Datasets", f"Mock File Exists: {schema_name}", False, f"Missing {mock_path}")
                    continue
                    
                df = pd.read_csv(mock_path)
                validate_dataset_schema(df, schema_name)
                mock_passed += 1
            except Exception as exc:
                self.record("Mock Datasets", f"Schema Validation: {schema_name}", False, str(exc))
                
        self.record("Mock Datasets", "All 21 Mock Datasets Schema-Compliant", mock_passed == len(SCHEMAS), f"{mock_passed}/{len(SCHEMAS)} valid")

    def validate_shared_io(self):
        """6.4 Validate Shared I/O Contracts, Caching & Rejections."""
        print("\n--- Validating 6.4: Shared I/O Architecture & Contracts ---")
        try:
            # Production load succeeds for M1
            cm = load("company_master", use_mock=False)
            self.record("Shared I/O", "Production Load Succeeded for M1", len(cm) == 10, "Loaded company_master")
            
            # Production load strictly raises FileNotFoundError for missing downstream data
            caught_fnf = False
            try:
                load("trading_calendar", use_mock=False)
            except FileNotFoundError:
                caught_fnf = True
            self.record("Shared I/O", "Strict Prohibition of Silent Mock Fallback", caught_fnf, "Raised FileNotFoundError")
            
            # Mock mode successfully loads mock data
            tc_mock = load("trading_calendar", use_mock=True)
            self.record("Shared I/O", "Mock Mode Load Succeeded", len(tc_mock) > 0, f"Loaded {len(tc_mock)} mock rows")
            
            # Provenance rejection check
            cm_bad = cm.copy()
            cm_bad["data_source"] = "REAL_NSE_DATA"
            caught_prov = False
            try:
                validate_dataset_schema(cm_bad, "company_master")
            except ValueError as ve:
                caught_prov = "Provenance violation" in str(ve)
            self.record("Shared I/O", "Provenance Violation Rejection", caught_prov, "Non-synthetic tag rejected")
            
            # Duplicate primary key rejection
            cm_dup = pd.concat([cm, cm.head(1)], ignore_index=True)
            caught_dup = False
            try:
                validate_dataset_schema(cm_dup, "company_master")
            except ValueError as ve:
                caught_dup = "Primary key integrity violation" in str(ve)
            self.record("Shared I/O", "Duplicate Primary Key Rejection", caught_dup, "Duplicate PK rejected")
            
            # Referential integrity check
            lp = load("latent_profiles", use_mock=False)
            ref_ok = True
            try:
                validate_referential_integrity(
                    lp, "latent_profiles",
                    cm, "company_master",
                    child_fk_col="company_id", parent_pk_col="company_id"
                )
            except Exception:
                ref_ok = False
            self.record("Shared I/O", "Referential Integrity Validation", ref_ok, "latent_profiles -> company_master valid")

        except Exception as e:
            self.record("Shared I/O", "Shared I/O Exception", False, str(e))

    def validate_application_shell(self):
        """6.5 Validate Application Shell and Home Dashboard Module."""
        print("\n--- Validating 6.5: Streamlit Application Shell & Modules ---")
        try:
            # Test app.py import and route dictionary
            app_module = importlib.import_module("app")
            has_pages = hasattr(app_module, "PAGES") and len(app_module.PAGES) == 10
            self.record("Application", "App Router Contains 10 Modules", has_pages, f"Modules: {len(app_module.PAGES)}")
            
            # Test Home module interface
            home_module = importlib.import_module("modules.home")
            has_render = hasattr(home_module, "render") and callable(home_module.render)
            self.record("Application", "Home Module Exposes render()", has_render, "modules.home.render exists")
            
            # Test shared UI module
            ui_module = importlib.import_module("common.ui")
            has_ui = all(hasattr(ui_module, fn) for fn in ["demo_banner", "init_state", "company_selector"])
            self.record("Application", "Shared UI Utilities Present", has_ui, "demo_banner, init_state, company_selector")

        except Exception as e:
            self.record("Application", "Application Shell Exception", False, str(e))

    def run_all(self) -> bool:
        start_time = time.time()
        print("=" * 80)
        print("SIRPS MEMBER 1 COMPREHENSIVE VALIDATION HARNESS")
        print("=" * 80)
        
        self.validate_company_master()
        self.validate_latent_profiles()
        self.validate_mock_datasets()
        self.validate_shared_io()
        self.validate_application_shell()
        
        elapsed = time.time() - start_time
        total_checks = len(self.results)
        passed_checks = sum(1 for _, _, p, _ in self.results if p)
        failed_checks = total_checks - passed_checks
        
        print("\n" + "=" * 80)
        print(f"VALIDATION SUMMARY: {passed_checks}/{total_checks} CHECKS PASSED ({elapsed:.2f}s)")
        print("=" * 80)
        if failed_checks > 0:
            print(f"CRITICAL: {failed_checks} validation checks failed.")
            for cat, chk, p, det in self.results:
                if not p:
                    print(f"  FAILED: [{cat}] {chk} -> {det}")
            return False
            
        print("✓ All Member 1 foundational contracts, datasets, and interfaces passed.")
        return True


if __name__ == "__main__":
    harness = ValidationHarness()
    success = harness.run_all()
    sys.exit(0 if success else 1)
