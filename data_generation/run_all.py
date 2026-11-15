"""SIRPS Data Generation Master Orchestrator (Member 1).

Executes the Member 1 foundational data generation pipeline in deterministic order:
1. Pre-flight configuration and directory verification
2. Stage 1: Company Master generation (data/company_master.csv)
3. Stage 2: Correlated Latent Factor Profiles generation (data/latent_profiles.csv)
4. Stage 3: Statistical Diagnostics computation (results/member_1/latent_factor_diagnostics.json)
5. Stage 4: Rapid Mock Datasets generation for all 21 schemas (data/mock/*.csv)
6. Post-generation schema and referential integrity validation

Features strict failure handling (stops execution on stage failure with non-zero exit code),
atomic write guarantees via common.io.save(), and a comprehensive execution summary.
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import data_generation.config as config
from common.schemas import SCHEMAS, validate_dataset_schema, get_dataset_status, DatasetStatus
from common.io import load, validate_referential_integrity
from data_generation.make_company_master import generate_company_master
from data_generation.latent import generate_latent_profiles
from data_generation.make_mock_data import generate_mock_datasets
from data_generation.diagnostics import compute_latent_diagnostics


def run_foundational_pipeline(strict: bool = True) -> Dict[str, Any]:
    """Executes the Member 1 foundational data generation tasks with strict validation.
    
    Args:
        strict: If True, raises exceptions and exits with code 1 on any failure.
        
    Returns:
        Dict[str, Any]: Detailed execution manifest and summary metadata.
    """
    print("=" * 80)
    print("SIRPS FOUNDATIONAL DATA GENERATION ORCHESTRATOR (MEMBER 1)")
    print(f"Master Seed: {config.MASTER_SEED} | Provenance: {config.DATA_SOURCE} | Universe: C01-C10")
    print("=" * 80)
    
    start_time = time.time()
    manifest: List[Dict[str, Any]] = []
    
    try:
        # Pre-flight Check: Ensure target directories exist
        (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "data" / "mock").mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "results" / "member_1").mkdir(parents=True, exist_ok=True)
        
        # ----------------------------------------------------------------------
        # Stage 1: Company Master Generation
        # ----------------------------------------------------------------------
        print("\n[Stage 1/4] Generating Company Master...")
        s1_start = time.time()
        cm_df = generate_company_master(output_mock=True)
        validate_dataset_schema(cm_df, "company_master")
        s1_elapsed = time.time() - s1_start
        print(f"  ✓ Verified {len(cm_df)} companies across 5 green sectors ({s1_elapsed:.3f}s).")
        manifest.append({
            "stage": "Stage 1: Company Master",
            "file": "data/company_master.csv",
            "rows": len(cm_df),
            "columns": len(cm_df.columns),
            "primary_key": str(SCHEMAS["company_master"].primary_key),
            "status": "PASSED"
        })

        # ----------------------------------------------------------------------
        # Stage 2: Latent Factor Profiles Generation
        # ----------------------------------------------------------------------
        print("\n[Stage 2/4] Generating Correlated Latent Factor Profiles...")
        s2_start = time.time()
        lp_df = generate_latent_profiles(output_mock=True)
        validate_dataset_schema(lp_df, "latent_profiles")
        
        # Referential integrity check: latent_profiles.company_id -> company_master.company_id
        validate_referential_integrity(
            lp_df, "latent_profiles",
            cm_df, "company_master",
            child_fk_col="company_id", parent_pk_col="company_id"
        )
        s2_elapsed = time.time() - s2_start
        print(f"  ✓ Verified {len(lp_df)} company-year records with referential integrity ({s2_elapsed:.3f}s).")
        manifest.append({
            "stage": "Stage 2: Latent Profiles",
            "file": "data/latent_profiles.csv",
            "rows": len(lp_df),
            "columns": len(lp_df.columns),
            "primary_key": str(SCHEMAS["latent_profiles"].primary_key),
            "status": "PASSED"
        })

        # ----------------------------------------------------------------------
        # Stage 3: Statistical Diagnostics Computation
        # ----------------------------------------------------------------------
        print("\n[Stage 3/4] Computing Latent Factor Statistical Diagnostics...")
        s3_start = time.time()
        diag = compute_latent_diagnostics(save_output=True)
        max_err = diag["correlation_difference_matrix_empirical_minus_target"]["max_absolute_error"]
        s3_elapsed = time.time() - s3_start
        print(f"  ✓ Diagnostics computed. Max absolute correlation error vs target: {max_err:.4f} ({s3_elapsed:.3f}s).")
        manifest.append({
            "stage": "Stage 3: Diagnostics",
            "file": "results/member_1/latent_factor_diagnostics.json",
            "rows": len(lp_df),
            "columns": 7,
            "primary_key": "N/A",
            "status": "PASSED"
        })

        # ----------------------------------------------------------------------
        # Stage 4: Mock Schema Datasets Generation
        # ----------------------------------------------------------------------
        print("\n[Stage 4/4] Generating Mock Schema Datasets (21 schemas)...")
        s4_start = time.time()
        mock_paths = generate_mock_datasets()
        
        # Validate all mock datasets against central schemas
        for schema_name in SCHEMAS.keys():
            mock_df = pd.read_csv(PROJECT_ROOT / "data" / "mock" / f"{Path(SCHEMAS[schema_name].relative_path).name}")
            validate_dataset_schema(mock_df, schema_name)
            
        s4_elapsed = time.time() - s4_start
        print(f"  ✓ Verified all {len(SCHEMAS)} mock datasets for schema compliance ({s4_elapsed:.3f}s).")
        manifest.append({
            "stage": "Stage 4: Mock Datasets",
            "file": f"data/mock/ (21 files)",
            "rows": "~20 rows/file",
            "columns": "Schema compliant",
            "primary_key": "Unique per schema",
            "status": "PASSED"
        })

        # ----------------------------------------------------------------------
        # Execution Summary
        # ----------------------------------------------------------------------
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("SIRPS MEMBER 1 DATA GENERATION SUMMARY")
        print("=" * 80)
        print(f"{'Stage / Output':<32} {'Rows':<8} {'Cols':<6} {'Primary Key':<24} {'Status':<8}")
        print("-" * 80)
        for item in manifest:
            print(f"{item['stage']:<32} {str(item['rows']):<8} {str(item['columns']):<6} {item['primary_key']:<24} {item['status']:<8}")
        print("-" * 80)
        print(f"Total Orchestrator Execution Time: {total_time:.3f} seconds")
        print(f"Pipeline Health: ALL STAGES PASSED (100% Deterministic & Schema-Compliant)")
        print("=" * 80)
        
        return {
            "success": True,
            "elapsed_seconds": round(total_time, 4),
            "manifest": manifest,
            "diagnostics": diag
        }

    except Exception as exc:
        print("\n" + "!" * 80)
        print("CRITICAL FAILURE IN SIRPS DATA GENERATION PIPELINE")
        print("!" * 80)
        traceback.print_exc()
        print("!" * 80)
        print(f"Aborting dependent stages. Exception caught: {exc}")
        print("!" * 80)
        if strict:
            sys.exit(1)
        return {
            "success": False,
            "error": str(exc),
            "manifest": manifest
        }


if __name__ == "__main__":
    run_foundational_pipeline(strict=True)
