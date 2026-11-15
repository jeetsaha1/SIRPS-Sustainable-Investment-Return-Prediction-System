# SIRPS Member 1 Validation Protocol & Engineering Runbook

**Module:** Member 1 — Project Lead & Integrator  
**Document Status:** Official Protocol & Runbook  
**Version:** 3.0.0 (Foundation Baseline)  
**Provenance Flag:** `SYNTHETIC_DEMO`  

---

## 1. Overview and Operational Principles

This protocol specifies the standard operating procedures for executing, validating, testing, and troubleshooting the foundational Member 1 components of the Sustainable Investment Return Prediction System (SIRPS).

All development must adhere to three foundational principles:
1. **Methodological Transparency:** All generated data is synthetic demonstration data (`SYNTHETIC_DEMO`). Real market data or active NSE tickers must never be introduced.
2. **Deterministic Reproducibility:** Fixed random seeds (`MASTER_SEED = 42`) guarantee bit-for-bit identity across repeated pipeline runs.
3. **Strict Interface Integrity:** All dataset reads and writes must pass through `common.io` and conform to `common.schemas`.

---

## 2. Standard Execution Commands

### 2.1 Running the Automated Test Suite (Pytest)
To execute the complete 31-test automated suite across all validation, interface, integration, archetype, temporal, and pipeline modules:
```powershell
python -m pytest -v
```
Expected output: `31 passed in < 2.0s`.

### 2.2 Running the Comprehensive Member 1 Validation Harness
To execute the standalone, 27-check validation harness covering all 5 core foundation areas:
```powershell
python tests/run_member1_validation.py
```
Expected output:
```
================================================================================
VALIDATION SUMMARY: 27/27 CHECKS PASSED
================================================================================
✓ All Member 1 foundational contracts, datasets, and interfaces passed.
```
*Exit Code: `0` on success, `1` on any failure.*

### 2.3 Running the Foundational Data Generation Orchestrator
To execute the 4-stage Member 1 data pipeline:
```powershell
python data_generation/run_all.py
```
*Stage 1: Generates `data/company_master.csv` (10 companies)*  
*Stage 2: Generates `data/latent_profiles.csv` (60 records)*  
*Stage 3: Computes and saves `results/member_1/latent_factor_diagnostics.json`*  
*Stage 4: Generates and validates all 21 mock datasets in `data/mock/`*  

### 2.4 Running the Streamlit Application Shell
```powershell
streamlit run app.py
```

### 2.5 Developing Individual Modules in Isolation
```powershell
streamlit run dev_run.py -- home
```

---

## 3. Dataset Lifecycle & Identification (Mock vs. Production)

SIRPS distinguishes four dataset deployment statuses via `common.schemas.DatasetStatus`:

| Lifecycle Status | Definition | Storage Location | Active Datasets in Foundation Release |
|---|---|---|---|
| **`FOUNDATION_GENERATED`** | Verified production datasets produced by Member 1 | `data/*.csv` | `company_master.csv`, `latent_profiles.csv` |
| **`MOCK_FIXTURE`** | Schema-compliant sample fixtures for rapid development | `data/mock/*.csv` | All 21 schema sample tables |
| **`FUTURE_MEMBER_GENERATED`** | Production datasets assigned to downstream members | `data/*.csv` or `results/*.csv` | *Pending M2–M7 implementation* |
| **`NOT_YET_AVAILABLE`** | Datasets neither in production nor mock directories | N/A | None |

### Checking Status in Code
```python
from common.schemas import get_dataset_status, DatasetStatus

status = get_dataset_status("company_master")
print(status)  # DatasetStatus.FOUNDATION_GENERATED

status_m2 = get_dataset_status("trading_calendar")
print(status_m2)  # DatasetStatus.MOCK_FIXTURE (Production pending Member 2)
```

---

## 4. Interpreting Statistical Diagnostic Outputs

The diagnostic utility (`data_generation/diagnostics.py`) writes its structured findings to `results/member_1/latent_factor_diagnostics.json`.

Key diagnostic fields to review:
1. **`correlation_analysis`**:
   - `theoretical_target_matrix`: Population correlation matrix $\mathbf{\Sigma}$ from research specification.
   - `empirical_sample_matrix`: Sample correlation matrix computed across the 60 generated company-year observations.
   - `max_absolute_error`: Maximum absolute difference between empirical and target entries (observed: $0.1315$).
   - `statistical_interpretation`: Clarifies that $1/\sqrt{60} \approx 0.129$ is an order-of-magnitude sampling heuristic, **not** an asymptotic standard error, because the panel features AR(1) temporal persistence ($\phi \approx 0.85\text{--}0.90$) and repeated firm measurements.
2. **`factor_summary_and_bounds`**:
   - First and second moments (mean, std, min, max) for `esg_latent_score`, `quality_latent_score`, `risk_latent_score`, `growth_latent_score`, `beta`, `idio_volatility`, and `base_drift`.
   - `in_bounds`: Confirms that all values strictly respect approved parameter ranges.
3. **`economic_coherence`**:
   - `corr_risk_latent_score_vs_beta` ($+0.798$, exceeds $+0.50$ target).
   - `corr_risk_latent_score_vs_idio_volatility` ($+0.726$, exceeds $+0.50$ target).
   - `corr_quality_latent_score_vs_beta` ($-0.469$, below $-0.20$ target).
   - `coherence_check_passed`: Evaluates to `True`.

---

## 5. Troubleshooting Common Failures

### 5.1 `FileNotFoundError: Production dataset '...' does not exist`
- **Cause:** Attempting to call `load("dataset_name", use_mock=False)` on a dataset that has not yet been generated by a downstream member.
- **Solution:** In Member 1 development and testing, explicitly pass `use_mock=True` to load developmental sample data:
  ```python
  df = load("trading_calendar", use_mock=True)
  ```
- **Note:** Never modify `common.io` to re-introduce silent mock fallback in production mode.

### 5.2 `ValueError: Primary key integrity violation in '...'`
- **Cause:** A generator created records with duplicate primary key tuples.
- **Solution:** Inspect the generator loop. Ensure all combinations of composite keys (e.g., `[experiment, model, scope]` in `metrics_summary`) are unique.

### 5.3 `ValueError: Provenance violation in '...'`
- **Cause:** A dataset contains rows where `data_source != 'SYNTHETIC_DEMO'` or lacks the mandatory `data_source` column.
- **Solution:** Ensure `data_source = 'SYNTHETIC_DEMO'` is added to the DataFrame prior to saving.

### 5.4 `ValueError: Referential integrity violation`
- **Cause:** A child dataset contains foreign keys (e.g., `company_id = 'C99'`) that do not exist in the parent table (`company_master`).
- **Solution:** Verify that all foreign keys belong strictly to the 10 approved companies (`C01` through `C10`).
