# SIRPS Architecture and Data Contracts Specification

**Module:** Member 1 — Project Lead & Integrator  
**Document Status:** Official Interface Specification  
**Version:** 2.0.0 (Foundation Baseline)  
**Provenance Flag:** `SYNTHETIC_DEMO`  

---

## 1. Executive Overview and Scope

This document specifies the authoritative architectural contracts, shared I/O protocol, schema definitions, and referential integrity rules governing the Sustainable Investment Return Prediction System (SIRPS).

SIRPS is organized across seven collaborative workstreams:
- **Member 1 (Integrator & Lead):** Foundation, company universe, archetypes, latent profiles, central schemas, shared I/O, application shell, orchestration.
- **Member 2 (Market Data):** Trading calendar, market index (NIFTY 50 proxy), India VIX proxy, daily company equity prices, corporate actions.
- **Member 3 (ESG & Sustainability):** Annual ESG ratings, E/S/G pillar scores, material controversies, dispute records.
- **Member 4 (Financials & Macro):** Quarterly fundamentals, financial ratios, macroeconomic indicators, market regime labels.
- **Member 5 (Feature Engineering):** Master integrated feature dataset, point-in-time joins, rolling aggregations, label generation ($H=63$ forward returns).
- **Member 6 (Modeling & Evaluation):** Predictive models (Linear/Ridge/Lasso, Random Forest, Gradient Boosting), walk-forward validation, ablation studies, SHAP explainability.
- **Member 7 (UI & Visualization):** Multi-page Streamlit application modules, interactive visual analytics, executive dashboard.

To prevent data contamination, schema drift, and interface regressions across members, all datasets consumed or produced within SIRPS must strictly adhere to the contracts defined herein.

---

## 2. Shared I/O Interface and Caching Protocol

All data access must route through the central interface in `common/io.py`:

```python
from common.io import (
    load_dataset,
    save_dataset,
    dataset_exists,
    validate_referential_integrity,
    clear_cache,
    DATA_DIR,
    MOCK_DIR,
)
```

### 2.1 Storage Layout
- Primary Canonical Directory: `data/` (`.csv` format, UTF-8 encoded).
- Mock/Development Directory: `data/mock/` (`.csv` format, UTF-8 encoded).
- Diagnostic Outputs: `results/member_1/` (`.json` format).

### 2.2 Strict Prohibition of Silent Mock Fallback
In early prototypes, loading routines frequently fell back to mock data if a production dataset was absent. In SIRPS v2.0:
- When `use_mock=False` (default, production mode), `load_dataset(name)` **must raise `FileNotFoundError`** if `data/{name}.csv` does not exist. It will **never** silently return mock data.
- When `use_mock=True` (development/testing mode), `load_dataset(name, use_mock=True)` loads from `data/mock/{name}.csv`. If neither exists, `FileNotFoundError` is raised.

### 2.3 In-Memory LRU Caching
- `load_dataset()` utilizes `@st.cache_data` (when executed within an active Streamlit runtime) or an in-memory dictionary cache to minimize disk I/O.
- Calls to `save_dataset()` automatically evict the target dataset from cache to ensure immediate consistency.
- Explicit cache invalidation is supported via `clear_cache()`.

### 2.4 Atomic Writes and Provenance Enactment
- `save_dataset(df, name)` automatically injects `provenance='SYNTHETIC_DEMO'` if missing.
- Schema conformity is verified against `common/schemas.py:DATASET_SCHEMAS[name]` via `validate_dataset_schema(df, name)` prior to disk serialization.
- Datasets are written atomically to prevent partial-write corruption.

---

## 3. The 21 Registered Central Schemas

SIRPS registers exactly 21 canonical schemas in `common/schemas.py`. Every schema defines:
1. `name`: Canonical identifier matching the CSV filename.
2. `primary_key`: Ordered tuple of column names guaranteeing row uniqueness.
3. `columns`: Ordered dictionary of `{column_name: expected_type}`.
4. `nullable_columns`: Set of columns permitted to hold `NaN`/`None` (e.g. forward return labels at sample edges, optional corporate action terms). All other columns strictly prohibit nulls.
5. `owner`: Workstream member responsible for data generation.

| # | Schema Name | Owner | Primary Key | Total Columns | Description |
|---|---|---|---|---|---|
| 1 | `company_master` | Member 1 | `['company_id']` | 13 | Fictional 10-company universe master metadata |
| 2 | `latent_profiles` | Member 1 | `['company_id', 'year']` | 12 | Correlated latent state factors (ESG, Quality, Risk, Growth, beta) |
| 3 | `trading_calendar` | Member 2 | `['date']` | 7 | Canonical trading dates (2020–2025, ~1,500 dates) |
| 4 | `market_index` | Member 2 | `['date']` | 8 | Benchmark index OHLCV, returns, and realized volatility |
| 5 | `india_vix` | Member 2 | `['date']` | 6 | Daily implied volatility index proxy |
| 6 | `company_daily_prices` | Member 2 | `['company_id', 'date']` | 13 | Daily equity OHLCV, unadjusted/adjusted closes, turnover |
| 7 | `corporate_actions` | Member 2 | `['company_id', 'ex_date', 'action_type']` | 7 | Splits, bonus issues, and cash dividends |
| 8 | `annual_esg_ratings` | Member 3 | `['company_id', 'year']` | 9 | Synthetic ESG composite, pillar scores, grade, and percentile |
| 9 | `esg_pillar_scores` | Member 3 | `['company_id', 'year', 'pillar']` | 8 | Granular sub-pillar metrics (Carbon, Water, Labor, Governance) |
| 10 | `controversies` | Member 3 | `['controversy_id']` | 9 | Incident log, severity score, category, sentiment impact |
| 11 | `quarterly_financials` | Member 4 | `['company_id', 'period_end']` | 10 | Quarterly P&L, balance sheet items, filed date, available date |
| 12 | `financial_ratios` | Member 4 | `['company_id', 'period_end']` | 9 | Profitability (ROE/ROCE), leverage, liquidity, efficiency |
| 13 | `macro_indicators` | Member 4 | `['date']` | 8 | Repo rate, CPI inflation, 10Y G-Sec yield, USD/INR, crude oil |
| 14 | `market_regimes` | Member 4 | `['date']` | 6 | Regime classifications (Bull, Bear, High Volatility, Crisis) |
| 15 | `feature_dataset` | Member 5 | `['company_id', 'date']` | 27 | Unified feature store, engineered signals, point-in-time joins |
| 16 | `model_predictions` | Member 6 | `['model_id', 'company_id', 'date']` | 7 | Out-of-sample forward return predictions ($H=63$), residuals |
| 17 | `metrics_summary` | Member 6 | `['experiment', 'model', 'scope']` | 11 | Evaluation metrics ($R^2$, RMSE, MAE, Sharpe, IC, Rank IC) |
| 18 | `feature_importance` | Member 6 | `['model', 'feature']` | 6 | Feature importance metrics (Gain, Permutation, SHAP global) |
| 19 | `local_contributions` | Member 6 | `['model', 'company_id', 'date', 'feature']` | 6 | Instance-level local feature explanations (SHAP values) |
| 20 | `experiment_results` | Member 6 | `['experiment_id']` | 9 | Walk-forward split metadata, hyperparameter logs, run time |
| 21 | `latest_predictions` | Member 7 | `['company_id', 'model']` | 8 | Most recent production prediction snapshot for UI consumption |

---

### 3.1 Dataset Lifecycle Status Taxonomy

SIRPS formalizes a 4-tier lifecycle status taxonomy for all registered datasets via `common.schemas.DatasetStatus`:
1. `FOUNDATION_GENERATED`: Verified production dataset generated by the Member 1 foundation pipeline (`data/company_master.csv`, `data/latent_profiles.csv`).
2. `MOCK_FIXTURE`: Schema-compliant developmental sample fixtures (`data/mock/*.csv`) providing zero-blocker parallel iteration for downstream members.
3. `FUTURE_MEMBER_GENERATED`: Production dataset reserved for downstream member execution (M2–M7). Marked as pending until that member's pipeline executes.
4. `NOT_YET_AVAILABLE`: Experimental or unregistered dataset.

---

## 4. Referential Integrity Constraints and Foreign Keys

Referential integrity across tables is validated via `validate_referential_integrity(child_df, child_name, child_fk_cols, parent_name, parent_pk_cols)` in `common/io.py`.

### Primary Relational Hierarchies:
1. **Company Master Anchor (`company_master.company_id`):**
   - `latent_profiles.company_id` $\subseteq$ `company_master.company_id`
   - `company_daily_prices.company_id` $\subseteq$ `company_master.company_id`
   - `corporate_actions.company_id` $\subseteq$ `company_master.company_id`
   - `annual_esg_ratings.company_id` $\subseteq$ `company_master.company_id`
   - `quarterly_financials.company_id` $\subseteq$ `company_master.company_id`
   - `financial_ratios.company_id` $\subseteq$ `company_master.company_id`
   - `feature_dataset.company_id` $\subseteq$ `company_master.company_id`
   - `model_predictions.company_id` $\subseteq$ `company_master.company_id`
   - `latest_predictions.company_id` $\subseteq$ `company_master.company_id`
   *(Allowed Domain: Exactly the 10 fictional IDs `['C01', 'C02', ..., 'C10']`)*

2. **Temporal Trading Calendar Anchor (`trading_calendar.date`):**
   - `market_index.date` $\subseteq$ `trading_calendar.date`
   - `india_vix.date` $\subseteq$ `trading_calendar.date`
   - `company_daily_prices.date` $\subseteq$ `trading_calendar.date`
   - `macro_indicators.date` $\subseteq$ `trading_calendar.date`
   - `market_regimes.date` $\subseteq$ `trading_calendar.date`
   - `feature_dataset.date` $\subseteq$ `trading_calendar.date`
   - `model_predictions.date` $\subseteq$ `trading_calendar.date`

3. **Latent Profile Join Convention:**
   - Any join from daily market data (`date`) to annual latent profiles **must use integer calendar year (`date.year == latent_profiles.year`)**.
   - `fiscal_year` (e.g. `'FY2020-21'`) is reserved for financial disclosure and human-readable reporting, and must **not** be used as a raw join key for daily market time-series.

---

## 5. Nullability and Data Type Enforcement

1. **Primary Keys:** Must **never** contain null values (`NaN`, `None`, empty string). Primary key uniqueness is strictly checked; duplicates raise `ValueError`.
2. **Provenance Column:** Must be present in all 21 tables and strictly equal to `'SYNTHETIC_DEMO'`.
3. **Approved Numeric Bounds:**
   - Latent factor scores (`esg_latent_score`, `quality_latent_score`, `risk_latent_score`, `growth_latent_score`): Strictly bounded in $[-2.00, +2.00]$.
   - Systematic beta (`beta`): Strictly positive, bounded in $[0.40, 1.80]$.
   - Annualised idiosyncratic volatility (`idio_volatility`): Strictly positive, bounded in $[0.10, 0.50]$.
   - Annualised base drift (`base_drift`): Bounded in $[-0.02, 0.18]$.
4. **Permitted Null Columns:**
   - In `feature_dataset`: Forward return target columns (e.g. `fwd_ret_63d`) are permitted to be null for the final 63 trading days of 2025 where future returns are unobservable.
   - In `corporate_actions`: `split_ratio` is nullable for dividend actions; `dividend_rate` is nullable for split actions.

---

## 6. Integration Checklist for Downstream Members

- [ ] **Member 2 (Market Data):** Ingest `trading_calendar` dates, `company_master` tickers, and `latent_profiles` ($\beta_i, \sigma_i, \mu_i$). Produce `market_index`, `india_vix`, `company_daily_prices`, and `corporate_actions` satisfying primary key uniqueness.
- [ ] **Member 3 (ESG Data):** Ingest `latent_profiles.esg_factor` as the generative driver for `annual_esg_ratings` and `esg_pillar_scores`.
- [ ] **Member 4 (Financials):** Ingest `latent_profiles` (`quality_factor`, `growth_factor`) to drive `quarterly_financials` and `financial_ratios`. Implement strict point-in-time availability dates (`available_date = period_end + 45d`).
- [ ] **Member 5 (Feature Store):** Execute point-in-time joins: `date >= available_date` for quarterly financials; `date.year == latent_profiles.year` for latent profiles. Sample every 5th trading day for prediction rows.
- [ ] **Member 6 (Modeling):** Train models on `feature_dataset` strictly preserving temporal train/test splits. Output `model_predictions`, `metrics_summary`, `feature_importance`, and `experiment_results`.
- [ ] **Member 7 (UI Application):** Route all data through `common.io.load_dataset()`. Render UI alerts if running in mock/demo mode.
