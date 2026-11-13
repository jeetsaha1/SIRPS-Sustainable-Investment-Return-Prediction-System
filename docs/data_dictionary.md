
# SIRPS – Data Dictionary and Schema Contract

**Project:** Sustainable Investment Return Prediction System
**Document Version:** 0.1
**Owner:** M1 – Project Lead
**Project Stage:** Prototype
**Python Version:** 3.12
**Market:** NSE, India

---

# 1. Purpose

This document defines the standard data structures, column names, data types, units and validation rules used throughout SIRPS.

All seven project members must follow this document when creating, processing, exchanging and integrating datasets.

The objective is to ensure that independently developed modules can communicate without column mismatches, inconsistent units or incompatible identifiers.

# 2. General Data Standards

## 2.1 File Formats

| Data Type | Format |
|---|---|
| Raw market data | CSV / Parquet |
| Processed datasets | Parquet |
| ESG extracted facts | CSV / Parquet |
| Predictions | CSV / Parquet |
| Configuration | YAML |
| Documentation | Markdown |

## 2.2 Naming Conventions

- Column names must use lowercase snake_case.
- Company identifiers must use `company_id`.
- Dates must use ISO format: YYYY-MM-DD.
- Monetary values must specify their currency and scale.
- Percentages must be represented consistently.
- Every dataset must have documented provenance.
- Raw data must remain separate from processed data.

## 2.3 Missing-Value Policy

Missing observations must never be automatically replaced with zero.

ESG status values:

| Status | Meaning |
|---|---|
| value | Valid reported value |
| NA | Not applicable |
| NR | Not reported |
| NC | Not comparable |

Other numerical missing values should be represented as null/NaN, with their reason documented where available.

# 3. Core Database Schemas

## 3.1 company_master

**Purpose:** Stores company identification, classification and eligibility information.

| Column | Data Type | Description |
|---|---|---|
| company_id | string | Unique company identifier |
| name | string | Official company name |
| isin | string | International Securities Identification Number |
| nse_symbol | string | NSE trading symbol(s) |
| sector | string | Industry sector |
| sub_sector | string | Industry sub-sector |
| universe_group | string | core / transition / control |
| eligibility_evidence | string | Supporting eligibility evidence |
| entry_date | date | Date company enters the panel |
| exit_date | date | Date company leaves the panel, if applicable |

**Primary key:** company_id

## 3.2 prices_daily

**Purpose:** Stores daily equity market observations.

| Column | Data Type | Description |
|---|---|---|
| company_id | string | Company identifier |
| date | date | Trading date |
| open | float | Opening price |
| high | float | Highest price |
| low | float | Lowest price |
| close | float | Closing price |
| adj_close | float | Split- and dividend-adjusted close |
| volume | integer | Trading volume |
| source | string | Data source |
| ingest_date | datetime | Data ingestion timestamp |

**Primary key:** company_id + date

**Owner:** M2

## 3.3 fundamentals_pit

**Purpose:** Stores point-in-time financial information.

| Column | Data Type | Description |
|---|---|---|
| company_id | string | Company identifier |
| period_end | date | Financial reporting period end |
| period_type | string | Quarterly / annual / TTM |
| item | string | Financial metric name |
| value | float | Reported financial value |
| basis | string | Consolidated / standalone |
| filed_timestamp | datetime | Actual filing timestamp |
| available_date | date | Date information becomes usable |

**Primary key:** company_id + period_end + period_type + item

**Owner:** M4

## 3.4 esg_facts

**Purpose:** Stores raw and standardised sustainability observations.

| Column | Data Type | Description |
|---|---|---|
| company_id | string | Company identifier |
| fiscal_year | integer | Financial reporting year |
| indicator_id | string | ESG indicator identifier |
| value | float | Extracted numerical value |
| unit_raw | string | Originally reported unit |
| unit_std | string | Standardised unit |
| status | string | value / NA / NR / NC |
| source_doc | string | Source document reference |
| page | integer | Document page number |
| filed_timestamp | datetime | Filing timestamp |
| available_date | date | Date information becomes usable |
| restated_flag | boolean | Indicates restated observation |
| entered_by | string | Initial data-entry member |
| verified_by | string | Independent verification member |

**Primary key:** company_id + fiscal_year + indicator_id

**Owner:** M3

## 3.5 regime_daily

**Purpose:** Stores market-regime classification results.

| Column | Data Type | Description |
|---|---|---|
| date | date | Trading date |
| close | float | NIFTY 50 closing value |
| ma50 | float | 50-day moving average |
| ma200 | float | 200-day moving average |
| drawdown_252 | float | 252-day drawdown |
| raw_regime | string | Initial regime classification |
| confirmed_regime | string | Confirmed Bull / Neutral / Bear regime |

**Primary key:** date

**Owner:** M4

## 3.6 model_dataset

**Purpose:** Contains the final modelling dataset generated by combining market, fundamental, technical and ESG information.

| Column | Data Type | Description |
|---|---|---|
| company_id | string | Company identifier |
| date | date | Prediction date |
| entry_date | date | Actual forecast entry date |
| target_end_date | date | Forecast exit date |
| features | structured data | Model input features |
| y_excess | float | Forward excess log return |
| y_abs | float | Forward absolute log return |
| regime | string | Start-of-window market regime |
| has_esg | boolean | Indicates ESG data availability |
| esg_available_date | date | ESG information availability date |

**Primary key:** company_id + date

**Owner:** M5

## 3.7 predictions

**Purpose:** Stores generated model predictions and associated metadata.

| Column | Data Type | Description |
|---|---|---|
| run_id | string | Experiment execution identifier |
| experiment | string | Experiment A / A' / B / C / P |
| model | string | Model name |
| company_id | string | Company identifier |
| date | date | Prediction date |
| y_hat_excess | float | Predicted excess log return |
| y_hat_abs | float | Predicted absolute log return |
| lower80 | float | Lower estimated 80% interval |
| upper80 | float | Upper estimated 80% interval |
| model_version | string | Model version |
| data_cutoff | date | Latest permissible data cutoff |

**Primary key:** run_id + experiment + model + company_id + date

**Owner:** M6 / M7

# 4. ESG Indicator Standards

The following eight indicators are pre-registered.

| Indicator ID | Indicator | Standard Unit |
|---|---|---|
| ESG01 | Renewable-energy share | Percentage |
| ESG02 | Energy intensity | Energy / turnover |
| ESG03 | Scope 1+2 emissions intensity | tCO2e / turnover |
| ESG04 | Water intensity | Water consumption / turnover |
| ESG05 | Waste recovery rate | Percentage |
| ESG06 | Lost-time injury frequency rate | Rate, documented denominator |
| ESG07 | Women in workforce | Percentage |
| ESG08 | Independent directors | Percentage |

The original reported value and unit must always be retained.

No undocumented unit conversions are permitted.

# 5. Target Calculation Standards

## 5.1 Primary Target

The primary target is the 63-trading-day forward excess log return.

y_excess = ln(P[t+64] / P[t+1]) - ln(TRI[t+64] / TRI[t+1])

Where:

- P represents the split- and dividend-adjusted equity close.
- TRI represents the NIFTY 50 Total Return Index.
- Entry occurs at t+1.
- Exit occurs at t+64.

## 5.2 Secondary Target

y_abs = ln(P[t+64] / P[t+1])

The absolute return is retained as a secondary prediction.

# 6. Data Availability and Point-in-Time Rules

1. Every financial and ESG observation must include an availability date.
2. Information becomes usable only after the filing timestamp plus one trading day.
3. First-reported values must be used for historical predictions.
4. Restated observations must be stored separately.
5. Features must not use future information.
6. Training-window statistics must be used for preprocessing.
7. Every modelling observation must retain its entry date and target end date.

# 7. Data Quality Validation

## 7.1 Market Data

- No duplicate company-date observations.
- No invalid negative prices.
- Missing trading dates must be investigated.
- Corporate-action discontinuities must be reviewed.
- Missing entry or exit prices require exclusion from target calculation.

## 7.2 ESG Data

- Every observation must have a source reference.
- Units must be validated.
- Page references must be retained.
- Missing statuses must be distinguished.
- PDF extraction must undergo independent verification.

## 7.3 Modelling Data

- Company identifiers must match company_master.
- Target dates must follow the approved forecast calendar.
- Feature availability must precede the prediction date.
- Train/test overlap must be prevented.
- Experiments A', B, C and P must use identical company-date keys and folds.

# 8. Dataset Ownership

| Dataset | Primary Owner | Reviewer |
|---|---|---|
| company_master | M1 / M5 | M2 |
| prices_daily | M2 | M5 |
| fundamentals_pit | M4 | M3 |
| esg_facts | M3 | M4 |
| regime_daily | M4 | M3 |
| model_dataset | M5 | M2 |
| predictions | M6 | M1 |
| Dashboard serving files | M7 | M1 |

# 9. Integration Rules

1. Every member must provide a two-company sample before full dataset production.
2. Dataset columns must follow this document.
3. Any schema modification requires M1 approval.
4. Breaking changes require a schema version update.
5. All shared outputs must be validated before integration.
6. No raw vendor data should be committed to the public GitHub repository.
7. Synthetic demonstration datasets must be labelled clearly.

# 10. Version History

| Version | Description | Owner |
|---|---|---|
| 0.1 | Initial schema contract | M1 |

**Status:** Draft for team review.

**Schema freeze:** End of Week 1, following member review and approval.

---

**End of Data Dictionary**
