# SIRPS Member 2 — Market Data Engine & Analytics Specification

**Project:** Sustainable Investment Return Prediction System (SIRPS)  
**Role:** Member 2 — Synthetic Market Data, Price Engine & Company Explorer  
**Document Type:** Technical Architecture & Subsystem Specification  
**Document Version:** 1.0.0 (Member 2 Final Release)  
**Git Branch:** `member-2-integration`  
**Foundation Baseline:** `member-1-integration`  
**Master Random Seed:** `42`  
**Data Provenance:** `SYNTHETIC_DEMO`  

---

## 1. Project and Member 2 Scope

Member 2 owns the **Synthetic Market Data, Price Engine & Company Explorer** components of the SIRPS research platform. In accordance with the authoritative project specification (`SIRPS_Prototype_Blueprint_and_Work_Distribution.pdf`, Page 8), Member 2's finalized responsibilities encompass:

1. Generating the official NSE synthetic trading calendar across the 6-year period 2020–2025 (`data/trading_calendar.csv`).
2. Synthesizing the daily market benchmark index (NIFTY 50 price proxy and Total Return Index / TRI) alongside an empirically calibrated India VIX volatility proxy (`data/index_daily.csv`).
3. Generating the corporate actions log (stock splits and cash dividends) across the 10-company universe (`data/corporate_actions.csv`).
4. Generating daily OHLCV bars and split-adjusted close prices for all 10 companies incorporating planted research signals (Momentum, Quality, and point-in-time ESG) (`data/prices_daily.csv`).
5. Implementing the interactive Company Explorer analytical module (`modules/company_explorer.py`) in Streamlit.
6. Authoring comprehensive data dictionary documentation (`docs/member_2_data_dictionary.md`) and validation protocols (`docs/member_2_validation_protocol.md`).
7. Providing comprehensive automated test suites covering market data generation, mathematical properties, and interactive UI behaviors (`tests/test_market_data.py`, `tests/test_company_explorer.py`).

**Scope Boundary Notice:** Member 2 explicitly does not implement Member 3 ESG disclosure generation or BRSR scoring, Member 4 quarterly fundamental balance sheet statements or regime classification models, Member 5 feature engineering pipelines, Member 6 machine learning models, or Member 7 portfolio optimization dashboards.

---

## 2. Technical Capabilities & Delivered Subsystems

- **Subsystem 1 — Mathematical Blueprint & Contract Specification:** Formulated the econometric blueprint for benchmark jump-diffusion, correlated volatility dynamics, corporate actions adjustments, and multi-factor company returns.
- **Subsystem 2 — Trading Calendar & Market Benchmark:** Generated the 1,526-day NSE exchange trading calendar (`data_generation/gen_calendar.py`) and synthesized the NIFTY 50 price index, NIFTY TRI, and India VIX proxy (`data_generation/gen_index.py`).
- **Subsystem 3 — Stochastic Volatility Calibration:** Formulated and calibrated the continuous mean-reverting Ornstein-Uhlenbeck VIX model with correlated standard normal market shocks ($\rho_v = 0.35$ leverage correlation) ensuring strictly positive values without artificial clipping.
- **Subsystem 4 — Corporate Actions & Company Price Simulation:** Implemented the corporate actions engine (`data_generation/gen_actions.py`, 52 actions) and simulated 15,260 daily OHLCV bars across all 10 companies (`data_generation/gen_prices.py`) with planted Momentum, Quality, and point-in-time ESG signals.
- **Subsystem 5 — Interactive Company Explorer UI Module:** Developed Module 2 (`modules/company_explorer.py`) in Streamlit featuring dynamic candlestick/line charts, raw vs. adjusted price toggles, causal full-history moving averages (MA50/MA200), benchmark excess returns, drawdown analytics, and CSV exports.
- **Subsystem 6 — Verification Harness & System Documentation:** Authored the data dictionary (`docs/member_2_data_dictionary.md`), validation protocol (`docs/member_2_validation_protocol.md`), and automated test suites (`tests/test_market_data.py`, `tests/test_company_explorer.py`), achieving 87/87 passing tests across the repository.

---

## 3. Production Datasets and Row Counts

All six production datasets are frozen in `data/`:

| Dataset Name | Relative Path | Rows | Columns | Primary Key | Foreign Key Reference |
|:---|:---|---:|---:|:---|:---|
| **Company Master** | `data/company_master.csv` | 10 | 17 | `company_id` | *(Root Table)* |
| **Latent Profiles** | `data/latent_profiles.csv` | 60 | 12 | `company_id, fiscal_year` | `company_id -> company_master` |
| **Trading Calendar** | `data/trading_calendar.csv` | 1,526 | 5 | `date` | *(Calendar Root)* |
| **Index Daily** | `data/index_daily.csv` | 1,526 | 5 | `date` | `date -> trading_calendar` |
| **Prices Daily** | `data/prices_daily.csv` | 15,260 | 9 | `company_id, date` | `company_id -> company_master`<br/>`date -> trading_calendar` |
| **Corporate Actions** | `data/corporate_actions.csv` | 52 | 7 | `action_id` | `company_id -> company_master`<br/>`ex_date -> trading_calendar` |

---

## 4. Generation Scripts

The reproducible generation scripts reside in `data_generation/`:

1. `data_generation/gen_calendar.py`: Generates the complete 2020–2025 NSE trading calendar with holiday exclusions, trading day indexing, and month-end indicators.
2. `data_generation/gen_index.py`: Simulates the synthetic NIFTY 50 benchmark, NIFTY TRI, and calibrated India VIX proxy under five macroeconomic regimes.
3. `data_generation/gen_actions.py`: Generates stock split records (C03 on 2022-08-11, C06 on 2023-04-13) and annual dividend payments (exactly 5 annual dividends per company for 2021–2025 = 50 annual dividends) across all 10 companies.
4. `data_generation/gen_prices.py`: Generates daily OHLCV bars, split adjustment multipliers, and adjusted close prices incorporating planted signals (Momentum, Quality, and point-in-time ESG).

---

## 5. Company Explorer Capabilities

The Company Explorer (`modules/company_explorer.py`) exposes a comprehensive interactive interface for evaluating the fictional universe:

- **Company Universe Selector:** Dropdown selector covering all 10 companies (`C01` to `C10`) displaying company name, ticker, and sector badge.
- **Date Range Filters:** Preset quick-selection buttons (Last 1 Year / 252 sessions, Last 3 Years / 756 sessions, Full Simulation / 1,526 sessions) and custom interactive start/end date pickers.
- **Warmup Period Notice:** Visual banner displayed when viewing data within the 2020 warmup year.
- **Summary KPI Banner:** Four core cards:
  1. *Latest Close Price* (₹ value, raw vs. adjusted toggle awareness).
  2. *Period Return* (% change over filtered window).
  3. *Annualised Volatility* (annualized standard deviation of daily log returns).
  4. *Cumulative Excess Return vs. NIFTY TRI* (arithmetic excess return vs. benchmark, with subtext displaying empirical $\beta$).
- **Tab 1: Price and Volume:**
  - Interactive Candlestick or Line price chart.
  - Price display mode toggle: **Split-Adjusted Close** vs. **Raw Close**.
  - Moving average overlays: MA50 (orange dashed) and MA200 (purple dashed) with causal lookback preservation.
  - Corporate action vertical markers indicating cash dividend and stock split events.
  - Trading volume bar chart colored by daily price direction.
- **Tab 2: Benchmark Comparison:**
  - Normalized rebased comparison chart (Company vs. NIFTY TRI rebased to 100 on start date).
  - Cumulative Excess Return spread chart showing historical performance differential.
- **Tab 3: Risk and Returns:**
  - Peak-to-trough historical drawdown chart and Maximum Drawdown (MDD) metric.
  - 21-day rolling annualized volatility chart.
  - Daily return distribution histogram with kernel density estimation and summary statistical table.
- **Tab 4: Corporate Actions:**
  - Tabular breakdown of historical dividends and stock splits with ex-dates and adjustment factors.
- **Tab 5: Fundamentals & ESG Preview:**
  - Preview card displaying quarterly fundamentals with explicit mock-data disclosure notice.
- **Tab 6: Historical Data & Export:**
  - Paginated, chronologically sorted OHLCV tabular view with direct CSV download button.

---

## 6. Data Contracts and Temporal Conventions

Downstream consumers (Members 3, 4, 5, 6, and 7) must observe the following conventions:

1. **Company Key:** The canonical join key across all modules is `company_id` (domain: `'C01'` through `'C10'`). Tickers should not be used as primary relational keys.
2. **Date Key & Joins:** Daily market tables must be joined using `date` (ISO-8601 `'YYYY-MM-DD'`). Positional joins are strictly prohibited.
3. **Latent Profile Joins:** Annual latent profiles (`data/latent_profiles.csv`) must be joined using `company_id` and calendar `year` (`int64`), matching the `year` column in `data/trading_calendar.csv`. Note that `data/latent_profiles.csv` contains both `fiscal_year` (string, e.g. `'FY2020-21'`) and integer `year` (`int64`, e.g. `2020`); the schema primary key is `['company_id', 'fiscal_year']`, while temporal joins with market/calendar tables use `['company_id', 'year']`. These keys are complementary and not contradictory.
4. **Price Mode for Modelling:** Predictive models (Member 6) and feature pipelines (Member 5) must use `adj_close` for return calculations to prevent artificial return spikes on stock split ex-dates.
5. **Warmup Year (2020):** 2020 data (256 sessions, indices 0–255) is reserved for indicator warmup (e.g., rolling returns, moving averages, initial volatility). Primary backtest evaluation should begin in 2021 (starting at index 256). Note: 252 is the standard annualisation scaling convention ($\sqrt{252}$), whereas the actual exchange trading calendar in 2020 contains 256 sessions (and 1,526 sessions across the full 2020–2025 simulation).
6. **Point-in-Time Lag:** Annual ESG information is published on July 15 of each calendar year. Downstream models must not use year $Y$'s ESG score prior to July 15 of year $Y$.

---

## 7. Test Results

The full unrestricted repository test suite passes with zero failures, errors, or warnings:

```
============================= test session starts =============================
platform win32 -- Python 3.10.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\srinj\Downloads\SIRPS
configfile: pytest.ini
testpaths: tests
collected 87 items

tests/test_archetypes.py .......                                         [  8%]
tests/test_company_explorer.py ..............                            [ 24%]
tests/test_integration.py ....                                           [ 28%]
tests/test_interface.py .....                                            [ 34%]
tests/test_market_data.py ....................................           [ 75%]
tests/test_pipeline_reproducibility.py ............                      [ 89%]
tests/test_temporal_conventions.py ..                                    [ 91%]
tests/validate_data.py ...                                               [100%]

============================= 87 passed in 7.37s ==============================
```

*Static Analysis & Verification:* Python bytecode compilation (`python -m compileall -q .`) completed with zero errors across all modules. (Note: Ruff and Black linters were unavailable on the execution environment; code quality was verified via pytest (87/87 tests passing) and Python bytecode compilation. Comprehensive verification confirmed schema compliance, primary/foreign key integrity, and absence of numerical anomalies.)

---

## 8. Dataset Integrity and SHA-256 Checksums

All six production CSV files have been verified against their established cryptographic checksums:

| Dataset | Relative Path | Expected SHA-256 Checksum | Verification Status |
|:---|:---|:---|:---:|
| `company_master.csv` | `data/company_master.csv` | `a9aa2c6ad892ce114e8b737485d94857e9712376d64c8ebf67862807cf5cf543` | **MATCH (100%)** |
| `latent_profiles.csv` | `data/latent_profiles.csv` | `3bfb6d477d3f6f52d1b204e08051064f16c608d4b64e5465c636d7b435fb9cfb` | **MATCH (100%)** |
| `trading_calendar.csv` | `data/trading_calendar.csv` | `a6a54485d86f3702008a9e087f43968458736b69c3c9ba7dee29a42280ee41dd` | **MATCH (100%)** |
| `index_daily.csv` | `data/index_daily.csv` | `3bd95805b448218980679146ef0a020bf3a40d004084df149d90cd5f37b7489b` | **MATCH (100%)** |
| `prices_daily.csv` | `data/prices_daily.csv` | `3c1b41fbd1d308dd43d12b4ea7fe0812b3c1fe74460d4d598775ae194af383f9` | **MATCH (100%)** |
| `corporate_actions.csv` | `data/corporate_actions.csv` | `7ea5058ceaa8501bb43965f097b3197e5d2bbcdad96754c298ad76db8af944d9` | **MATCH (100%)** |

All 21 mock fixture files under `data/mock/` remain completely untouched and identical to the foundation baseline.

---

## 9. Branch and Integration Architecture

- **Active Branch:** `member-2-integration`
- **Foundation Baseline:** `member-1-integration`
- **Integration Boundary:** Member 2 builds upon the Member 1 foundation schemas, company master, and latent profiles, providing the market data layer for downstream workstreams (Members 3–7).

---

## 10. Files Added and Modified

### Tracked Modifications:
- `tests/test_archetypes.py`: Isolated missing-dataset test via monkeypatch to test `FileNotFoundError` without relying on filesystem absence.

### Member 2 Deliverables:
- **Datasets:**
  - `data/trading_calendar.csv`
  - `data/index_daily.csv`
  - `data/corporate_actions.csv`
  - `data/prices_daily.csv`
- **Generators:**
  - `data_generation/gen_calendar.py`
  - `data_generation/gen_index.py`
  - `data_generation/gen_actions.py`
  - `data_generation/gen_prices.py`
- **Application Modules:**
  - `modules/company_explorer.py`
- **Documentation:**
  - `docs/member_2_data_dictionary.md`
  - `docs/member_2_validation_protocol.md`
  - `docs/member_2_system_specification.md`
- **Test Suites:**
  - `tests/test_market_data.py`
  - `tests/test_company_explorer.py`

---

## 11. Known Limitations

1. **Synthetic Nature:** All financial data is generated for research demonstration and educational evaluation. It must not be cited as empirical market history.
2. **Preview Mode for Fundamentals:** The quarterly fundamentals preview in Company Explorer currently displays mock data from `data/mock/fundamentals_quarterly.csv`. Downstream Member 4 will replace this with production fundamental accounting statements upon integration.
3. **Daily Granularity:** Intraday bars (tick or minute-level) are not generated; daily OHLCV bars model intraday high/low ranges via Garman-Klass volatility scaling.

---

## 12. Dependencies on Member 3 and Member 4

- **Member 3 (ESG Disclosures & Scores):** Member 3 will consume `data/company_master.csv` and `data/latent_profiles.csv` to generate annual BRSR facts and ESG Pillar scores. Member 2's planted signal `ESG_SIGNAL = 0.04` will be evaluated downstream against Member 3's published ESG scores.
- **Member 4 (Fundamentals & Market Regimes):** Member 4 will consume `data/company_master.csv`, `data/trading_calendar.csv`, and `data/index_daily.csv` to generate quarterly fundamental statements (`fundamentals_quarterly.csv`) and daily regime classifications (`regime_daily.csv`). Upon Member 4 completion, Company Explorer's Tab 5 fundamentals preview can be toggled to production mode.

---

## 13. Exact Downstream Integration Instructions

To integrate Member 2 market data into downstream pipelines:

```python
from common.io import load_dataset

# 1. Load canonical market data
calendar_df = load_dataset("trading_calendar", use_mock=False)
index_df = load_dataset("index_daily", use_mock=False)
prices_df = load_dataset("prices_daily", use_mock=False)
actions_df = load_dataset("corporate_actions", use_mock=False)

# 2. Join company prices with benchmark index
company_prices = prices_df[prices_df["company_id"] == "C01"]
merged_df = company_prices.merge(index_df, on="date", how="inner")

# 3. Calculate total return using adjusted close to prevent split shocks
merged_df["daily_return"] = merged_df["adj_close"].pct_change()
merged_df["bench_return"] = merged_df["nifty_tri"].pct_change()

# 4. Join annual latent profiles using 'company_id' and calendar 'year'
profiles_df = load_dataset("latent_profiles", use_mock=False)
calendar_lookup = calendar_df[["date", "year"]].drop_duplicates()
merged_df = merged_df.merge(calendar_lookup, on="date", how="inner")
merged_df = merged_df.merge(profiles_df, on=["company_id", "year"], how="inner")
```
