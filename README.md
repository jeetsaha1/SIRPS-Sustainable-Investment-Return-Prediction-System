# Sustainable Investment Return Prediction System (SIRPS)

> **Academic Research Prototype** | Multi-Horizon Sustainable Alpha & BRSR ESG Integration  
> **Master Seed:** `42` | **Scope:** `prototype_plus` | **Provenance:** `SYNTHETIC_DEMO`  
> **Version:** `2.0.0` (Member 1 Foundation & Member 2 Market Data Integration)

---

## 1. Project Overview & Problem Statement

The **Sustainable Investment Return Prediction System (SIRPS)** is an academic research platform designed to investigate whether corporate sustainability disclosures—specifically quantitative reporting under India's **SEBI Business Responsibility and Sustainability Reporting (BRSR)** framework—provide statistically significant out-of-sample predictive power for medium-term equity returns ($H = 63$ trading days).

### 1.1 Problem Statement
While environmental, social, and governance (ESG) factors are increasingly prominent in quantitative finance, empirical evaluations in emerging markets face several critical econometric challenges:
1. **Lookahead Bias & Filing Lags:** Corporate sustainability reports and quarterly accounting statements are published weeks or months after fiscal period ends. Modeling must enforce strict point-in-time information availability.
2. **Endogeneity & Factor Confounding:** High sustainability ratings frequently correlate with large market capitalization, high financial quality, and lower debt levels. Isolate-able "sustainability alpha" requires controlling for standard risk factors (Market Beta, Quality, Momentum, and Growth).
3. **Non-Linear Dynamics & Explainability:** Traditional linear factor models may overlook non-linear interactions between sustainability indicators and macroeconomic volatility regimes. Machine learning models must be coupled with rigorous econometric hypothesis tests (Clark-West) and explainability metrics (SHAP).

### 1.2 Synthetic Demonstration Scope
To ensure complete academic transparency, strict deterministic reproducibility, and zero ethical or proprietary conflicts with actively traded National Stock Exchange (NSE) securities, SIRPS operates entirely within an engineered, econometrically constrained synthetic economy (`SYNTHETIC_DEMO`). Real market data or active NSE tickers are never used. All stochastic processes—including geometric jump-diffusion price paths, continuous log-Ornstein-Uhlenbeck volatility indices, corporate actions, and BRSR disclosures—are calibrated to mirror stylized empirical facts of the Indian financial market.

---

## 2. Research Objectives

1. **Foundational Architecture & Relational Contracts:** Standardize a catalog of 21 dataset schemas, enforcing primary/foreign key integrity, non-nullability constraints, and atomic file I/O across multi-member research streams.
2. **Correlated Latent State-Space Modeling:** Drive firm evolution via a 4-dimensional vector autoregressive AR(1) state space ($\text{ESG}, \text{Quality}, \text{Risk}, \text{Growth}$) parameterized with Cholesky-factored target cross-correlations and qualitative archetype baseline offsets.
3. **Realistic Synthetic Market Data & Volatility Engine:** Model a 6-year daily trading history (2020–2025, 1,526 sessions) featuring a regime-switching NIFTY 50 benchmark, Total Return Index (TRI), and an India VIX volatility proxy calibrated with empirical negative leverage dynamics ($\rho_v = 0.35$).
4. **Corporate Actions & Split Continuity:** Model stock splits and cash dividend distributions with backward split-adjustment multipliers ($C_t = \prod_{s \ge t} k_s$) to guarantee price continuity and eliminate false economic crash artifacts.
5. **Interactive Analytical Visualization:** Deliver an interactive Streamlit Company Explorer module with full-history causal moving averages (MA50/MA200), benchmark excess returns, drawdown analytics, and dataset export capabilities.

---

## 3. Seven-Member Responsibility Matrix

The SIRPS initiative divides research and development across seven specialized roles:

| Member | Role | Research & Implementation Scope | Project Status |
|:---|:---|:---|:---|
| **Member 1** | **Project Lead & Integrator** | System architecture, 21 dataset schemas, company master universe, qualitative archetypes, correlated latent factor engine, statistical diagnostics, Streamlit application shell, automated validation harness | **Complete (Foundation Baseline)** |
| **Member 2 (Current)** | **Market Data Lead** | NSE trading calendar (2020–2025), synthetic NIFTY 50 benchmark & TRI, calibrated India VIX proxy, corporate actions log, daily company OHLCV simulation, Company Explorer module | **Complete (Market Data Integration)** |
| Member 3 | ESG Framework Lead | Quantitative BRSR fact generator, missingness pattern simulation, multi-tier ESG pillar scoring engine, ESG Analysis module | Contract Specified & Mock Ready |
| Member 4 | Financials & Regime Lead | Quarterly financial accounting statements, ratio analysis, 10-day causal macroeconomic regime classification filter, Market Regime module | Contract Specified & Mock Ready |
| Member 5 | Dataset & Features Lead | Master feature engineering store, point-in-time relational joins, rolling technical indicators, 63-day forward return labeling, Return Prediction module | Contract Specified & Mock Ready |
| Member 6 | Machine Learning Lead | Purged walk-forward cross-validation, baseline and ESG-augmented ML models, Clark-West out-of-sample hypothesis tests, SHAP local/global explainability | Contract Specified & Mock Ready |
| Member 7 | Dashboard & Serving Lead | Portfolio risk analytics, mean-variance and ESG-tilted optimization backtesting, interactive executive dashboards, Methodology module | UI Shell & Router Ready |

---

## 4. Technical Architecture & System Design

```
+---------------------------------------------------------------------------------+
|                       SIRPS STREAMLIT UI SHELL (app.py)                         |
|   1. Home Dashboard  |  2. Company Explorer (M2)  |  ...  |  10. Methodology   |
+----------------------------------------+----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                         COMMON CORE INFRASTRUCTURE                              |
|   - common.schemas : 21 Data Contracts, Lifecycle Status, Type Enforcement     |
|   - common.io      : Atomic File I/O, Cache Management, Referential Integrity   |
|   - common.ui      : Persistent Demo Banner, Shared Session State, Selectors   |
+----------------------------------------+----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                   MEMBER 1 & MEMBER 2 DATA PIPELINE                             |
|   - M1: make_company_master.py -> data/company_master.csv (10 firms)           |
|   - M1: latent.py              -> data/latent_profiles.csv (60 firm-years)      |
|   - M2: gen_calendar.py        -> data/trading_calendar.csv (1,526 sessions)    |
|   - M2: gen_index.py           -> data/index_daily.csv (NIFTY + TRI + VIX)      |
|   - M2: gen_actions.py         -> data/corporate_actions.csv (52 actions)       |
|   - M2: gen_prices.py          -> data/prices_daily.csv (15,260 OHLCV bars)     |
+---------------------------------------------------------------------------------+
```

### 4.1 Econometric & Stochastic Formulations

#### Latent State-Space Evolution (Member 1)
$$\mathbf{X}_{i,t} = \boldsymbol{\mu}_i + \mathbf{\Phi} \odot (\mathbf{X}_{i,t-1} - \boldsymbol{\mu}_i) + \boldsymbol{\sigma}_{\text{innov}} \odot (\mathbf{L}\mathbf{Z}_{i,t})$$
- Target cross-factor correlation matrix $\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^T$ is strictly symmetric and positive definite.
- Qualitative archetypes in `data_generation/company_archetypes.json` determine firm baseline vectors: $\boldsymbol{\mu}_i = \boldsymbol{\mu}_{s(i)} + \boldsymbol{\delta}_i$.

#### Calibrated India VIX Volatility Proxy (Member 2)
The synthetic volatility index is modeled as a continuous mean-reverting Ornstein-Uhlenbeck process in log-volatility space with correlated market innovations:
$$x_t = \log(\text{VIX}_t)$$
$$\Delta x_t = \kappa (\log \bar{\theta}_{R_t} - x_{t-1}) \Delta t + \sigma_x \left(-\rho_v z_{m, t} + \sqrt{1 - \rho_v^2} \, z_{v, t}\right)$$
- Mean reversion speed: $\kappa = 0.08$
- Volatility of log-VIX: $\sigma_x = 0.04$
- Leverage correlation: $\rho_v = 0.35$ (market drops trigger asymmetric spikes in volatility)
- Regime targets: $\bar{\theta}_{R_t} \in [13.0, 32.0]$

#### Multi-Factor Company Price Engine (Member 2)
Daily company log-returns follow a multi-factor specification:
$$R_{i, t} = \alpha_{i, t} \Delta t + \beta_{i, t} R_{m, t} + \gamma_s R_{s, t} + \sigma_{\epsilon, i} \sqrt{\Delta t} \, z_{i, t}$$
- **Planted Signals:** Momentum ($\text{MOM} = 0.02$), Quality ($\text{QUAL} = 0.02$), and point-in-time ESG ($\text{ESG} = 0.04$).
- **Point-in-Time Lag Rule:** The annual ESG drift contribution remains exactly $0.0$ until July 15 of filing year $Y$, strictly preventing lookahead leakage.

---

## 5. Technology Stack

- **Runtime Environment:** Python 3.10 / 3.11
- **Numerical & Financial Computing:** NumPy (`>=1.24.0`), SciPy (`>=1.10.0`), Pandas (`>=2.0.0`)
- **Web Application & Interactive Visualization:** Streamlit (`>=1.30.0`), Plotly (`>=5.18.0`)
- **Testing & Quality Assurance:** Pytest (`>=7.4.0`), Python `compileall`
- **Data Architecture:** Standard CSV (UTF-8 encoding), JSON for archetypes and diagnostics

---

## 6. Project Directory Structure

```
SIRPS/
├── app.py                         # Multi-page Streamlit application router (10 modules)
├── dev_run.py                     # Single-module development runner
├── requirements.txt               # Pinned runtime dependencies
├── pytest.ini                     # Pytest suite configuration
├── README.md                      # Comprehensive academic project documentation
├── common/                        # Core shared architecture (Owner: M1)
│   ├── schemas.py                 # Central dataset catalog (21 schemas), keys, nullability
│   ├── io.py                      # Standard I/O, cache management, provenance enforcement
│   └── ui.py                      # Persistent demonstration banner, session state, selectors
├── data/                          # Production datasets (M1 Foundation & M2 Market Data)
│   ├── company_master.csv         # 10 fictional sustainable companies (M1)
│   ├── latent_profiles.csv        # 60 annual company-year latent profiles (M1)
│   ├── trading_calendar.csv       # 1,526 NSE trading days 2020–2025 (M2)
│   ├── index_daily.csv            # Daily NIFTY 50 close, NIFTY TRI, and India VIX (M2)
│   ├── prices_daily.csv           # 15,260 daily OHLCV bars across 10 companies (M2)
│   ├── corporate_actions.csv      # 52 corporate actions: 2 splits, 50 annual dividends (M2)
│   └── mock/                      # 21 schema-compliant mock CSV fixtures for development
├── data_generation/               # Econometric data generators (Owner: M1 & M2–M5)
│   ├── config.py                  # Global parameters, sector baselines, bounds, signal constants
│   ├── company_archetypes.json    # Qualitative business model specifications (C01–C10)
│   ├── make_company_master.py     # Company master generator & schema validator (M1)
│   ├── latent.py                  # Correlated 4D latent factor generator (M1)
│   ├── diagnostics.py             # Empirical statistical diagnostics utility (M1)
│   ├── make_mock_data.py          # Rapid mock dataset generator (M1)
│   ├── gen_calendar.py            # NSE trading calendar generator (M2)
│   ├── gen_index.py               # Benchmark index & calibrated VIX simulator (M2)
│   ├── gen_actions.py             # Corporate actions generator (M2)
│   ├── gen_prices.py              # Daily company OHLCV price simulator (M2)
│   └── run_all.py                 # 4-stage foundational pipeline orchestrator
├── modules/                       # Streamlit UI modules (Owner: M1–M7)
│   ├── home.py                    # Module 1: Home / Research Overview Dashboard (M1)
│   └── company_explorer.py        # Module 2: Interactive Company Explorer (M2)
├── models/                        # Econometric and machine learning models (Owner: M6)
├── results/                       # Diagnostic outputs and experimental results
│   └── member_1/                  # Member 1 factor diagnostics
│       └── latent_factor_diagnostics.json
├── tests/                         # Automated test suite and validation harness
│   ├── run_member1_validation.py  # Standalone 27-check validation harness
│   ├── test_archetypes.py         # Archetype baseline mapping & schema tests
│   ├── test_company_explorer.py   # 14 Company Explorer unit & UI analytics tests
│   ├── test_integration.py        # Reproducibility, Cholesky, & economic coherence tests
│   ├── test_interface.py          # Application shell & UI contract tests
│   ├── test_market_data.py        # 21 market data generation & mathematical tests
│   ├── test_pipeline_reproducibility.py # 12 pipeline orchestration & failure tests
│   ├── test_temporal_conventions.py     # Calendar-year vs fiscal-year key consistency tests
│   └── validate_data.py           # Data bounds, provenance, and nullability tests
└── docs/                          # Formal architectural documentation & specifications
    ├── member_1_foundation.md     # M1 architectural blueprint & econometric formulations
    ├── member_1_system_specification.md  # M1 system architecture & foundation specification
    ├── member_1_validation_protocol.md # M1 validation protocol & test runbook
    ├── member_2_data_dictionary.md     # M2 data dictionary across all 6 active datasets
    ├── member_2_system_specification.md  # M2 market data & analytics specification
    ├── member_2_validation_protocol.md # M2 validation protocol & mathematical runbook
    ├── architecture_and_data_contracts.md # Shared I/O protocol and 21 central schemas
    ├── archetype_factor_mapping.md     # Formal qualitative-to-quantitative baseline mapping
    └── temporal_conventions.md         # Date, year, fiscal-year semantics and lag rules
```

---

## 7. Fictional Company Universe (C01–C10)

The canonical universe is frozen in `data/company_master.csv` ($10 \text{ rows} \times 17 \text{ columns}$):

| ID | Company Name | Ticker | Sector | Sub-Sector | Listing Date | Base Price |
|:---|:---|:---|:---|:---|:---:|:---:|
| **C01** | Surya Grid Renewables Ltd | SURYAGRID | Renewable Power | Solar generation | 2012-06-18 | ₹412.50 |
| **C02** | Vayu Windpower Ltd | VAYUWIND | Renewable Power | Wind generation | 2014-11-04 | ₹285.00 |
| **C03** | Jaldhara Water Tech Ltd | JALDHARA | Water | Water & wastewater treatment | 2010-03-22 | ₹609.80 |
| **C04** | Navchetna Recycling Ltd | NAVCHETNA | Waste and Bio | Recycling | 2016-08-19 | ₹215.40 |
| **C05** | Voltway Mobility Ltd | VOLTWAY | EV and Batteries | Electric vehicles | 2017-04-11 | ₹540.00 |
| **C06** | Ampere Cell Energy Ltd | AMPERECELL | EV and Batteries | Battery manufacturing | 2018-09-25 | ₹780.00 |
| **C07** | Harita Hydro Power Ltd | HARITAHYD | Renewable Power | Hydro generation | 2008-01-15 | ₹320.00 |
| **C08** | GridSmart Systems Ltd | GRIDSMART | Efficiency and Grid | Grid efficiency equipment | 2015-12-07 | ₹495.00 |
| **C09** | Kiran Efficiency Solutions Ltd | KIRANEFF | Efficiency and Grid | Energy-efficiency products | 2013-05-30 | ₹185.00 |
| **C10** | Biogas Bharat Energy Ltd | BIOGASBH | Waste and Bio | Waste-to-energy | 2019-02-14 | ₹260.00 |

---

## 8. Production Datasets Summary

| Dataset | File Location | Rows | Description | Primary Key |
|:---|:---|---:|:---|:---|
| `company_master` | `data/company_master.csv` | 10 | Universe master across 5 green sectors | `company_id` |
| `latent_profiles` | `data/latent_profiles.csv` | 60 | Annual 4D correlated factor state profiles | `company_id, fiscal_year` |
| `trading_calendar` | `data/trading_calendar.csv` | 1,526 | NSE exchange calendar (2020–2025) | `date` |
| `index_daily` | `data/index_daily.csv` | 1,526 | NIFTY 50 close, NIFTY TRI, and India VIX | `date` |
| `prices_daily` | `data/prices_daily.csv` | 15,260 | Daily OHLCV and split-adjusted close | `company_id, date` |
| `corporate_actions` | `data/corporate_actions.csv` | 52 | 2 stock splits (C03, C06) and 50 dividends | `action_id` |

---

## 9. Installation & Execution Guide

### 9.1 Environment Setup
```powershell
# Clone the repository
git clone https://github.com/jeetsaha1/SIRPS-Sustainable-Investment-Return-Prediction-System.git
cd SIRPS-Sustainable-Investment-Return-Prediction-System

# Checkout Member 2 integration branch
git checkout member-2-integration

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 9.2 Generating Datasets (Controlled Pipeline)
```powershell
# Member 1 Foundation Pipeline
python data_generation/run_all.py

# Member 2 Market Data Pipeline (Optional - production files are pre-generated)
python data_generation/gen_calendar.py
python data_generation/gen_index.py
python data_generation/gen_actions.py
python data_generation/gen_prices.py
```

### 9.3 Launching the Interactive Web Application
```powershell
streamlit run app.py
```
- **Module 1 (Home Dashboard):** System overview, fictional universe summary, and project health metrics.
- **Module 2 (Company Explorer):** Interactive candlestick/line charts, raw vs. adjusted price toggles, moving averages (MA50/MA200), benchmark excess returns, drawdown analytics, and OHLCV CSV downloads.

---

## 10. Testing & Quality Assurance

### 10.1 Running the Automated Test Suite (Pytest)
```powershell
pytest -v
```
**Outcome:** **87 passed in ~7.4s** (0 failed, 0 errors, 0 warnings across all 8 test modules).

### 10.2 Standalone Validation Harness
```powershell
python tests/run_member1_validation.py
```
**Outcome:** **27/27 checks passed** covering foundational schemas, bounds compliance, AR(1) persistence, and routing.

### 10.3 Python Bytecode Compilation
```powershell
python -m compileall -q .
```
**Outcome:** Exit Code 0 (clean compilation across all repository files).

---

## 11. Future Scope & Downstream Roadmap

- **Member 3 Integration:** Ingest quantitative BRSR sustainability disclosures, missingness flags, and pillar-level ESG scores into Module 3.
- **Member 4 Integration:** Implement quarterly balance sheets, income statements, cash flows, and macroeconomic regime classifications into Module 4.
- **Member 5 Integration:** Generate point-in-time feature store with 63-day forward return prediction labels into Module 5.
- **Member 6 Integration:** Fit purged walk-forward cross-validated ML models, execute Clark-West out-of-sample tests, and compute SHAP explanations into Modules 6 and 9.
- **Member 7 Integration:** Implement mean-variance and ESG-constrained portfolio optimization backtests with interactive executive visualizations into Modules 7 and 8.

---

## 12. Academic Documentation Links

- **Member 1 Foundation Blueprint:** [`docs/member_1_foundation.md`](docs/member_1_foundation.md)
- **Member 1 System Specification:** [`docs/member_1_system_specification.md`](docs/member_1_system_specification.md)
- **Member 1 Validation Protocol:** [`docs/member_1_validation_protocol.md`](docs/member_1_validation_protocol.md)
- **Member 2 Data Dictionary:** [`docs/member_2_data_dictionary.md`](docs/member_2_data_dictionary.md)
- **Member 2 System Specification:** [`docs/member_2_system_specification.md`](docs/member_2_system_specification.md)
- **Member 2 Validation Protocol:** [`docs/member_2_validation_protocol.md`](docs/member_2_validation_protocol.md)
- **Data Contracts & Schemas:** [`docs/architecture_and_data_contracts.md`](docs/architecture_and_data_contracts.md)
- **Archetype Factor Mapping:** [`docs/archetype_factor_mapping.md`](docs/archetype_factor_mapping.md)
- **Temporal Conventions:** [`docs/temporal_conventions.md`](docs/temporal_conventions.md)
