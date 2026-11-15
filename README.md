# Sustainable Investment Return Prediction System (SIRPS)

> **Academic Research Prototype** | Multi-Horizon Sustainable Alpha & BRSR ESG Integration  
> **Master Seed:** `42` | **Scope:** `prototype_plus` | **Provenance:** `SYNTHETIC_DEMO`  
> **Version:** `1.0.0` (Member 1 Foundation Baseline)

---

## 1. Project Overview & Problem Statement

The **Sustainable Investment Return Prediction System (SIRPS)** is an academic research platform designed to investigate whether corporate sustainability metrics—specifically disclosures under India's **SEBI Business Responsibility and Sustainability Reporting (BRSR)** framework—possess statistically significant out-of-sample predictive power for medium-term equity returns ($H = 63$ trading days).

### 1.1 Problem Statement
While environmental, social, and governance (ESG) factors are widely cited in modern portfolio management, empirical academic evaluations frequently suffer from lookahead bias, unstandardized voluntary reporting, endogenous data revisions, and circular correlation structures. In emerging markets such as India, the transition from voluntary BRSR guidelines to mandatory core assurance requires a rigorous econometric framework to evaluate:
1. Whether quantitative BRSR performance metrics generate persistent alpha after controlling for standard equity risk factors (Market Beta, Quality, Momentum, and Growth).
2. How point-in-time disclosure lags (e.g., annual reporting availability) impact quantitative factor performance in realistic market simulations.
3. Whether machine learning models (e.g., Ridge, Random Forests, XGBoost) combined with modern explainability tools (SHAP) can isolate non-linear return contributions attributable strictly to sustainable operational practices.

### 1.2 Synthetic Demonstration Scope
To ensure complete academic transparency, full reproducibility, and avoid ethical or proprietary conflicts with actively traded National Stock Exchange (NSE) securities, this research prototype operates entirely on an engineered, econometrically constrained synthetic economy (`SYNTHETIC_DEMO`). Real market data or active NSE tickers are never used; instead, all stochastic processes (factor persistence, jump-diffusions, financial statements, and disclosure scores) are calibrated to mirror stylized empirical facts of the Indian equity market.

---

## 2. Research Objectives

1. **Foundational Architecture & Relational Contracts:** Establish a standardized data catalog (21 schemas), strict referential integrity rules, and atomic I/O protocols to support multi-member parallel research workflows.
2. **Coherent Latent State-Space Modeling:** Engineer a 4-dimensional correlated factor framework ($\text{ESG}, \text{Quality}, \text{Risk}, \text{Growth}$) driven by vector autoregressive AR(1) dynamics and Cholesky factorizations.
3. **Qualitative Business Archetype Grounding:** Map qualitative industry fundamentals (capital intensity, revenue predictability, technology dependence) to quantitative baseline adjustments for each fictional firm.
4. **Point-in-Time Temporal Consistency:** Formalize calendar-year vs. fiscal-year semantics, 45-day financial statement lags, and July 15 annual ESG filing release dates to strictly prevent lookahead leakage.
5. **Decoupled Rapid Prototyping:** Provide schema-compliant mock dataset fixtures across all 21 schemas to allow downstream modules (market pricing, ESG scoring, financial modeling, machine learning, and portfolio optimization) to develop independently.

---

## 3. Seven-Member Responsibility Matrix

The SIRPS initiative divides research and development across seven specialized roles:

| Member | Role | Research & Implementation Scope | Project Status |
|:---|:---|:---|:---|
| **Member 1 (Current)** | **Project Lead & Integrator** | System architecture, 21 dataset schemas, company master universe, qualitative archetypes, correlated latent factor engine, statistical diagnostics, Streamlit application shell, automated validation harness | **Complete (Foundation Baseline)** |
| Member 2 | Market Data Lead | NSE trading calendar (2020–2025), synthetic NIFTY 50 benchmark, India VIX proxy, corporate actions log, daily company OHLCV simulation, Company Explorer module | Contract Specified & Mock Ready |
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
|   1. Home Dashboard  |  2. Company Explorer  |  ...  |  10. Research Methodology|
+----------------------------------------+----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                         COMMON CORE INFRASTRUCTURE                              |
|   - common.schemas : 21 Data Contracts, Lifecycle Status, Type Enforcement     |
|   - common.io      : Atomic File I/O, Cache Invalidation, Referential Integrity|
|   - common.ui      : Persistent Demo Banner, Shared Session State, Selectors   |
+----------------------------------------+----------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                   MEMBER 1 DATA GENERATION ENGINE (run_all.py)                  |
|   - Stage 1 : make_company_master.py -> data/company_master.csv (10 firms)      |
|   - Stage 2 : latent.py             -> data/latent_profiles.csv (60 firm-years) |
|   - Stage 3 : diagnostics.py        -> results/member_1/latent_factor_diag.json|
|   - Stage 4 : make_mock_data.py     -> data/mock/*.csv (21 schema fixtures)     |
+---------------------------------------------------------------------------------+
```

### 4.1 Correlated Latent Factor Engine
Firms evolve across an annual 4-dimensional latent state space:
$$\mathbf{X}_{i,t} = \begin{bmatrix} \text{ESG}_{i,t} \\ \text{Quality}_{i,t} \\ \text{Risk}_{i,t} \\ \text{Growth}_{i,t} \end{bmatrix}$$

- **Target Cross-Factor Correlation Matrix $\mathbf{\Sigma}$:**
  Calibrated to empirical financial literature (positive ESG-Quality correlation, negative ESG-Risk correlation). Positive definiteness verified via Cholesky decomposition $\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^T$.
- **Vector Autoregressive Dynamics (AR(1)):**
  $$\mathbf{X}_{i,t} = \boldsymbol{\mu}_i + \mathbf{\Phi} \odot (\mathbf{X}_{i,t-1} - \boldsymbol{\mu}_i) + \boldsymbol{\sigma}_{\text{innov}} \odot (\mathbf{L}\mathbf{Z}_{i,t})$$
  with persistence parameter $\phi \approx 0.85\text{--}0.90$.
- **Qualitative Archetype Linkage:**
  $$\boldsymbol{\mu}_i = \boldsymbol{\mu}_{s(i)} + \boldsymbol{\delta}_i$$
  where $\boldsymbol{\delta}_i$ is derived from qualitative business archetypes (`data_generation/company_archetypes.json`).

---

## 5. Technology Stack

- **Core Runtime:** Python 3.10 / 3.11
- **Scientific Computing & Linear Algebra:** NumPy (`>=1.24.0`), SciPy (`>=1.10.0`), Pandas (`>=2.0.0`)
- **Web Application & Interactive Visualization:** Streamlit (`>=1.30.0`), Plotly (`>=5.18.0`)
- **Testing & Quality Assurance:** Pytest (`>=7.4.0`), Python `compileall`
- **Data Serialization:** Standard CSV with UTF-8 encoding, JSON for metadata and diagnostics

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
├── data/                          # Production datasets
│   ├── company_master.csv         # 10 fictional sustainable companies
│   ├── latent_profiles.csv        # 60 annual company-year latent profiles (2020–2025)
│   └── mock/                      # 21 schema-compliant mock CSV fixtures
├── data_generation/               # Econometric data generators (Owner: M1 & M2–M5)
│   ├── config.py                  # Global parameters, sector baselines, bounds, signal constants
│   ├── company_archetypes.json    # Qualitative business model specifications (C01–C10)
│   ├── make_company_master.py     # Company master generator & schema validator
│   ├── latent.py                  # Correlated 4D latent factor generator (AR(1) + Cholesky)
│   ├── diagnostics.py             # Empirical statistical diagnostics utility
│   ├── make_mock_data.py          # Rapid mock dataset generator (unique PK combinations)
│   └── run_all.py                 # 4-stage pipeline orchestrator with fail-stop handling
├── modules/                       # Streamlit UI modules (Owner: M1–M7)
│   └── home.py                    # Module 1: Home / Research Overview Dashboard
├── models/                        # Econometric and machine learning models (Owner: M6)
├── results/                       # Diagnostic outputs and experimental results
│   └── member_1/                  # Member 1 factor diagnostics
│       └── latent_factor_diagnostics.json
├── tests/                         # Automated test suite and validation harness
│   ├── run_member1_validation.py  # Standalone 27-check validation harness
│   ├── test_archetypes.py         # Archetype baseline mapping & schema tests
│   ├── test_integration.py        # Reproducibility, Cholesky, & economic coherence tests
│   ├── test_interface.py          # Application shell & UI contract tests
│   ├── test_pipeline_reproducibility.py # 12 pipeline orchestration & failure-injection tests
│   ├── test_temporal_conventions.py     # Calendar-year vs fiscal-year key consistency tests
│   └── validate_data.py           # Data bounds, provenance, and nullability tests
└── docs/                          # Formal architectural documentation & specifications
    ├── member_1_foundation.md     # Architectural blueprint & econometric formulations
    ├── member_1_system_specification.md  # System architecture & foundation specification
    ├── member_1_validation_protocol.md # Official validation protocol & test runbook
    ├── architecture_and_data_contracts.md # Shared I/O protocol and 21 central schemas
    ├── archetype_factor_mapping.md# Formal qualitative-to-quantitative baseline mapping
    └── temporal_conventions.md    # Date, year, fiscal-year semantics and lag rules
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

## 8. Installation & Execution Guide

### 8.1 Environment Setup
```powershell
# Clone the repository
git clone https://github.com/jeetsaha1/SIRPS-Sustainable-Investment-Return-Prediction-System.git
cd SIRPS-Sustainable-Investment-Return-Prediction-System

# Checkout Member 1 integration branch
git checkout member-1-integration

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 8.2 Regenerating Foundational Datasets
```powershell
python data_generation/run_all.py
```
*Executes all 4 generation stages: company master, latent profiles, statistical diagnostics, and mock fixtures with atomic write safety.*

### 8.3 Launching the Interactive Web Application
```powershell
streamlit run app.py
```
*Navigates across the prototype. Module 1 (Home) is fully functional; unmerged downstream modules cleanly display integration placeholders.*

### 8.4 Developing Modules in Isolation
```powershell
streamlit run dev_run.py -- home
```

---

## 9. Testing & Quality Assurance

### 9.1 Running the Automated Test Suite (Pytest)
```powershell
pytest -v
```
**Outcome:** **31 passed in ~3.2s** (0 failed, 0 errors, 0 warnings across all test modules).

### 9.2 Running the Comprehensive Validation Harness
```powershell
python tests/run_member1_validation.py
```
**Outcome:** **27/27 checks passed** covering primary key uniqueness, referential integrity, bounds compliance, AR(1) persistence, and application routing.

### 9.3 Python Bytecode Compilation
```powershell
python -m compileall -q .
```
**Outcome:** Exit Code 0 (clean compilation across all repository files).

---

## 10. Future Scope & Downstream Roadmap

- **Member 2 Integration:** Ingest trading calendar, synthetic NIFTY 50 and India VIX index proxies, corporate actions, and daily company OHLCV price histories into the Company Explorer.
- **Member 3 Integration:** Ingest quantitative BRSR sustainability disclosures, missingness flags, and pillar-level ESG scores.
- **Member 4 Integration:** Implement quarterly balance sheets, income statements, cash flows, and macroeconomic regime classifications.
- **Member 5 Integration:** Generate point-in-time feature store with 63-day forward return prediction labels.
- **Member 6 Integration:** Fit purged walk-forward cross-validated ML models, execute Clark-West out-of-sample tests, and compute SHAP explanations.
- **Member 7 Integration:** Implement mean-variance and ESG-constrained portfolio optimization backtests with interactive executive visualizations.

---

## 11. Academic Citation & Documentation Links

- **Foundation Blueprint:** [`docs/member_1_foundation.md`](docs/member_1_foundation.md)
- **System Specification:** [`docs/member_1_system_specification.md`](docs/member_1_system_specification.md)
- **Validation Protocol:** [`docs/member_1_validation_protocol.md`](docs/member_1_validation_protocol.md)
- **Data Contracts & Schemas:** [`docs/architecture_and_data_contracts.md`](docs/architecture_and_data_contracts.md)
- **Archetype Mapping:** [`docs/archetype_factor_mapping.md`](docs/archetype_factor_mapping.md)
- **Temporal Conventions:** [`docs/temporal_conventions.md`](docs/temporal_conventions.md)
