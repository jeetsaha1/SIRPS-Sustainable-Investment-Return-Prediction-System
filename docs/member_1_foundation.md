# SIRPS Member 1 — Research Foundation & Architectural Blueprint

**Project:** Sustainable Investment Return Prediction System (SIRPS)  
**Role:** Member 1 — Project Lead & Integrator  
**Document Version:** 2.0.0 (Foundation Baseline)  
**Master Random Seed:** `42`  
**Dataset Provenance:** `SYNTHETIC_DEMO`  

---

## 1. Project Background and System Architecture

### 1.1 System Objective & Research Scope
The Sustainable Investment Return Prediction System (SIRPS) is an econometric and machine learning research platform designed to evaluate whether sustainability disclosures (such as SEBI Business Responsibility and Sustainability Reports / BRSR) contain incremental predictive signal for multi-asset equity returns in Indian capital markets.

The research prototype implements a synthetic demonstration universe (`SYNTHETIC_DEMO`) to mathematically validate data contracts, factor correlations, corporate actions adjustments, and point-in-time temporal alignment before live market ingestion.

### 1.2 Core Specifications and Architecture
1. **System Objective**: Build an end-to-end research prototype to test whether SEBI BRSR ESG disclosures provide incremental predictive power for 63-trading-day forward excess returns over baseline technical and fundamental models.
2. **Methodological Transparency**: All entities, prices, and ESG facts are explicitly synthetic demonstration data (`SYNTHETIC_DEMO`). The research prototype serves to prove the econometric validity and code integrity of the pipeline prior to ingesting empirical NSE/BRSR disclosures.
3. **Role of Member 1 (Project Lead & Integrator)**: Establish the architecture, central schemas (`common/schemas.py`), secure I/O utilities (`common/io.py`), UI helpers (`common/ui.py`), global simulation configuration (`data_generation/config.py`), company universe (`data/company_master.csv`), annual latent profiles (`data/latent_profiles.csv`), main Streamlit application shell (`app.py`), development runner (`dev_run.py`), Home dashboard (`modules/home.py`), mock schema datasets, statistical diagnostics (`data_generation/diagnostics.py`), and integration test suite (`tests/`).

---

## 2. Company Universe Design (C01–C10)

### 2.1 Universe Composition & Constraints
The system establishes exactly 10 fictional sustainable companies across 5 green sectors. All entities satisfy:
- `company_id`: C01–C10 (unique primary key)
- `ticker`: 8–10 character uppercase ticker (unique)
- `universe_group`: `core`
- `listing_date`: strictly prior to `2020-01-01`
- `base_price`: ₹150.00 to ₹900.00
- `is_synthetic`: `True`
- `data_source`: `SYNTHETIC_DEMO`

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

### 2.2 Sector Distribution
- **Renewable Power:** 3 companies (C01, C02, C07)
- **Water:** 1 company (C03)
- **Waste and Bio:** 2 companies (C04, C10)
- **EV and Batteries:** 2 companies (C05, C06)
- **Efficiency and Grid:** 2 companies (C08, C09)

### 2.3 Business Archetypes & Research Characteristics
The company universe is parametrized in `data_generation/company_archetypes.json`, capturing 7 auditable research characteristics per firm:
1. `business_model`
2. `capital_intensity` (`Low-Medium`, `Medium`, `High`, `Very High`)
3. `growth_profile` (`Medium`, `High`, `Very High`)
4. `revenue_stability` (`Medium`, `High`)
5. `technology_dependence` (`Medium`, `High`, `Very High`)
6. `business_risk_type` (Sector-specific exposures such as commodity price volatility, grid curtailment, municipal receivables, and feedstock supply)
7. `profile_version` (`v1.0`)

---

## 3. Correlated Latent-Factor Econometric Architecture

### 3.1 Motivation
To avoid circular file dependencies between price generation (M2), BRSR disclosures (M3), and quarterly balance sheets (M4), all downstream generators condition their stochastic draws on an underlying annual latent profile:
$$\mathbf{X}_{i,t} = [\text{ESG}_{i,t}, \text{Quality}_{i,t}, \text{Risk}_{i,t}, \text{Growth}_{i,t}]^T$$

### 3.2 Target Correlation Structure & Mathematical Verification
The research specification dictates the following cross-factor correlation matrix:

$$\mathbf{\Sigma} = \begin{bmatrix}
1.00 & 0.30 & -0.15 & 0.10 \\
0.30 & 1.00 & -0.55 & 0.20 \\
-0.15 & -0.55 & 1.00 & 0.25 \\
0.10 & 0.20 & 0.25 & 1.00
\end{bmatrix}$$

**Symmetry and Positive Definiteness Proof:**
1. $\mathbf{\Sigma} = \mathbf{\Sigma}^T$ (Symmetric).
2. The eigenvalues of $\mathbf{\Sigma}$ computed numerically are:
   $$\lambda_1 \approx 0.29988, \quad \lambda_2 \approx 0.80391, \quad \lambda_3 \approx 1.19831, \quad \lambda_4 \approx 1.69789$$
   Since all eigenvalues $\lambda_k > 0$, the matrix is **strictly positive definite**.
3. The lower-triangular Cholesky factor $\mathbf{L}$ satisfying $\mathbf{L}\mathbf{L}^T = \mathbf{\Sigma}$ exists uniquely:
   $$\mathbf{L} = \begin{bmatrix}
   1.00000 & 0.00000 & 0.00000 & 0.00000 \\
   0.30000 & 0.95394 & 0.00000 & 0.00000 \\
   -0.15000 & -0.52938 & 0.83502 & 0.00000 \\
   0.10000 & 0.17821 & 0.43034 & 0.87923
   \end{bmatrix}$$

### 3.3 Archetype-to-Factor Mathematical Linkage
In the foundation architecture, the qualitative parameters from `company_archetypes.json` are mathematically linked to the company-level baseline factor vector $\boldsymbol{\mu}_i$:
$$\boldsymbol{\mu}_i = \boldsymbol{\mu}_{s(i)} + \boldsymbol{\delta}_i$$
where $\boldsymbol{\mu}_{s(i)}$ is the sector baseline vector and $\boldsymbol{\delta}_i$ is the firm-level adjustment computed via `compute_archetype_baseline_adjustments(company)` in `data_generation/config.py`:
- `growth_profile`: maps to $\Delta \mu_{\text{growth}}$ ($+0.00$ to $+0.12$) and $\Delta \mu_{\text{risk}}$ ($+0.00$ to $+0.08$).
- `revenue_stability`: maps to $\Delta \mu_{\text{quality}}$ ($+0.00$ to $+0.08$) and $\Delta \mu_{\text{risk}}$ ($-0.08$ to $+0.00$).
- `technology_dependence`: maps to $\Delta \mu_{\text{growth}}$ ($+0.00$ to $+0.08$) and $\Delta \mu_{\text{risk}}$ ($+0.00$ to $+0.06$).
- `capital_intensity`: maps to $\Delta \mu_{\text{risk}}$ ($+0.00$ to $+0.06$) and $\Delta \mu_{\text{quality}}$ ($-0.04$ to $+0.00$).

This guarantees that two firms within the same sector (e.g., C01 Solar vs C07 Hydro in Renewable Power) exhibit distinct, economically coherent trajectories without breaking the bounded $[-2.0, +2.0]$ factor domain. See [archetype_factor_mapping.md](file:///c:/Users/srinj/Downloads/SIRPS/docs/archetype_factor_mapping.md) for full derivations.

### 3.4 State-Space Transition Dynamics
For each firm $i$, the factor vector evolves across the 6 years (2020 through 2025):
- **Initialization at $t = 0$ (2020):**
  $$\mathbf{X}_{i,0} = \boldsymbol{\mu}_i + \boldsymbol{\sigma}_{\text{init}} \odot (\mathbf{L} \mathbf{Z}_{i,0}), \quad \mathbf{Z}_{i,0} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_4)$$
  where $\boldsymbol{\sigma}_{\text{init}} = [0.70, 0.65, 0.60, 0.60]^T$.

- **Annual AR(1) Evolution for $t = 1 \dots 5$ (2021–2025):**
  $$\mathbf{X}_{i,t} = \boldsymbol{\mu}_i + \mathbf{\Phi} \odot (\mathbf{X}_{i,t-1} - \boldsymbol{\mu}_i) + \boldsymbol{\sigma}_{\text{innov}} \odot (\mathbf{L} \mathbf{Z}_{i,t})$$
  where $\mathbf{\Phi} = [0.90, 0.85, 0.85, 0.85]^T$ and $\boldsymbol{\sigma}_{\text{innov}} = [0.20, 0.25, 0.25, 0.25]^T$.
  All scores are clamped to $[-2.0, +2.0]$.

### 3.5 Economic Coherence of Derived Parameters
To ensure structural financial consistency without double-counting risk:
1. **Systematic Beta ($\beta_{i,t}$):**
   $$\beta_{i,t} = \text{clip}\left(\beta_{\text{sector}} + 0.12 \cdot \text{Risk}_{i,t} - 0.08 \cdot \text{Quality}_{i,t} + \epsilon_{\beta}, \; 0.40, \; 1.80\right)$$
   *Empirical correlation in generated data:* $\text{corr}(\text{Risk}, \beta) = +0.7975$, $\text{corr}(\text{Quality}, \beta) = -0.4688$.
2. **Idiosyncratic Volatility ($\sigma_{\epsilon, i,t}$):**
   $$\sigma_{\epsilon, i,t} = \text{clip}\left(\sigma_{\text{sector}} + 0.04 \cdot \text{Risk}_{i,t} + 0.02 \cdot \text{Growth}_{i,t} - 0.02 \cdot \text{Quality}_{i,t} + \epsilon_{\sigma}, \; 0.10, \; 0.50\right)$$
   *Empirical correlation in generated data:* $\text{corr}(\text{Risk}, \sigma_{\epsilon}) = +0.7256$.
3. **Base Annualised Drift ($\mu_{\text{base}, i,t}$):**
   $$\mu_{\text{base}, i,t} = \text{clip}\left(\mu_{\text{drift}, \text{sector}} + 0.015 \cdot \text{Growth}_{i,t} + 0.010 \cdot \text{Quality}_{i,t} + \epsilon_{\mu}, \; -0.02, \; 0.18\right)$$

### 3.6 Empirical Correlation Verification ($N=60$ observations)
Statistical diagnostics generated by `data_generation/diagnostics.py` and stored in `results/member_1/latent_factor_diagnostics.json` demonstrate:
- **Target vs. Empirical:**
  - $\text{corr}(\text{ESG}, \text{Quality})$: Target $+0.30$, Empirical $+0.2326$ (diff: $-0.0674$)
  - $\text{corr}(\text{ESG}, \text{Risk})$: Target $-0.15$, Empirical $-0.1050$ (diff: $+0.0450$)
  - $\text{corr}(\text{ESG}, \text{Growth})$: Target $+0.10$, Empirical $+0.1647$ (diff: $+0.0647$)
  - $\text{corr}(\text{Quality}, \text{Risk})$: Target $-0.55$, Empirical $-0.6815$ (diff: $-0.1315$)
  - $\text{corr}(\text{Quality}, \text{Growth})$: Target $+0.20$, Empirical $+0.2373$ (diff: $+0.0373$)
  - $\text{corr}(\text{Risk}, \text{Growth})$: Target $+0.25$, Empirical $+0.1906$ (diff: $-0.0594$)
- Maximum absolute error is $0.1315$. Note: $1/\sqrt{60} \approx 0.129$ is an order-of-magnitude sample-size heuristic and **not** a formal asymptotic standard error for hypothesis testing, as the panel data features AR(1) temporal persistence ($\phi = 0.85\text{--}0.90$) and repeated firm measurements.
- Zero bound violations across all 60 company-year records.

---

## 4. Temporal Conventions & Non-Leakage Architecture

### 4.1 Resolution of Temporal Key Conventions
The semantics of time variables and relational keys are formalized in [temporal_conventions.md](file:///c:/Users/srinj/Downloads/SIRPS/docs/temporal_conventions.md):
- **Calendar Year (`year`):** Integer (2020–2025). Used as the **sole relational primary key** for joins between market trading data and annual latent profiles (`date.year == latent_profiles.year`).
- **Fiscal Year Label (`fiscal_year`):** Descriptive string (e.g. `'FY2020-21'`). Strictly preserved for accounting disclosures, reporting, and financial statement presentation. Never used as a raw mathematical join key for market dates.
- **Reference Dates Walkthrough:**
  - `2020-01-01`: Warm-up start (Calendar Year 2020, maps to `year=2020`, `fiscal_year='FY2020-21'`).
  - `2020-03-31`: FY2019-20 fiscal year-end.
  - `2020-04-01`: Start of Indian FY2020-21 (still within Calendar Year 2020).
  - `2025-09-30`: Q2 FY2025-26 fiscal end (Calendar Year 2025, maps to `year=2025`).
  - `2025-12-31`: Simulation cutoff (Calendar Year 2025, maps to `year=2025`).

### 4.2 Point-in-Time Availability & Anti-Leakage
To prevent look-ahead bias:
- Annual ESG BRSR disclosures for year $T$ become observable at $T+1$ (filing delay of ~105 calendar days).
- Quarterly financials for quarter ending $t$ become observable at $t + 45$ calendar days.
- Feature pipeline joins strictly enforce `market_date >= filing_available_date`.

---

## 5. Software Architecture & Shared Contracts

### 5.1 Central Schemas (`common/schemas.py`)
Centralizes metadata, column data types, required constraints, primary keys, and foreign keys for all **21 prototype datasets**. Features:
- Explicit `nullable_columns` registration per schema.
- Automated `validate_dataset_schema(df, schema_name)` executing primary key uniqueness, non-null enforcement, provenance verification, and numeric finite checks.

### 5.2 I/O Layer (`common/io.py`)
- Provides `load_dataset(name, use_mock=False)` and `save_dataset(df, name)`.
- **Eliminated silent fallback:** When `use_mock=False`, raises `FileNotFoundError` if production data is missing.
- Added `validate_referential_integrity(child_df, child_name, child_fk_cols, parent_name, parent_pk_cols)` to verify relational constraints across datasets.
- Thread-safe caching and automatic cache invalidation on save.

### 5.3 UI Layer (`common/ui.py`) & Home Dashboard (`modules/home.py`)
- Standard persistent banner (`demo_banner()`).
- Session state initialization (`init_state()`).
- Shared company selector (`company_selector()`).
- Updated `modules/home.py` to transparently distinguish completed Member 1 foundational data (`company_master.csv`, `latent_profiles.csv`) from downstream mock datasets. Removed premature claims of fabricated market price statistics.

### 5.4 Rapid Mock Datasets (`data/mock/`)
All 21 mock datasets generated via `data_generation/make_mock_data.py` strictly satisfy primary key uniqueness (zero duplicates) and schema conformance.

---

## 6. Automated Verification Results

All automated unit and integration tests execute successfully via `python -m pytest`:
- Verified 10-company universe integrity and archetypes.
- Verified latent factor bounds, persistence, and correlation properties.
- Verified schema conformance and duplicate primary key rejection.
- Verified referential integrity checks and non-silent mock fallback.
- Verified application shell and Home module contracts.
