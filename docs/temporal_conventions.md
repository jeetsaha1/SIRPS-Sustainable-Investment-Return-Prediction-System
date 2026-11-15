# SIRPS Temporal Conventions & Information Availability Protocol

**Module:** Member 1 — Architecture & Foundation  
**Version:** 2.0.0 (Foundation Baseline)  
**Status:** Authoritative Architectural Standard  
**Cross-References:** `common/schemas.py`, `data_generation/latent.py`, `data_generation/config.py`  

---

## 1. Executive Summary & Core Definitions

In quantitative asset pricing research, temporal alignment errors (lookahead bias, backward information leakage, and calendar mismatches) invalidate empirical conclusions. This document defines the exact joining rules and temporal boundaries across the SIRPS prototype.

### Core Temporal Fields
1. **`date`** (`str`, ISO `YYYY-MM-DD`):
   The exact calendar trading session date on the National Stock Exchange (NSE). Valid weekdays only; strictly increasing; no weekend dates.
2. **`year`** (`int`, `2020` to `2025`):
   The integer calendar year of a trading date or annual simulation cycle. This is the **primary joining key** between daily market series and annual latent firm profiles.
3. **`fiscal_year`** (`str`, e.g., `FY2021-22`):
   The standard Indian Financial Year (running from April 1 of year $T$ to March 31 of year $T+1$). Used for corporate accounting filings (BRSR ESG disclosures and annual balance sheets).
4. **`period_end`** (`str`, ISO `YYYY-MM-DD`):
   The final calendar day of a financial reporting period (e.g., `2024-03-31` for annual, `2024-09-30` for Q2).
5. **`filed_date`** (`str`, ISO `YYYY-MM-DD`):
   The statutory date on which the company submits its disclosure to the exchange and regulatory authorities.
6. **`available_date`** (`str`, ISO `YYYY-MM-DD`):
   The first date on which the filed information is processed, public, and commercially ingestible by market participants.

---

> [!NOTE]
> The July 15 annual availability date and the 45-day quarterly reporting lag are **configurable synthetic prototype modelling assumptions** (defined in `data_generation/config.py` as `ANNUAL_ESG_DISCLOSURE_AVAILABILITY_MONTH = 7`, `ANNUAL_ESG_DISCLOSURE_AVAILABILITY_DAY = 15`, and `QUARTERLY_FUNDAMENTALS_AVAILABILITY_LAG_DAYS = 45`). They are designed to prevent look-ahead bias in simulation, and should **not** be construed as statutory regulatory filing deadlines.

## 2. Calendar Year vs. Indian Fiscal Year Alignment

### 2.1 The Indian Fiscal Cycle
Unlike western markets where fiscal years typically coincide with calendar years, the Indian financial reporting year runs:
$$\text{April 1 (Year } T\text{)} \quad \longrightarrow \quad \text{March 31 (Year } T+1\text{)}$$
- **Quarter 1 (Q1)**: April 1 – June 30
- **Quarter 2 (Q2)**: July 1 – September 30
- **Quarter 3 (Q3)**: October 1 – December 31
- **Quarter 4 (Q4)**: January 1 – March 31

### 2.2 Annual Latent Profile Joining Convention
The latent profile dataset `data/latent_profiles.csv` contains both an integer `year` (`2020`–`2025`) and a formatted `fiscal_year` string (`FY2020-21`–`FY2025-26`).

**MANDATORY RULE FOR DOWNSTREAM GENERATORS (M2–M6):**
> **All joins between daily trading dates and latent profiles MUST join on `(company_id, year)` where `year = date.year`.**

The string `fiscal_year` is maintained as a descriptive cross-reference for corporate disclosures. Downstream modules must not use string slicing on `fiscal_year` to infer market trading dates.

---

## 3. Walkthrough of Specific Reference Dates

| Reference Date | Calendar Year | Active Fiscal Period | Latent Profile Joined (`year`) | Information Status & Permitted Features |
|:---|:---:|:---:|:---:|:---|
| **January 15, 2020** | `2020` | Q4 of FY2019-20 | `2020` (`FY2020-21` annual profile) | Warm-up phase. Only filings with `available_date < 2020-01-15` (e.g., FY2018-19 annual, Q2 FY2019-20) may be conditioned upon. No BRSR data exists. |
| **March 25, 2020** | `2020` | Q4 of FY2019-20 | `2020` | COVID-19 shock window. FY2019-20 financial statements have **not** ended yet (`period_end = 2020-03-31`). Any model operating on this date must only use Q3 FY2019-20 (filed Feb 2020). |
| **April 15, 2020** | `2020` | Q1 of FY2020-21 | `2020` | New Indian fiscal year begins. Although FY2019-20 ended on March 31, annual results are **not yet filed** (annual results are filed in May–June). Models must NOT use FY2019-20 full-year figures on this date. |
| **September 30, 2025 (Reporting Period End)** | `2025` | Q2 of FY2025-26 | `2025` | The fiscal quarter ends on this date, but books close over subsequent weeks. Under SEBI LODR regulations, Q2 results are filed 38–45 days later (`filed_date` ≈ Nov 12, 2025). |
| **December 31, 2025** | `2025` | Q3 of FY2025-26 | `2025` | Prototype simulation cutoff. Q2 FY2025-26 results (filed Nov 2025) are active and available. Q3 FY2025-26 results (which end Dec 31) are **unfiled** and must not appear in any feature vector. |

---

## 4. Strict Point-in-Time Non-Leakage Architecture

### 4.1 The Availability Horizon Formula
In empirical backtesting, lookahead bias occurs when an analyst assumes that information from period $T$ is available on the period-end date $T_{\text{end}}$. 

In SIRPS, all fundamental statements and BRSR facts include two distinct timestamps:
- `filed_date`: Actual corporate submission date.
- `available_date`: The next immediate trading session following `filed_date`.

**Point-in-Time Constraint:**
$$\text{Feature Observation at Date } t \text{ may include disclosure } k \iff \text{available\_date}_k + 1 \text{ trading day} \le t$$

The $+1$ trading day buffer guarantees that trading decisions at market open on date $t$ only utilize information that was already fully public and disseminated before the market opened.

### 4.2 Planted Signals Non-Leakage Rule
The three planted signals in `data_generation/config.py` are strictly governed:
1. `MOMENTUM_SIGNAL = 0.02`: Uses rolling 63-day and 12-to-1 month past price returns strictly prior to date $t$.
2. `QUALITY_SIGNAL = 0.02`: Uses published quarterly balance sheets satisfying $\text{available\_date} + 1 \le t$.
3. `ESG_SIGNAL = 0.04`: **Must remain completely dormant and inactive for each firm until its annual BRSR filing has been formally published and processed:**
   $$\text{ESG Signal Active}_{i,t} \iff t \ge \text{available\_date}_{\text{BRSR}, i, \text{FY}} + 1$$
   Since Indian BRSR disclosures for FY2021-22 were filed in July 2022, models evaluated in 2021 and H1 2022 have zero access to ESG scores. This mirrors real market history.
