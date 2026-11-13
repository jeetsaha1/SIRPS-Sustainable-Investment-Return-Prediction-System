
# SIRPS – Research Protocol

**Project:** Sustainable Investment Return Prediction System
**Protocol Version:** 0.1 (Initial Draft)
**Project Stage:** Prototype / Pre-implementation
**Environment:** Python 3.12
**Market:** Indian National Stock Exchange (NSE)

---

## 1. Project Overview

SIRPS is a research-oriented decision-support prototype designed to estimate the 63-trading-day forward performance of sustainability-related equities listed on the Indian NSE.

The project investigates whether company-level Environmental, Social and Governance (ESG) information provides additional predictive value beyond conventional financial, technical and market information.

The system also examines whether predictive performance differs across Bull, Neutral and Bear market conditions.

SIRPS is an academic research prototype. It is not an automated trading system and does not provide investment advice.

## 2. Research Objectives

The project has the following primary objectives:

1. Develop a documented universe of Indian sustainability-related NSE equities.
2. Collect and standardise market, financial and ESG information.
3. Develop a reproducible Bull, Neutral and Bear market-regime classifier.
4. Predict 63-trading-day forward excess returns using machine learning.
5. Evaluate the additional predictive contribution of ESG information.
6. Analyse model performance across different market regimes.
7. Develop a Streamlit dashboard to present predictions and research results.

## 3. Research Questions

**RQ1:** Do machine learning models outperform simple baselines in predicting 63-day forward excess returns?

**RQ2:** Does incorporating ESG information improve predictive performance compared with models that use only financial, technical and market information?

**RQ3:** Does the predictive contribution of ESG information differ across Bull, Neutral and Bear market regimes?

**RQ4:** Do sustainability-focused companies differ from matched conventional companies in return and downside-risk behaviour?

**RQ5:** Can the developed models generalise to companies excluded from their training datasets?

RQ4 and RQ5 are secondary-scope investigations.

## 4. Company Universe

### 4.1 Target Universe

- Market: Indian NSE-listed equities.
- Target universe: 20–30 sustainability-related companies.
- Minimum required universe: 15 companies.
- Main-board NSE equity instruments.

### 4.2 Eligibility Criteria

A company must satisfy the following requirements:

1. At least 252 trading days of historical price information.
2. At least 50% of revenue, installed capacity or order book (where revenue is not segmented) attributable to approved sustainability-related activities.
3. Average daily traded value above the documented liquidity threshold.
4. Annual reports and BRSR/BRR filings available for at least the two latest financial years.
5. Eligibility supported by documented evidence and reviewed by two members.

Approved activities include:

- Renewable power generation.
- Electric vehicle and battery manufacturing.
- Water and wastewater treatment.
- Waste management and recycling.
- Grid and energy-efficiency solutions.

Companies with 30–50% qualifying sustainability activity are classified as transition companies and excluded from the main universe.

Delisted and merged companies must be retained until their last trading date where historical records are available.

The universe must be frozen using documented eligibility evidence, not historical stock-price performance.

## 5. Forecast Target Definition

### 5.1 Forecast Horizon

The prediction horizon is 63 trading days.

The forecast is generated after the close of trading day t.

- Decision time: After close of day t.
- Entry: Close of day t+1.
- Exit: Close of day t+64.

### 5.2 Primary Target

The primary target is the 63-day forward excess log return relative to NIFTY 50 TRI.

### 5.3 Secondary Target

The secondary target is the absolute 63-day forward log return.

The primary target is used for market-relative prediction and ranking. Absolute return is retained as a secondary prediction and display metric.

### 5.4 Observation Sampling

One observation per company is generated every five trading days.

Each observation must store:

- company_id
- date
- entry_date
- target_end_date
- y_excess
- y_abs

## 6. Market Regime Classification

The system classifies market conditions into:

1. Bull
2. Neutral
3. Bear

Regime classification will use trailing NIFTY 50 market information.

The classifier must be causal and use a 10-trading-day confirmation rule.

Regime assignment must not use future market information.

Performance comparisons will use the regime at the beginning of the forecast window.

Sensitivity analysis will be conducted for the regime-classification methodology.

## 7. ESG Information

### 7.1 ESG Data Source

The primary ESG information source is company BRSR reporting through exchange filings.

Structured filings should be preferred.

PDF-based extraction is the fallback method.

### 7.2 Pre-registered ESG Indicators

The following eight indicators will be used:

1. Renewable-energy share.
2. Energy intensity.
3. Emissions intensity (Scope 1+2 per turnover).
4. Water intensity.
5. Waste recovery rate.
6. Lost-time injury frequency rate.
7. Women in workforce percentage.
8. Independent directors percentage.

An ESG coverage ratio will also be calculated.

### 7.3 Missing-Value Policy

Missing ESG observations must never be automatically replaced with zero.

The following statuses must remain distinguishable:

- Value
- NA: Not applicable
- NR: Not reported
- NC: Not comparable

Raw values and standardised units must both be retained.

### 7.4 ESG Quality Assurance

- Structured filing extraction must be spot-checked.
- PDF extraction must be independently double-entered on a random 20% sample.
- Range and unit checks must cover 100% of extracted observations.
- Target extraction error rate: below 2%.

The indicator list is frozen at the end of Week 1. Changes require documented justification and approval.

## 8. Machine Learning Models

The project will evaluate:

### Baseline Models
- Zero-return baseline.
- Expanding-mean baseline.
- Momentum-rank baseline.
- Simple-factor Ridge baseline.

### Primary Machine Learning Models
1. Ridge Regression.
2. Random Forest.
3. XGBoost.

All models must use reproducible configurations and documented random seeds.

Model hyperparameters must be frozen according to the approved experimental protocol.

## 9. Experimental Design

The central experiment evaluates the additional predictive contribution of ESG information.

The experiments include:

- Experiment A: General model benchmarking.
- Experiment A': Model without ESG information.
- Experiment B: Model with ESG information.
- Experiment C: Company identity control.
- Experiment P: Shuffled-ESG placebo.

The primary ESG comparison is A' versus B using identical observations and validation folds.

Identity-control and placebo experiments are secondary-scope investigations.

## 10. Validation Methodology

The project will use expanding-window, time-respecting walk-forward validation.

The validation protocol includes:

- 63-trading-day purge.
- 5-trading-day embargo.
- Frozen hyperparameters on the development window.
- A single final holdout evaluation.

No training observation may have a target period overlapping the test period.

All ESG and financial features must respect their actual availability dates.

The final holdout must be evaluated only once after the research design and model choices are frozen.

## 11. Evaluation Metrics

The project will use:

- Mean Squared Forecast Error (MSFE).
- Out-of-sample R-squared.
- Rank Information Coefficient (Rank-IC).
- Clark-West test for the primary nested-model comparison.
- Heteroskedasticity and autocorrelation consistent (HAC) inference.
- Bootstrap confidence intervals.
- Holm adjustment for multiple secondary comparisons.

Performance must also be reported across market regimes with observation coverage counts.

## 12. Data Integrity and Leakage Prevention

The following rules are mandatory:

1. All datasets must contain documented source information.
2. All dates must use ISO YYYY-MM-DD format.
3. Features must only use information available at the prediction date.
4. Restated financial values must not replace first-reported values in historical predictions.
5. ESG availability dates must be respected.
6. Training-window statistics must be used for preprocessing.
7. Feature and target leakage tests must be performed.
8. All model experiments must be reproducible.

## 13. Prototype Data Policy

The initial judges' demonstration may use AI-generated synthetic datasets.

Synthetic data must:

- Be generated reproducibly using a fixed random seed.
- Have consistent company identifiers and dates.
- Follow the approved data schemas.
- Be clearly labelled as synthetic demonstration data.
- Never be presented as actual NSE historical observations.
- Never be presented as genuine company-reported ESG information.

Real-world research results must use verified historical sources and documented availability dates.

## 14. Dashboard Requirements

The application will use Streamlit.

It will display:

- Project overview and market conditions.
- Company sustainability profiles.
- Predicted investment returns.
- Bull, Neutral and Bear market comparisons.
- Model performance and research findings.

Predictions must display the model version, data cutoff, forecast horizon and estimated range where available.

All outputs must include an academic prototype and not-investment-advice disclaimer.

## 15. Team Responsibilities

| Member | Responsibility |
|---|---|
| M1 | Project protocol, architecture, integration and governance |
| M2 | Market data and ESG quality assurance |
| M3 | ESG extraction and indicator processing |
| M4 | Fundamentals, market regimes and backtesting |
| M5 | Dataset integration, features and targets |
| M6 | Machine learning and statistical evaluation |
| M7 | Dashboard, serving and ESG quality assurance |

Each member must submit a two-company sample before full-scale data production.

## 16. Governance and Version Control

- `main` is reserved for reviewed integration.
- Every member must work on an independent feature branch.
- Pull requests require one reviewer from the assigned reviewer pair.
- Automated tests must pass before merging.
- Schema changes require a pull request and version update.
- Weekly integration builds must be maintained.
- All experiments must be recorded in results/runs.csv.

## 17. Current Protocol Status

Version 0.1 is the initial research-protocol draft.

The protocol will be reviewed and frozen as protocol-v1 before the ESG experimental results are examined.

Any subsequent changes must be documented with reasons, approval and version history.

---

**End of Protocol**
