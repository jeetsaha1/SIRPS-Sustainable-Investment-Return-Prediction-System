# SIRPS Company Archetype to Latent Factor Mapping Specification

**Module:** Member 1 — Architecture & Foundation  
**Version:** 2.0.0 (Foundation Baseline)  
**Status:** Implemented & Verified in `data_generation/config.py` and `data_generation/latent.py`  
**Dataset Reference:** `data/company_master.csv`, `data/latent_profiles.csv`  

---

## 1. Overview & Theoretical Motivation

In the SIRPS econometric simulation, companies are not interchangeable random draws. Each company represents a specific sustainable business archetype operating within India's transition economy. 

While sector group baselines ($\boldsymbol{\mu}_{\text{sector}}$) provide macroeconomic anchors, companies within the same sector exhibit substantial variation in operational structure, technological risk, capital intensity, and revenue predictability.

To capture this intra-sector heterogeneity without introducing ad-hoc noise, Member 1 implements a deterministic, bounded, and economically grounded mapping:

$$\boldsymbol{\mu}_i = \boldsymbol{\mu}_{\text{sector}(i)} + \boldsymbol{\delta}_i$$

where $\boldsymbol{\mu}_{\text{sector}(i)}$ is the sector baseline vector and $\boldsymbol{\delta}_i = [\delta_{\text{esg}}, \delta_{\text{qual}}, \delta_{\text{risk}}, \delta_{\text{growth}}]^T$ represents the archetype baseline adjustment.

---

## 2. Archetype Mapping Table

The categorical business characteristics defined in `company_archetypes.json` and preserved in `company_master.csv` map to latent factor offsets as follows:

| Characteristic | Category / Level | Latent Factor Affected | Direction | Magnitude ($\Delta$) | Economic Rationale |
|:---|:---|:---:|:---:|:---:|:---|
| **Growth Profile** | `Very High` | Growth | $+$ | $+0.10$ | Aggressive market expansion, high addressable market (TAM), rapid compounding. |
| | `High` | Growth | $+$ | $+0.05$ | Above-average adoption curve in green industrial transition. |
| | `Medium-High` | Growth | $+$ | $+0.02$ | Steady secular expansion tracking GDP and industrial capex. |
| | `Medium` | Growth | $-$ | $-0.05$ | Mature utility or municipal asset with regulated return caps. |
| **Revenue Stability** | `High` | Quality<br>Risk | $+$<br>$-$ | $+0.08$<br>$-0.08$ | Long-term Power Purchase Agreements (PPAs) or municipal contracts provide predictable cash flows, insulating balance sheets. |
| | `Medium-High` | Quality<br>Risk | $+$<br>$-$ | $+0.04$<br>$-0.04$ | Moderate counterparty diversification with semi-regulated commercial pricing. |
| | `Medium` | Quality<br>Risk | Neutral | $0.00$<br>$0.00$ | Baseline exposure to market demand cycles and off-taker payment delays. |
| **Technology Dependence**| `Very High` | Risk<br>Growth | $+$<br>$+$ | $+0.08$<br>$+0.04$ | Rapid cell chemistry/hardware obsolescence cycle, high technological disruption risk paired with breakthrough potential. |
| | `High` | Risk<br>Growth | $+$<br>$+$ | $+0.04$<br>$+0.02$ | Specialized engineering exposure (smart grid IoT, power electronics, complex solar supply chains). |
| | `Medium` | Risk<br>Growth | Neutral | $0.00$<br>$0.00$ | Proven, mature technologies (onshore wind, civil water filtration, hydro turbines). |
| **Capital Intensity** | `Very High` | Risk<br>Quality | $+$<br>$-$ | $+0.06$<br>$-0.04$ | Gigawatt-scale capex requires heavy debt financing, increasing debt-service vulnerability during interest rate hikes. |
| | `High` | Risk<br>Quality | $+$<br>$-$ | $+0.03$<br>$-0.02$ | Substantial asset construction outlays and project financing leverage. |
| | `Medium-High` | Risk<br>Quality | $+$<br>$-$ | $+0.01$<br>$-0.01$ | Moderate plant and equipment capex relative to operating cash flows. |
| | `Medium` | Risk<br>Quality | Neutral | $0.00$<br>$0.00$ | Standard asset-turnover operations (e.g., recycling sorting facilities). |
| | `Low-Medium` | Risk<br>Quality | $-$<br>$+$ | $-0.03$<br>$+0.03$ | Asset-light product solutions and energy service contracts; low leverage. |

---

## 3. Firm-by-Firm Baseline Adjustments

Applying the mapping matrix to the 10 fictional companies yields the following exact baseline parameters:

| ID | Company Name | Sector | Baseline $\boldsymbol{\mu}_{\text{sector}}$ | $\delta_{\text{qual}}$ | $\delta_{\text{risk}}$ | $\delta_{\text{growth}}$ | Adjusted Mean $\boldsymbol{\mu}_i$ [ESG, Qual, Risk, Growth] |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **C01** | Surya Grid Renewables Ltd | Renewable Power | `[0.45, 0.10, 0.05, 0.20]` | $-0.02$ | $+0.07$ | $+0.07$ | `[0.45, 0.08, 0.12, 0.27]` |
| **C02** | Vayu Windpower Ltd | Renewable Power | `[0.45, 0.10, 0.05, 0.20]` | $-0.02$ | $+0.03$ | $+0.02$ | `[0.45, 0.08, 0.08, 0.22]` |
| **C03** | Jaldhara Water Tech Ltd | Water | `[0.55, 0.30, -0.30, 0.00]` | $+0.07$ | $-0.07$ | $-0.05$ | `[0.55, 0.37, -0.37, -0.05]` |
| **C04** | Navchetna Recycling Ltd | Waste and Bio | `[0.25, -0.05, 0.15, 0.15]` | $0.00$ | $0.00$ | $+0.05$ | `[0.25, -0.05, 0.15, 0.20]` |
| **C05** | Voltway Mobility Ltd | EV and Batteries | `[0.30, 0.05, 0.35, 0.40]` | $-0.02$ | $+0.07$ | $+0.07$ | `[0.30, 0.03, 0.42, 0.47]` |
| **C06** | Ampere Cell Energy Ltd | EV and Batteries | `[0.30, 0.05, 0.35, 0.40]` | $-0.04$ | $+0.14$ | $+0.14$ | `[0.30, 0.01, 0.49, 0.54]` |
| **C07** | Harita Hydro Power Ltd | Renewable Power | `[0.45, 0.10, 0.05, 0.20]` | $+0.04$ | $-0.02$ | $-0.05$ | `[0.45, 0.14, 0.03, 0.15]` |
| **C08** | GridSmart Systems Ltd | Efficiency and Grid | `[0.40, 0.20, -0.10, 0.15]` | $+0.08$ | $-0.04$ | $+0.04$ | `[0.40, 0.28, -0.14, 0.19]` |
| **C09** | Kiran Efficiency Solutions Ltd| Efficiency and Grid | `[0.40, 0.20, -0.10, 0.15]` | $+0.07$ | $-0.07$ | $+0.02$ | `[0.40, 0.27, -0.17, 0.17]` |
| **C10** | Biogas Bharat Energy Ltd | Waste and Bio | `[0.25, -0.05, 0.15, 0.15]` | $-0.02$ | $+0.03$ | $+0.05$ | `[0.25, -0.07, 0.18, 0.20]` |

### Intra-Sector Differentiation Highlights
1. **Renewable Power**:
   - C07 (Hydro) exhibits the highest Quality ($0.14$) and lowest Risk ($0.03$) due to stable base-load hydrology contracts, contrasting with C01 (Solar, Risk $0.12$, Growth $0.27$) which experiences higher technological turnover and accelerated capacity growth.
2. **EV & Batteries**:
   - C06 (Battery gigafactory) carries higher Growth ($0.54$) and higher operational Risk ($0.49$) than C05 (EV assembly, Growth $0.47$, Risk $0.42$), reflecting upstream mineral cycle and cell chemistry obsolescence risks.
3. **Efficiency & Grid**:
   - C08 (Grid utility hardware) and C09 (Energy-efficiency consumer products) reflect low risk and high quality, but C08 possesses higher technological leverage.

---

## 4. Preservation of Mathematical Structure

1. **Bounds Protection**: The maximum adjustment vector magnitude is $\max |\boldsymbol{\delta}| = 0.14$, which is modest relative to the allowed score domain $[-2.0, +2.0]$. In the generated 60-observation dataset, all scores remain strictly inside $[-1.38, +1.93]$.
2. **Correlation Matrix Preservation**: Adjustments shift the company mean $\boldsymbol{\mu}_i$ deterministically. The stochastic innovation shocks $\boldsymbol{\eta}_{i,t} = \boldsymbol{\sigma}_{\text{innov}} \odot (\mathbf{L}\mathbf{Z}_{i,t})$ are governed strictly by the Cholesky factor $\mathbf{L}$, ensuring that the target cross-sectional correlation structure is preserved across all simulated paths.
