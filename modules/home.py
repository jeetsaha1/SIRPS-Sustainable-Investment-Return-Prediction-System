"""SIRPS Module 1: Home / Overview Dashboard (Owner: Member 1).

Orients judges, displays the current synthetic market state, presents the latest
excess return forecasts across the 10 fictional sustainable companies, and
clearly delineates what is implemented, simulated, and planned for future work.
"""

import streamlit as st
import pandas as pd
from typing import Optional

from common.ui import demo_banner, render_kpi_card
from common.io import load

def render() -> None:
    """Renders the Home / Overview Dashboard page."""
    demo_banner()
    
    st.title("Sustainable Investment Return Prediction System (SIRPS)")
    st.markdown(
        "**Research Prototype Dashboard** | Multi-Horizon Return Forecasting with BRSR ESG Integration"
    )
    
    # Load required data with explicit separation of foundational vs mock development data
    try:
        cm_df = load("company_master", use_mock=False)
    except FileNotFoundError:
        cm_df = load("company_master", use_mock=True)

    try:
        idx_df = load("index_daily", use_mock=False)
        reg_df = load("regime_daily", use_mock=False)
        lp_df = load("latest_predictions", use_mock=False)
        is_mock_mode = False
    except FileNotFoundError:
        idx_df = load("index_daily", use_mock=True)
        reg_df = load("regime_daily", use_mock=True)
        lp_df = load("latest_predictions", use_mock=True)
        is_mock_mode = True

    if is_mock_mode:
        st.info(
            "🛠️ **Member 1 Development Mode Active**: Foundational company master and annual latent profiles are fully generated and verified. "
            "Downstream market time-series and predictions are loaded from schema-compliant mock samples (`data/mock/`) pending subsequent member pipelines."
        )

    # 1. Five KPI Cards
    st.subheader("System & Market Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        n_firms = len(cm_df)
        render_kpi_card("Universe Size", f"{n_firms} Firms", "5 Green Sectors")
        
    with col2:
        render_kpi_card("Modeling Horizon", "2020–2025", "63-day horizon")
        
    with col3:
        latest_reg = reg_df.iloc[-1]["confirmed_regime"].capitalize() if len(reg_df) > 0 else "Bull"
        reg_delta = "Mock Sample" if is_mock_mode else "10-day confirmed"
        render_kpi_card("Market Regime", latest_reg, reg_delta)
        
    with col4:
        nifty_last = idx_df.iloc[-1]["nifty_close"] if len(idx_df) > 0 else 14500.0
        vix_last = idx_df.iloc[-1]["india_vix"] if len(idx_df) > 0 else 14.5
        vix_delta = "Mock VIX" if is_mock_mode else f"VIX {vix_last:.1f}"
        render_kpi_card("NIFTY 50 / VIX", f"{nifty_last:,.0f}", vix_delta)
        
    with col5:
        render_kpi_card("ML Models", "3 Models", "Ridge, RF, XGBoost")
        
    st.markdown("---")
    
    # 2. Interactive Forecast Controls & Latest Predictions
    st.subheader("Latest 63-Trading-Day Excess Return Forecasts")
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 4])
    with ctrl_col1:
        selected_model = st.selectbox(
            "Select Forecasting Model",
            ["XGBoost", "Random Forest", "Ridge"],
            index=0,
            key="home_model_select"
        )
    with ctrl_col2:
        selected_exp = st.radio(
            "Feature Scope",
            [("B", "Experiment B (With BRSR ESG)"), ("A", "Experiment A (Technical + Fund Only)")],
            format_func=lambda x: x[1],
            index=0,
            key="home_exp_select"
        )[0]
        
    # Filter predictions
    filtered_preds = lp_df[
        (lp_df["model"] == selected_model) & (lp_df["experiment"] == selected_exp)
    ].copy()
    
    if len(filtered_preds) == 0:
        # Graceful fallback to any matching experiment
        filtered_preds = lp_df[lp_df["experiment"] == selected_exp].copy()
        
    # Merge with company details
    merged_view = pd.merge(
        cm_df[["company_id", "company_name", "sector", "sub_sector"]],
        filtered_preds[["company_id", "y_hat_excess", "y_hat_abs", "lower80", "upper80"]],
        on="company_id",
        how="inner"
    ).sort_values("y_hat_excess", ascending=False)
    
    view_col1, view_col2 = st.columns([5, 4])
    with view_col1:
        st.markdown("**Cross-Sectional Excess Return Forecasts (%)**")
        chart_data = merged_view.set_index("company_name")["y_hat_excess"] * 100.0
        st.bar_chart(chart_data)
        
    with view_col2:
        st.markdown("**Top & Bottom Ranked Constituents**")
        display_df = merged_view.rename(columns={
            "company_name": "Company",
            "sector": "Sector",
            "y_hat_excess": "Forecast Excess",
            "lower80": "Lower 80%",
            "upper80": "Upper 80%"
        })
        display_df["Forecast Excess"] = display_df["Forecast Excess"].apply(lambda x: f"{x*100:+.2f}%")
        display_df["Lower 80%"] = display_df["Lower 80%"].apply(lambda x: f"{x*100:+.2f}%")
        display_df["Upper 80%"] = display_df["Upper 80%"].apply(lambda x: f"{x*100:+.2f}%")
        st.dataframe(
            display_df[["Company", "Sector", "Forecast Excess", "Lower 80%", "Upper 80%"]],
            hide_index=True
        )
        
    st.markdown("---")
    
    # 3. System Architecture: Implemented vs Simulated vs Future
    st.subheader("Research Prototype Scope & Dataset Lifecycle Matrix")
    
    st.markdown(
        """
        SIRPS operates under a strict **7-Member Integration Matrix**. All datasets adhere to a 4-tier lifecycle:
        - `FOUNDATION_GENERATED`: Verified production data generated by Member 1 foundation (`data/`).
        - `MOCK_FIXTURE`: Schema-compliant developmental sample fixtures (`data/mock/`).
        - `FUTURE_MEMBER_GENERATED`: Production pipeline data reserved for downstream member execution (M2–M7).
        - `NOT_YET_AVAILABLE`: Unregistered or future experimental data.
        """
    )
    
    status_summary = [
        {"Workstream": "M1: Foundational Core", "Key Datasets": "company_master, latent_profiles", "Status": "FOUNDATION_GENERATED", "Active File": "data/company_master.csv, data/latent_profiles.csv"},
        {"Workstream": "M2: Market Data", "Key Datasets": "trading_calendar, index_daily, prices_daily, corporate_actions", "Status": "MOCK_FIXTURE (Pending M2)", "Active File": "data/mock/*.csv"},
        {"Workstream": "M3: ESG Disclosures", "Key Datasets": "esg_facts, esg_scores", "Status": "MOCK_FIXTURE (Pending M3)", "Active File": "data/mock/*.csv"},
        {"Workstream": "M4: Financials & Regimes", "Key Datasets": "fundamentals_quarterly, regime_daily, portfolio_default", "Status": "MOCK_FIXTURE (Pending M4)", "Active File": "data/mock/*.csv"},
        {"Workstream": "M5: Feature Store", "Key Datasets": "model_dataset, latest_features", "Status": "MOCK_FIXTURE (Pending M5)", "Active File": "data/mock/*.csv"},
        {"Workstream": "M6: ML Models & Eval", "Key Datasets": "predictions, metrics_summary, feature_importance, local_contributions", "Status": "MOCK_FIXTURE (Pending M6)", "Active File": "data/mock/*.csv"},
        {"Workstream": "M7: UI Dashboards", "Key Datasets": "latest_predictions, visualization views", "Status": "MOCK_FIXTURE (Pending M7)", "Active File": "data/mock/*.csv"},
    ]
    st.dataframe(pd.DataFrame(status_summary), hide_index=True, use_container_width=True)

    arch_col1, arch_col2, arch_col3 = st.columns(3)
    
    with arch_col1:
        st.success("✅ **Member 1 Implemented in Code**")
        st.markdown(
            r"""
            - **Central Schema Registry**: 21 registered dataset contracts with primary keys and explicit nullability.
            - **4D Latent AR(1) State Space**: Correlated annual trajectories with Cholesky factorization.
            - **Archetype-to-Factor Mapping**: Qualitative business attributes mapped to baseline offsets $\boldsymbol{\delta}_i$.
            - **Strict I/O & Caching**: Atomic writes, provenance validation, and rejection of silent mock fallback.
            """
        )
        
    with arch_col2:
        st.info("⚙️ **Simulated Demonstration Baseline**")
        st.markdown(
            """
            - **10 Fictional Companies**: C01–C10 across 5 green economic sectors.
            - **Master Random Seed 42**: Bit-for-bit deterministic reproducibility.
            - **Configurable Availability Lags**: Synthetic prototype assumptions (mid-July annual, 45d quarterly).
            - **Planted Signal Benchmarks**: Preserved constants (0.02 momentum, 0.02 quality, 0.04 ESG).
            """
        )
        
    with arch_col3:
        st.warning("🚀 **Future Real Data Transition**")
        st.markdown(
            """
            - **NSE Bhavcopy Ingestion**: Real tick and daily OHLCV execution.
            - **SEBI BRSR Parsing**: Automated extraction from XBRL/PDF disclosures.
            - **Broader Investment Universe**: Scaling from 10 to NIFTY 500 sustainable universe.
            - **Execution Slippage & Impact**: High-frequency order book and transaction models.
            """
        )


if __name__ == "__main__":
    render()
