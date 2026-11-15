"""SIRPS Common UI Components and State Management.

This module provides standard UI elements, disclaimers, navigation controls,
and synchronized session state across all 10 Streamlit modules.
"""

from typing import Optional, List, Dict, Any
import streamlit as st

def demo_banner() -> None:
    """Renders the required synthetic demonstration banner across all pages."""
    st.markdown(
        """
        <div style="background-color: #FFF3CD; border-left: 6px solid #FFC107; padding: 12px 16px; margin-bottom: 20px; border-radius: 4px;">
            <strong style="color: #856404; font-size: 1.05rem;">⚠️ SYNTHETIC DEMONSTRATION DATA ONLY</strong>
            <p style="color: #856404; margin: 4px 0 0 0; font-size: 0.90rem;">
                All entities (C01–C10), market paths, BRSR ESG disclosures, and financial figures are synthetically generated
                for research methodology demonstration. This system does NOT contain empirical NSE data or real investment advice.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

def init_state() -> None:
    """Initializes standard session state keys if not already present."""
    defaults = {
        "company_id": "C01",
        "model_name": "XGBoost",
        "experiment": "B",  # Experiment B: Full Model (with ESG)
        "nav_target": "1 Home",
        "chart_type": "Line",
        "fiscal_year": "FY2024-25"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def company_selector(label: str = "Select Company", key: str = "company_selector") -> str:
    """Renders a standard company selector dropdown synchronized with st.session_state['company_id']."""
    companies = [
        ("C01", "C01 - Surya Grid Renewables Ltd (Renewable Power)"),
        ("C02", "C02 - Vayu Windpower Ltd (Renewable Power)"),
        ("C03", "C03 - Jaldhara Water Tech Ltd (Water)"),
        ("C04", "C04 - Navchetna Recycling Ltd (Waste and Bio)"),
        ("C05", "C05 - Voltway Mobility Ltd (EV and Batteries)"),
        ("C06", "C06 - Ampere Cell Energy Ltd (EV and Batteries)"),
        ("C07", "C07 - Harita Hydro Power Ltd (Renewable Power)"),
        ("C08", "C08 - GridSmart Systems Ltd (Efficiency and Grid)"),
        ("C09", "C09 - Kiran Efficiency Solutions Ltd (Efficiency and Grid)"),
        ("C10", "C10 - Biogas Bharat Energy Ltd (Waste and Bio)"),
    ]
    
    current_id = st.session_state.get("company_id", "C01")
    index_match = 0
    for idx, (cid, _) in enumerate(companies):
        if cid == current_id:
            index_match = idx
            break
            
    selected_tuple = st.selectbox(
        label,
        options=companies,
        index=index_match,
        format_func=lambda x: x[1],
        key=key
    )
    
    selected_id = selected_tuple[0]
    st.session_state["company_id"] = selected_id
    return selected_id

def render_kpi_card(title: str, value: str, delta: Optional[str] = None, delta_color: str = "normal") -> None:
    """Helper to render uniform KPI metric cards."""
    st.metric(label=title, value=value, delta=delta, delta_color=delta_color)
