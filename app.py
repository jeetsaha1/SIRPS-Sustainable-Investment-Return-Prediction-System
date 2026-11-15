"""SIRPS Streamlit Multi-Module Application Shell (Owner: Member 1).

Main entrypoint for the Sustainable Investment Return Prediction System.
Routes across all 10 prototype modules with unified session state,
persistent synthetic demonstration banner, and graceful handling for unmerged modules.
"""

import sys
import importlib
import traceback
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.ui import demo_banner, init_state

st.set_page_config(
    page_title="SIRPS Research Prototype",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Registry of all 10 planned prototype modules
PAGES = {
    "1 Home": "modules.home",
    "2 Company Explorer": "modules.company_explorer",
    "3 ESG Analysis": "modules.esg_analysis",
    "4 Market Regime": "modules.market_regime",
    "5 Return Prediction": "modules.return_prediction",
    "6 Model Comparison": "modules.model_comparison",
    "7 ESG Contribution": "modules.esg_contribution",
    "8 Portfolio and Risk": "modules.portfolio_risk",
    "9 Explainability": "modules.explainability",
    "10 Methodology": "modules.methodology",
}

# Initialize shared session state
init_state()

# Sidebar branding and navigation
st.sidebar.title("🌱 SIRPS Prototype")
st.sidebar.caption("Sustainable Investment Return Prediction System")

choice = st.sidebar.radio("Navigation", list(PAGES.keys()), key="nav")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Prototype Details**
    - Scope: `prototype_plus`
    - Scenario: `historical_plus`
    - Factor Design: `correlated`
    - Master Seed: `42`
    - Provenance: `SYNTHETIC_DEMO`
    """
)

# Module dispatch with exception handling
try:
    module_path = PAGES[choice]
    mod = importlib.import_module(module_path)
    if hasattr(mod, "render") and callable(mod.render):
        mod.render()
    else:
        st.error(f"Module '{module_path}' does not define a callable 'render()' entrypoint.")
except ModuleNotFoundError as e:
    # If the missing module is the current page itself, display a clean placeholder
    demo_banner()
    st.header(f"{choice}")
    st.info(
        f"ℹ️ **Status: Pending Downstream Implementation**\n\n"
        f"The module `{PAGES[choice]}` has not yet been merged into this branch.\n"
        f"In accordance with the 7-Member Responsibility Matrix, this module is assigned to a downstream team member.\n\n"
        f"Foundational mock datasets are ready in `data/mock/` for development."
    )
except Exception:
    st.error("This page encountered an unhandled exception during rendering:")
    st.code(traceback.format_exc())
