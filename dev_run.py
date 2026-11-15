"""SIRPS Developer Page Runner (Member 1).

Allows developing and smoke-testing any individual Streamlit module in isolation
with the standard demonstration banner and session state initialized.

Usage:
    streamlit run dev_run.py -- home
    streamlit run dev_run.py -- company_explorer
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

st.set_page_config(page_title="SIRPS Dev Runner", layout="wide")
init_state()

# Determine target module from CLI arguments or fallback to 'home'
args = sys.argv[1:]
target = "home"
for arg in args:
    if not arg.startswith("--") and not arg.startswith("-"):
        target = arg
        break

st.sidebar.title("🛠️ Developer Runner")
module_input = st.sidebar.text_input("Module name (inside modules/)", value=target)

try:
    mod_name = f"modules.{module_input.replace('.py', '')}"
    mod = importlib.import_module(mod_name)
    if hasattr(mod, "render") and callable(mod.render):
        mod.render()
    else:
        st.error(f"Module '{mod_name}' does not implement 'def render():'")
except ModuleNotFoundError:
    demo_banner()
    st.warning(f"Module 'modules.{module_input}' not found. Please verify that the file exists in 'modules/'.")
except Exception:
    st.error("Error running module:")
    st.code(traceback.format_exc())
