"""Original app.py entry point for Streamlit compatibility."""

import streamlit as st

from sirps.common.io import project_root
from sirps.common.ui import render_header, render_predictions

st.set_page_config(page_title="SIRPS", page_icon="S", layout="wide")
render_header()
render_predictions(project_root() / "data" / "serving" / "predictions.csv")
