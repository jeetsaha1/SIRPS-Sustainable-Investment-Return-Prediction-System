import streamlit as st

from sirps.common.io import project_root
from sirps.common.ui import render_header, render_predictions

PREDICTIONS_PATH = project_root() / "data" / "serving" / "predictions.csv"

st.set_page_config(page_title="SIRPS", page_icon="S", layout="wide")
render_header()
render_predictions(PREDICTIONS_PATH)
