from pathlib import Path

import pandas as pd
import streamlit as st

PREDICTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "serving" / "predictions.csv"

st.set_page_config(page_title="SIRPS", page_icon="S", layout="wide")
st.title("SIRPS research prototype")
st.caption("Sustainable Investment Return Prediction System")
st.warning("Academic prototype only. This is not investment advice.")

if not PREDICTIONS_PATH.exists():
    st.info("No precomputed predictions are available. Run `make pilot` first.")
else:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    st.subheader("Synthetic pilot predictions")
    st.dataframe(predictions, use_container_width=True, hide_index=True)
    st.caption("The displayed pilot data is synthetic and is not historical NSE data.")
