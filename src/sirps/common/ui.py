"""Shared dashboard presentation helpers with lazy Streamlit imports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DISCLAIMER = "Academic prototype only. This is not investment advice."


def render_header(title: str = "SIRPS research prototype") -> None:
    import streamlit as st

    st.title(title)
    st.caption("Sustainable Investment Return Prediction System")
    st.warning(DISCLAIMER)


def render_predictions(path: str | Path) -> None:
    import streamlit as st

    source = Path(path)
    if not source.exists():
        st.info("No precomputed predictions are available. Run `python run_all.py` first.")
        return
    predictions = pd.read_csv(source)
    st.subheader("Precomputed predictions")
    st.dataframe(predictions, use_container_width=True, hide_index=True)
