"""SIRPS Module 2: Company Explorer & Interactive Market Analytics (Owner: Member 2).

Provides comprehensive exploration of synthetic market price history, technical indicators,
benchmark-relative total return comparison against NIFTY TRI, drawdown risk analytics,
and corporate action tracking across all 10 fictional sustainable companies.

Econometric & Financial Methodology:
1. Total-Return Performance:
   Uses total-return adjusted close (`adj_close`) for all performance, return, volatility,
   drawdown, and benchmark comparisons, ensuring zero false economic drops from stock splits.
2. Raw vs. Adjusted Price Toggle:
   Allows toggling between raw traded `close` (displaying mechanical ~50% split drop on ex-dates)
   and continuous `adj_close`.
3. Benchmark Total Return Comparison:
   Compares company `adj_close` against `nifty_tri` (NIFTY Total Return Index, including dividend
   reinvestment), rebased to 100 at the start of the selected observation window:
       P_t^{rebased} = (P_t / P_0) * 100
4. Risk Metrics:
   - Annualized Volatility: sigma_{ann} = std(r_{daily}) * sqrt(252)
   - Drawdown: DD_t = P_t / max_{s <= t}(P_s) - 1.0
   - Maximum Drawdown: MDD = min_t(DD_t)
   - Historical Beta: Cov(r_i, r_m) / Var(r_m)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from common.io import load
from common.ui import demo_banner, render_kpi_card

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. PURE CALCULATION & DATA PROCESSING HELPERS (ISOLATED & TESTABLE)
# ==============================================================================

def filter_by_company(df: pd.DataFrame, company_id: str) -> pd.DataFrame:
    """Filter DataFrame by company_id, preserving all records if column missing."""
    if "company_id" not in df.columns:
        return df
    filtered = df[df["company_id"] == company_id].copy()
    if "date" in filtered.columns:
        filtered = filtered.sort_values("date").reset_index(drop=True)
    elif "ex_date" in filtered.columns:
        filtered = filtered.sort_values("ex_date").reset_index(drop=True)
    return filtered


def filter_by_date_range(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """Filter DataFrame within inclusive date range [start_date, end_date].

    Handles reversed dates by swapping them and guarantees chronological ordering.
    """
    if date_col not in df.columns or len(df) == 0:
        return df.copy()

    # Guarantee proper string ordering
    s_date, e_date = min(start_date, end_date), max(start_date, end_date)

    # Convert to string format if datetime
    series = df[date_col]
    if pd.api.types.is_datetime64_any_dtype(series):
        date_strs = series.dt.strftime("%Y-%m-%d")
    else:
        date_strs = series.astype(str)

    mask = (date_strs >= s_date) & (date_strs <= e_date)
    filtered = df.loc[mask].copy()
    return filtered.sort_values(date_col).reset_index(drop=True)


def calculate_kpis(
    prices_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    price_col: str = "adj_close",
) -> Dict[str, Any]:
    """Calculate key performance indicators and risk statistics for a company.

    Args:
        prices_df: Company price history DataFrame.
        benchmark_df: Optional benchmark index DataFrame (with nifty_tri and nifty_close).
        price_col: Price column used for total-return calculations (default 'adj_close').

    Returns:
        Dictionary of formatted KPI metrics and numerical values.
    """
    if len(prices_df) == 0 or price_col not in prices_df.columns:
        return {
            "last_price": 0.0,
            "cum_return": 0.0,
            "ann_vol": 0.0,
            "mdd": 0.0,
            "excess_return": 0.0,
            "beta": 1.0,
            "n_obs": 0,
        }

    prices = prices_df[price_col].values.astype(float)
    n_obs = len(prices)
    last_p = float(prices[-1])
    first_p = float(prices[0])

    # Cumulative return
    cum_ret = float(last_p / first_p - 1.0) if first_p > 0 else 0.0

    # Annualized volatility
    if n_obs > 1:
        daily_rets = np.diff(np.log(prices))
        ann_vol = float(np.std(daily_rets, ddof=1) * np.sqrt(252.0))
    else:
        ann_vol = 0.0

    # Maximum drawdown
    peaks = np.maximum.accumulate(prices)
    dds = (prices / peaks) - 1.0
    mdd = float(np.min(dds)) if len(dds) > 0 else 0.0

    # Benchmark-relative metrics
    excess_ret = 0.0
    beta = 1.0

    if benchmark_df is not None and len(benchmark_df) > 0 and n_obs > 1:
        # Date-aligned merge
        df_p = prices_df[["date", price_col]].copy()
        df_b = benchmark_df.copy()

        df_p["d_str"] = pd.to_datetime(df_p["date"]).dt.strftime("%Y-%m-%d")
        df_b["d_str"] = pd.to_datetime(df_b["date"]).dt.strftime("%Y-%m-%d")

        merged = pd.merge(
            df_p, df_b[["d_str", "nifty_tri", "nifty_close"]], on="d_str"
        ).sort_values("d_str")

        if len(merged) > 1:
            b_first = float(merged["nifty_tri"].iloc[0])
            b_last = float(merged["nifty_tri"].iloc[-1])
            b_cum = float(b_last / b_first - 1.0) if b_first > 0 else 0.0
            excess_ret = float(cum_ret - b_cum)

            # Empirical beta
            c_log_rets = np.diff(np.log(merged[price_col].values))
            b_log_rets = np.diff(np.log(merged["nifty_close"].values))
            var_b = float(np.var(b_log_rets))
            if var_b > 1e-8:
                beta = float(np.cov(c_log_rets, b_log_rets)[0, 1] / var_b)

    return {
        "last_price": last_p,
        "cum_return": cum_ret,
        "ann_vol": ann_vol,
        "mdd": mdd,
        "excess_return": excess_ret,
        "beta": beta,
        "n_obs": n_obs,
    }


def calculate_drawdown_series(
    prices_df: pd.DataFrame,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Calculate running peak and drawdown time-series.

    Formula:
        DD_t = (P_t / max_{s <= t} P_s) - 1.0
    """
    if len(prices_df) == 0:
        return pd.DataFrame(columns=["date", "drawdown", "peak"])

    df = prices_df[["date", price_col]].copy().sort_values("date").reset_index(drop=True)
    prices = df[price_col].values.astype(float)
    peaks = np.maximum.accumulate(prices)
    drawdowns = (prices / peaks) - 1.0

    df["drawdown"] = drawdowns
    df["peak"] = peaks
    return df


def calculate_moving_averages(
    prices_df: pd.DataFrame,
    windows: Tuple[int, ...] = (50, 200),
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Calculate causal simple moving averages using past data only.

    Requires exactly w actual observations before displaying a value (min_periods=w).
    Early observations (< w) are NaN, ensuring partial averages are not misrepresented.
    """
    df = prices_df.copy().sort_values("date").reset_index(drop=True)
    for w in windows:
        df[f"ma_{w}"] = df[price_col].rolling(window=w, min_periods=w).mean()
    return df


def rebase_to_100(
    company_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    comp_col: str = "adj_close",
    bench_col: str = "nifty_tri",
) -> pd.DataFrame:
    """Rebase company adjusted close and benchmark TRI to 100 at common start date.

    Formula:
        P_t^{rebased} = (P_t / P_0) * 100
        B_t^{rebased} = (B_t / B_0) * 100
    """
    if len(company_df) == 0 or len(benchmark_df) == 0:
        return pd.DataFrame(columns=["date", "company_rebased", "benchmark_rebased", "excess_spread"])

    c_df = company_df[["date", comp_col]].copy()
    b_df = benchmark_df[["date", bench_col]].copy()

    c_df["d_str"] = pd.to_datetime(c_df["date"]).dt.strftime("%Y-%m-%d")
    b_df["d_str"] = pd.to_datetime(b_df["date"]).dt.strftime("%Y-%m-%d")

    merged = pd.merge(c_df, b_df, on="d_str").sort_values("d_str").reset_index(drop=True)
    if len(merged) == 0:
        return pd.DataFrame(columns=["date", "company_rebased", "benchmark_rebased", "excess_spread"])

    c_base = float(merged[comp_col].iloc[0])
    b_base = float(merged[bench_col].iloc[0])

    c_rebased = (merged[comp_col] / c_base) * 100.0 if c_base > 0 else 100.0
    b_rebased = (merged[bench_col] / b_base) * 100.0 if b_base > 0 else 100.0

    result = pd.DataFrame({
        "date": merged["date_x"],
        "d_str": merged["d_str"],
        "company_rebased": c_rebased,
        "benchmark_rebased": b_rebased,
        "excess_spread": c_rebased - b_rebased,
    })
    return result


def calculate_rolling_volatility(
    prices_df: pd.DataFrame,
    window: int = 63,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Calculate rolling 63-day annualized volatility from daily log returns."""
    df = prices_df[["date", price_col]].copy().sort_values("date").reset_index(drop=True)
    if len(df) <= 1:
        df["rolling_vol"] = 0.0
        return df

    log_rets = np.zeros(len(df))
    log_rets[1:] = np.diff(np.log(df[price_col].values))
    df["log_ret"] = log_rets

    # Rolling sample standard deviation annualized
    df["rolling_vol"] = df["log_ret"].rolling(window=window, min_periods=10).std(ddof=1) * np.sqrt(252.0)
    df["rolling_vol"] = df["rolling_vol"].fillna(0.0)
    return df


def calculate_returns_distribution(
    prices_df: pd.DataFrame,
    price_col: str = "adj_close",
) -> Dict[str, float]:
    """Calculate return distribution summary statistics."""
    if len(prices_df) <= 1:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 0.0, "positive_days_pct": 0.0}

    rets = np.diff(np.log(prices_df[price_col].values))
    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0

    # Sample skewness and kurtosis
    if std_ret > 1e-8:
        z = (rets - mean_ret) / std_ret
        skew = float(np.mean(z**3))
        kurt = float(np.mean(z**4) - 3.0)  # Excess kurtosis
    else:
        skew = 0.0
        kurt = 0.0

    pos_pct = float((rets > 0).mean() * 100.0)

    return {
        "mean": mean_ret * 252.0,  # Annualized
        "std": std_ret * np.sqrt(252.0),  # Annualized
        "skew": skew,
        "kurt": kurt,
        "positive_days_pct": pos_pct,
    }


# ==============================================================================
# 2. UI RENDERING COMPONENTS
# ==============================================================================

def render_company_header_and_metadata(comp_row: pd.Series) -> None:
    """Render company identification header and archetype metadata."""
    st.markdown(
        f"""
        ### {comp_row['company_name']} (`{comp_row['ticker']}`)
        **Company ID**: `{comp_row['company_id']}` | **Sector**: `{comp_row['sector']}` | **Sub-Sector**: `{comp_row['sub_sector']}`
        """
    )

    with st.expander("ℹ️ Company Profile & Business Archetype Characteristics", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Business Model**: {comp_row.get('business_model', 'N/A')}")
            st.markdown(f"**Capital Intensity**: `{comp_row.get('capital_intensity', 'N/A')}`")
            st.markdown(f"**Listing Date**: `{comp_row.get('listing_date', 'N/A')}`")
        with c2:
            st.markdown(f"**Growth Profile**: `{comp_row.get('growth_profile', 'N/A')}`")
            st.markdown(f"**Revenue Stability**: `{comp_row.get('revenue_stability', 'N/A')}`")
            st.markdown(f"**Base Price (2020)**: `₹{float(comp_row.get('base_price', 0.0)):.2f}`")
        with c3:
            st.markdown(f"**Tech Dependence**: `{comp_row.get('technology_dependence', 'N/A')}`")
            st.markdown(f"**Universe Group**: `{comp_row.get('universe_group', 'N/A')}`")
            st.markdown(f"**Data Provenance**: `{comp_row.get('data_source', 'SYNTHETIC_DEMO')}`")

        st.caption(f"**Core Business Risk**: {comp_row.get('business_risk_type', 'N/A')}")


def render_kpi_dashboard(kpis: Dict[str, Any]) -> None:
    """Render the 5 primary KPI summary cards."""
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        render_kpi_card(
            "Latest Adjusted Close",
            f"₹{kpis['last_price']:,.2f}",
            "Total-Return Basis",
        )
    with c2:
        ret_val = kpis["cum_return"]
        ret_str = f"{ret_val*100:+.2f}%"
        delta_col = "normal" if ret_val >= 0 else "inverse"
        render_kpi_card("Selected Period Return", ret_str, f"{kpis['n_obs']} sessions", delta_col)
    with c3:
        vol_str = f"{kpis['ann_vol']*100:.1f}% p.a."
        render_kpi_card("Annualized Volatility", vol_str, "Daily returns * sqrt(252)")
    with c4:
        mdd_val = kpis["mdd"]
        mdd_str = f"{mdd_val*100:.1f}%"
        render_kpi_card("Maximum Drawdown", mdd_str, "Peak-to-Trough", "inverse")
    with c5:
        excess_val = kpis["excess_return"]
        excess_str = f"{excess_val*100:+.1f}% pts"
        render_kpi_card("Cumulative Excess Return vs. NIFTY TRI", excess_str, f"Empirical Beta {kpis['beta']:.2f}")


def render_price_chart_tab(
    company_df: pd.DataFrame,
    corporate_actions_df: pd.DataFrame,
    company_id: str,
    full_company_df: Optional[pd.DataFrame] = None,
) -> None:
    """Render interactive price history chart with overlays and volume panel."""
    st.subheader("Price History & Volume")

    # Controls
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2, 1.5, 1.5, 2])
    with ctrl_col1:
        price_mode = st.radio(
            "Price Series",
            ["Adjusted Close (Total Return)", "Raw Traded Close (Split Impacted)"],
            index=0,
            horizontal=True,
            key="ce_price_mode",
        )
    with ctrl_col2:
        show_ma50 = st.checkbox("Overlay MA 50", value=True, key="ce_ma50")
        show_ma200 = st.checkbox("Overlay MA 200", value=False, key="ce_ma200")
    with ctrl_col3:
        show_actions = st.checkbox("Corporate Action Markers", value=True, key="ce_actions")
        chart_style = st.selectbox("Chart Style", ["Line Chart", "Area Chart"], index=0, key="ce_style")

    selected_col = "adj_close" if "Adjusted" in price_mode else "close"

    # Calculate moving averages using full historical chronology to prevent lookback truncation
    base_df_for_ma = full_company_df if full_company_df is not None else company_df
    df_full_ma = calculate_moving_averages(base_df_for_ma, (50, 200), selected_col)

    if full_company_df is not None:
        filter_dates = set(pd.to_datetime(company_df["date"]).dt.strftime("%Y-%m-%d"))
        df_full_ma["d_str"] = pd.to_datetime(df_full_ma["date"]).dt.strftime("%Y-%m-%d")
        df_chart = df_full_ma[df_full_ma["d_str"].isin(filter_dates)].copy().sort_values("date").reset_index(drop=True)
    else:
        df_chart = df_full_ma

    df_chart["dt"] = pd.to_datetime(df_chart["date"])

    # Base price chart
    y_title = f"{'Adjusted' if selected_col == 'adj_close' else 'Raw Traded'} Price (₹)"
    base = alt.Chart(df_chart).encode(
        x=alt.X("dt:T", title="Date", axis=alt.Axis(format="%b %Y")),
    )

    if chart_style == "Line Chart":
        price_line = base.mark_line(color="#1f77b4", strokeWidth=2).encode(
            y=alt.Y(f"{selected_col}:Q", title=y_title, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:N", title="Date"),
                alt.Tooltip(f"{selected_col}:Q", title="Price (₹)", format=",.2f"),
                alt.Tooltip("volume:Q", title="Volume", format=","),
            ],
        )
    else:
        price_line = base.mark_area(
            color="#1f77b4",
            opacity=0.3,
            line={"color": "#1f77b4", "strokeWidth": 2},
        ).encode(
            y=alt.Y(f"{selected_col}:Q", title=y_title, scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:N", title="Date"),
                alt.Tooltip(f"{selected_col}:Q", title="Price (₹)", format=",.2f"),
                alt.Tooltip("volume:Q", title="Volume", format=","),
            ],
        )

    chart_layers = [price_line]

    # Moving average overlays
    if show_ma50:
        ma50_line = base.mark_line(color="#ff7f0e", strokeWidth=1.5, strokeDash=[4, 4]).encode(
            y=f"ma_50:Q",
            tooltip=[alt.Tooltip("date:N"), alt.Tooltip("ma_50:Q", title="MA 50", format=",.2f")],
        )
        chart_layers.append(ma50_line)

    if show_ma200:
        ma200_line = base.mark_line(color="#2ca02c", strokeWidth=1.5, strokeDash=[2, 2]).encode(
            y=f"ma_200:Q",
            tooltip=[alt.Tooltip("date:N"), alt.Tooltip("ma_200:Q", title="MA 200", format=",.2f")],
        )
        chart_layers.append(ma200_line)

    # Corporate actions vertical annotations
    if show_actions and len(corporate_actions_df) > 0:
        comp_actions = filter_by_company(corporate_actions_df, company_id)
        if len(comp_actions) > 0:
            comp_actions["dt"] = pd.to_datetime(comp_actions["ex_date"])
            min_date = df_chart["dt"].min()
            max_date = df_chart["dt"].max()
            comp_actions = comp_actions[(comp_actions["dt"] >= min_date) & (comp_actions["dt"] <= max_date)]

            if len(comp_actions) > 0:
                action_rules = alt.Chart(comp_actions).mark_rule(
                    color="#d62728", strokeWidth=1.5, strokeDash=[5, 5]
                ).encode(
                    x="dt:T",
                    tooltip=[
                        alt.Tooltip("ex_date:N", title="Ex-Date"),
                        alt.Tooltip("action_type:N", title="Action"),
                        alt.Tooltip("description:N", title="Details"),
                        alt.Tooltip("ratio_or_amount:Q", title="Ratio / DPS"),
                    ],
                )
                chart_layers.append(action_rules)

    main_price_chart = alt.layer(*chart_layers).properties(
        height=380,
        title=f"{company_id} Price History ({price_mode})",
    )

    st.altair_chart(main_price_chart, use_container_width=True)

    # Volume Chart Panel
    vol_chart = alt.Chart(df_chart).mark_bar(color="#7f7f7f", opacity=0.7).encode(
        x=alt.X("dt:T", title="Date", axis=alt.Axis(format="%b %Y")),
        y=alt.Y("volume:Q", title="Volume (Shares)", axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("date:N", title="Date"), alt.Tooltip("volume:Q", title="Volume", format=",")],
    ).properties(height=140, title="Daily Trading Volume")

    st.altair_chart(vol_chart, use_container_width=True)


def render_benchmark_comparison_tab(
    rebased_df: pd.DataFrame,
    company_id: str,
    kpis: Dict[str, Any],
) -> None:
    """Render rebased performance chart vs synthetic NIFTY TRI."""
    st.subheader("Benchmark Comparison: Company vs. NIFTY TRI")
    st.markdown(
        """
        **Total Return Comparison Methodology**:
        Both series are rebased to **100.0** at the initial date of the selected window.
        - **Company Line**: Total-return adjusted closing price (`adj_close`).
        - **Benchmark Line**: NIFTY Total Return Index (`nifty_tri`), reflecting 1.30% continuous dividend reinvestment.
        """
    )

    if len(rebased_df) == 0:
        st.warning("Insufficient common observation dates between company and benchmark in this window.")
        return

    df_plot = rebased_df.copy()
    df_plot["dt"] = pd.to_datetime(df_plot["date"])

    # Melt for Altair multi-line comparison
    df_melt = pd.melt(
        df_plot,
        id_vars=["dt", "date"],
        value_vars=["company_rebased", "benchmark_rebased"],
        var_name="Series",
        value_name="Rebased_Value",
    )
    df_melt["Series"] = df_melt["Series"].map({
        "company_rebased": f"{company_id} (Total Return)",
        "benchmark_rebased": "NIFTY TRI (Synthetic Benchmark)",
    })

    comp_chart = alt.Chart(df_melt).mark_line(strokeWidth=2).encode(
        x=alt.X("dt:T", title="Date", axis=alt.Axis(format="%b %Y")),
        y=alt.Y("Rebased_Value:Q", title="Rebased Level (Base = 100)", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(
                domain=[f"{company_id} (Total Return)", "NIFTY TRI (Synthetic Benchmark)"],
                range=["#1f77b4", "#2ca02c"],
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[
            alt.Tooltip("date:N", title="Date"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("Rebased_Value:Q", title="Rebased Level", format=",.2f"),
        ],
    ).properties(height=360)

    st.altair_chart(comp_chart, use_container_width=True)

    # Spread Chart (Cumulative Excess Return)
    spread_chart = alt.Chart(df_plot).mark_area(
        color="#1f77b4", opacity=0.25, line={"color": "#1f77b4", "strokeWidth": 1.5}
    ).encode(
        x=alt.X("dt:T", title="Date", axis=alt.Axis(format="%b %Y")),
        y=alt.Y("excess_spread:Q", title="Excess Spread (pts)"),
        tooltip=[alt.Tooltip("date:N", title="Date"), alt.Tooltip("excess_spread:Q", title="Spread (pts)", format="+,.2f")],
    ).properties(height=160, title="Cumulative Excess Return Spread vs. NIFTY TRI")

    st.altair_chart(spread_chart, use_container_width=True)


def render_returns_and_risk_tab(company_df: pd.DataFrame) -> None:
    """Render returns distribution, drawdown underwater curve, and rolling volatility."""
    st.subheader("Returns & Drawdown Risk Analytics")

    dd_df = calculate_drawdown_series(company_df)
    rvol_df = calculate_rolling_volatility(company_df, window=63)
    dist_stats = calculate_returns_distribution(company_df)

    dd_df["dt"] = pd.to_datetime(dd_df["date"])
    rvol_df["dt"] = pd.to_datetime(rvol_df["date"])

    c1, c2 = st.columns([3, 2])

    with c1:
        # Underwater drawdown chart
        dd_chart = alt.Chart(dd_df).mark_area(
            color="#d62728", opacity=0.4, line={"color": "#d62728", "strokeWidth": 1.5}
        ).encode(
            x=alt.X("dt:T", title="Date", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("drawdown:Q", title="Drawdown", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("date:N", title="Date"), alt.Tooltip("drawdown:Q", title="Drawdown", format=".2%")],
        ).properties(height=260, title="Underwater Drawdown Curve (DD_t = P_t / Peak_t - 1)")

        st.altair_chart(dd_chart, use_container_width=True)

    with c2:
        st.markdown("#### Return Distribution Statistics")
        st.markdown(
            f"""
            - **Annualized Mean Drift**: `{dist_stats['mean']*100:+.2f}%`
            - **Annualized Volatility**: `{dist_stats['std']*100:.2f}%`
            - **Skewness**: `{dist_stats['skew']:+.3f}`
            - **Excess Kurtosis**: `{dist_stats['kurt']:+.3f}`
            - **Positive Session Share**: `{dist_stats['positive_days_pct']:.1f}%`
            """
        )

    # Rolling 63-day volatility chart
    st.markdown("#### Rolling 63-Trading-Day Annualized Volatility")
    vol_chart = alt.Chart(rvol_df[rvol_df["rolling_vol"] > 0]).mark_line(
        color="#9467bd", strokeWidth=2
    ).encode(
        x=alt.X("dt:T", title="Date", axis=alt.Axis(format="%b %Y")),
        y=alt.Y("rolling_vol:Q", title="Annualized Volatility", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("date:N", title="Date"), alt.Tooltip("rolling_vol:Q", title="63d Vol", format=".2%")],
    ).properties(height=220)

    st.altair_chart(vol_chart, use_container_width=True)


def render_corporate_actions_tab(
    corporate_actions_df: pd.DataFrame,
    company_id: str,
) -> None:
    """Render corporate actions table and accounting details."""
    st.subheader(f"Corporate Actions for {company_id}")

    comp_actions = filter_by_company(corporate_actions_df, company_id)

    if len(comp_actions) == 0:
        st.info("No corporate actions recorded for this company in the selected simulation period.")
        return

    splits = comp_actions[comp_actions["action_type"] == "split"]
    dividends = comp_actions[comp_actions["action_type"] == "dividend"]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Corporate Actions", len(comp_actions))
        st.caption(f"Stock Splits: {len(splits)} | Annual Dividends: {len(dividends)}")
    with c2:
        if len(dividends) > 0:
            avg_dps = float(dividends["ratio_or_amount"].mean())
            st.metric("Average Dividend per Share", f"₹{avg_dps:.2f}")

    # Format table for display
    display_df = comp_actions.copy()
    display_df["ex_date"] = pd.to_datetime(display_df["ex_date"]).dt.strftime("%Y-%m-%d")
    display_df["action_type"] = display_df["action_type"].str.upper()

    def format_amount(row):
        if row["action_type"] == "SPLIT":
            return f"{row['ratio_or_amount']:.1f}:1 Split"
        else:
            return f"₹{row['ratio_or_amount']:.2f} / share"

    display_df["Benefit / Ratio"] = display_df.apply(format_amount, axis=1)

    st.dataframe(
        display_df[["action_id", "ex_date", "action_type", "Benefit / Ratio", "description", "data_source"]].rename(
            columns={
                "action_id": "Action ID",
                "ex_date": "Ex-Date",
                "action_type": "Action Type",
                "description": "Description",
                "data_source": "Source",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "💡 **Split-Adjustment Note**: Production `adj_close` is already total-return adjusted. "
        "The mechanical price adjustment occurs only on the raw `close` series, avoiding any duplicate adjustments."
    )


def render_ohlcv_data_tab(company_df: pd.DataFrame) -> None:
    """Render historical OHLCV data table with download options."""
    st.subheader("Historical OHLCV Data")

    st.markdown(
        f"Displaying **{len(company_df)}** trading sessions. Sort order: newest sessions first."
    )

    df_display = company_df.copy().sort_values("date", ascending=False).reset_index(drop=True)
    df_display["date"] = pd.to_datetime(df_display["date"]).dt.strftime("%Y-%m-%d")

    # Format numeric columns
    format_cols = ["open", "high", "low", "close", "adj_close"]
    for col in format_cols:
        df_display[col] = df_display[col].apply(lambda x: f"₹{x:,.2f}")
    df_display["volume"] = df_display["volume"].apply(lambda x: f"{x:,}")

    st.dataframe(
        df_display[["date", "open", "high", "low", "close", "adj_close", "volume", "data_source"]],
        hide_index=True,
        use_container_width=True,
    )

    # Download CSV
    csv_bytes = company_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download OHLCV Data (CSV)",
        data=csv_bytes,
        file_name=f"{company_df['company_id'].iloc[0]}_prices_daily.csv",
        mime="text/csv",
    )


def render_fundamentals_preview_tab(
    company_id: str,
) -> None:
    """Render downstream Member 4 quarterly fundamentals preview if available."""
    st.subheader("Quarterly Fundamentals Preview (Downstream Member 4)")

    try:
        df_fq = load("fundamentals_quarterly", use_mock=True)
        is_available = True
    except Exception:
        is_available = False

    if not is_available:
        st.info("ℹ️ **Status: Pending Downstream Implementation**\n\nQuarterly fundamentals dataset will be generated by Member 4.")
        return

    st.caption("🛠️ **Development Notice**: Loaded from schema-compliant mock sample (`data/mock/fundamentals_quarterly.csv`).")

    comp_fq = filter_by_company(df_fq, company_id)
    if len(comp_fq) == 0:
        st.warning(f"No mock quarterly records found for {company_id}.")
        return

    comp_fq = comp_fq.sort_values("period_end").reset_index(drop=True)

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Latest Quarter", f"{comp_fq.iloc[-1]['fiscal_quarter']} ({comp_fq.iloc[-1]['fiscal_year']})")
    with m2:
        rev_cr = comp_fq.iloc[-1]["revenue"]
        st.metric("Revenue (₹ Cr)", f"₹{rev_cr:,.1f} Cr")
    with m3:
        pat_cr = comp_fq.iloc[-1]["net_profit"]
        st.metric("Net Profit (₹ Cr)", f"₹{pat_cr:,.1f} Cr")
    with m4:
        eps_val = comp_fq.iloc[-1]["eps"]
        st.metric("Basic EPS", f"₹{eps_val:.2f}")

    # Fundamentals table
    show_cols = ["fiscal_year", "fiscal_quarter", "period_end", "revenue", "ebit", "net_profit", "eps", "total_debt", "total_equity"]
    available_cols = [c for c in show_cols if c in comp_fq.columns]
    st.dataframe(comp_fq[available_cols], hide_index=True, use_container_width=True)


# ==============================================================================
# 3. MAIN MODULE ENTRYPOINT
# ==============================================================================

def render() -> None:
    """Main rendering entrypoint for Module 2: Company Explorer."""
    demo_banner()

    st.title("🌱 Company Explorer")
    st.markdown(
        "**Market Dynamics, Total-Return Analytics, and Corporate Actions for Sustainable Equities**"
    )

    # 1. Load Data Assets
    try:
        cm_df = load("company_master", use_mock=False)
        pr_df = load("prices_daily", use_mock=False)
        idx_df = load("index_daily", use_mock=False)
        ca_df = load("corporate_actions", use_mock=False)
    except FileNotFoundError as e:
        st.error(f"Required production dataset missing: {e}. Please ensure Member 2 generation scripts have completed.")
        return

    # 2. Company Selector
    company_options = [
        (row["company_id"], f"{row['company_id']} — {row['company_name']} ({row['sector']})")
        for _, row in cm_df.sort_values("company_id").iterrows()
    ]
    comp_ids = [opt[0] for opt in company_options]

    current_id = st.session_state.get("company_id", "C01")
    if current_id not in comp_ids:
        current_id = comp_ids[0]

    current_idx = comp_ids.index(current_id)

    col_sel1, col_sel2 = st.columns([3, 3])

    with col_sel1:
        selected_tuple = st.selectbox(
            "Select Sustainable Company",
            options=company_options,
            index=current_idx,
            format_func=lambda x: x[1],
            key="ce_company_select",
        )
        selected_company_id = selected_tuple[0]
        st.session_state["company_id"] = selected_company_id

    # Filter company prices
    company_full_prices = filter_by_company(pr_df, selected_company_id)
    if len(company_full_prices) == 0:
        st.error(f"No price records found for {selected_company_id}.")
        return

    # 3. Date Range Selection
    all_dates = pd.to_datetime(company_full_prices["date"]).dt.strftime("%Y-%m-%d").tolist()
    min_date_str = all_dates[0]
    max_date_str = all_dates[-1]

    with col_sel2:
        date_preset = st.radio(
            "Observation Window Preset",
            ["Last 1 Year (252 sessions)", "Last 3 Years (756 sessions)", "Full Simulation (2020–2025)", "Custom Range"],
            index=0,
            horizontal=True,
            key="ce_date_preset",
        )

    if date_preset == "Last 1 Year (252 sessions)":
        start_date_str = all_dates[-min(252, len(all_dates))]
        end_date_str = max_date_str
    elif date_preset == "Last 3 Years (756 sessions)":
        start_date_str = all_dates[-min(756, len(all_dates))]
        end_date_str = max_date_str
    elif date_preset == "Full Simulation (2020–2025)":
        start_date_str = min_date_str
        end_date_str = max_date_str
    else:  # Custom Range
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            start_date_str = st.date_input(
                "Start Date",
                value=pd.to_datetime(all_dates[-min(252, len(all_dates))]),
                min_value=pd.to_datetime(min_date_str),
                max_value=pd.to_datetime(max_date_str),
            ).strftime("%Y-%m-%d")
        with c_d2:
            end_date_str = st.date_input(
                "End Date",
                value=pd.to_datetime(max_date_str),
                min_value=pd.to_datetime(min_date_str),
                max_value=pd.to_datetime(max_date_str),
            ).strftime("%Y-%m-%d")

    # Filter by date range
    filtered_prices = filter_by_date_range(company_full_prices, start_date_str, end_date_str)
    filtered_benchmark = filter_by_date_range(idx_df, start_date_str, end_date_str)

    if len(filtered_prices) == 0:
        st.warning(f"No trading sessions found for {selected_company_id} between {start_date_str} and {end_date_str}.")
        return

    # Check for warm-up year notification
    if start_date_str < "2021-01-01":
        st.info(
            "ℹ️ **Simulation Notice**: The selected window includes observations from the **2020 warm-up period**. "
            "In SIRPS research methodology, 2020 serves as the warm-up calibration baseline; formal predictive model evaluation starts in 2021."
        )

    # 4. Render Metadata & KPIs
    comp_row = cm_df[cm_df["company_id"] == selected_company_id].iloc[0]
    render_company_header_and_metadata(comp_row)

    kpis = calculate_kpis(filtered_prices, filtered_benchmark)
    render_kpi_dashboard(kpis)

    st.markdown("---")

    # 5. Interactive Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Price & Volume",
        "🎯 Benchmark Comparison",
        "⚡ Returns & Risk",
        "📑 Corporate Actions",
        "📊 Historical OHLCV",
        "🏢 Fundamentals Preview",
    ])

    with tab1:
        render_price_chart_tab(
            filtered_prices,
            ca_df,
            selected_company_id,
            full_company_df=company_full_prices,
        )

    with tab2:
        rebased_df = rebase_to_100(filtered_prices, filtered_benchmark)
        render_benchmark_comparison_tab(rebased_df, selected_company_id, kpis)

    with tab3:
        render_returns_and_risk_tab(filtered_prices)

    with tab4:
        render_corporate_actions_tab(ca_df, selected_company_id)

    with tab5:
        render_ohlcv_data_tab(filtered_prices)

    with tab6:
        render_fundamentals_preview_tab(selected_company_id)


if __name__ == "__main__":
    render()
