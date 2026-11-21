"""SIRPS Module 3: ESG Analysis & Corporate Sustainability Scorecard (Owner: Member 3).

Provides comprehensive exploration of synthetic BRSR ESG disclosures, pillar score breakdowns,
cross-sectional percentile benchmarking against the green universe, and multi-year reporting trends.

Methodology & Architectural Alignment:
1. Long-Format BRSR Disclosures:
   Reads data/esg_facts.csv (320 rows across 8 core indicators, 10 companies, 4 fiscal years).
2. Authoritative Percentile Scoring & 60% Coverage Rule:
   Reads data/esg_scores.csv (derived via common/scoring.py). Displays E, S, G pillar scores
   and total ESG score. Explains why a pillar score is null when reported indicators fall below
   the 60% threshold.
3. Strict Point-in-Time Traceability:
   Displays corporate filing date (mid-July) and point-in-time trading availability date.
4. Interactive Benchmarking:
   - Scorecard: Radar chart comparing firm against universe median across E, S, G pillars.
   - Indicators: Granular table displaying status ('value', 'NR', 'NA'), source document, and page.
   - Peers: Ranked cross-sectional bar charts benchmarking the firm across all 10 companies.
   - Trend: Multi-year historical progression across all 4 fiscal years.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from common.io import load
from common.ui import demo_banner, company_selector, render_kpi_card
from common.scoring import add_indicator_percentiles, INDICATOR_CONFIG, PILLAR_INDICATORS

logger = logging.getLogger(__name__)


# ==============================================================================
# PURE DATA PROCESSING & METRIC EXTRACTION HELPERS (TESTABLE)
# ==============================================================================

def get_company_esg_summary(
    df_scores: pd.DataFrame, company_id: str, fiscal_year: str
) -> Optional[Dict[str, Any]]:
    """Extracts score and coverage metrics for a specific company and fiscal year."""
    subset = df_scores[(df_scores["company_id"] == company_id) & (df_scores["fiscal_year"] == fiscal_year)]
    if len(subset) == 0:
        return None
    row = subset.iloc[0]
    return {
        "company_id": company_id,
        "fiscal_year": fiscal_year,
        "available_date": str(row["available_date"])[:10],
        "e_score": row["e_score"] if pd.notna(row["e_score"]) else None,
        "s_score": row["s_score"] if pd.notna(row["s_score"]) else None,
        "g_score": row["g_score"] if pd.notna(row["g_score"]) else None,
        "esg_total": row["esg_total"] if pd.notna(row["esg_total"]) else None,
        "coverage_ratio": float(row["coverage_ratio"]),
        "e_coverage": float(row["e_coverage"]),
        "s_coverage": float(row["s_coverage"]),
        "g_coverage": float(row["g_coverage"]),
    }


def get_universe_pillar_medians(df_scores: pd.DataFrame, fiscal_year: str) -> Dict[str, float]:
    """Calculates median pillar scores across all companies for a given fiscal year."""
    fy_scores = df_scores[df_scores["fiscal_year"] == fiscal_year]
    return {
        "E": float(fy_scores["e_score"].dropna().median()) if len(fy_scores["e_score"].dropna()) > 0 else 50.0,
        "S": float(fy_scores["s_score"].dropna().median()) if len(fy_scores["s_score"].dropna()) > 0 else 50.0,
        "G": float(fy_scores["g_score"].dropna().median()) if len(fy_scores["g_score"].dropna()) > 0 else 50.0,
        "Total": float(fy_scores["esg_total"].dropna().median()) if len(fy_scores["esg_total"].dropna()) > 0 else 50.0,
    }


# ==============================================================================
# MAIN PAGE RENDERER
# ==============================================================================

def render() -> None:
    """Renders the ESG Analysis module in the Streamlit application."""
    demo_banner()

    st.title("🌱 ESG Analysis & Corporate Sustainability Scorecard")
    st.markdown(
        """
        Analyze synthetic Business Responsibility and Sustainability Reporting (BRSR) disclosures,
        cross-sectional pillar performance, reporting coverage, and peer rankings across the Indian green universe.
        """
    )

    # --------------------------------------------------------------------------
    # 1. Load Datasets via Shared I/O
    # --------------------------------------------------------------------------
    try:
        company_master = load("company_master", use_mock=False)
        df_facts_raw = load("esg_facts", use_mock=False)
        df_scores = load("esg_scores", use_mock=False)
    except FileNotFoundError as exc:
        st.error(f"Required ESG datasets not found: {exc}")
        st.info("Please generate production datasets via `python data_generation/gen_esg.py`.")
        return
    except Exception as exc:
        st.error(f"Error loading ESG data: {exc}")
        return

    # Attach computed direction-adjusted percentiles for detailed inspection
    df_facts = add_indicator_percentiles(df_facts_raw)

    # --------------------------------------------------------------------------
    # 2. Control Filters (Company & Fiscal Year)
    # --------------------------------------------------------------------------
    col_c, col_fy = st.columns([2, 1])

    with col_c:
        cid = company_selector(label="Select Company", key="esg_company_select")
        company_info = company_master[company_master["company_id"] == cid].iloc[0]
        st.caption(f"**Sector:** {company_info['sector']} | **Ticker:** `{company_info['ticker']}`")

    with col_fy:
        available_fys = sorted(df_facts["fiscal_year"].unique().tolist())
        default_fy = "FY2024-25" if "FY2024-25" in available_fys else available_fys[-1]
        selected_fy = st.selectbox(
            "Select Reporting Fiscal Year",
            options=available_fys,
            index=available_fys.index(default_fy) if default_fy in available_fys else 0,
            key="esg_fy_select"
        )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 3. Top Executive KPI Scorecards
    # --------------------------------------------------------------------------
    summary = get_company_esg_summary(df_scores, cid, selected_fy)
    if summary is None:
        st.warning(f"No score data available for company {cid} in fiscal year {selected_fy}.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)

    # KPI 1: Total ESG Score
    with col1:
        tot = summary["esg_total"]
        if tot is not None:
            st.metric("Total ESG Score", f"{tot:.1f} / 100", delta=None)
            if tot >= 70:
                st.caption("🟢 **Leader** (Top Quintile)")
            elif tot >= 40:
                st.caption("🟡 **Average** (Middle Quintiles)")
            else:
                st.caption("🔴 **Laggard** (Bottom Quintile)")
        else:
            st.metric("Total ESG Score", "N/A*")
            st.caption("⚠️ *Insufficient Coverage (<60%)*")

    # KPI 2: Environmental (E)
    with col2:
        e_sc = summary["e_score"]
        e_cov_pct = int(summary["e_coverage"] * 100)
        e_val = f"{e_sc:.1f}" if e_sc is not None else "N/A"
        st.metric("Environmental (E)", e_val, delta=f"{e_cov_pct}% Cov")
        st.caption(f"5 indicators | {int(summary['e_coverage'] * 5)}/5 reported")

    # KPI 3: Social (S)
    with col3:
        s_sc = summary["s_score"]
        s_cov_pct = int(summary["s_coverage"] * 100)
        s_val = f"{s_sc:.1f}" if s_sc is not None else "N/A"
        st.metric("Social (S)", s_val, delta=f"{s_cov_pct}% Cov")
        st.caption(f"2 indicators | {int(summary['s_coverage'] * 2)}/2 reported")

    # KPI 4: Governance (G)
    with col4:
        g_sc = summary["g_score"]
        g_cov_pct = int(summary["g_coverage"] * 100)
        g_val = f"{g_sc:.1f}" if g_sc is not None else "N/A"
        st.metric("Governance (G)", g_val, delta=f"{g_cov_pct}% Cov")
        st.caption(f"1 indicator | {int(summary['g_coverage'] * 1)}/1 reported")

    # KPI 5: Reporting Coverage & Provenance
    with col5:
        cov_tot_pct = int(summary["coverage_ratio"] * 100)
        reported_count = int(round(summary["coverage_ratio"] * 8))
        st.metric("Overall Coverage", f"{cov_tot_pct}%", delta=f"{reported_count}/8 Indicators")
        st.caption(f"Available: `{summary['available_date']}`")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 4. Interactive Tabs: Scorecard | Indicators | Peers | Trend
    # --------------------------------------------------------------------------
    tab_scorecard, tab_indicators, tab_peers, tab_trend = st.tabs([
        "📊 Scorecard & Benchmark",
        "📋 Indicator Disclosures",
        "🏆 Peer Comparison",
        "📈 Multi-Year Trend",
    ])

    # ==========================================================================
    # TAB 1: SCORECARD & RADAR BENCHMARK
    # ==========================================================================
    with tab_scorecard:
        col_rad, col_exp = st.columns([3, 2])

        with col_rad:
            st.subheader(f"Pillar Profile: {company_info['company_name']} vs. Universe Median")
            medians = get_universe_pillar_medians(df_scores, selected_fy)

            categories = ["Environmental (E)", "Social (S)", "Governance (G)"]

            company_vals = [
                summary["e_score"] if summary["e_score"] is not None else 0.0,
                summary["s_score"] if summary["s_score"] is not None else 0.0,
                summary["g_score"] if summary["g_score"] is not None else 0.0,
            ]
            median_vals = [medians["E"], medians["S"], medians["G"]]

            # Close the polygon for radar
            categories_closed = categories + [categories[0]]
            company_closed = company_vals + [company_vals[0]]
            median_closed = median_vals + [median_vals[0]]

            fig_radar = go.Figure()

            # Universe Median Polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=median_closed,
                theta=categories_closed,
                fill="toself",
                name="Universe Median",
                line=dict(color="#6c757d", dash="dash"),
                fillcolor="rgba(108, 117, 125, 0.15)",
            ))

            # Company Polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=company_closed,
                theta=categories_closed,
                fill="toself",
                name=f"{cid} ({company_info['ticker']})",
                line=dict(color="#2e7d32", width=3),
                fillcolor="rgba(46, 125, 50, 0.30)",
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
                    angularaxis=dict(tickfont=dict(size=12)),
                ),
                showlegend=True,
                margin=dict(l=40, r=40, t=30, b=30),
                height=380,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_exp:
            st.subheader("Methodology & 60% Coverage Rule")
            st.markdown(
                r"""
                **Pillar Scoring Rules:**
                - Scores are **cross-sectional percentile ranks (0–100)** calculated within the 10-firm green universe for each fiscal year.
                - Direction is adjusted: lower values receive higher scores for intensity metrics (energy, emissions, water) and workplace injuries (LTIFR).

                **The 60% Coverage Rule:**
                - A pillar score is computed **only if the firm reports at least 60% of indicators** in that pillar.
                - **Environmental (E):** Requires $\ge 3$ of 5 indicators reported.
                - **Social (S):** Requires 2 of 2 indicators reported ($1/2 = 50\% < 60\%$).
                - **Governance (G):** Requires 1 of 1 indicator reported ($1/1 = 100\% \ge 60\%$).

                **Total ESG Score (`esg_total`):**
                - Equal-weighted average of the three pillars:
                  $$\\text{ESG Total} = \\frac{E + S + G}{3}$$
                - If **any** pillar fails the 60% coverage threshold, the total score is marked **N/A** (not evaluated) to prevent misleading aggregation.
                """
            )

            if summary["esg_total"] is None:
                st.warning(
                    f"⚠️ **Coverage Notice for {cid}:** One or more pillars had insufficient disclosures for {selected_fy}. "
                    "Under the 60% coverage rule, a composite ESG Total score is omitted to avoid distorting model signals."
                )

    # ==========================================================================
    # TAB 2: INDICATOR DISCLOSURES
    # ==========================================================================
    with tab_indicators:
        st.subheader(f"BRSR Indicator Disclosures ({selected_fy})")

        c_filter, c_toggle = st.columns([1, 1])
        with c_filter:
            pillar_filter = st.selectbox(
                "Filter by Pillar",
                options=["All Pillars", "Environmental (E)", "Social (S)", "Governance (G)"],
                index=0
            )
        with c_toggle:
            view_mode = st.radio(
                "Table Display Mode",
                options=["Reported Values & Percentiles", "Standardized Percentile Scores Only"],
                horizontal=True
            )

        firm_facts = df_facts[(df_facts["company_id"] == cid) & (df_facts["fiscal_year"] == selected_fy)].copy()

        if pillar_filter == "Environmental (E)":
            firm_facts = firm_facts[firm_facts["pillar"] == "E"]
        elif pillar_filter == "Social (S)":
            firm_facts = firm_facts[firm_facts["pillar"] == "S"]
        elif pillar_filter == "Governance (G)":
            firm_facts = firm_facts[firm_facts["pillar"] == "G"]

        display_rows = []
        for _, r in firm_facts.iterrows():
            ind_id = r["indicator_id"]
            cfg = INDICATOR_CONFIG.get(ind_id, {})
            direction_desc = "Lower is better (Intensities/Injuries)" if cfg.get("lower_is_better") else "Higher is better (Positive Share)"

            # Formatted value display
            if r["status"] == "value" and pd.notna(r["value"]):
                formatted_val = f"{r['value']:,.2f} {r['unit']}"
                pct_str = f"{r['percentile']:.1f} / 100" if pd.notna(r["percentile"]) else "N/A"
                status_badge = "✅ Reported"
            elif r["status"] == "NR":
                formatted_val = "Not Reported"
                pct_str = "—"
                status_badge = "⚠️ NR (Non-Reporting)"
            elif r["status"] == "NA":
                formatted_val = "Not Applicable"
                pct_str = "—"
                status_badge = "⚪ NA (Non-Applicable)"
            else:
                formatted_val = "—"
                pct_str = "—"
                status_badge = f"{r['status']}"

            display_rows.append({
                "Pillar": r["pillar"],
                "Indicator": r["indicator_name"],
                "Status": status_badge,
                "Reported Value": formatted_val,
                "Direction": direction_desc,
                "Percentile Score": pct_str,
                "Source Document": r["source_doc"],
                "Page": r["source_page"],
            })

        df_display = pd.DataFrame(display_rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.caption("All BRSR disclosures are point-in-time verified against audited synthetic annual filings.")

    # ==========================================================================
    # TAB 3: PEER COMPARISON
    # ==========================================================================
    with tab_peers:
        st.subheader(f"Cross-Sectional Peer Benchmarking ({selected_fy})")

        metric_choices = [
            ("esg_total", "Composite ESG Total Score"),
            ("e_score", "Environmental (E) Pillar Score"),
            ("s_score", "Social (S) Pillar Score"),
            ("g_score", "Governance (G) Pillar Score"),
        ]

        c_p_select, c_p_cov = st.columns([2, 1])
        with c_p_select:
            selected_metric_key = st.selectbox(
                "Select Comparison Metric",
                options=[m[0] for m in metric_choices],
                format_func=lambda x: dict(metric_choices)[x]
            )

        # Extract universe values for this metric in the chosen FY
        peer_scores = df_scores[df_scores["fiscal_year"] == selected_fy].copy()
        peer_scores = peer_scores.merge(company_master[["company_id", "company_name", "sector", "ticker"]], on="company_id")

        peer_scores["metric_value"] = peer_scores[selected_metric_key]
        peer_scores = peer_scores.sort_values("metric_value", ascending=True).reset_index(drop=True)

        # Color coding: Highlight the currently selected firm
        colors = ["#2e7d32" if c == cid else "#90caf9" for c in peer_scores["company_id"]]

        fig_bar = go.Figure(go.Bar(
            x=peer_scores["metric_value"],
            y=peer_scores["ticker"] + " (" + peer_scores["company_id"] + ")",
            orientation="h",
            marker=dict(color=colors),
            text=peer_scores["metric_value"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "N/A"),
            textposition="outside",
        ))

        fig_bar.update_layout(
            title=f"{dict(metric_choices)[selected_metric_key]} across all 10 Companies",
            xaxis=dict(title="Score (0–100)", range=[0, 110]),
            yaxis=dict(title=""),
            height=400,
            margin=dict(l=10, r=40, t=40, b=30),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.caption(f"Green bar indicates the currently inspected firm (**{cid}**). Missing values reflect insufficient reporting under the 60% rule.")

    # ==========================================================================
    # TAB 4: MULTI-YEAR TREND
    # ==========================================================================
    with tab_trend:
        st.subheader(f"Historical ESG Trajectory: {company_info['company_name']} (FY2021-22 to FY2024-25)")

        firm_history = df_scores[df_scores["company_id"] == cid].sort_values("fiscal_year")

        fig_trend = go.Figure()

        fig_trend.add_trace(go.Scatter(
            x=firm_history["fiscal_year"],
            y=firm_history["esg_total"],
            mode="lines+markers",
            name="Total ESG",
            line=dict(color="#1b5e20", width=4),
            marker=dict(size=9),
        ))
        fig_trend.add_trace(go.Scatter(
            x=firm_history["fiscal_year"],
            y=firm_history["e_score"],
            mode="lines+markers",
            name="Environmental (E)",
            line=dict(color="#4caf50", width=2, dash="dash"),
        ))
        fig_trend.add_trace(go.Scatter(
            x=firm_history["fiscal_year"],
            y=firm_history["s_score"],
            mode="lines+markers",
            name="Social (S)",
            line=dict(color="#2196f3", width=2, dash="dot"),
        ))
        fig_trend.add_trace(go.Scatter(
            x=firm_history["fiscal_year"],
            y=firm_history["g_score"],
            mode="lines+markers",
            name="Governance (G)",
            line=dict(color="#ff9800", width=2, dash="dashdot"),
        ))

        fig_trend.update_layout(
            title=f"Multi-Year Pillar & Total Score Progression ({cid})",
            xaxis=dict(title="Fiscal Year"),
            yaxis=dict(title="Score (0–100)", range=[0, 105]),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Planted signal disclosure statement as per Blueprint Section 10
        st.info(
            "💡 **Research Methodology Note:** The synthetic generator incorporates a planted ESG drift of "
            "strength 0.04 per standard deviation per 63 trading days, active only after the respective `available_date`. "
            "Recovering this relationship in predictive modeling demonstrates pipeline calibration rather than empirical real-market predictability."
        )


if __name__ == "__main__":
    # Test execution via bare Python
    print("ESG Analysis module successfully compiled.")
