"""SIRPS Mock Dataset Generator (Member 1).

Generates 20-row schema-compliant mock sample CSVs in data/mock/ for all 20 prototype
datasets. This establishes the contract and unlocks independent parallel work
for UI modules, unit tests, and downstream teammates (M2–M7).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.schemas import SCHEMAS, FILES
from common.io import save
import data_generation.config as config

MOCK_DIR: Path = PROJECT_ROOT / "data" / "mock"

def generate_mock_datasets() -> None:
    """Generates 20-row valid mock CSVs for every schema."""
    MOCK_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.MASTER_SEED)
    
    companies = [f"C{i:02d}" for i in range(1, 11)]
    dates = pd.date_range("2021-01-04", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
    
    # 1. trading_calendar
    df_cal = pd.DataFrame({
        "date": dates,
        "trading_day_index": list(range(20)),
        "year": [int(d[:4]) for d in dates],
        "is_month_end": [False] * 19 + [True],
        "data_source": "SYNTHETIC_DEMO"
    })
    save(df_cal, "trading_calendar", is_mock=True)
    
    # 2. index_daily
    df_idx = pd.DataFrame({
        "date": dates,
        "nifty_close": [round(14000.0 + i * 25.0 + rng.normal(0, 30), 2) for i in range(20)],
        "nifty_tri": [round(14200.0 + i * 26.0 + rng.normal(0, 30), 2) for i in range(20)],
        "india_vix": [round(14.5 + rng.normal(0, 1.2), 2) for _ in range(20)],
        "data_source": "SYNTHETIC_DEMO"
    })
    save(df_idx, "index_daily", is_mock=True)
    
    # 3. prices_daily (20 rows: 2 days for all 10 companies)
    price_rows = []
    for d in dates[:2]:
        for cid in companies:
            adj = round(300.0 + rng.uniform(50, 400), 2)
            price_rows.append({
                "company_id": cid,
                "date": d,
                "open": round(adj * 0.99, 2),
                "high": round(adj * 1.02, 2),
                "low": round(adj * 0.98, 2),
                "close": adj,
                "adj_close": adj,
                "volume": int(rng.integers(50000, 500000)),
                "data_source": "SYNTHETIC_DEMO"
            })
    df_prices = pd.DataFrame(price_rows)
    save(df_prices, "prices_daily", is_mock=True)
    
    # 4. corporate_actions
    ca_rows = []
    for i in range(20):
        ca_rows.append({
            "action_id": f"CA{i+1:03d}",
            "company_id": companies[i % 10],
            "ex_date": dates[i % len(dates)],
            "action_type": "dividend" if i % 4 != 0 else "split",
            "ratio_or_amount": 4.5 if i % 4 != 0 else 2.0,
            "description": f"Synthetic corporate action {i+1}",
            "data_source": "SYNTHETIC_DEMO"
        })
    df_ca = pd.DataFrame(ca_rows)
    save(df_ca, "corporate_actions", is_mock=True)
    
    # 5. esg_facts
    indicators = [
        ("renewable_share_pct", "Renewable energy share", "E", "%"),
        ("energy_intensity", "Energy intensity", "E", "GJ per Rs cr"),
        ("emission_intensity", "GHG emission intensity", "E", "tCO2e per Rs cr"),
        ("water_intensity", "Water intensity", "E", "kL per Rs cr"),
        ("waste_recovery_pct", "Waste recovery rate", "E", "%"),
        ("ltifr", "Lost time injury frequency rate", "S", "per million hours"),
        ("women_workforce_pct", "Women in workforce", "S", "%"),
        ("independent_directors_pct", "Independent directors", "G", "%"),
    ]
    facts_rows = []
    for i in range(20):
        ind = indicators[i % len(indicators)]
        facts_rows.append({
            "fact_id": i + 1,
            "company_id": companies[i % 10],
            "fiscal_year": "FY2023-24",
            "indicator_id": ind[0],
            "indicator_name": ind[1],
            "pillar": ind[2],
            "unit": ind[3],
            "value": round(float(rng.uniform(10, 85)), 2) if i % 7 != 0 else np.nan,
            "status": "value" if i % 7 != 0 else "NR",
            "source_doc": f"BRSR_SYNTH_{companies[i % 10]}_FY2023-24.pdf",
            "source_page": 20 + i,
            "filed_date": "2024-07-15",
            "available_date": "2024-07-16",
            "data_source": "SYNTHETIC_DEMO"
        })
    df_facts = pd.DataFrame(facts_rows)
    save(df_facts, "esg_facts", is_mock=True)
    
    # 6. esg_scores
    scores_rows = []
    for i in range(20):
        cid = companies[i % 10]
        fy = "FY2023-24" if i < 10 else "FY2024-25"
        e = round(float(rng.uniform(40, 90)), 1)
        s = round(float(rng.uniform(40, 90)), 1)
        g = round(float(rng.uniform(40, 90)), 1)
        scores_rows.append({
            "company_id": cid,
            "fiscal_year": fy,
            "available_date": "2024-07-16" if fy == "FY2023-24" else "2025-07-15",
            "e_score": e,
            "s_score": s,
            "g_score": g,
            "esg_total": round((e + s + g) / 3.0, 1),
            "coverage_ratio": 0.875,
            "e_coverage": 0.80,
            "s_coverage": 1.00,
            "g_coverage": 1.00,
            "data_source": "SYNTHETIC_DEMO"
        })
    df_scores = pd.DataFrame(scores_rows)
    save(df_scores, "esg_scores", is_mock=True)
    
    # 7. fundamentals_quarterly
    fund_rows = []
    for i in range(20):
        cid = companies[i % 10]
        rev = round(float(rng.uniform(800, 3000)), 1)
        ebit = round(rev * 0.22, 1)
        np_val = round(ebit * 0.65, 1)
        fund_rows.append({
            "company_id": cid,
            "period_end": "2025-06-30" if i < 10 else "2025-09-30",
            "fiscal_year": "FY2025-26",
            "fiscal_quarter": 1 if i < 10 else 2,
            "filed_date": "2025-08-10" if i < 10 else "2025-11-12",
            "available_date": "2025-08-11" if i < 10 else "2025-11-13",
            "revenue": rev,
            "ebit": ebit,
            "net_profit": np_val,
            "interest_expense": round(ebit * 0.2, 1),
            "operating_cash_flow": round(ebit * 0.85, 1),
            "total_assets": round(rev * 4.5, 1),
            "total_debt": round(rev * 1.5, 1),
            "total_equity": round(rev * 2.2, 1),
            "eps": round(np_val / 25.0, 2),
            "shares_outstanding": 25.0,
            "data_source": "SYNTHETIC_DEMO"
        })
    df_fund = pd.DataFrame(fund_rows)
    save(df_fund, "fundamentals_quarterly", is_mock=True)
    
    # 8. regime_daily
    regime_rows = []
    for i, d in enumerate(dates):
        close_val = 15000.0 + i * 20.0
        regime_rows.append({
            "date": d,
            "nifty_close": close_val,
            "ma50": close_val * 0.98,
            "ma200": close_val * 0.95,
            "high_252": close_val * 1.05,
            "drawdown_252": -0.05,
            "raw_regime": "bull",
            "confirmed_regime": "bull",
            "regime_code": 1,
            "data_source": "SYNTHETIC_DEMO"
        })
    df_regime = pd.DataFrame(regime_rows)
    save(df_regime, "regime_daily", is_mock=True)
    
    # 9. model_dataset (20 rows)
    md_rows = []
    for i in range(20):
        cid = companies[i % 10]
        d = dates[i % len(dates)]
        md_rows.append({
            "company_id": cid,
            "date": d,
            "entry_date": dates[(i + 1) % len(dates)],
            "target_end_date": dates[(i + 5) % len(dates)],
            "y_excess": round(float(rng.normal(0.015, 0.08)), 4),
            "y_abs": round(float(rng.normal(0.035, 0.09)), 4),
            "regime": "bull",
            "regime_code": 1,
            "data_source": "SYNTHETIC_DEMO"
        })
    df_md = pd.DataFrame(md_rows)
    save(df_md, "model_dataset", is_mock=True)
    
    # 10. latest_features (10 rows, pad to 20 or 10 rows)
    lf_rows = []
    for cid in companies:
        lf_rows.append({
            "company_id": cid,
            "date": "2025-12-31",
            "regime": "bull",
            "regime_code": 1,
            "data_source": "SYNTHETIC_DEMO"
        })
    df_lf = pd.DataFrame(lf_rows)
    save(df_lf, "latest_features", is_mock=True)
    
    # 11. fold_definitions
    fold_rows = []
    for f in range(1, 21):
        fold_rows.append({
            "fold_id": f,
            "train_start": "2021-01-04",
            "train_end": "2023-12-29",
            "test_start": "2024-01-02",
            "test_end": "2024-06-28",
            "data_source": "SYNTHETIC_DEMO"
        })
    df_folds = pd.DataFrame(fold_rows)
    save(df_folds, "fold_definitions", is_mock=True)
    
    # 12. predictions (20 rows: 10 companies × 2 experiments)
    pred_rows = []
    for exp in ["A", "B"]:
        for cid in companies:
            pred_rows.append({
                "experiment": exp,
                "model": "XGBoost",
                "company_id": cid,
                "date": "2025-12-31",
                "y_true": round(float(rng.normal(0.02, 0.07)), 4),
                "y_hat": round(float(rng.normal(0.02, 0.05)), 4),
                "fold_id": 1,
                "data_source": "SYNTHETIC_DEMO"
            })
    df_preds = pd.DataFrame(pred_rows)
    save(df_preds, "predictions", is_mock=True)
    
    # 13. latest_predictions (20 rows: 10 companies × 2 experiments)
    lp_rows = []
    for exp in ["A", "B"]:
        for cid in companies:
            y_hat = round(float(rng.normal(0.025, 0.04)), 4)
            lp_rows.append({
                "experiment": exp,
                "model": "XGBoost",
                "company_id": cid,
                "date": "2025-12-31",
                "y_hat_excess": y_hat,
                "y_hat_abs": round(y_hat + 0.03, 4),
                "lower80": round(y_hat - 0.05, 4),
                "upper80": round(y_hat + 0.05, 4),
                "data_source": "SYNTHETIC_DEMO"
            })
    df_lp = pd.DataFrame(lp_rows)
    save(df_lp, "latest_predictions", is_mock=True)
    
    # 14. metrics_summary (27 rows: 3 experiments × 3 models × 3 scopes, unique PK)
    metrics_rows = []
    models = ["Ridge", "Random Forest", "XGBoost"]
    scopes = ["all", "bull", "bear"]
    experiments = ["A", "Aprime", "B"]
    counter = 0
    for exp in experiments:
        for m in models:
            for s in scopes:
                metrics_rows.append({
                    "experiment": exp,
                    "model": m,
                    "scope": s,
                    "n_dates": 62,
                    "n_obs": 620,
                    "msfe": round(0.016 + counter * 0.0002, 4),
                    "rmse": round(0.128 + counter * 0.0008, 4),
                    "mae": round(0.098 + counter * 0.0008, 4),
                    "oos_r2_zero": round(0.014 - (counter % 3) * 0.004, 3),
                    "oos_r2_expmean": round(0.011 - (counter % 3) * 0.003, 3),
                    "rank_ic_mean": round(0.085 + (counter % 5) * 0.01, 3),
                    "rank_ic_ir": round(0.45 + (counter % 5) * 0.05, 2),
                    "directional_accuracy": round(0.535 + (counter % 4) * 0.01, 3),
                    "topk_spread": round(0.012 + (counter % 4) * 0.002, 3),
                    "data_source": "SYNTHETIC_DEMO"
                })
                counter += 1
    df_metrics = pd.DataFrame(metrics_rows)
    save(df_metrics, "metrics_summary", is_mock=True)
    
    # 15. experiment_results (18 rows: 3 hypotheses × 3 models × 2 scopes, unique PK)
    exp_res_rows = []
    hypotheses = [("H1", "B vs Aprime (ESG Value Add)"), ("H2", "C vs B (Contextual ESG)"), ("H3", "P vs B (Placebo ESG)")]
    counter = 0
    for h_code, h_note in hypotheses:
        for m in models:
            for s in ["all", "bull"]:
                exp_res_rows.append({
                    "hypothesis": h_code,
                    "model": m,
                    "scope": s,
                    "delta_msfe": round(-0.0018 + (counter % 3) * 0.0005, 4),
                    "ci_low": -0.0052,
                    "ci_high": 0.0014,
                    "cw_stat": round(1.35 + (counter % 4) * 0.2, 2),
                    "cw_pvalue": round(0.088 - (counter % 4) * 0.015, 3),
                    "n_dates": 62,
                    "inferential_flag": True,
                    "note": h_note,
                    "data_source": "SYNTHETIC_DEMO"
                })
                counter += 1
    df_exp_res = pd.DataFrame(exp_res_rows)
    save(df_exp_res, "experiment_results", is_mock=True)
    
    # 16. placebo_distribution
    placebo_rows = []
    for i in range(20):
        placebo_rows.append({
            "model": "XGBoost",
            "permutation_id": i + 1,
            "delta_msfe_vs_aprime": round(float(rng.normal(0.0001, 0.0015)), 5),
            "data_source": "SYNTHETIC_DEMO"
        })
    df_placebo = pd.DataFrame(placebo_rows)
    save(df_placebo, "placebo_distribution", is_mock=True)
    
    # 17. feature_importance (32 rows: 2 experiments × 2 models × 8 features, unique PK)
    features = ["ret_21d", "mom_12_1", "vol_63d", "esg_total", "e_score", "roe", "rev_growth_yoy", "debt_to_equity"]
    fi_rows = []
    for exp in ["A", "B"]:
        for m in ["Ridge", "XGBoost"]:
            for rk, feat in enumerate(features, start=1):
                fi_rows.append({
                    "experiment": exp,
                    "model": m,
                    "feature": feat,
                    "feature_group": "ESG" if "esg" in feat or "score" in feat else ("Technical" if "ret" in feat or "vol" in feat or "mom" in feat else "Fundamental"),
                    "importance_perm": round(0.0025 - rk * 0.0002, 5),
                    "importance_rank": rk,
                    "data_source": "SYNTHETIC_DEMO"
                })
    df_fi = pd.DataFrame(fi_rows)
    save(df_fi, "feature_importance", is_mock=True)
    
    # 18. local_contributions (20 rows: 10 companies × 2 features, unique PK)
    lc_rows = []
    for cid in companies:
        for feat in ["mom_12_1", "esg_total"]:
            lc_rows.append({
                "model": "XGBoost",
                "experiment": "B",
                "company_id": cid,
                "date": "2025-12-31",
                "feature": feat,
                "feature_value": round(float(rng.uniform(0.1, 80.0)), 2),
                "contribution": round(float(rng.normal(0.003, 0.006)), 5),
                "base_value": 0.0150,
                "data_source": "SYNTHETIC_DEMO"
            })
    df_lc = pd.DataFrame(lc_rows)
    save(df_lc, "local_contributions", is_mock=True)
    
    # 19. portfolio_default
    port_rows = []
    for i, d in enumerate(dates):
        port_rows.append({
            "date": d,
            "strategy": "SIRPS_Top3_ESG",
            "portfolio_value": round(100000.0 * (1.0 + 0.001 * i + rng.normal(0, 0.004)), 2),
            "daily_return": round(float(rng.normal(0.0008, 0.009)), 4),
            "cumulative_return": round(0.001 * i, 4),
            "drawdown": round(-abs(float(rng.normal(0.02, 0.015))), 4),
            "turnover": 0.15 if i % 5 == 0 else 0.0,
            "data_source": "SYNTHETIC_DEMO"
        })
    df_port = pd.DataFrame(port_rows)
    save(df_port, "portfolio_default", is_mock=True)
    
    print(f"✓ Successfully generated mock sample datasets for all 20 schemas in {MOCK_DIR}")

if __name__ == "__main__":
    generate_mock_datasets()
