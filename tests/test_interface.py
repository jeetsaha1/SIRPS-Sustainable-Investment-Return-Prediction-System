"""SIRPS Interface Contract Test Suite (Member 1).

Verifies the interface contracts between the main application, common modules,
and individual feature modules:
1. Standard UI render() entrypoints
2. App page routing table integrity
3. Common UI helper exports
4. Common I/O loader and saver exports
5. Central schema definitions
"""

import sys
import importlib
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_home_module_interface():
    """Verifies that modules.home can be imported and exposes a callable render()."""
    import modules.home as home
    assert hasattr(home, "render"), "modules.home must have a 'render' attribute"
    assert callable(home.render), "'render' in modules.home must be callable"

def test_common_ui_interface():
    """Verifies standard common.ui function signatures and exports."""
    import common.ui as ui
    assert hasattr(ui, "demo_banner") and callable(ui.demo_banner)
    assert hasattr(ui, "init_state") and callable(ui.init_state)
    assert hasattr(ui, "company_selector") and callable(ui.company_selector)
    assert hasattr(ui, "render_kpi_card") and callable(ui.render_kpi_card)

def test_common_io_interface():
    """Verifies standard common.io function signatures and exports."""
    import common.io as io
    assert hasattr(io, "load") and callable(io.load)
    assert hasattr(io, "save") and callable(io.save)
    assert hasattr(io, "get_file_path") and callable(io.get_file_path)

def test_schemas_catalog_interface():
    """Verifies that schemas catalog includes all expected prototype datasets."""
    from common.schemas import SCHEMAS, FILES
    
    expected_datasets = [
        "company_master", "latent_profiles", "trading_calendar", "index_daily",
        "prices_daily", "corporate_actions", "esg_facts", "esg_scores",
        "fundamentals_quarterly", "regime_daily", "model_dataset", "latest_features",
        "fold_definitions", "predictions", "latest_predictions", "metrics_summary",
        "experiment_results", "placebo_distribution", "feature_importance",
        "local_contributions", "portfolio_default"
    ]
    
    for ds in expected_datasets:
        assert ds in FILES, f"Dataset '{ds}' missing from FILES dictionary"
        assert ds in SCHEMAS, f"Dataset '{ds}' missing from SCHEMAS dictionary"

def test_app_router_contract():
    """Verifies that app.py defines the standard 10-page routing table."""
    import app
    assert hasattr(app, "PAGES"), "app.py must define 'PAGES'"
    pages = app.PAGES
    assert len(pages) == 10, f"Expected 10 prototype pages, found {len(pages)}"
    assert "1 Home" in pages
    assert pages["1 Home"] == "modules.home"
