import csv
from pathlib import Path

import pandas as pd
import pytest
import yaml

from sirps.common.calendar import TradingCalendar

ROOT = Path(__file__).resolve().parents[1]


def test_trading_calendar_uses_explicit_sessions():
    calendar = TradingCalendar(pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-05"]))

    assert calendar.add("2024-01-02", 1) == pd.Timestamp("2024-01-04")
    assert calendar.add("2024-01-05", 0) == pd.Timestamp("2024-01-05")
    assert not calendar.contains("2024-01-03")
    with pytest.raises(ValueError, match="not in the trading calendar"):
        calendar.add("2024-01-03", 1)


def test_universe_and_pilot_configuration_are_loadable():
    with (ROOT / "config" / "universe_eligibility.yaml").open(encoding="utf-8") as file:
        universe = yaml.safe_load(file)
    with (ROOT / "config" / "pilot.yaml").open(encoding="utf-8") as file:
        pilot = yaml.safe_load(file)

    assert universe["status"] == "draft_for_review"
    assert universe["eligibility"]["independent_reviewers_required"] == 2
    assert pilot["is_real_market_data"] is False
    assert pilot["companies"] == ["SYN001", "SYN002"]


def test_source_register_and_run_registry_have_contract_headers():
    for relative_path, required in {
        "config/source_register.csv": {"source_id", "domain", "verification_status"},
        "results/runs.csv": {"run_id", "git_commit", "config_hash", "experiment", "model"},
    }.items():
        with (ROOT / relative_path).open(encoding="utf-8", newline="") as file:
            fields = set(csv.DictReader(file).fieldnames or [])
        assert required <= fields
