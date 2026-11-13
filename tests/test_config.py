import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_settings_yaml():
    path = ROOT / "config" / "settings.yaml"

    with open(path, encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    assert settings["project"]["name"] == "SIRPS"
    assert settings["project"]["python_version"] == "3.12"
    assert settings["market"]["exchange"] == "NSE"
    assert settings["market"]["forecast_horizon"] == 63


def test_esg_indicator_count():
    indicators = read_csv(ROOT / "config" / "esg_indicators.csv")

    assert len(indicators) == 8


def test_esg_indicator_ids():
    indicators = read_csv(ROOT / "config" / "esg_indicators.csv")

    expected = {f"ESG{i:02d}" for i in range(1, 9)}
    actual = {row["indicator_id"] for row in indicators}

    assert actual == expected


def test_esg_required_columns():
    indicators = read_csv(ROOT / "config" / "esg_indicators.csv")

    required = {
        "indicator_id",
        "indicator_name",
        "esg_pillar",
        "definition",
        "calculation_method",
        "standard_unit",
        "preferred_source",
        "directionality",
        "validation_rule",
    }

    assert required.issubset(indicators[0].keys())


def test_esg_directionality():
    indicators = read_csv(ROOT / "config" / "esg_indicators.csv")

    allowed = {"positive", "negative"}

    for row in indicators:
        assert row["directionality"] in allowed


def test_universe_schema():
    path = ROOT / "config" / "universe.csv"

    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])

    required = {
        "company_id",
        "company_name",
        "isin",
        "nse_symbol",
        "sector",
        "universe_group",
        "eligibility_status",
        "evidence_reference",
    }

    assert required.issubset(columns)


def test_universe_allows_initial_empty_template():
    records = read_csv(ROOT / "config" / "universe.csv")

    # Empty company records are intentional during initialization.
    assert isinstance(records, list)
