import csv
from pathlib import Path

import pandas as pd

from sirps.common.schemas import SCHEMA_VERSION, validate_schema

ROOT = Path(__file__).resolve().parents[1]


def test_company_master_pilot_is_present_and_schema_compliant():
    path = ROOT / "data" / "processed" / "company_master.csv"
    frame = pd.read_csv(path, dtype={"company_id": str, "isin": str, "nse_symbol": str})

    validate_schema(frame, "company_master", metadata={"schema_version": SCHEMA_VERSION})
    assert len(frame) == 2
    assert set(frame["company_id"]) == {"SYN001", "SYN002"}
    assert set(frame["universe_group"]) == {"core"}


def test_company_master_has_no_duplicate_ids():
    path = ROOT / "data" / "processed" / "company_master.csv"
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    ids = [row["company_id"] for row in rows]
    assert len(ids) == len(set(ids))
