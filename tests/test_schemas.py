import pandas as pd
import pytest

from sirps.common.schemas import SCHEMA_VERSION, validate_schema
from sirps.common.synthetic import make_pilot_prices


def test_pilot_is_deterministic_and_schema_compliant():
    first = make_pilot_prices()
    second = make_pilot_prices()

    pd.testing.assert_frame_equal(first, second)
    validate_schema(first, "prices_daily", metadata={"schema_version": SCHEMA_VERSION})
    assert set(first["company_id"]) == {"SYN001", "SYN002", "NIFTY50_TRI_SYNTHETIC"}


def test_schema_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(pd.DataFrame({"company_id": ["SYN001"]}), "prices_daily")


def test_schema_rejects_invalid_controlled_values():
    frame = pd.DataFrame(
        {
            "company_id": ["SYN001"],
            "name": ["Synthetic One"],
            "isin": ["SYNTHETIC"],
            "nse_symbol": ["SYN1"],
            "sector": ["Synthetic"],
            "sub_sector": ["Synthetic"],
            "universe_group": ["invalid"],
            "eligibility_evidence": ["synthetic pilot"],
            "entry_date": [None],
            "exit_date": [None],
        }
    )
    with pytest.raises(ValueError, match="Invalid universe_group"):
        validate_schema(frame, "company_master")
