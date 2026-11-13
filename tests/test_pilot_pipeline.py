from pathlib import Path

from sirps.common.schemas import SCHEMA_VERSION, validate_schema
from sirps.common.synthetic import make_pilot_prices
from sirps.features.pilot import build_pilot_dataset
from sirps.modeling.pilot import make_pilot_predictions
from sirps.pilot import run


def test_pilot_flows_from_prices_to_predictions(tmp_path: Path):
    dataset = build_pilot_dataset(make_pilot_prices(periods=180))
    predictions = make_pilot_predictions(dataset)

    validate_schema(dataset, "model_dataset", metadata={"schema_version": SCHEMA_VERSION})
    validate_schema(predictions, "predictions", metadata={"schema_version": SCHEMA_VERSION})
    assert not dataset.empty
    assert set(predictions["model"]) == {"Ridge"}

    output_path = run(tmp_path)
    assert output_path.exists()
    assert (tmp_path / "pilot_model_dataset.parquet").exists()
