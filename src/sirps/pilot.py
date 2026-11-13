"""Run the deterministic two-company initialization pilot."""

from __future__ import annotations

from pathlib import Path

from sirps.common.synthetic import make_pilot_prices
from sirps.features.pilot import build_pilot_dataset
from sirps.modeling.pilot import make_pilot_predictions


def run(output_dir: str | Path = "data/serving") -> Path:
    """Generate synthetic pilot serving data and return its prediction path."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prices = make_pilot_prices(periods=180)
    dataset = build_pilot_dataset(prices)
    predictions = make_pilot_predictions(dataset)
    dataset.to_parquet(output / "pilot_model_dataset.parquet", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    return output / "predictions.csv"


if __name__ == "__main__":
    print(run())
