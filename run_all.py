"""Run the complete deterministic development pipeline."""

from data_generation.config import GenerationConfig
from data_generation.mock_data import generate_mock_data

from sirps.pilot import run as run_pilot


def main() -> None:
    paths = generate_mock_data("data/mock", GenerationConfig())
    serving_path = run_pilot("data/serving")
    print(f"Generated {len(paths)} mock tables")
    print(f"Generated serving output: {serving_path}")


if __name__ == "__main__":
    main()
