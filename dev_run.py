"""Original development-run entry point retained for M1 compatibility."""

from data_generation.config import GenerationConfig
from data_generation.mock_data import generate_mock_data


def main() -> None:
    paths = generate_mock_data("data/mock", GenerationConfig())
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
