from pathlib import Path

import pandas as pd
from data_generation.config import GenerationConfig
from data_generation.mock_data import generate_mock_data

from sirps.common.io import read_table
from sirps.common.schemas import SCHEMA_VERSION
from sirps.latent import generate_latent_profiles


def test_pdf_required_mock_dataset_set(tmp_path: Path):
    paths = generate_mock_data(tmp_path, GenerationConfig(periods=180))
    expected = {
        "company_master",
        "prices_daily",
        "fundamentals_pit",
        "esg_facts",
        "regime_daily",
        "model_dataset",
        "predictions",
        "latent_profiles",
    }

    assert set(paths) == expected
    for schema, path in paths.items():
        frame = read_table(path, schema=schema)
        assert not frame.empty


def test_latent_profiles_are_reproducible():
    first = generate_latent_profiles(seed=2026)
    second = generate_latent_profiles(seed=2026)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["source_label"]) == {"synthetic_latent_profile"}


def test_generation_config_matches_project_contract():
    config = GenerationConfig.from_project(Path(__file__).parents[1])

    assert config.seed == 2026
    assert config.forecast_horizon == 63
    assert config.sample_step == 5
    assert SCHEMA_VERSION == "1.0"
