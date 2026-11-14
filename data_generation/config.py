"""Configuration contract for the M1 development-data generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 2026
    company_ids: tuple[str, ...] = ("SYN001", "SYN002")
    periods: int = 180
    forecast_horizon: int = 63
    sample_step: int = 5
    source_label: str = "synthetic_mock"

    @classmethod
    def from_project(cls, root: str | Path) -> GenerationConfig:
        """Load the stable generator settings without coupling to model code."""
        import yaml

        path = Path(root) / "config" / "pilot.yaml"
        with path.open(encoding="utf-8") as file:
            settings = yaml.safe_load(file)
        return cls(
            seed=int(settings["seed"]),
            company_ids=tuple(settings["companies"]),
            source_label=str(settings["source_label"]),
        )
