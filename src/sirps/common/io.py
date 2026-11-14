"""Shared, schema-aware file I/O for the SIRPS pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .schemas import SCHEMA_VERSION, validate_schema


def project_root() -> Path:
    """Return the repository root for an installed or editable checkout."""
    return Path(__file__).resolve().parents[3]


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return value


def read_table(path: str | Path, *, schema: str | None = None) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError(f"Unsupported table format: {source.suffix}")
    if schema is not None:
        validate_schema(frame, schema, metadata={"schema_version": SCHEMA_VERSION})
    return frame


def write_table(frame: pd.DataFrame, path: str | Path, *, schema: str | None = None) -> Path:
    destination = Path(path)
    if schema is not None:
        validate_schema(frame, schema, metadata={"schema_version": SCHEMA_VERSION})
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".parquet":
        frame.to_parquet(destination, index=False)
    elif destination.suffix.lower() == ".csv":
        frame.to_csv(destination, index=False)
    else:
        raise ValueError(f"Unsupported table format: {destination.suffix}")
    return destination
