"""SIRPS Common I/O Utilities and Provenance Validation.

This module provides standard loading and saving routines across all modules,
verifying column completeness and mandatory synthetic provenance disclaimers.
It safely integrates with Streamlit caching when available while supporting
unattended CLI and pytest execution.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from common.schemas import FILES, SCHEMAS, validate_dataset_schema

# Project root directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Check if Streamlit runtime is active
try:
    import streamlit as st
    import streamlit.runtime as rt
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

def _has_active_streamlit_runtime() -> bool:
    return HAS_STREAMLIT and rt.exists()

def get_file_path(name: str, use_mock: bool = False) -> Path:
    """Resolves the absolute path for a logical dataset name."""
    if name not in FILES:
        raise KeyError(f"Unknown dataset logical name: '{name}'. Available: {list(FILES.keys())}")
    
    rel_path = FILES[name]
    if use_mock:
        # Map data/foo.csv to data/mock/foo.csv or results/foo.csv to data/mock/foo.csv
        parts = rel_path.split("/")
        if len(parts) >= 2:
            rel_path = f"data/mock/{parts[1]}"
            
    return PROJECT_ROOT / rel_path

def _read_and_validate_csv(file_path: Path, schema_name: str) -> pd.DataFrame:
    """Reads a CSV from disk and performs strict schema and integrity validation."""
    if not file_path.exists():
        raise FileNotFoundError(f"Required dataset file not found at: {file_path}")
    
    schema = SCHEMAS.get(schema_name)
    date_cols = schema.date_columns if schema else []
    
    df = pd.read_csv(file_path)
    
    # Run comprehensive schema, primary key, nullability, and provenance validator
    if schema is not None:
        validate_dataset_schema(df, schema_name)
        
    # Parse date columns if present
    for d_col in date_cols:
        if d_col in df.columns:
            df[d_col] = pd.to_datetime(df[d_col], errors="coerce")
            
    return df

def load_data_uncached(name: str, use_mock: bool = False) -> pd.DataFrame:
    """Loads and validates a dataset without Streamlit caching.
    
    Strictly avoids silent fallbacks to mock data in production mode (use_mock=False).
    """
    file_path = get_file_path(name, use_mock=use_mock)
    
    if not file_path.exists():
        if not use_mock:
            raise FileNotFoundError(
                f"Production dataset '{name}' does not exist at '{file_path}'. "
                f"Silent fallback to mock data is disabled in production mode. "
                f"If you are developing or testing without full generated data, pass use_mock=True explicitly."
            )
        else:
            raise FileNotFoundError(f"Mock dataset '{name}' not found at '{file_path}'.")
            
    return _read_and_validate_csv(file_path, name)

_cached_load = None

def load(name: str, use_mock: bool = False) -> pd.DataFrame:
    """Loads a dataset, using Streamlit cache if in Streamlit runtime, or uncached in CLI/pytest."""
    global _cached_load
    if _has_active_streamlit_runtime():
        if _cached_load is None:
            _cached_load = st.cache_data(show_spinner=False)(load_data_uncached)
        return _cached_load(name, use_mock=use_mock)
    return load_data_uncached(name, use_mock=use_mock)

def clear_cache() -> None:
    """Clears Streamlit cache if runtime is active."""
    global _cached_load
    if _has_active_streamlit_runtime() and _cached_load is not None:
        _cached_load.clear()

def save(df: pd.DataFrame, name: str, is_mock: bool = False) -> Path:
    """Validates schema and provenance and atomically writes dataframe to its designated destination."""
    file_path = get_file_path(name, use_mock=is_mock)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure data_source is present
    if "data_source" not in df.columns:
        df = df.copy()
        df["data_source"] = "SYNTHETIC_DEMO"
        
    # Validate full schema before writing
    if name in SCHEMAS:
        validate_dataset_schema(df, name)
            
    # Atomic write pattern: write to hidden .tmp file, then atomically replace
    tmp_path = file_path.with_name(f".{file_path.name}.tmp")
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, file_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
                
    clear_cache()
    return file_path


# Standard interface aliases
load_dataset = load
save_dataset = save
DATA_DIR: Path = PROJECT_ROOT / "data"
MOCK_DIR: Path = PROJECT_ROOT / "data/mock"

def validate_referential_integrity(
    child_df: pd.DataFrame,
    child_schema_name: str,
    *args,
    **kwargs
) -> None:
    """Validates that foreign keys in child_df exist in parent_df.
    
    Supports both schema-driven lookup:
        validate_referential_integrity(child_df, child_schema, parent_df, parent_schema)
    and explicit column arguments:
        validate_referential_integrity(child_df, child_name, child_fk, parent_name, parent_pk, parent_df)
    """
    parent_df = kwargs.get("parent_df")
    parent_schema_name = kwargs.get("parent_schema_name")
    child_fk_col = kwargs.get("child_fk_col")
    parent_pk_col = kwargs.get("parent_pk_col")

    if args:
        if isinstance(args[0], pd.DataFrame):
            parent_df = args[0]
            if len(args) > 1 and isinstance(args[1], str):
                parent_schema_name = args[1]
        elif isinstance(args[0], (str, list)):
            child_fk_col = args[0] if isinstance(args[0], str) else args[0][0]
            if len(args) > 1 and isinstance(args[1], str):
                parent_schema_name = args[1]
            if len(args) > 2 and isinstance(args[2], (str, list)):
                parent_pk_col = args[2] if isinstance(args[2], str) else args[2][0]
            if len(args) > 3 and isinstance(args[3], pd.DataFrame):
                parent_df = args[3]

    if child_fk_col and parent_pk_col and parent_df is not None:
        if child_fk_col in child_df.columns and parent_pk_col in parent_df.columns:
            child_keys = set(child_df[child_fk_col].dropna().unique())
            parent_keys = set(parent_df[parent_pk_col].dropna().unique())
            orphans = child_keys - parent_keys
            if orphans:
                raise ValueError(
                    f"Referential integrity violation: {child_schema_name}.{child_fk_col} contains "
                    f"foreign keys not found in {parent_schema_name}.{parent_pk_col}: {orphans}"
                )
        return

    schema = SCHEMAS.get(child_schema_name)
    if not schema or not schema.foreign_keys or parent_df is None or not parent_schema_name:
        return
        
    for fk_col, target in schema.foreign_keys.items():
        target_schema, target_col = target.split(".")
        if target_schema == parent_schema_name:
            if fk_col in child_df.columns and target_col in parent_df.columns:
                child_keys = set(child_df[fk_col].dropna().unique())
                parent_keys = set(parent_df[target_col].dropna().unique())
                orphans = child_keys - parent_keys
                if orphans:
                    raise ValueError(
                        f"Referential integrity failure: {child_schema_name}.{fk_col} contains "
                        f"foreign keys not found in {parent_schema_name}.{target_col}: {orphans}"
                    )


