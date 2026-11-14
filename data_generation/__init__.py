"""Deterministic mock-data generation for development and demonstrations."""

from .config import GenerationConfig
from .mock_data import generate_mock_data

__all__ = ["GenerationConfig", "generate_mock_data"]
