"""Data loading and synthetic generation."""

from __future__ import annotations

from .generator import generate_load_series, load_series_from_csv

__all__ = ["generate_load_series", "load_series_from_csv"]
