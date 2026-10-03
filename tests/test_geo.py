"""Smoke tests for the geographic conversion package."""

from src.geo import coordinate_converter


def test_coordinate_converter_module_imports() -> None:
    """Ensure the coordinate converter module is importable."""
    assert coordinate_converter is not None
