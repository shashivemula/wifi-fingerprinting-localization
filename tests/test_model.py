"""Smoke tests for the WKNN model package."""

from src.model import wknn_localizer


def test_wknn_module_imports() -> None:
    """Ensure the WKNN localizer module is importable."""
    assert wknn_localizer is not None
