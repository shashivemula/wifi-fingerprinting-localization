"""Smoke tests for the prediction package."""

from src.prediction import predictor


def test_predictor_module_imports() -> None:
    """Ensure the predictor module is importable."""
    assert predictor is not None
