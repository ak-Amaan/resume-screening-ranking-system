"""Tests for synthetic candidate-ranking data generation."""

from __future__ import annotations

from ml.features import FEATURE_COLUMNS
from ml.synthetic_data import TARGET_COLUMN, generate_synthetic_dataset


def test_generate_synthetic_dataset_has_required_columns_and_score_range() -> None:
    """Generate synthetic features and bounded target labels."""
    dataset = generate_synthetic_dataset(n_samples=100, random_state=42)

    assert dataset.shape == (100, len(FEATURE_COLUMNS) + 1)
    assert list(dataset.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]
    assert dataset[TARGET_COLUMN].between(0.0, 100.0).all()
    assert dataset[FEATURE_COLUMNS].isna().sum().sum() == 0

