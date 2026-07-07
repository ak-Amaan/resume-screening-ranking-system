"""Tests for Random Forest candidate ranking training."""

from __future__ import annotations

from pathlib import Path

from ml.synthetic_data import generate_synthetic_dataset
from ml.train_model import load_training_model, train_candidate_ranking_model


def test_train_candidate_ranking_model_saves_artifacts(tmp_path: Path) -> None:
    """Train a small model and save model plus feature importance artifacts."""
    dataset = generate_synthetic_dataset(n_samples=120, random_state=7)
    model_path = tmp_path / "model.pkl"
    plot_path = tmp_path / "feature_importance.png"

    model, metrics = train_candidate_ranking_model(
        dataset=dataset,
        model_path=model_path,
        feature_importance_path=plot_path,
        random_state=42,
    )
    loaded_model = load_training_model(model_path)

    assert model_path.exists()
    assert plot_path.exists()
    assert plot_path.read_bytes().startswith(b"\x89PNG")
    assert metrics.mae >= 0.0
    assert metrics.rmse >= 0.0
    assert hasattr(model, "predict")
    assert hasattr(loaded_model, "predict")

