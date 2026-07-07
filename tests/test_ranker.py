"""Tests for model loading, prediction, and ranking."""

from __future__ import annotations

from pathlib import Path

import joblib

from ml.features import FEATURE_COLUMNS
from ml.ranker import load_model, predict_candidate_scores, rank_candidates
from ml.synthetic_data import TARGET_COLUMN, generate_synthetic_dataset
from ml.train_model import train_candidate_ranking_model


def test_prediction_ranking_and_model_loading(tmp_path: Path) -> None:
    """Load a saved model, predict candidate scores, and sort ranks descending."""
    dataset = generate_synthetic_dataset(n_samples=120, random_state=11)
    model_path = tmp_path / "model.pkl"
    plot_path = tmp_path / "feature_importance.png"
    model, _ = train_candidate_ranking_model(
        dataset=dataset,
        model_path=model_path,
        feature_importance_path=plot_path,
        random_state=42,
    )
    loaded_model = load_model(model_path)

    candidate_features = dataset.head(5).drop(columns=[TARGET_COLUMN]).copy()
    candidate_features.insert(
        0,
        "Candidate Name",
        [f"Candidate {index}" for index in range(1, 6)],
    )
    predictions = predict_candidate_scores(loaded_model, candidate_features)
    ranked = rank_candidates(candidate_features, model=model)

    assert predictions["Predicted Score"].between(0.0, 100.0).all()
    assert predictions["Confidence Estimate"].between(0.0, 100.0).all()
    assert ranked["Rank"].tolist() == [1, 2, 3, 4, 5]
    assert ranked["Predicted Score"].is_monotonic_decreasing
    assert set(FEATURE_COLUMNS).issubset(ranked.columns)
    assert joblib.load(model_path) is not None

