"""Candidate scoring, sorting, and ranking output module."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml.features import FEATURE_COLUMNS
from ml.train_model import DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> RandomForestRegressor:
    """Load a saved candidate ranking model.

    Args:
        model_path: Path to a Joblib model artifact.

    Returns:
        Trained RandomForestRegressor.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate ranking model not found: {path}")
    logger.info("Loading candidate ranking model from %s", path)
    return joblib.load(path)


def predict_candidate_scores(
    model: RandomForestRegressor,
    feature_vectors: pd.DataFrame,
) -> pd.DataFrame:
    """Predict candidate suitability scores and confidence estimates.

    Args:
        model: Trained RandomForestRegressor.
        feature_vectors: DataFrame containing Step 3 feature columns.

    Returns:
        DataFrame with predicted scores and confidence estimates.
    """
    _validate_feature_vectors(feature_vectors)
    x = feature_vectors[FEATURE_COLUMNS]
    predictions = np.clip(model.predict(x), 0.0, 100.0)
    confidence = _estimate_confidence(model, x)
    return pd.DataFrame(
        {
            "Predicted Score": predictions,
            "Confidence Estimate": confidence,
        }
    )


def rank_candidates(
    candidate_features: pd.DataFrame,
    model: RandomForestRegressor | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Rank candidates by predicted suitability score.

    Args:
        candidate_features: DataFrame containing candidate metadata and features.
        model: Optional preloaded model.
        model_path: Model path used if no model is provided.

    Returns:
        Ranked DataFrame sorted by predicted score descending.
    """
    ranking_model = model if model is not None else load_model(model_path)
    predictions = predict_candidate_scores(ranking_model, candidate_features)

    ranked = candidate_features.reset_index(drop=True).copy()
    ranked["Predicted Score"] = predictions["Predicted Score"]
    ranked["Confidence Estimate"] = predictions["Confidence Estimate"]
    ranked = ranked.sort_values(
        by=["Predicted Score", "Confidence Estimate"],
        ascending=[False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    return ranked


def _estimate_confidence(
    model: RandomForestRegressor,
    feature_vectors: pd.DataFrame,
) -> np.ndarray:
    if not hasattr(model, "estimators_"):
        return np.full(len(feature_vectors), 75.0)

    feature_array = feature_vectors.to_numpy()
    tree_predictions = np.vstack(
        [estimator.predict(feature_array) for estimator in model.estimators_]
    )
    prediction_std = tree_predictions.std(axis=0)
    return np.clip(100.0 - prediction_std * 3.0, 0.0, 100.0)


def _validate_feature_vectors(feature_vectors: pd.DataFrame) -> None:
    missing_columns = [column for column in FEATURE_COLUMNS if column not in feature_vectors]
    if missing_columns:
        raise ValueError(f"Feature vectors missing columns: {missing_columns}")
