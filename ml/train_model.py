"""Random Forest Regressor training and model persistence module."""

from __future__ import annotations

import logging
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from ml.features import FEATURE_COLUMNS
from ml.synthetic_data import (
    TARGET_COLUMN,
    load_or_generate_synthetic_dataset,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/model.pkl")
DEFAULT_FEATURE_IMPORTANCE_PATH = Path("output/feature_importance.png")
RANDOM_STATE = 42


@dataclass(slots=True)
class TrainingMetrics:
    """Evaluation metrics for the trained candidate ranking model."""

    mae: float
    rmse: float
    r2_score: float
    cross_validation_rmse_mean: float
    cross_validation_rmse_std: float


def train_candidate_ranking_model(
    dataset: pd.DataFrame | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    feature_importance_path: str | Path = DEFAULT_FEATURE_IMPORTANCE_PATH,
    random_state: int = RANDOM_STATE,
) -> tuple[RandomForestRegressor, TrainingMetrics]:
    """Train and persist a RandomForestRegressor for candidate ranking.

    Args:
        dataset: Optional synthetic dataset. If omitted, data is loaded or generated.
        model_path: Destination path for the Joblib model artifact.
        feature_importance_path: Destination PNG path for feature importances.
        random_state: Seed used for splitting and model training.

    Returns:
        Trained model and evaluation metrics.
    """
    training_data = dataset if dataset is not None else load_or_generate_synthetic_dataset()
    _validate_training_dataset(training_data)

    x = training_data[FEATURE_COLUMNS]
    y = training_data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=random_state,
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    logger.info("Training RandomForestRegressor on %d samples.", len(x_train))
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    metrics = TrainingMetrics(
        mae=float(mean_absolute_error(y_test, predictions)),
        rmse=rmse,
        r2_score=float(r2_score(y_test, predictions)),
        cross_validation_rmse_mean=0.0,
        cross_validation_rmse_std=0.0,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        model,
        x,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    metrics = TrainingMetrics(
        mae=metrics.mae,
        rmse=metrics.rmse,
        r2_score=metrics.r2_score,
        cross_validation_rmse_mean=float(-cv_scores.mean()),
        cross_validation_rmse_std=float(cv_scores.std()),
    )

    save_model(model, model_path)
    save_feature_importance_plot(
        model.feature_importances_,
        FEATURE_COLUMNS,
        feature_importance_path,
    )
    print_evaluation_metrics(metrics)
    return model, metrics


def save_model(model: RandomForestRegressor, model_path: str | Path) -> Path:
    """Save a trained model with Joblib."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved trained model to %s", path)
    return path


def load_training_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> RandomForestRegressor:
    """Load a trained model artifact."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")
    return joblib.load(path)


def print_evaluation_metrics(metrics: TrainingMetrics) -> None:
    """Print model evaluation metrics clearly."""
    print("\nModel Evaluation Metrics")
    print(f"MAE: {metrics.mae:.4f}")
    print(f"RMSE: {metrics.rmse:.4f}")
    print(f"R2 Score: {metrics.r2_score:.4f}")
    print(
        "5-Fold CV RMSE: "
        f"{metrics.cross_validation_rmse_mean:.4f} "
        f"+/- {metrics.cross_validation_rmse_std:.4f}"
    )


def save_feature_importance_plot(
    importances: np.ndarray,
    feature_names: list[str],
    output_path: str | Path = DEFAULT_FEATURE_IMPORTANCE_PATH,
) -> Path:
    """Save a simple PNG bar chart of model feature importances.

    This project intentionally avoids adding a plotting dependency. The chart is
    rendered into an RGB image buffer and written as a valid PNG with the Python
    standard library.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1000, 620
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_feature_importance_chart(image, importances, feature_names)
    _write_png(path, image)
    logger.info("Saved feature importance plot to %s", path)
    return path


def _validate_training_dataset(dataset: pd.DataFrame) -> None:
    missing_columns = [
        column for column in [*FEATURE_COLUMNS, TARGET_COLUMN] if column not in dataset
    ]
    if missing_columns:
        raise ValueError(f"Training dataset missing columns: {missing_columns}")


def _draw_feature_importance_chart(
    image: np.ndarray,
    importances: np.ndarray,
    feature_names: list[str],
) -> None:
    height, width, _ = image.shape
    left, right, top, bottom = 330, 50, 40, 60
    chart_width = width - left - right
    chart_height = height - top - bottom
    order = np.argsort(importances)[::-1]
    max_importance = float(importances[order[0]]) if len(importances) else 1.0
    max_importance = max(max_importance, 1e-9)
    bar_gap = 8
    bar_height = max(12, (chart_height - bar_gap * len(order)) // max(len(order), 1))

    _draw_text(image, 30, 16, "Feature Importance")
    _draw_line(image, left, top, left, height - bottom, color=(40, 40, 40))
    _draw_line(
        image,
        left,
        height - bottom,
        width - right,
        height - bottom,
        color=(40, 40, 40),
    )

    for index, feature_index in enumerate(order):
        y = top + index * (bar_height + bar_gap)
        importance = float(importances[feature_index])
        bar_width = int((importance / max_importance) * chart_width)
        _draw_text(image, 20, y + 3, feature_names[feature_index][:38])
        _fill_rect(
            image,
            left,
            y,
            left + bar_width,
            y + bar_height,
            color=(48, 111, 173),
        )
        _draw_text(image, left + bar_width + 8, y + 3, f"{importance:.3f}")


def _fill_rect(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    height, width, _ = image.shape
    image[max(y0, 0) : min(y1, height), max(x0, 0) : min(x1, width)] = color


def _draw_line(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    if x0 == x1:
        _fill_rect(image, x0, min(y0, y1), x0 + 2, max(y0, y1) + 1, color)
    elif y0 == y1:
        _fill_rect(image, min(x0, x1), y0, max(x0, x1) + 1, y0 + 2, color)


def _draw_text(image: np.ndarray, x: int, y: int, text: str) -> None:
    cursor_x = x
    for character in text:
        pattern = _FONT.get(character.upper(), _FONT.get(" ", []))
        for row_index, row in enumerate(pattern):
            for col_index, pixel in enumerate(row):
                if pixel == "1":
                    _fill_rect(
                        image,
                        cursor_x + col_index * 2,
                        y + row_index * 2,
                        cursor_x + col_index * 2 + 2,
                        y + row_index * 2 + 2,
                        color=(30, 30, 30),
                    )
        cursor_x += 12


def _write_png(path: Path, image: np.ndarray) -> None:
    height, width, _ = image.shape
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw_rows, level=6)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


_FONT = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11110", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    train_candidate_ranking_model()
