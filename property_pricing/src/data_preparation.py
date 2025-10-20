"""Data preparation module for property price classification."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


LOGGER = logging.getLogger(__name__)

BASE_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_PATH / "data" / "property_prices.csv"
TRAIN_DATA_PATH = BASE_PATH / "data" / "train.pkl"
MODELS_PATH = BASE_PATH / "models"


COLUMN_MAPPING = {
    "price_aprox_usd": "price",
    "surface_covered_in_m2": "surface_covered",
    "surface_total_in_m2": "surface_total",
    "barrio": "neighborhood",
}

COLUMNS_TO_KEEP = [
    "price",
    "surface_covered",
    "surface_total",
    "floor",
    "rooms",
    "expenses",
    "neighborhood",
    "property_type",
]

CATEGORICAL_COLUMNS = ["neighborhood", "property_type"]
NUMERIC_COLUMNS = [
    "price",
    "surface_covered",
    "surface_total",
    "floor",
    "rooms",
    "expenses",
    "price_per_m2",
    "room_density",
    "expenses_ratio",
]


class DataPreparationError(Exception):
    """Custom exception for data preparation errors."""


def _ensure_directories() -> None:
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Ensured data and model directories exist.")


def _load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        LOGGER.error("❌ Error: No se encuentra property_prices.csv en la carpeta data/")
        raise FileNotFoundError(
            "❌ Error: No se encuentra property_prices.csv en la carpeta data/"
        )
    LOGGER.info("Loading raw dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    LOGGER.debug("Raw dataset shape: %%s", df.shape)
    return df


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: new for col, new in COLUMN_MAPPING.items() if col in df.columns}
    LOGGER.debug("Applying column rename mapping: %s", rename_map)
    df = df.rename(columns=rename_map)
    return df


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in COLUMNS_TO_KEEP if col not in df.columns]
    if missing:
        LOGGER.error("Dataset is missing required columns: %s", missing)
        raise DataPreparationError(
            f"❌ Error: Columnas requeridas ausentes en el dataset: {', '.join(missing)}"
        )
    selection = df[COLUMNS_TO_KEEP].copy()
    LOGGER.debug("Selected required columns. Shape: %%s", selection.shape)
    return selection


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(df)
    df = df.dropna()
    df = df[(df["surface_covered"] > 0) & (df["price"] > 0)]
    price_threshold = df["price"].quantile(0.99)
    df = df[df["price"] <= price_threshold]
    LOGGER.info(
        "Cleaned dataset from %s to %s rows (price outlier threshold: %.2f)",
        initial_rows,
        len(df),
        price_threshold,
    )
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["price_per_m2"] = df["price"] / df["surface_covered"]
    df["room_density"] = np.where(
        df["surface_total"] > 0,
        df["rooms"] / df["surface_total"],
        0.0,
    )
    df["expenses_ratio"] = df["expenses"] / df["price"]
    LOGGER.debug("Engineered features added: price_per_m2, room_density, expenses_ratio")
    return df


def _encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encoders: Dict[str, LabelEncoder] = {}
    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(df[column].astype(str))
        if "Unknown" not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_, "Unknown")
        df[column] = encoded
        encoders[column] = encoder
        LOGGER.debug("Encoded categorical column '%s' with %s classes", column, len(encoder.classes_))
    joblib.dump(encoders, MODELS_PATH / "encoders.pkl")
    LOGGER.info("Label encoders saved to %s", MODELS_PATH / "encoders.pkl")
    return df, encoders


def _create_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
    p33 = df["price_per_m2"].quantile(0.33)
    p66 = df["price_per_m2"].quantile(0.66)

    def categorize(value: float) -> str:
        if value <= p33:
            return "BARATO"
        if value <= p66:
            return "REGULAR"
        return "CARO"

    target = df["price_per_m2"].apply(categorize)
    LOGGER.info("Target distribution: %s", target.value_counts(normalize=True).to_dict())
    return df, target, {"p33": float(p33), "p66": float(p66)}


def prepare_data(
    csv_path: Path | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, LabelEncoder], Dict[str, float]]:
    """Prepare dataset for training.

    Returns the processed feature matrix, target vector, feature columns,
    fitted label encoders, and percentile thresholds used for categorisation.
    """

    _ensure_directories()
    csv_path = csv_path or DATA_PATH
    df = _load_dataset(csv_path)
    df = _rename_columns(df)
    df = _select_columns(df)
    df = _clean_data(df)
    df = _engineer_features(df)
    df, encoders = _encode_categoricals(df)
    df, target, percentiles = _create_target(df)

    feature_columns = COLUMNS_TO_KEEP + ["price_per_m2", "room_density", "expenses_ratio"]
    features = df[feature_columns].copy()

    payload = {
        "X": features,
        "y": target,
        "feature_columns": feature_columns,
        "percentiles": percentiles,
    }
    joblib.dump(payload, TRAIN_DATA_PATH)
    LOGGER.info(
        "Prepared dataset saved to %s with %s rows and %s features",
        TRAIN_DATA_PATH,
        len(features),
        len(feature_columns),
    )

    return features, target, feature_columns, encoders, percentiles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_data()
