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

STANDARDIZED_COLUMNS = {
    "price",
    "surface_covered",
    "surface_total",
    "floor",
    "rooms",
    "expenses",
    "neighborhood",
    "property_type",
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
    if STANDARDIZED_COLUMNS.issubset(df.columns):
        LOGGER.info("Detected dataset in standardized format. No renaming required.")
        return df

    rename_map = {col: new for col, new in COLUMN_MAPPING.items() if col in df.columns}
    if not rename_map:
        LOGGER.error(
            "Dataset columns %s do not match expected formats.", list(df.columns)
        )
        raise DataPreparationError(
            "❌ Error: El dataset no contiene las columnas requeridas para el formato "
            "original ni el actualizado."
        )

    LOGGER.debug("Applying column rename mapping: %s", rename_map)
    df = df.rename(columns=rename_map)
    return df


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "price",
        "surface_covered",
        "surface_total",
        "floor",
        "rooms",
        "expenses",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in COLUMNS_TO_KEEP if col not in df.columns]
    if missing:
        LOGGER.error("Dataset is missing required columns: %s", missing)
        raise DataPreparationError(
            f"❌ Error: Columnas requeridas ausentes en el dataset: {', '.join(missing)}"
        )
    selection = df[COLUMNS_TO_KEEP].copy()
    selection = _coerce_numeric_columns(selection)
    LOGGER.debug("Selected required columns. Shape: %s", selection.shape)
    return selection


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(df)
    LOGGER.info("Initial dataset: %s rows", initial_rows)
    
    if initial_rows == 0:
        LOGGER.warning("Dataset already empty before cleaning")
        return df

    # ✅ CORREGIDO: No eliminar por surface_covered/surface_total nulos inicialmente
    # Solo eliminar filas donde price es nulo o no positivo
    df_clean = df.dropna(subset=["price"])
    after_dropna_price = len(df_clean)
    LOGGER.info("After dropping NA in price: %s rows", after_dropna_price)
    
    # Filtrar por precio positivo
    df_clean = df_clean[df_clean["price"] > 0]
    after_positive_price = len(df_clean)
    LOGGER.info("After positive price filter: %s rows", after_positive_price)
    
    # ✅ CORREGIDO: Imputar valores nulos en otras columnas numéricas antes de usar
    # estos valores para completar superficies faltantes.
    numeric_imputations = {"floor": 0, "rooms": 1, "expenses": 0}
    for column, default_value in numeric_imputations.items():
        if column in df_clean.columns:
            null_count = df_clean[column].isna().sum()
            if null_count > 0:
                df_clean[column] = df_clean[column].fillna(default_value)
                LOGGER.info(
                    "Imputed %s null values in column '%s' with default %s",
                    null_count,
                    column,
                    default_value,
                )

    # Rooms must be at least one to avoid zero divisions and unrealistic surfaces.
    if "rooms" in df_clean.columns:
        below_min_rooms = (df_clean["rooms"] < 1).sum()
        if below_min_rooms:
            LOGGER.info(
                "Adjusted %s rows with rooms < 1 to the minimum value 1",
                below_min_rooms,
            )
        df_clean["rooms"] = df_clean["rooms"].clip(lower=1)

    # ✅ NUEVO: Manejar surface_covered y surface_total nulos
    df_clean["surface_covered"] = df_clean["surface_covered"].fillna(
        df_clean["surface_total"]
    )
    df_clean["surface_total"] = df_clean["surface_total"].fillna(
        df_clean["surface_covered"]
    )

    mask_both_empty = df_clean["surface_covered"].isna() & df_clean["surface_total"].isna()
    if mask_both_empty.any():
        rooms_reference = df_clean.loc[mask_both_empty, "rooms"].fillna(1)
        default_surface = rooms_reference * 30  # 30m² por habitación como estimación
        df_clean.loc[mask_both_empty, "surface_covered"] = default_surface
        df_clean.loc[mask_both_empty, "surface_total"] = default_surface
        LOGGER.info(
            "Estimated surface for %s rows with missing surface data",
            mask_both_empty.sum(),
        )

    df_clean["surface_total"] = np.maximum(
        df_clean["surface_total"], df_clean["surface_covered"]
    )

    # Filtrar por superficie positiva
    df_clean = df_clean[(df_clean["surface_covered"] > 0) & (df_clean["surface_total"] > 0)]
    after_surface_filter = len(df_clean)
    LOGGER.info("After surface filter: %s rows", after_surface_filter)
    
    # Eliminar outliers de precio (conservador)
    if after_surface_filter > 0:
        price_threshold = df_clean["price"].quantile(0.99)
        df_clean = df_clean[df_clean["price"] <= price_threshold]
        after_outliers = len(df_clean)
        LOGGER.info("After outlier removal: %s rows (threshold: $%.2f)", after_outliers, price_threshold)
    else:
        LOGGER.warning("No data left after surface filter")
        return df_clean

    LOGGER.info(
        "Final cleaned dataset: %s rows (from original %s)",
        len(df_clean),
        initial_rows,
    )
    
    if len(df_clean) == 0:
        LOGGER.error("❌ Dataset is completely empty after cleaning!")
        # Debug detallado
        LOGGER.info("Debug info - Column analysis:")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            LOGGER.info("  %s: %s nulls, %s unique values", col, null_count, unique_count)
            if null_count == len(df):
                LOGGER.info("    ⚠️ Column '%s' is completely empty!", col)
    
    return df_clean

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # ✅ MEJORADO: Calcular price_per_m2 con manejo de division por cero
    df["price_per_m2"] = np.where(
        df["surface_covered"] > 0,
        df["price"] / df["surface_covered"],
        df["price"] / df["surface_total"]  # Fallback a surface_total
    )
    
    # ✅ MEJORADO: Calcular room_density con manejo de division por cero
    df["room_density"] = np.where(
        df["surface_total"] > 0,
        df["rooms"] / df["surface_total"],
        0.0,
    )
    
    # ✅ MEJORADO: Calcular expenses_ratio con manejo de division por cero
    df["expenses_ratio"] = np.where(
        df["price"] > 0,
        df["expenses"] / df["price"],
        0.0,
    )
    
    LOGGER.info("Engineered 3 new features with robust calculations")
    
    # Verificar que no hay valores infinitos o nulos en las nuevas features
    for feature in ["price_per_m2", "room_density", "expenses_ratio"]:
        if df[feature].isnull().any() or np.isinf(df[feature]).any():
            LOGGER.warning("Found invalid values in %s, cleaning...", feature)
            df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
            df[feature] = df[feature].fillna(0)
    
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
    # ✅ MEJORADO: Validar que tenemos datos para crear el target
    if df.empty:
        raise DataPreparationError("Cannot create target: dataset is empty")
    
    if "price_per_m2" not in df.columns:
        raise DataPreparationError("Cannot create target: price_per_m2 feature is missing")
    
    # Calcular percentiles
    p33 = df["price_per_m2"].quantile(0.33)
    p66 = df["price_per_m2"].quantile(0.66)
    
    LOGGER.info("Price per m² percentiles - p33: $%.2f, p66: $%.2f", p33, p66)

    def categorize(value: float) -> str:
        if value <= p33:
            return "BARATO"
        if value <= p66:
            return "REGULAR"
        return "CARO"

    target = df["price_per_m2"].apply(categorize)
    
    # Log de distribución
    target_distribution = target.value_counts()
    LOGGER.info("Target distribution:")
    for category, count in target_distribution.items():
        percentage = (count / len(target)) * 100
        LOGGER.info("  %s: %s (%.1f%%)", category, count, percentage)
    
    return df, target, {"p33": float(p33), "p66": float(p66)}

def prepare_data(
    csv_path: Path | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, LabelEncoder], Dict[str, float]]:
    """Prepare dataset for training."""

    _ensure_directories()
    csv_path = csv_path or DATA_PATH
    
    try:
        df = _load_dataset(csv_path)
        df = _rename_columns(df)
        df = _select_columns(df)
        df = _clean_data(df)
        
        # ✅ VALIDACIÓN CRÍTICA: Verificar que hay datos después de la limpieza
        if df.empty:
            raise DataPreparationError(
                "El dataset está vacío después de la limpieza. "
                "Revise el archivo CSV y los criterios de limpieza."
            )
            
        df = _engineer_features(df)
        df, encoders = _encode_categoricals(df)
        df, target, percentiles = _create_target(df)

        # ✅ CORREGIR: Las columnas categóricas ya están codificadas, no incluirlas dos veces
        feature_columns = [col for col in COLUMNS_TO_KEEP if col not in CATEGORICAL_COLUMNS] + [
            "price_per_m2", "room_density", "expenses_ratio"
        ] + CATEGORICAL_COLUMNS  # Las categóricas ya están codificadas

        features = df[feature_columns].copy()

        payload = {
            "X": features,
            "y": target,
            "feature_columns": feature_columns,
            "percentiles": percentiles,
        }
        joblib.dump(payload, TRAIN_DATA_PATH)
        LOGGER.info(
            "✅ Prepared dataset saved to %s with %s rows and %s features",
            TRAIN_DATA_PATH,
            len(features),
            len(feature_columns),
        )

        return features, target, feature_columns, encoders, percentiles
        
    except Exception as e:
        LOGGER.error("❌ Error in data preparation: %s", str(e))
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_data()
