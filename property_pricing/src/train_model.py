"""Model training module for property price classification."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cambiar import relativo por absoluto
import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_preparation import NUMERIC_COLUMNS, MODELS_PATH, prepare_data


LOGGER = logging.getLogger(__name__)

MODEL_PATH = MODELS_PATH / "price_classifier.pkl"
SCALER_PATH = MODELS_PATH / "scaler.pkl"
COLUMNS_PATH = MODELS_PATH / "model_columns.pkl"
CONFUSION_MATRIX_PATH = MODELS_PATH / "confusion_matrix.png"
METADATA_PATH = MODELS_PATH / "metadata.json"
TEST_BUNDLE_PATH = MODELS_PATH / "price_classifier_bundle.pkl"
MODEL_VERSION = "1.1.0"

RANDOM_FOREST_PARAMS: Dict[str, object] = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

EARLY_STOPPING_STEP = 50
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA = 1e-3
NUMERIC_DRIFT_THRESHOLD = 0.5
CATEGORICAL_DRIFT_THRESHOLD = 0.2
CATEGORICAL_COLUMNS = ["neighborhood", "property_type"]

def _build_cross_validation_pipeline(n_estimators: int) -> Pipeline:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, NUMERIC_COLUMNS)],
        remainder="passthrough",
    )
    classifier = RandomForestClassifier(
        **{**RANDOM_FOREST_PARAMS, "n_estimators": n_estimators}
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _plot_confusion_matrix(cm: np.ndarray, labels: Iterable[str], save_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.ylabel("Valor real")
    plt.xlabel("Predicción")
    plt.title("Matriz de confusión - Clasificador de precios")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    LOGGER.info("Confusion matrix saved to %s", save_path)


def _log_feature_importances(model: RandomForestClassifier, feature_names: Iterable[str]) -> List[Tuple[str, float]]:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    LOGGER.info("Top feature importances:")
    for feature, importance in importances.head(10).items():
        LOGGER.info("  - %s: %.4f", feature, importance)
    return [(feature, float(importance)) for feature, importance in importances.items()]


def _assess_distribution_shift(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {
        "numeric_drift": {},
        "categorical_drift": {},
        "target_drift": {},
    }

    for column in NUMERIC_COLUMNS:
        train_mean = X_train[column].mean()
        test_mean = X_test[column].mean()
        train_std = X_train[column].std(ddof=0) or 1.0
        drift_score = abs(train_mean - test_mean) / train_std
        metrics["numeric_drift"][column] = float(drift_score)
        if drift_score > NUMERIC_DRIFT_THRESHOLD:
            LOGGER.warning(
                "Significant numeric drift detected for '%s': %.3f",
                column,
                drift_score,
            )

    for column in CATEGORICAL_COLUMNS:
        train_dist = X_train[column].value_counts(normalize=True)
        test_dist = X_test[column].value_counts(normalize=True)
        categories = set(train_dist.index).union(test_dist.index)
        drift_score = 0.5 * sum(
            abs(train_dist.get(cat, 0.0) - test_dist.get(cat, 0.0))
            for cat in categories
        )
        metrics["categorical_drift"][column] = float(drift_score)
        if drift_score > CATEGORICAL_DRIFT_THRESHOLD:
            LOGGER.warning(
                "Categorical distribution drift detected for '%s': %.3f",
                column,
                drift_score,
            )

    train_target = y_train.value_counts(normalize=True)
    test_target = y_test.value_counts(normalize=True)
    classes = set(train_target.index).union(test_target.index)
    target_drift = 0.5 * sum(
        abs(train_target.get(cls, 0.0) - test_target.get(cls, 0.0))
        for cls in classes
    )
    metrics["target_drift"] = {
        "total_variation": float(target_drift),
    }
    if target_drift > CATEGORICAL_DRIFT_THRESHOLD:
        LOGGER.warning("Target distribution drift detected: %.3f", target_drift)

    return metrics


def _train_with_early_stopping(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[RandomForestClassifier, Dict[str, object]]:
    X_inner, X_val, y_inner, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    best_model: RandomForestClassifier | None = None
    best_score = -np.inf
    best_estimators = RANDOM_FOREST_PARAMS["n_estimators"]
    no_improvement = 0
    history: List[Dict[str, float]] = []

    max_estimators = RANDOM_FOREST_PARAMS["n_estimators"]
    for n_estimators in range(EARLY_STOPPING_STEP, max_estimators + EARLY_STOPPING_STEP, EARLY_STOPPING_STEP):
        n_current = min(n_estimators, max_estimators)
        LOGGER.info("Training RandomForest with %s estimators for early stopping", n_current)
        model = RandomForestClassifier(
            **{**RANDOM_FOREST_PARAMS, "n_estimators": n_current}
        )
        model.fit(X_inner, y_inner)
        score = model.score(X_val, y_val)
        history.append({"n_estimators": float(n_current), "validation_accuracy": float(score)})
        LOGGER.info("Validation accuracy with %s estimators: %.4f", n_current, score)

        if score > best_score + EARLY_STOPPING_MIN_DELTA:
            best_score = score
            best_model = model
            best_estimators = n_current
            no_improvement = 0
            LOGGER.debug("New best model found at %s estimators", n_current)
        else:
            no_improvement += 1
            LOGGER.debug(
                "No improvement observed. Patience counter: %s/%s",
                no_improvement,
                EARLY_STOPPING_PATIENCE,
            )

        if no_improvement >= EARLY_STOPPING_PATIENCE and n_current >= EARLY_STOPPING_STEP:
            LOGGER.info("Early stopping triggered at %s estimators", n_current)
            break

        if n_current == max_estimators:
            LOGGER.info("Maximum number of estimators (%s) reached during early stopping", max_estimators)
            break

    if best_model is None:
        raise RuntimeError("Early stopping failed to produce a trained model.")

    summary = {
        "best_estimators": int(best_estimators),
        "best_validation_accuracy": float(best_score),
        "history": history,
    }
    return best_model, summary


def _serialize_metadata(metadata: Dict[str, object]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    LOGGER.info("Model metadata saved to %s", METADATA_PATH)


def _export_model_bundle(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    feature_columns: Iterable[str],
    encoders: Dict[str, object],
    percentiles: Dict[str, float],
    metadata: Dict[str, object],
) -> Path:
    """Bundle key artefacts into a single file for downstream testing."""

    bundle_payload = {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(feature_columns),
        "encoders": encoders,
        "percentiles": percentiles,
        "metadata": metadata,
        "exported_at": datetime.utcnow().isoformat() + "Z",
    }
    joblib.dump(bundle_payload, TEST_BUNDLE_PATH)
    LOGGER.info("Testing bundle saved to %s", TEST_BUNDLE_PATH)
    return TEST_BUNDLE_PATH


def train_model() -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Train the RandomForest classifier and persist all artefacts."""

    features, target, feature_columns, encoders, percentiles = prepare_data()
    
    # ✅ VALIDACIÓN CRÍTICA: Verificar que hay datos para entrenar
    if len(features) == 0:
        raise ValueError("No hay datos para entrenar el modelo. El dataset está vacío.")
        
    LOGGER.info("Starting training pipeline with %s samples", len(features))

    # ✅ VERIFICAR: Asegurar que las columnas categóricas están en los datos
    LOGGER.info("Feature columns: %s", feature_columns)
    LOGGER.info("Data types: %s", features.dtypes)
    
    # Verificar que todas las columnas requeridas existen
    missing_columns = [col for col in feature_columns if col not in features.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in features: {missing_columns}")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    
    # ✅ VERIFICACIÓN ADICIONAL: Que el split no esté vacío
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Train/Test split resulted in empty sets. Train: {len(X_train)}, Test: {len(X_test)}"
        )
        
    LOGGER.info("Train/Test split: %s train rows, %s test rows", len(X_train), len(X_test))

    # Resto del código permanece igual...
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # ✅ FILTRAR: Solo escalar columnas numéricas que existen
    numeric_cols_to_scale = [col for col in NUMERIC_COLUMNS if col in X_train_scaled.columns]
    LOGGER.info("Scaling numeric columns: %s", numeric_cols_to_scale)
    
    X_train_scaled[numeric_cols_to_scale] = scaler.fit_transform(X_train_scaled[numeric_cols_to_scale])
    X_test_scaled[numeric_cols_to_scale] = scaler.transform(X_test_scaled[numeric_cols_to_scale])
    
    drift_metrics = _assess_distribution_shift(X_train, X_test, y_train, y_test)

    _, early_summary = _train_with_early_stopping(X_train_scaled, y_train)
    best_estimators = early_summary["best_estimators"]

    final_model = RandomForestClassifier(
        **{**RANDOM_FOREST_PARAMS, "n_estimators": best_estimators}
    )
    final_model.fit(X_train_scaled, y_train)
    LOGGER.info("Final model trained with %s estimators", best_estimators)

    y_pred = final_model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, digits=4)
    LOGGER.info("Classification report:\n%s", report)

    cm = confusion_matrix(y_test, y_pred, labels=final_model.classes_)
    _plot_confusion_matrix(cm, final_model.classes_, CONFUSION_MATRIX_PATH)

    cv_pipeline = _build_cross_validation_pipeline(best_estimators)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        cv_pipeline,
        features,
        target,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1,
    )
    LOGGER.info("Cross-validation accuracy scores: %s", np.round(cv_scores, 4).tolist())
    LOGGER.info("Cross-validation mean accuracy: %.4f", cv_scores.mean())

    feature_importances = _log_feature_importances(final_model, feature_columns)

    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_columns, COLUMNS_PATH)
    LOGGER.info("Model artefacts saved under %s", MODELS_PATH)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "hyperparameters": {**RANDOM_FOREST_PARAMS, "n_estimators": best_estimators},
        "early_stopping": early_summary,
        "dataset": {
            "total_samples": int(len(features)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "feature_columns": feature_columns,
            "percentiles": percentiles,
        },
        "performance": {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "cross_validation_scores": cv_scores.tolist(),
            "cross_validation_mean": float(cv_scores.mean()),
        },
        "distribution_checks": drift_metrics,
        "feature_importances": feature_importances,
    }
    _serialize_metadata(metadata)
    _export_model_bundle(final_model, scaler, feature_columns, encoders, percentiles, metadata)

    return final_model, percentiles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()