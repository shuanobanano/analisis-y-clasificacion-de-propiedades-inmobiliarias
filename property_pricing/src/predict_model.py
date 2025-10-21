"""Prediction utilities for the property price classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# Cambiar import relativo por absoluto
import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_preparation import MODELS_PATH, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
REQUIRED_FIELDS = {
    "price",
    "surface_covered",
    "surface_total",
    "rooms",
    "floor",
    "expenses",
    "neighborhood",
    "property_type",
}

NUMERIC_VALIDATIONS = {
    "price": lambda value: value > 0,
    "surface_covered": lambda value: value > 0,
    "surface_total": lambda value: value > 0,
    "rooms": lambda value: value >= 1,
    "floor": lambda value: value >= 0,
    "expenses": lambda value: value >= 0,
}


class PropertyPricePredictor:
    """Property price classification helper."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        base_models_path = MODELS_PATH
        model_file = Path(model_path) if model_path else base_models_path / "price_classifier.pkl"
        scaler_file = base_models_path / "scaler.pkl"
        columns_file = base_models_path / "model_columns.pkl"
        encoders_file = base_models_path / "encoders.pkl"

        required_files = [model_file, scaler_file, columns_file, encoders_file]
        if not all(file.exists() for file in required_files):
            raise FileNotFoundError("❌ Error: Modelo no encontrado. Ejecute train_model.py primero")

        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.model_columns: List[str] = joblib.load(columns_file)
        self.encoders: Dict[str, object] = joblib.load(encoders_file)
        self.class_labels = list(self.model.classes_)

    def _validate_payload(self, property_data: Dict) -> Dict[str, float]:
        missing_fields = [field for field in REQUIRED_FIELDS if field not in property_data]
        if missing_fields:
            raise ValueError(
                f"❌ Error: Falta campo requerido '{missing_fields[0]}' en propiedad {property_data.get('id', 'desconocida')}"
            )

        numeric_values: Dict[str, float] = {}
        for field, validator in NUMERIC_VALIDATIONS.items():
            try:
                numeric_values[field] = float(property_data[field])
            except (TypeError, ValueError):
                raise ValueError(
                    f"❌ Error: Valor inválido para '{field}' en propiedad {property_data.get('id', 'desconocida')}"
                ) from None
            if not validator(numeric_values[field]):
                raise ValueError(
                    f"❌ Error: Valor fuera de rango para '{field}' en propiedad {property_data.get('id', 'desconocida')}"
                )

        numeric_values["rooms"] = int(numeric_values["rooms"])
        numeric_values["floor"] = int(numeric_values["floor"])
        return numeric_values

    def _encode_categorical(self, column: str, value: str) -> int:
        encoder = self.encoders[column]
        value_str = str(value)
        if value_str not in encoder.classes_:
            value_str = "Unknown"
        return int(encoder.transform([value_str])[0])

    def preprocess_property(self, property_data: Dict) -> np.ndarray:
        """Transform a raw property payload into the model feature vector."""

        numeric_values = self._validate_payload(property_data)

        data = {**property_data, **numeric_values}
        data["price_per_m2"] = data["price"] / data["surface_covered"]
        data["room_density"] = (
            data["rooms"] / data["surface_total"] if data["surface_total"] > 0 else 0.0
        )
        data["expenses_ratio"] = data["expenses"] / data["price"]

        features = {}
        for column in self.model_columns:
            if column in CATEGORICAL_COLUMNS:
                features[column] = self._encode_categorical(column, data[column])
            else:
                features[column] = float(data[column])

        features_df = pd.DataFrame([features])
        features_df[NUMERIC_COLUMNS] = self.scaler.transform(features_df[NUMERIC_COLUMNS])
        return features_df[self.model_columns].to_numpy()

    def predict_single_property(self, property_data: Dict) -> Dict:
        """Predict the price category for a single property."""

        feature_vector = self.preprocess_property(property_data)
        probabilities = self.model.predict_proba(feature_vector)[0]
        prediction_index = int(np.argmax(probabilities))
        prediction_label = self.class_labels[prediction_index]

        price_per_m2 = float(property_data["price"]) / float(property_data["surface_covered"])

        result = {
            "prediction": prediction_label,
            "confidence": float(probabilities[prediction_index]),
            "price_per_m2": float(price_per_m2),
            "probabilities": {
                label: float(prob) for label, prob in zip(self.class_labels, probabilities)
            },
            "status": "success",
        }
        result["report"] = self.generate_detailed_report(result)
        return result

    def predict_from_jsonl(self, jsonl_path: str) -> List[Dict]:
        """Predict property categories from a JSON Lines file."""

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"❌ Error: No se encuentra el archivo {jsonl_path}")

        predictions: List[Dict] = []
        with path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    predictions.append(
                        {
                            "status": "error",
                            "error": f"❌ Error JSON línea {index}: {error.msg}",
                        }
                    )
                    continue

                try:
                    predictions.append(self.predict_single_property(payload))
                except ValueError as error:
                    predictions.append({"status": "error", "error": str(error)})
        return predictions

    def generate_detailed_report(self, prediction_result: Dict) -> str:
        """Generate a human readable report for a prediction result."""

        recommendation_map = {
            "BARATO": "Excelente oportunidad de compra. Considere ofertar pronto.",
            "REGULAR": "Precio acorde al mercado. Compare con propiedades similares.",
            "CARO": "Precio elevado. Negocie o espere una opción más conveniente.",
        }

        prediction = prediction_result["prediction"]
        confidence_pct = prediction_result["confidence"] * 100
        price_per_m2 = prediction_result["price_per_m2"]
        probabilities = prediction_result["probabilities"]

        report_lines = [
            "🏠 ANÁLISIS DE PROPIEDAD",
            "",
            f"📊 Clasificación: {prediction}",
            f"🎯 Confianza del modelo: {confidence_pct:.1f}%",
            f"💰 Precio por m²: ${price_per_m2:,.2f} USD",
            "",
            f"💬 Recomendación: {recommendation_map.get(prediction, 'Sin recomendación disponible.')}",
            "",
            "📈 Probabilidades:",
            f"   - BARATO: {probabilities.get('BARATO', 0.0) * 100:.1f}%",
            f"   - REGULAR: {probabilities.get('REGULAR', 0.0) * 100:.1f}%",
            f"   - CARO: {probabilities.get('CARO', 0.0) * 100:.1f}%",
        ]
        return "\n".join(report_lines)


if __name__ == "__main__":
    predictor = PropertyPricePredictor()
    sample_property = {
        "price": 120000,
        "surface_covered": 60,
        "surface_total": 70,
        "rooms": 3,
        "floor": 5,
        "expenses": 15000,
        "neighborhood": "Palermo",
        "property_type": "apartment",
    }
    print(predictor.generate_detailed_report(predictor.predict_single_property(sample_property)))
