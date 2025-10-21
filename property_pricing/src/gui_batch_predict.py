"""Utilities for batch predictions to power the Electron GUI."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

import joblib
import pandas as pd

from predict_model import PropertyPricePredictor, REQUIRED_FIELDS


def _load_from_json(path: Path) -> List[dict]:
    """Load property payloads from JSON or JSONL files."""

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Try JSON Lines first
    lines = [line for line in text.splitlines() if line.strip()]
    if all(line.startswith("{") for line in lines):
        try:
            return [json.loads(line) for line in lines]
        except json.JSONDecodeError:
            pass

    data = json.loads(text)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return list(data)
    raise ValueError("❌ Error: Formato JSON no soportado para predicciones")


def _load_from_csv(path: Path) -> List[dict]:
    """Load property payloads from CSV files."""

    dataframe = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_FIELDS if column not in dataframe.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"❌ Error: Faltan columnas requeridas en CSV: {missing}")
    return dataframe.to_dict(orient="records")


def _load_from_joblib(path: Path) -> List[dict]:
    """Load property payloads from joblib serialized files."""

    data = joblib.load(path)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, Iterable):
        return list(data)
    raise ValueError("❌ Error: Contenido de joblib inválido. Se esperaba lista o diccionario")


def _load_from_markdown(path: Path) -> List[dict]:
    """Extract property payloads from Markdown files."""

    text = path.read_text(encoding="utf-8")

    # Look for JSON fenced code blocks first
    json_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if json_blocks:
        payloads: List[dict] = []
        for block in json_blocks:
            block = block.strip()
            if not block:
                continue
            data = json.loads(block)
            if isinstance(data, dict):
                payloads.append(data)
            elif isinstance(data, list):
                payloads.extend(data)
        if payloads:
            return payloads

    # Fallback: try to interpret Markdown tables
    table_lines = [line for line in text.splitlines() if "|" in line and not line.strip().startswith("#")]
    if table_lines:
        headers = [header.strip() for header in table_lines[0].split("|") if header.strip()]
        rows = []
        for line in table_lines[2:]:  # Skip header and separator rows
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if len(cells) != len(headers):
                continue
            rows.append(dict(zip(headers, cells)))
        if rows:
            return rows

    raise ValueError(
        "❌ Error: No se encontraron datos válidos en el archivo Markdown. "
        "Incluya un bloque JSON o una tabla con los campos requeridos."
    )


def load_properties_from_file(file_path: Path) -> List[dict]:
    """Load property payloads from a supported file type."""

    if not file_path.exists():
        raise FileNotFoundError(f"❌ Error: No se encuentra el archivo {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return _load_from_json(file_path)
    if suffix == ".csv":
        return _load_from_csv(file_path)
    if suffix in {".joblib", ".pkl"}:
        return _load_from_joblib(file_path)
    if suffix in {".md", ".markdown"}:
        return _load_from_markdown(file_path)
    raise ValueError("❌ Error: Formato de archivo no soportado para predicciones GUI")


def run_batch_prediction(file_path: Path) -> List[dict]:
    """Generate predictions for all payloads contained in *file_path*."""

    predictor = PropertyPricePredictor()
    payloads = load_properties_from_file(file_path)
    if not payloads:
        return []

    results: List[dict] = []
    for payload in payloads:
        try:
            results.append(predictor.predict_single_property(payload))
        except ValueError as error:
            results.append({"status": "error", "error": str(error)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch predictions for the GUI front-end")
    parser.add_argument("file_path", type=Path, help="Ruta al archivo de propiedades")
    args = parser.parse_args()

    try:
        results = run_batch_prediction(args.file_path)
        response = {"status": "success", "results": results}
    except Exception as error:  # pylint: disable=broad-except
        response = {"status": "error", "error": str(error)}

    print(json.dumps(response, ensure_ascii=False))


if __name__ == "__main__":
    main()
