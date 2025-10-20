"""End-to-end execution script for the property price classifier."""
from __future__ import annotations

from .data_preparation import prepare_data
from .train_model import train_model


def main() -> None:
    try:
        features, target, _, _, _ = prepare_data()
        print(f"Datos procesados: {len(features)} registros listos para entrenamiento.")
        train_model()
        print("Entrenamiento completado con éxito.")
    except FileNotFoundError as error:
        print(error)


if __name__ == "__main__":
    main()
