"""End-to-end execution script for the property price classifier."""
from __future__ import annotations

import logging
import sys
import os

# Agregar el directorio actual al path para imports absolutos
sys.path.append(os.path.dirname(__file__))

from data_preparation import prepare_data, DataPreparationError
from train_model import train_model

# Configurar logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def main() -> None:
    try:
        LOGGER.info("🚀 Starting data preparation...")
        features, target, _, _, _ = prepare_data()
        
        if len(features) == 0:
            LOGGER.error("❌ No data available for training. Check your dataset.")
            return
            
        LOGGER.info("✅ Data preparation completed: %s records ready for training", len(features))
        
        LOGGER.info("🧠 Starting model training...")
        train_model()
        LOGGER.info("🎉 Training completed successfully!")
        
    except FileNotFoundError as error:
        LOGGER.error("📁 File error: %s", error)
    except DataPreparationError as error:
        LOGGER.error("🔄 Data preparation error: %s", error)
    except ValueError as error:
        LOGGER.error("❌ Validation error: %s", error)
    except Exception as error:
        LOGGER.error("💥 Unexpected error: %s", error)
        raise

if __name__ == "__main__":
    main()