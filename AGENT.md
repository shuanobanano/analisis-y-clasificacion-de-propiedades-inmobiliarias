OBJECTIVE
Build a robust property price classification system that trains on Kaggle data and classifies properties as "BARATO", "REGULAR", or "CARO" via JSON input through a GUI interface.


🔧 TECHNICAL SPECIFICATIONS - DO NOT DEVIATE
DEPENDENCIES (requirements.txt)
txt
pandas==1.5.3
scikit-learn==1.2.2
numpy==1.24.3
joblib==1.2.0
matplotlib==3.7.1
seaborn==0.12.2
DATA PREPARATION (data_preparation.py)
MUST IMPLEMENT EXACTLY:

Column Mapping:

price_aprox_usd → price

surface_covered_in_m2 → surface_covered

surface_total_in_m2 → surface_total

barrio → neighborhood

Keep: floor, rooms, expenses, property_type

Data Cleaning:

Remove rows with ANY missing values

Filter: surface_covered > 0 AND price > 0

Remove price outliers (top 1%)

Feature Engineering:

price_per_m2 = price / surface_covered

room_density = rooms / surface_total

expenses_ratio = expenses / price

Categorical Encoding:

Use LabelEncoder for neighborhood and property_type

Save encoders to models/encoders.pkl

Target Variable Creation:

Calculate 33rd and 66th percentiles of price_per_m2

Categories:

price_per_m2 ≤ p33 → "BARATO"

p33 < price_per_m2 ≤ p66 → "REGULAR"

price_per_m2 > p66 → "CARO"

MODEL TRAINING (train_model.py)
MUST USE THESE EXACT PARAMETERS:

python
RandomForestClassifier(
    n_estimators=200,      # DO NOT CHANGE
    max_depth=15,          # DO NOT CHANGE
    min_samples_split=10,  # DO NOT CHANGE
    min_samples_leaf=5,    # DO NOT CHANGE
    max_features='sqrt',   # DO NOT CHANGE
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
REQUIRED VALIDATION:

80/20 train/test split with stratification

5-fold cross-validation

Classification report and confusion matrix

Feature importance analysis

PREDICTION SYSTEM (predict_model.py)
REQUIRED CLASS STRUCTURE:

python
class PropertyPricePredictor:
    def __init__(self, model_path='../models/price_classifier.pkl')
    def preprocess_property(self, property_data: Dict) -> np.ndarray
    def predict_single_property(self, property_data: Dict) -> Dict
    def predict_from_jsonl(self, jsonl_path: str) -> List[Dict]
    def generate_detailed_report(self, prediction_result: Dict) -> str
JSON INPUT FORMAT - STRICT SCHEMA:

json
{
  "price": number > 0,
  "surface_covered": number > 0,
  "surface_total": number > 0,
  "rooms": integer >= 1,
  "floor": integer >= 0,
  "expenses": number >= 0,
  "neighborhood": string,
  "property_type": string
}
🚀 EXECUTION PIPELINE - FOLLOW EXACTLY
STEP 1: DATA DOWNLOAD
bash
# MANUAL STEP - User must download from:
# https://www.kaggle.com/datasets/martinbasualdo/property-prices-in-caba-zonaprop-data
# Save as: property_pricing/data/property_prices.csv
STEP 2: INITIAL SETUP
bash
cd property_pricing
pip install -r requirements.txt
STEP 3: RUN COMPLETE PIPELINE
bash
python src/run_pipeline.py
EXPECTED OUTPUT:

text
🚀 INICIANDO PIPELINE DE CLASIFICACIÓN DE PROPIEDADES
📊 1. PREPARANDO DATOS...
✅ Datos preparados correctamente
🤖 2. ENTRENANDO MODELO...
📊 Precisión validación cruzada: 0.XXX (+/- 0.XXX)
📈 Reporte de Clasificación:
🎯 Importancia de Features:
✅ Modelo entrenado y guardado correctamente
🎯 3. PROBANDO PREDICCIONES...
🏠 ANÁLISIS DE PROPIEDAD...
✅ Pipeline ejecutado exitosamente
STEP 4: INDIVIDUAL PREDICTIONS
bash
python src/predict_model.py data/test_properties.jsonl
📊 EXPECTED MODEL PERFORMANCE
MINIMUM ACCEPTANCE CRITERIA:

Cross-validation accuracy: > 75%

Precision/Recall per class: > 70%

Confidence scores: > 60% for correct predictions

🎪 OUTPUT FORMAT REQUIREMENTS
PREDICTION REPORT TEMPLATE:
text
🏠 ANÁLISIS DE PROPIEDAD

📊 Clasificación: BARATO/REGULAR/CARO
🎯 Confianza del modelo: XX.X%
💰 Precio por m²: $X,XXX.XX USD

💬 Recomendación: [Contextual message based on category]

📈 Probabilidades:
   - BARATO: XX.X%
   - REGULAR: XX.X%
   - CARO: XX.X%
ERROR HANDLING - REQUIRED SCENARIOS:
Missing CSV: "❌ Error: No se encuentra property_prices.csv en la carpeta data/"

Invalid JSON: "❌ Error JSON línea X: [specific error]"

Missing required fields: "❌ Error: Falta campo requerido 'price' en propiedad X"

Model not trained: "❌ Error: Modelo no encontrado. Ejecute train_model.py primero"

🔍 VALIDATION CHECKLIST
PRE-EXECUTION CHECKS:
Kaggle dataset downloaded to correct location

requirements.txt installed

Directory structure exists

Sufficient disk space (>500MB)

POST-EXECUTION VERIFICATION:
train.pkl file generated (>10MB)

All .pkl files in models/ directory

Confusion matrix image generated

Cross-validation scores printed

Feature importance analysis displayed

PREDICTION VERIFICATION:
JSONL file with valid schema

All properties receive classification

Confidence scores between 0-1

Detailed reports generated for each property

⚠️ COMMON PITFALLS TO AVOID
DATA ISSUES:

Don't forget to handle division by zero in feature engineering

Ensure proper outlier removal (top 1% prices only)

Verify categorical encoding preserves all categories

MODEL ISSUES:

Use exact RandomForest parameters specified

Always set random_state=42 for reproducibility

Save ALL components: model, scaler, encoders, columns

PREDICTION ISSUES:

Handle unknown categories in neighborhood/property_type

Validate JSON input schema strictly

Provide meaningful error messages

📝 INTEGRATION WITH ELECTRON/GUI
EXPORT FORMAT FOR GUI:

typescript
interface PredictionResult {
  prediction: "BARATO" | "REGULAR" | "CARO";
  confidence: number;
  price_per_m2: number;
  probabilities: {
    BARATO: number;
    REGULAR: number;
    CARO: number;
  };
  status: "success" | "error";
  error?: string;
}
USAGE PATTERN:

python
# In Electron main process
predictor = PropertyPricePredictor()
results = predictor.predict_from_jsonl(uploaded_file_path)
# Send results to React frontend via IPC
🎯 SUCCESS CRITERIA
The implementation is successful when:

Pipeline runs end-to-end without errors

Model achieves >75% cross-validation accuracy

Predictions generate detailed, actionable reports

System handles edge cases gracefully

All files are properly saved and loaded


NOTE TO AGENT: Follow this specification EXACTLY. Do not modify hyperparameters, file structure, or data processing steps without explicit approval. This ensures consistency and reproducibility across all implementations.
