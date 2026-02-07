from pathlib import Path
import coremltools as ct
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ROOT / "models"

XGB_JSON = MODELS / "purchase_predictor.json"
OUT_MLMODEL = MODELS / "PurchasePredictor.mlmodel"

print("=== convert.py starting ===")
print("ROOT:", ROOT)
print("MODELS:", MODELS)
print("Looking for:", XGB_JSON)

if not XGB_JSON.exists():
    raise FileNotFoundError(f"Missing model file: {XGB_JSON}")

# Load model
model = xgb.XGBClassifier()
model.load_model(str(XGB_JSON))
print("Loaded XGBoost model OK")

print("Attempting CoreML conversion...")

coreml_model = ct.converters.xgboost.convert(
    model,
    mode="classifier",
    class_labels=[0, 1],
    input_features=[
        ("distance_to_merchant", "Double"),
        ("hour_of_day", "Double"),
        ("is_weekend", "Double"),
        ("budget_utilization", "Double"),
        ("merchant_regret_rate", "Double"),
        ("dwell_time", "Double"),
    ],
)

print("Conversion succeeded. Saving to:", OUT_MLMODEL)
coreml_model.save(str(OUT_MLMODEL))

print("Saved:", OUT_MLMODEL)
print("=== convert.py done ===")
