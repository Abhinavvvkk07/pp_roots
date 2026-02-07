import json
import os
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_training_data.csv"
MODEL_PATH = ROOT / "models" / "purchase_predictor.json"
META_PATH = ROOT / "models" / "purchase_predictor_meta.json"

TARGET = "purchase_occurred"

# ---- Load data ----
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {DATA_PATH}. Run generate_data.py first.")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Loaded {len(df)} rows")
print("Label distribution:")
print(y.value_counts(normalize=True))

# ---- Split (stratified) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---- Train XGBoost ----
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
)

print("\nTraining model...")
model.fit(X_train, y_train)

# ---- Evaluate ----
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)

print("\n--- Metrics ---")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")

# ---- Save ----
os.makedirs(ROOT / "models", exist_ok=True)
model.save_model(str(MODEL_PATH))

meta = {
    "model_type": "xgboost",
    "feature_names": list(X.columns),
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ Saved model to:", MODEL_PATH)
print("✅ Saved metadata to:", META_PATH)
