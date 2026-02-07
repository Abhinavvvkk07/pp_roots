import os
import json
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ----------------------------
# Config (easy to tweak)
# ----------------------------
DATA_PATH = "data/synthetic_training_data.csv"
MODEL_PATH = "models/purchase_predictor.json"
META_PATH = "models/purchase_predictor_meta.json"

# Choose a probability threshold for "send nudge"
# You can tune this later after seeing metrics.
DEFAULT_THRESHOLD = 0.70

# ----------------------------
# 1) Load data
# ----------------------------
df = pd.read_csv(DATA_PATH)

TARGET = "purchase_occurred"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES]
y = df[TARGET]

print(f"Loaded {len(df)} rows")
print("Label distribution:")
print(y.value_counts(normalize=True).rename("ratio"))

# ----------------------------
# 2) Train/test split (stratified)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,   # IMPORTANT: keeps same 0/1 ratio in train and test
)

# ----------------------------
# 3) Train XGBoost
# ----------------------------
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=200,         # a bit more capacity (still fast)
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42,
)

print("\nTraining model...")
model.fit(X_train, y_train)

# ----------------------------
# 4) Evaluate with probabilities + threshold
# ----------------------------
probs = model.predict_proba(X_test)[:, 1]  # probability of class 1
threshold = DEFAULT_THRESHOLD
preds = (probs >= threshold).astype(int)

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
auc = roc_auc_score(y_test, probs)

cm = confusion_matrix(y_test, preds)

print("\n--- Metrics (threshold-based) ---")
print(f"Threshold: {threshold:.2f}")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}   (How often nudges are correct)")
print(f"Recall   : {rec:.3f}   (How many purchases we catch)")
print(f"F1       : {f1:.3f}")
print(f"AUC      : {auc:.3f}   (overall separability)")

print("\nConfusion Matrix [[TN FP],[FN TP]]:")
print(cm)

print("\n--- Classification Report ---")
print(classification_report(y_test, preds, zero_division=0))

# ----------------------------
# 5) Quick threshold sweep (to choose a good nudge threshold)
# ----------------------------
print("\n--- Threshold sweep (pick what you like) ---")
for t in [0.50, 0.60, 0.70, 0.80, 0.90]:
    p = (probs >= t).astype(int)
    p_prec = precision_score(y_test, p, zero_division=0)
    p_rec = recall_score(y_test, p, zero_division=0)
    p_f1 = f1_score(y_test, p, zero_division=0)
    print(f"t={t:.2f}  precision={p_prec:.3f}  recall={p_rec:.3f}  f1={p_f1:.3f}")

# ----------------------------
# 6) Save model + metadata (feature order matters!)
# ----------------------------
os.makedirs("models", exist_ok=True)

model.save_model(MODEL_PATH)

meta = {
    "model_type": "xgboost",
    "feature_names": FEATURES,
    "threshold": threshold,
    "notes": "Probability threshold used for nudges. Keep feature order consistent at inference.",
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved model to: {MODEL_PATH}")
print(f"Saved metadata to: {META_PATH}")

# ----------------------------
# 7) Demo prediction (one example)
# ----------------------------
demo_example = {
    "distance_to_merchant": 30,
    "hour_of_day": 23,
    "is_weekend": 1,
    "budget_utilization": 0.95,
    "merchant_regret_rate": 0.8,
    "dwell_time": 300,
}
demo_df = pd.DataFrame([demo_example])[FEATURES]
demo_prob = model.predict_proba(demo_df)[:, 1][0]
demo_pred = int(demo_prob >= threshold)

print("\n--- Demo example ---")
print("Inputs:", demo_example)
print(f"Predicted probability of purchase: {demo_prob:.3f}")
print(f"Nudge? {'YES' if demo_pred == 1 else 'NO'} (threshold={threshold:.2f})")
