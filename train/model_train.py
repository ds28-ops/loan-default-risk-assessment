import pandas as pd
import os
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────── Paths ────────────────
DATA_PATH = "/mnt/data/processed/train_ready.csv"
MODEL_DIR = "/mnt/data/models"
REPORT_DIR = os.path.join(MODEL_DIR, "xgb_risk_classifier")
os.makedirs(REPORT_DIR, exist_ok=True)

# ──────────────── Load and Preprocess ────────────────
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop(columns=["risk_level"])
y = df["risk_level"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Encode categorical features
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# ──────────────── Model Training ────────────────
USE_HYPERPARAM_TUNING = False  # Set to True to enable grid search

if USE_HYPERPARAM_TUNING:
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
    }
    clf = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"), param_grid, cv=3, scoring='f1_weighted', verbose=1)
else:
    clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

clf.fit(X_train, y_train)

# ──────────────── Evaluation ────────────────
y_pred = clf.predict(X_test)

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Save classification report
with open(os.path.join(REPORT_DIR, "classification_report.json"), "w") as f:
    json.dump(report, f, indent=4)

# Save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
plt.close()

# ──────────────── Save Artifacts ────────────────
joblib.dump(clf, os.path.join(REPORT_DIR, "xgb_model.pkl"))
joblib.dump(label_encoder, os.path.join(REPORT_DIR, "label_encoder.pkl"))

print(f"✅ Model training complete. Artifacts and reports saved to {REPORT_DIR}")
