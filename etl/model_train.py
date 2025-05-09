import pandas as pd
import os
import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────── Paths ────────────────
BASE_PATH = "/mnt/object/loan-default-data"
TRAIN_PATH = os.path.join(BASE_PATH, "train", "train_clean.csv")
VAL_PATH = os.path.join(BASE_PATH, "val", "val_clean.csv")

MODEL_DIR = os.path.join(BASE_PATH, "models", "xgb_risk_classifier")
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────── Load and Preprocess ────────────────
df_train = pd.read_csv(TRAIN_PATH)
df_val = pd.read_csv(VAL_PATH)

X_train, y_train = df_train.drop(columns=["risk_level"]), df_train["risk_level"]
X_val, y_val = df_val.drop(columns=["risk_level"]), df_val["risk_level"]

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)

for df in [X_train, X_val]:
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# ──────────────── Model Training ────────────────
USE_HYPERPARAM_TUNING = False

mlflow.set_experiment("loan-default-xgb")

with mlflow.start_run() as run:
    if USE_HYPERPARAM_TUNING:
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
        }
        base_model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
        clf = GridSearchCV(base_model, param_grid, cv=3, scoring='f1_weighted', verbose=1)
    else:
        clf = XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.9
        )

    clf.fit(X_train, y_train_enc, eval_set=[(X_val, y_val_enc)], verbose=False)

    # ──────────────── Evaluation ────────────────
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val_enc, y_pred)
    f1 = f1_score(y_val_enc, y_pred, average='weighted')
    cm = confusion_matrix(y_val_enc, y_pred)
    report = classification_report(y_val_enc, y_pred, target_names=label_encoder.classes_, output_dict=True)

    # ──────────────── MLflow Logging ────────────────
    mlflow.log_params(clf.get_params() if not USE_HYPERPARAM_TUNING else clf.best_params_)

    mlflow.log_metrics({
        "val_accuracy": acc,
        "val_f1_score": f1
    })

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="plots")

    # Artifacts
    model_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    joblib.dump(clf, model_path)
    joblib.dump(label_encoder, encoder_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.log_artifact(encoder_path, artifact_path="model")

    report_path = os.path.join(MODEL_DIR, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact(report_path, artifact_path="metrics")

print(f"✅ Training complete. Artifacts saved in {MODEL_DIR} and logged to MLflow.")
