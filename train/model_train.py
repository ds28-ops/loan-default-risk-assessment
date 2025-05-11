import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
from ray.train.sklearn import SklearnTrainer
from ray.train import RunConfig, ScalingConfig, Checkpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def train_fn(config):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(config["experiment_name"])

    df = pd.read_csv(config["dataset_path"])
    df[config["label_col"]] = df[config["label_col"]].map({"Low": 0, "Medium": 1, "High": 2})
    X = df.drop(columns=[config["label_col"]])
    y = df[config["label_col"]]

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params({k: config[k] for k in ["n_estimators", "learning_rate", "max_depth", "n_splits"]})

        kf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["random_state"])
        fold_accuracies = []

        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model = XGBClassifier(
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                max_depth=config["max_depth"],
                random_state=config["random_state"],
                objective=config["objective"],
                num_class=config["num_class"],
                eval_metric=config["eval_metric"]
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            fold_accuracies.append(acc)

            mlflow.log_metric(f"fold_{fold+1}_accuracy", acc)

        avg_acc = np.mean(fold_accuracies)
        mlflow.log_metric("avg_kfold_accuracy", avg_acc)

        joblib.dump(model, config["save_path"])
        mlflow.log_artifact(config["save_path"])

config = {
    "n_splits": 5,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "random_state": 42,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "experiment_name": "loan-risk-xgboost",
    "run_name": "xgb-ray-run",
    "save_path": "loan_risk_model_ray.pth",
    "dataset_path": "/mnt/object/LoanData/train_transformed.csv",
    "label_col": "risk_level"
}

trainer = SklearnTrainer(
    run_config=RunConfig(name="xgb-ray-run1"),
    scaling_config=ScalingConfig(num_workers=1, use_gpu=False, resources_per_worker={"CPU": 4}),
    datasets=None,
    train_loop_config=config,
    train_loop_per_worker=train_fn
)

result = trainer.fit()
