import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb

from ray.train.xgboost import XGBoostTrainer
from ray.train import ScalingConfig, RunConfig

# -------------------- Config --------------------
CONFIG = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "random_state": 42,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "experiment_name": "loan-risk-xgboost",
    "run_name": "xgb-ray-run",
    "dataset_path": "/mnt/object/LoanData/train_transformed.csv",
    "label_col": "risk_level"
}

# -------------------- Training Function --------------------
def train_func(config):
    df = pd.read_csv(config["dataset_path"])
    df[config["label_col"]] = df[config["label_col"]].map({"Low": 0, "Medium": 1, "High": 2})
    
    X = df.drop(columns=[config["label_col"]])
    y = df[config["label_col"]]

    dtrain = xgb.DMatrix(X, label=y)

    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params({
            "n_estimators": config["n_estimators"],
            "learning_rate": config["learning_rate"],
            "max_depth": config["max_depth"]
        })

        booster = xgb.train(
            params={
                "objective": config["objective"],
                "num_class": config["num_class"],
                "eval_metric": config["eval_metric"],
                "learning_rate": config["learning_rate"],
                "max_depth": config["max_depth"]
            },
            dtrain=dtrain,
            num_boost_round=config["n_estimators"]
        )

        booster.save_model("model.json")
        mlflow.xgboost.log_model(booster, artifact_path="xgb-model")
        mlflow.log_artifact("model.json")

# -------------------- Ray Trainer --------------------
trainer = XGBoostTrainer(
    scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    label_column=CONFIG["label_col"],
    params={
        "objective": CONFIG["objective"],
        "num_class": CONFIG["num_class"],
        "eval_metric": CONFIG["eval_metric"],
        "learning_rate": CONFIG["learning_rate"],
        "max_depth": CONFIG["max_depth"]
    },
    num_boost_round=CONFIG["n_estimators"],
    datasets={"train": pd.read_csv(CONFIG["dataset_path"])},
    run_config=RunConfig(name="xgb-ray-job", verbose=1),
    train_loop_config=CONFIG
)

if __name__ == "__main__":
    trainer.fit()
