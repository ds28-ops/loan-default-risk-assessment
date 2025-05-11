import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
import ray
from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback
from ray.train import ScalingConfig, RunConfig

# Config
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
    "dataset_path": "/mnt/object/train_transformed.csv",
    "label_col": "risk_level"
}
data_path = os.getenv("DATA_PATH")
# Load dataset as Ray Data
df = pd.read_csv(os.path.join(data_path, "train_transformed.csv"))
df[CONFIG["label_col"]] = df[CONFIG["label_col"]].map({"Low": 0, "Medium": 1, "High": 2})
ds = ray.data.from_pandas(df)

# Training function
def train_func(config):
    import mlflow
    import mlflow.xgboost
    import xgboost as xgb
    from ray.train import get_dataset_shard

    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params({
            "n_estimators": config["n_estimators"],
            "learning_rate": config["learning_rate"],
            "max_depth": config["max_depth"]
        })

        shard = get_dataset_shard("train").to_pandas()
        X = shard.drop(columns=[config["label_col"]])
        y = shard[config["label_col"]]

        dtrain = xgb.DMatrix(X, label=y)

        booster = xgb.train(
            params={
                "objective": config["objective"],
                "num_class": config["num_class"],
                "eval_metric": config["eval_metric"],
                "learning_rate": config["learning_rate"],
                "max_depth": config["max_depth"]
            },
            dtrain=dtrain,
            num_boost_round=config["n_estimators"],
            callbacks=[RayTrainReportCallback()]
        )

        booster.save_model("model.json")
        mlflow.xgboost.log_model(booster, artifact_path="xgb-model")
        mlflow.log_artifact("model.json")


# Set up Ray Trainer
trainer = XGBoostTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    datasets={"train": ds},
    run_config=RunConfig(name="xgb-ray-job"),
    train_loop_config=CONFIG
)

if __name__ == "__main__":
    result = trainer.fit()
