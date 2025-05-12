import os
import pandas as pd
import ray
from ray import tune
from ray.air.integrations.mlflow import setup_mlflow
import xgboost as xgb

CONFIG = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "dataset_path": "/mnt/object/train_transformed.csv",
    "label_col": "risk_level",
    "experiment_name": "loan-risk-xgboost",
    "run_name": "xgb-ray-tune",
    "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:8000")
}

def train_fn(config):
    mlflow = setup_mlflow(
        config=config,
        experiment_name=config["experiment_name"],
        tracking_uri=config["tracking_uri"],
        run_name=config["run_name"]
    )

    df = pd.read_csv(config["dataset_path"])
    df[config["label_col"]] = df[config["label_col"]].map({"Low": 0, "Medium": 1, "High": 2})
    X = df.drop(columns=[config["label_col"]])
    y = df[config["label_col"]]

    dtrain = xgb.DMatrix(X, label=y)

    booster = xgb.train(
        params={
            "objective": config["objective"],
            "num_class": config["num_class"],
            "eval_metric": config["eval_metric"],
            "learning_rate": config["learning_rate"],
            "max_depth": config["max_depth"],
            "tree_method": "hist"
        },
        dtrain=dtrain,
        num_boost_round=config["n_estimators"]
    )

    booster.save_model("model.pth")
    mlflow.log_artifact("model.pth")
    mlflow.log_params({
        "n_estimators": config["n_estimators"],
        "learning_rate": config["learning_rate"],
        "max_depth": config["max_depth"]
    })
    mlflow.log_metric("final_iteration", config["n_estimators"])
    tune.report(success=1)

if __name__ == "__main__":
    tuner = tune.Tuner(
        tune.with_resources(train_fn, resources={"cpu": 4}),
        param_space=CONFIG,
        run_config=tune.RunConfig(
            name="xgb-ray-tune"
        ),
        tune_config=tune.TuneConfig(num_samples=1)
    )

    tuner.fit()
