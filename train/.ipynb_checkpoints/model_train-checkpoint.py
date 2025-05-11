import os
import ray
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from ray.train.xgboost import XGBoostTrainer, XGBoostCheckpoint
from ray.train import ScalingConfig, RunConfig, FailureConfig

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

# Load data
df = pd.read_csv(CONFIG["dataset_path"])
df[CONFIG["label_col"]] = df[CONFIG["label_col"]].map({"Low": 0, "Medium": 1, "High": 2})
ds = ray.data.from_pandas(df)

# Define trainer
trainer = XGBoostTrainer(
    label_column=CONFIG["label_col"],
    params={
        "objective": CONFIG["objective"],
        "num_class": CONFIG["num_class"],
        "eval_metric": CONFIG["eval_metric"],
        "learning_rate": CONFIG["learning_rate"],
        "max_depth": CONFIG["max_depth"],
        "tree_method": "hist"
    },
    datasets={"train": ds},
    num_boost_round=CONFIG["n_estimators"],
    scaling_config=ScalingConfig(
        num_workers=2,
        resources_per_worker={"CPU": 4},
        use_gpu=False
    ),
    run_config=RunConfig(
        name="xgb-ray-job",
        failure_config=FailureConfig(max_failures=2)
    )
)

if __name__ == "__main__":
    result = trainer.fit()

    # Load model from Ray checkpoint
    booster = XGBoostCheckpoint.from_checkpoint(result.checkpoint).get_model()
    booster.save_model("model.pth")

    # Log to MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8000"))
    mlflow.set_experiment(CONFIG["experiment_name"])
    with mlflow.start_run(run_name=CONFIG["run_name"]):
        mlflow.log_params({
            "n_estimators": CONFIG["n_estimators"],
            "learning_rate": CONFIG["learning_rate"],
            "max_depth": CONFIG["max_depth"]
        })
        mlflow.xgboost.log_model(booster, artifact_path="xgb-model")
        mlflow.log_artifact("model.pth")
