import ray
from ray.train.xgboost import XGBoostTrainer
from ray.train import ScalingConfig, RunConfig, FailureConfig
import pandas as pd
import os
CONFIG = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "random_state": 42,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "experiment_name": "loan-risk-xgboost",
    "dataset_path": "/mnt/object/train_transformed.csv",
    "label_col": "risk_level"
}
data_path = os.getenv("DATA_PATH")
# Load dataset
df = pd.read_csv(os.path.join(data_path, "train_transformed.csv")) 
df[CONFIG["label_col"]] = df[CONFIG["label_col"]].map({"Low": 0, "Medium": 1, "High": 2})

# Convert to Ray Dataset
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
        "tree_method": "hist",
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