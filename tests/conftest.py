import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import joblib
from src.evaluate import transform, predict




MODEL_PATH = "/mnt/block/minio_data/mlflow-artifacts/1/19ade86ae7634e788676cd21117edbfa/artifacts/model.pth"
VAL_DATA_PATH = "/mnt/object/val/val_transformed.csv"
IMP_FEATURES_ONLY = "/mnt/object/val/template_test/non_top_downscaled.csv"
NO_IMP_FEATURES = "/mnt/object/val/template_test/top_downscaled.csv"
NOISY_NON_IMP = "/mnt/object/val/template_test/non_top_upscaled.csv"
NOISY_IMP_FEATURES = "/mnt/object/val/template_test/top_upscaled.csv"
RANDOM_NOISE_PATH = "/mnt/object/val/template_test/random_noise.csv"

@pytest.fixture(scope="session")
def model():
    return joblib.load(MODEL_PATH)

@pytest.fixture(scope="session")
def val_data():
    return pd.read_csv(VAL_DATA_PATH)

@pytest.fixture(scope="session")
def imp_features_only_data():
    return pd.read_csv(IMP_FEATURES_ONLY)

@pytest.fixture(scope="session")
def no_imp_features_data():
    return pd.read_csv(NO_IMP_FEATURES)

@pytest.fixture(scope="session")
def noisy_non_imp_data():
    return pd.read_csv(NOISY_NON_IMP)

@pytest.fixture(scope="session")
def noisy_imp_features_data():
    return pd.read_csv(NOISY_IMP_FEATURES)

@pytest.fixture(scope="session")
def random_noise_data():
    return pd.read_csv(RANDOM_NOISE_PATH)


@pytest.fixture(scope="session")
def val_predictions(model, val_data):
    X, y = transform(val_data)
    y_pred, probs = predict(model, X)
    return {
        "y_true": y,
        "y_pred": y_pred,
        "probs": probs
    }

@pytest.fixture(scope="session")
def imp_features_only_predictions(model, imp_features_only_data):
    X, y = transform(imp_features_only_data)
    y_pred, probs = predict(model, X)
    return {
        "y_true": y,
        "y_pred": y_pred,
        "probs": probs
    }

@pytest.fixture(scope="session")
def no_imp_features_predictions(model, no_imp_features_data):
    X, y = transform(no_imp_features_data)
    y_pred, probs = predict(model, X)
    return {
        "y_true": y,
        "y_pred": y_pred,
        "probs": probs
    }

@pytest.fixture(scope="session")
def noisy_non_imp_predictions(model, noisy_non_imp_data):
    X, y = transform(noisy_non_imp_data)
    y_pred, probs = predict(model, X)
    return {
        "y_true": y,
        "y_pred": y_pred,
        "probs": probs
    }

@pytest.fixture(scope="session")
def noisy_imp_features_predictions(model, noisy_imp_features_data):
    X, y = transform(noisy_imp_features_data)
    y_pred, probs = predict(model, X)
    return {
        "y_true": y,
        "y_pred": y_pred,
        "probs": probs
    }

@pytest.fixture(scope="session")
def random_noise_predictions(model, random_noise_data):
    X, y = transform(random_noise_data)
    y_pred, probs = predict(model, X)
    return {
        "y_true": y,
        "y_pred": y_pred,
        "probs": probs
    }

