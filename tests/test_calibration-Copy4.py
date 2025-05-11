from sklearn.metrics import f1_score
import numpy as np

def test_cal_f1(cal_predictions):
    y_true = cal_predictions["y_true"]
    y_pred = cal_predictions["y_pred"]
    f1 = f1_score(y_true, y_pred, average="macro")
    assert f1 >= 0.70, f"Cal F1 macro too low: {f1:.3f}"

def test_cal_class_confidence(cal_predictions):
    pred_classes = cal_predictions["probs"].argmax(axis=1)
    confidences = cal_predictions["probs"].max(axis=1)
    n_classes = cal_predictions["probs"].shape[1]
    for c in range(n_classes):
        mask = pred_classes == c
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        assert avg_conf >= 0.4, f"Cal class {c} confidence too low: {avg_conf:.3f}"
