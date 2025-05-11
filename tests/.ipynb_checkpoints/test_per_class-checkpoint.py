from sklearn.metrics import f1_score
import numpy as np

def test_val_per_class_f1(val_predictions):
    y_true = val_predictions["y_true"]
    y_pred = val_predictions["y_pred"]
    f1s = f1_score(y_true, y_pred, average=None)
    for i, f1 in enumerate(f1s):
        assert f1 >= 0.60, f"Class {i} F1 too low: {f1:.3f}"

def test_val_class_confidence(val_predictions):
    pred_classes = val_predictions["probs"].argmax(axis=1)
    confidences = val_predictions["probs"].max(axis=1)
    n_classes = val_predictions["probs"].shape[1]
    for c in range(n_classes):
        mask = pred_classes == c
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        assert avg_conf >= 0.5, f"Class {c} confidence too low: {avg_conf:.3f}"
