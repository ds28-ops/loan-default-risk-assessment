from sklearn.metrics import f1_score

def test_val_overall_f1(val_predictions):
    y_true = val_predictions["y_true"]
    y_pred = val_predictions["y_pred"]
    f1 = f1_score(y_true, y_pred, average="macro")
    assert f1 >= 0.75, f"F1 macro too low: {f1:.3f}"
