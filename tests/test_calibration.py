from sklearn.metrics import f1_score
import numpy as np

def avg_class_conf(probs, pred_classes):
    n_classes = probs.shape[1]
    confs = {}
    for c in range(n_classes):
        mask = pred_classes == c
        if mask.sum() == 0:
            confs[c] = None
        else:
            confs[c] = probs[mask].max(axis=1).mean()
    return confs


def test_imp_features_only(imp_features_only_predictions):
    y_true = imp_features_only_predictions["y_true"]
    y_pred = imp_features_only_predictions["y_pred"]
    probs = imp_features_only_predictions["probs"]
    f1 = f1_score(y_true, y_pred, average="macro")
    assert f1 >= 0.10, f"F1 too low with only important features: {f1:.3f}"

    confs = avg_class_conf(probs, probs.argmax(axis=1))
    for c in confs:
        if confs[c] is not None:
            assert confs[c] >= 0.30, f"Confidence too low for class {c}: {confs[c]:.3f}"

def test_no_imp_features(no_imp_features_predictions):
    y_true = no_imp_features_predictions["y_true"]
    y_pred = no_imp_features_predictions["y_pred"]
    probs = no_imp_features_predictions["probs"]
    f1 = f1_score(y_true, y_pred, average="macro")
    assert f1 >= 0.15, f"F1 too low with no important features: {f1:.3f}"

    confs = avg_class_conf(probs, probs.argmax(axis=1))
    for c in confs:
        if confs[c] is not None:
            assert confs[c] >= 0.30, f"Confidence too low for class {c}: {confs[c]:.3f}"

def test_noisy_non_imp(noisy_non_imp_predictions):
    y_true = noisy_non_imp_predictions["y_true"]
    y_pred = noisy_non_imp_predictions["y_pred"]
    probs = noisy_non_imp_predictions["probs"]
    f1 = f1_score(y_true, y_pred, average="macro")
    assert f1 >= 0.10, f"F1 too low with noise in non-important features: {f1:.3f}"

    confs = avg_class_conf(probs, probs.argmax(axis=1))
    for c in confs:
        if confs[c] is not None:
            assert confs[c] >= 0.30, f"Confidence too low for class {c}: {confs[c]:.3f}"

def test_noisy_imp_features(noisy_imp_features_predictions):
    y_true = noisy_imp_features_predictions["y_true"]
    y_pred = noisy_imp_features_predictions["y_pred"]
    probs = noisy_imp_features_predictions["probs"]
    f1 = f1_score(y_true, y_pred, average="macro")
    assert f1 <= 0.10, f"F1 unexpectedly high with noise in important features: {f1:.3f}"

    confs = avg_class_conf(probs, probs.argmax(axis=1))
    for c in confs:
        if confs[c] is not None:
            assert confs[c] <= 0.30, f"Confidence too high for class {c} under noisy-imp: {confs[c]:.3f}"

def test_random_noise(random_noise_predictions):
    y_true = random_noise_predictions["y_true"]
    y_pred = random_noise_predictions["y_pred"]
    probs = random_noise_predictions["probs"]
    f1 = f1_score(y_true, y_pred, average="macro")
    assert f1 <= 0.1, f"F1 unexpectedly high on random noise: {f1:.3f}"

    confs = avg_class_conf(probs, probs.argmax(axis=1))
    for c in confs:
        if confs[c] is not None:
            assert confs[c] <= 0.35, f"Confidence too high for class {c} under random noise: {confs[c]:.3f}"
