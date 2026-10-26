import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score

def binary_metrics(y_true, probabilities, threshold=0.5):
    y = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) == 0:
        raise ValueError("Cannot evaluate empty data")
    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Labels/probabilities mismatch or non-finite probabilities")
    pred = (p >= threshold).astype(int)
    one_class = np.unique(y).size < 2
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)) if not one_class else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if not one_class else 0.0,
        "single_class_eval": bool(one_class),
        "num_samples": int(len(y)),
    }
