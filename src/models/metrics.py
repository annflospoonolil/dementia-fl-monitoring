import numpy as np

try:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def _numpy_binary_metrics(y: np.ndarray, pred: np.ndarray, p: np.ndarray):
    """Pure NumPy fallback for binary classification metrics when scikit-learn is not installed."""
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    tpr = rec
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = 1.0 - fpr
    balanced_acc = (tpr + tnr) / 2.0

    # Approximate PR-AUC using trapezoidal rule if multiple thresholds exist
    order = np.argsort(-p)
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted == 1)
    cum_fp = np.cumsum(y_sorted == 0)
    precisions = cum_tp / (cum_tp + cum_fp)
    recalls = cum_tp / np.maximum(cum_tp[-1], 1)
    # Simple trapezoidal PR-AUC
    pr_auc = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])) if len(y) > 1 else 0.0

    return float(prec), float(rec), float(f1), float(pr_auc), float(balanced_acc)


def binary_metrics(y_true, probabilities, threshold=0.5):
    y = np.asarray(y_true).astype(int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) == 0:
        raise ValueError("Cannot evaluate empty data")
    if len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Labels/probabilities mismatch or non-finite probabilities")
    pred = (p >= threshold).astype(int)
    one_class = np.unique(y).size < 2

    if _SKLEARN_AVAILABLE:
        return {
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "pr_auc": float(average_precision_score(y, p)) if not one_class else 0.0,
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if not one_class else 0.0,
            "single_class_eval": bool(one_class),
            "num_samples": int(len(y)),
        }
    else:
        prec, rec, f1, pr_auc, balanced_acc = _numpy_binary_metrics(y, pred, p)
        return {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "pr_auc": pr_auc if not one_class else 0.0,
            "balanced_accuracy": balanced_acc if not one_class else 0.0,
            "single_class_eval": bool(one_class),
            "num_samples": int(len(y)),
        }


def reconstruction_metrics(x_true, x_pred):
    """Compute summary reconstruction errors (MSE, MAE, RMSE) across feature vectors."""
    yt = np.asarray(x_true, dtype=np.float32)
    yp = np.asarray(x_pred, dtype=np.float32)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot evaluate empty features")
    diff = yt - yp
    sample_mse = np.mean(diff ** 2, axis=-1)
    sample_mae = np.mean(np.abs(diff), axis=-1)
    return {
        "mean_mse": float(np.mean(sample_mse)),
        "median_mse": float(np.median(sample_mse)),
        "std_mse": float(np.std(sample_mse)),
        "mean_mae": float(np.mean(sample_mae)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "num_samples": int(len(yt)),
    }
