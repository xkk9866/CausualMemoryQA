import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def preference_bias(cm, iteration=20):
    """Bradley-Terry preference bias, copied from EmoDynamiX `metrics.py`."""
    eps = 1e-6
    k = cm.shape[0]
    p = [1.0] * k
    for _ in range(iteration):
        for i in range(k):
            n = sum(cm[j, i] * p[j] / (p[i] + p[j] + eps) for j in range(k))
            d = sum(cm[i, j] / (p[i] + p[j] + eps) for j in range(k))
            p[i] = n / d
        p = [p[i] * k / sum(p) for i in range(k)]
    return (sum((p[i] - np.mean(p)) ** 2 for i in range(k)) / k) ** 0.5


def report(y, pred, k):
    y, pred = np.asarray(y), np.asarray(pred)
    cm = confusion_matrix(y, pred, labels=list(range(k)))
    return {
        "acc": 100 * accuracy_score(y, pred),
        "macro_f1": 100 * f1_score(y, pred, average="macro", labels=list(range(k))),
        "weighted_f1": 100 * f1_score(y, pred, average="weighted", labels=list(range(k))),
        "bias": preference_bias(cm),
    }
