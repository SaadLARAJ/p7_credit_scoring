from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def business_cost_score(y_true, y_pred, fn_cost: float = 10.0, fp_cost: float = 1.0) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fn * fn_cost + fp * fp_cost
    max_cost = (fn + tp) * fn_cost + (fp + tn) * fp_cost
    return 1 - total_cost / max_cost


def optimal_threshold(y_true, y_proba, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 50)
    best_threshold, best_score = 0.5, -np.inf
    for thr in grid:
        preds = (y_proba >= thr).astype(int)
        score = business_cost_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = thr
    return best_threshold, best_score
