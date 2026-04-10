"""Threshold tuning for classification models with fairness constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ThresholdResult:
    """Result of threshold optimization."""

    optimal_threshold: float
    metric_value: float
    metric_name: str
    fairness_constraint: Optional[str] = None
    thresholds_evaluated: int = 0
    all_results: List[Dict] = field(default_factory=list)


def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    if metric == "accuracy":
        return (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    elif metric == "precision":
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    elif metric == "recall":
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    elif metric == "f1":
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    elif metric == "balanced_accuracy":
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return (tpr + tnr) / 2
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _disparate_impact(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    groups = np.unique(sensitive)
    rates = []
    for g in groups:
        mask = sensitive == g
        if mask.sum() > 0:
            rates.append(y_pred[mask].mean())
    if not rates or max(rates) == 0:
        return 0.0
    return min(rates) / max(rates)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    sensitive: Optional[pd.Series] = None,
    min_disparate_impact: float = 0.8,
    n_thresholds: int = 100,
) -> ThresholdResult:
    """Find the optimal classification threshold with optional fairness constraint.

    Sweeps thresholds and selects the one that maximizes the target metric
    while satisfying the disparate impact constraint (four-fifths rule).

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        metric: Metric to optimize.
        sensitive: Optional protected attribute for fairness constraint.
        min_disparate_impact: Minimum disparate impact ratio required.
        n_thresholds: Number of thresholds to evaluate.

    Returns:
        ThresholdResult with optimal threshold and evaluation details.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    best_threshold = 0.5
    best_value = -1.0
    all_results = []

    sensitive_arr = np.asarray(sensitive) if sensitive is not None else None
    constraint_name = f"disparate_impact >= {min_disparate_impact}" if sensitive is not None else None

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        value = _compute_metric(y_true, y_pred, metric)

        passes_fairness = True
        di = None
        if sensitive_arr is not None:
            di = _disparate_impact(y_pred, sensitive_arr)
            if di < min_disparate_impact:
                passes_fairness = False

        result = {"threshold": float(t), metric: float(value), "passes_fairness": passes_fairness}
        if di is not None:
            result["disparate_impact"] = float(di)
        all_results.append(result)

        if passes_fairness and value > best_value:
            best_value = value
            best_threshold = float(t)

    return ThresholdResult(
        optimal_threshold=best_threshold,
        metric_value=best_value,
        metric_name=metric,
        fairness_constraint=constraint_name,
        thresholds_evaluated=len(thresholds),
        all_results=all_results,
    )
