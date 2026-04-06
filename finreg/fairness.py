"""Fairness metrics across protected attributes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class GroupMetrics:
    """Metrics for a single group within a protected attribute."""

    group_name: str
    size: int
    positive_rate: float
    true_positive_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None


@dataclass
class FairnessReport:
    """Fairness assessment across protected attributes."""

    attribute: str
    groups: List[GroupMetrics]
    demographic_parity_diff: float
    equalized_odds_diff: Optional[float] = None
    disparate_impact_ratio: Optional[float] = None
    passes_four_fifths_rule: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attribute": self.attribute,
            "demographic_parity_diff": self.demographic_parity_diff,
            "equalized_odds_diff": self.equalized_odds_diff,
            "disparate_impact_ratio": self.disparate_impact_ratio,
            "passes_four_fifths_rule": self.passes_four_fifths_rule,
            "groups": [
                {
                    "group": g.group_name,
                    "size": g.size,
                    "positive_rate": g.positive_rate,
                    "tpr": g.true_positive_rate,
                    "fpr": g.false_positive_rate,
                }
                for g in self.groups
            ],
        }


def fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: pd.Series,
) -> FairnessReport:
    """Compute fairness metrics for a protected attribute.

    Args:
        y_true: Ground truth labels (binary).
        y_pred: Predicted labels (binary).
        sensitive: Protected attribute values for each sample.

    Returns:
        FairnessReport with demographic parity, equalized odds,
        and disparate impact metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    groups_list: List[GroupMetrics] = []
    unique_groups = sorted(sensitive.unique())

    for group in unique_groups:
        mask = sensitive == group
        group_true = y_true[mask]
        group_pred = y_pred[mask]
        size = int(mask.sum())
        positive_rate = float(group_pred.mean()) if size > 0 else 0.0

        tpr = None
        fpr = None
        if size > 0:
            positives = group_true == 1
            negatives = group_true == 0
            if positives.sum() > 0:
                tpr = float(group_pred[positives].mean())
            if negatives.sum() > 0:
                fpr = float(group_pred[negatives].mean())

        groups_list.append(
            GroupMetrics(
                group_name=str(group),
                size=size,
                positive_rate=positive_rate,
                true_positive_rate=tpr,
                false_positive_rate=fpr,
            )
        )

    # Demographic parity: max difference in positive rates
    pos_rates = [g.positive_rate for g in groups_list]
    dp_diff = max(pos_rates) - min(pos_rates)

    # Equalized odds: max difference in TPR
    tprs = [g.true_positive_rate for g in groups_list if g.true_positive_rate is not None]
    eo_diff = (max(tprs) - min(tprs)) if len(tprs) >= 2 else None

    # Disparate impact ratio (four-fifths rule)
    di_ratio = None
    passes_rule = None
    if min(pos_rates) > 0:
        di_ratio = min(pos_rates) / max(pos_rates)
        passes_rule = di_ratio >= 0.8

    return FairnessReport(
        attribute=sensitive.name or "unknown",
        groups=groups_list,
        demographic_parity_diff=dp_diff,
        equalized_odds_diff=eo_diff,
        disparate_impact_ratio=di_ratio,
        passes_four_fifths_rule=passes_rule,
    )
