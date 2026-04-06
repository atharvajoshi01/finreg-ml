"""Model explainability using SHAP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class ExplanationReport:
    """Container for SHAP-based model explanations."""

    shap_values: np.ndarray
    feature_names: List[str]
    base_value: float
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_importance:
            mean_abs = np.abs(self.shap_values).mean(axis=0)
            self.feature_importance = {
                name: float(val)
                for name, val in sorted(
                    zip(self.feature_names, mean_abs), key=lambda x: -x[1]
                )
            }

    def top_features(self, n: int = 10) -> Dict[str, float]:
        return dict(list(self.feature_importance.items())[:n])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_value": self.base_value,
            "feature_importance": self.feature_importance,
            "n_samples": self.shap_values.shape[0],
            "n_features": self.shap_values.shape[1],
        }


def compute_explanations(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> ExplanationReport:
    """Compute SHAP explanations for a fitted model.

    Args:
        model: A fitted scikit-learn compatible estimator.
        X: Feature matrix to explain.
        max_samples: Max samples for SHAP background dataset.

    Returns:
        ExplanationReport with SHAP values and feature importance.
    """
    import shap

    sample = X if len(X) <= max_samples else X.sample(max_samples, random_state=42)

    # Use appropriate explainer based on model type
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)

    shap_values = explainer.shap_values(sample)

    # For binary classification, shap_values may be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    base_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else float(explainer.expected_value)
    )

    return ExplanationReport(
        shap_values=np.array(shap_values),
        feature_names=list(X.columns),
        base_value=float(base_value),
    )
