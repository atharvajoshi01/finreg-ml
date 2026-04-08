"""Data drift detection for post-deployment monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DriftResult:
    """Drift detection result for a single feature."""

    feature: str
    statistic: float
    p_value: float
    drifted: bool
    method: str
    reference_mean: float
    current_mean: float
    shift: float


@dataclass
class DriftReport:
    """Drift detection report across all features."""

    n_features: int
    n_drifted: int
    results: List[DriftResult] = field(default_factory=list)
    threshold: float = 0.05

    @property
    def has_drift(self) -> bool:
        return self.n_drifted > 0

    @property
    def drift_fraction(self) -> float:
        return self.n_drifted / self.n_features if self.n_features > 0 else 0.0

    def drifted_features(self) -> List[str]:
        return [r.feature for r in self.results if r.drifted]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_drift": self.has_drift,
            "n_features": self.n_features,
            "n_drifted": self.n_drifted,
            "drift_fraction": self.drift_fraction,
            "threshold": self.threshold,
            "drifted_features": self.drifted_features(),
            "details": [
                {
                    "feature": r.feature,
                    "statistic": r.statistic,
                    "p_value": r.p_value,
                    "drifted": r.drifted,
                    "method": r.method,
                    "reference_mean": r.reference_mean,
                    "current_mean": r.current_mean,
                    "shift": r.shift,
                }
                for r in self.results
            ],
        }


def _ks_test(reference: np.ndarray, current: np.ndarray) -> tuple:
    """Two-sample Kolmogorov-Smirnov test."""
    from scipy import stats

    stat, p_value = stats.ks_2samp(reference, current)
    return float(stat), float(p_value)


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index.

    PSI < 0.1: no significant shift
    PSI 0.1-0.25: moderate shift
    PSI > 0.25: significant shift
    """
    eps = 1e-4
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1,
    )
    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

    ref_counts = np.clip(ref_counts, eps, None)
    cur_counts = np.clip(cur_counts, eps, None)

    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    method: str = "ks",
    threshold: float = 0.05,
    features: Optional[List[str]] = None,
) -> DriftReport:
    """Detect data drift between reference and current datasets.

    Args:
        reference: Training/reference data.
        current: New/production data to check for drift.
        method: Detection method — "ks" (Kolmogorov-Smirnov) or "psi"
            (Population Stability Index).
        threshold: p-value threshold for KS test (default 0.05),
            or PSI threshold (default 0.25 if method="psi").
        features: Specific features to check. If None, checks all
            numeric columns present in both DataFrames.

    Returns:
        DriftReport with per-feature drift results.
    """
    if method == "psi" and threshold == 0.05:
        threshold = 0.25  # PSI default

    if features is None:
        ref_numeric = reference.select_dtypes(include=[np.number]).columns
        cur_numeric = current.select_dtypes(include=[np.number]).columns
        features = list(set(ref_numeric) & set(cur_numeric))

    results: List[DriftResult] = []

    for feat in sorted(features):
        ref_vals = reference[feat].dropna().values
        cur_vals = current[feat].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        ref_mean = float(np.mean(ref_vals))
        cur_mean = float(np.mean(cur_vals))
        shift = cur_mean - ref_mean

        if method == "ks":
            statistic, p_value = _ks_test(ref_vals, cur_vals)
            drifted = p_value < threshold
        elif method == "psi":
            statistic = _psi(ref_vals, cur_vals)
            p_value = 0.0  # PSI doesn't produce a p-value
            drifted = statistic > threshold
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ks' or 'psi'.")

        results.append(DriftResult(
            feature=feat,
            statistic=statistic,
            p_value=p_value,
            drifted=drifted,
            method=method,
            reference_mean=ref_mean,
            current_mean=cur_mean,
            shift=shift,
        ))

    n_drifted = sum(1 for r in results if r.drifted)

    return DriftReport(
        n_features=len(results),
        n_drifted=n_drifted,
        results=results,
        threshold=threshold,
    )
