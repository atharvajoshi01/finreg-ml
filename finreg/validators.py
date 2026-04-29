"""Input validation for training data quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class ValidationIssue:
    """A single data quality issue."""

    feature: str
    check: str
    severity: str  # "error", "warning"
    detail: str


@dataclass
class ValidationReport:
    """Data quality validation report."""

    n_features: int
    n_samples: int
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ValidationReport({status}, n_samples={self.n_samples}, "
            f"n_features={self.n_features}, errors={self.n_errors}, "
            f"warnings={self.n_warnings})"
        )

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "n_features": self.n_features,
            "n_samples": self.n_samples,
            "errors": self.n_errors,
            "warnings": self.n_warnings,
            "issues": [
                {
                    "feature": i.feature,
                    "check": i.check,
                    "severity": i.severity,
                    "detail": i.detail,
                }
                for i in self.issues
            ],
        }


def validate_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    max_missing_pct: float = 0.1,
    max_constant_pct: float = 0.95,
    check_target_balance: bool = True,
    min_samples: int = 50,
) -> ValidationReport:
    """Validate training data quality before model fitting.

    Checks for common data quality issues that can silently degrade
    model performance or violate compliance requirements.

    Args:
        X: Training feature matrix.
        y: Training target.
        max_missing_pct: Maximum allowed missing value fraction per feature.
        max_constant_pct: Flag features where one value dominates above this threshold.
        check_target_balance: Whether to check for class imbalance.
        min_samples: Minimum number of samples required.

    Returns:
        ValidationReport with issues found.
    """
    issues: List[ValidationIssue] = []

    # Sample count check
    if len(X) < min_samples:
        issues.append(ValidationIssue(
            feature="_dataset",
            check="min_samples",
            severity="error",
            detail=f"Only {len(X)} samples, minimum required is {min_samples}.",
        ))

    # Per-feature checks
    for col in X.columns:
        series = X[col]

        # Missing values
        missing_pct = series.isna().mean()
        if missing_pct > max_missing_pct:
            issues.append(ValidationIssue(
                feature=col,
                check="missing_values",
                severity="error" if missing_pct > 0.5 else "warning",
                detail=f"{missing_pct:.1%} missing values (threshold: {max_missing_pct:.1%}).",
            ))

        # Constant or near-constant features
        if series.nunique() <= 1:
            issues.append(ValidationIssue(
                feature=col,
                check="constant_feature",
                severity="warning",
                detail="Feature has zero variance (constant).",
            ))
        elif pd.api.types.is_numeric_dtype(series):
            mode_pct = series.value_counts(normalize=True).iloc[0]
            if mode_pct > max_constant_pct:
                issues.append(ValidationIssue(
                    feature=col,
                    check="near_constant",
                    severity="warning",
                    detail=f"Single value dominates {mode_pct:.1%} of data.",
                ))

        # Infinite values
        if pd.api.types.is_numeric_dtype(series):
            inf_count = np.isinf(series.dropna()).sum()
            if inf_count > 0:
                issues.append(ValidationIssue(
                    feature=col,
                    check="infinite_values",
                    severity="error",
                    detail=f"{inf_count} infinite values found.",
                ))

    # Target checks
    if check_target_balance and y.nunique() == 2:
        minority_pct = y.value_counts(normalize=True).min()
        if minority_pct < 0.05:
            issues.append(ValidationIssue(
                feature="_target",
                check="class_imbalance",
                severity="warning",
                detail=f"Minority class is {minority_pct:.1%} of data. "
                       f"Model may not learn minority class well.",
            ))

    # Feature-target leakage check (perfect correlation)
    for col in X.select_dtypes(include=[np.number]).columns:
        corr = X[col].corr(y.astype(float))
        if abs(corr) > 0.99:
            issues.append(ValidationIssue(
                feature=col,
                check="target_leakage",
                severity="error",
                detail=f"Near-perfect correlation ({corr:.4f}) with target. "
                       f"Possible data leakage.",
            ))

    return ValidationReport(
        n_features=len(X.columns),
        n_samples=len(X),
        issues=issues,
    )
