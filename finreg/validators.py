"""Input validation for training data quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Union

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


# ---------------------------------------------------------------------------
# Temporal integrity
# ---------------------------------------------------------------------------

@dataclass
class TemporalLeak:
    """A single point-in-time correctness violation."""

    feature: str
    n_offending: int
    max_lag: Optional[pd.Timedelta]
    detail: str


@dataclass
class TemporalIntegrityReport:
    """Result of a point-in-time correctness check on a training dataset.

    A feature is "temporally clean" if, for every row, its value's as-of
    timestamp is no later than the row's prediction timestamp. This catches a
    very common class of silent leakage in credit, insurance, and any model
    where the data store keeps moving even after the decision is locked in:
    the model accidentally trains on information that wasn't available at
    decision time but is available at fit time.
    """

    n_samples: int
    n_features_checked: int
    leaks: List[TemporalLeak] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.leaks) == 0

    @property
    def n_leaking_features(self) -> int:
        return len(self.leaks)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"TemporalIntegrityReport({status}, n_samples={self.n_samples}, "
            f"n_features_checked={self.n_features_checked}, "
            f"leaking_features={self.n_leaking_features})"
        )

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "n_samples": self.n_samples,
            "n_features_checked": self.n_features_checked,
            "n_leaking_features": self.n_leaking_features,
            "leaks": [
                {
                    "feature": leak.feature,
                    "n_offending": leak.n_offending,
                    "max_lag_seconds": (
                        leak.max_lag.total_seconds() if leak.max_lag is not None else None
                    ),
                    "detail": leak.detail,
                }
                for leak in self.leaks
            ],
        }


def _coerce_to_timestamp(values, name: str) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not parse {name} as timestamps. Pass datetime64-compatible values."
        ) from exc


def validate_temporal_integrity(
    X: pd.DataFrame,
    prediction_timestamps: Union[pd.Series, pd.Index],
    feature_timestamps: Optional[Union[pd.DataFrame, Mapping[str, pd.Series]]] = None,
    feature_timestamp_columns: Optional[Mapping[str, str]] = None,
    tolerance: pd.Timedelta = pd.Timedelta(0),
) -> TemporalIntegrityReport:
    """Check point-in-time correctness of training features.

    For each row, every feature's as-of timestamp must be no later than the
    prediction timestamp (plus an optional tolerance). Any violation is a
    candidate leak: the model is being trained on information that was not
    yet available at the decision point but is present at fit time.

    Two ways to declare feature timestamps, pick whichever is closer to the
    shape of your feature store:

    * ``feature_timestamps``: a DataFrame the same shape as ``X`` (or a mapping
      of feature name to ``pd.Series``) whose values are the as-of timestamps
      for the corresponding cell in ``X``.
    * ``feature_timestamp_columns``: a mapping ``{feature: timestamp_column}``
      where ``timestamp_column`` is a column inside ``X`` whose value is the
      as-of timestamp for ``feature``.

    Features not covered by either mapping are skipped (they're assumed to be
    pre-validated by an earlier stage). The skipped features are not counted
    in ``n_features_checked``.

    Args:
        X: Training feature matrix.
        prediction_timestamps: One timestamp per row of ``X``, the moment at
            which the prediction would have been made.
        feature_timestamps: Per-cell timestamps. Either a DataFrame with the
            same index and columns as ``X`` (a subset of columns is fine), or
            a mapping ``{feature: Series}`` indexed like ``X``.
        feature_timestamp_columns: Per-column timestamp mapping.
        tolerance: How much later than the prediction timestamp a feature is
            allowed to be before it counts as a leak. Defaults to zero
            (strict). Useful for tolerating small clock-skew artifacts.

    Returns:
        TemporalIntegrityReport. ``passed`` is True iff no leaks were found.

    Raises:
        ValueError: if neither timestamp source is provided, if the prediction
            timestamps cannot be parsed, or if the per-cell DataFrame's index
            does not match ``X``.
    """
    if feature_timestamps is None and feature_timestamp_columns is None:
        raise ValueError(
            "Provide either feature_timestamps (per-cell) or "
            "feature_timestamp_columns (per-column) so the validator knows "
            "which features carry as-of timestamps."
        )

    pred_ts = _coerce_to_timestamp(prediction_timestamps, "prediction_timestamps")
    if len(pred_ts) != len(X):
        raise ValueError(
            f"prediction_timestamps has length {len(pred_ts)} but X has "
            f"{len(X)} rows."
        )
    pred_ts.index = X.index

    leaks: List[TemporalLeak] = []
    checked: set = set()

    # Per-column mode
    if feature_timestamp_columns:
        for feature, ts_col in feature_timestamp_columns.items():
            if feature not in X.columns:
                continue
            if ts_col not in X.columns:
                raise ValueError(
                    f"feature_timestamp_columns references column '{ts_col}' "
                    f"which is not present in X."
                )
            ts = _coerce_to_timestamp(X[ts_col], f"X[{ts_col!r}]")
            ts.index = X.index
            offending_mask = ts > (pred_ts + tolerance)
            n_off = int(offending_mask.sum())
            if n_off > 0:
                lag = (ts[offending_mask] - pred_ts[offending_mask]).max()
                leaks.append(
                    TemporalLeak(
                        feature=feature,
                        n_offending=n_off,
                        max_lag=lag,
                        detail=(
                            f"{n_off} of {len(X)} rows have {feature} as-of "
                            f"timestamp later than the prediction timestamp "
                            f"(max lag: {lag})."
                        ),
                    )
                )
            checked.add(feature)

    # Per-cell mode
    if feature_timestamps is not None:
        if isinstance(feature_timestamps, pd.DataFrame):
            ts_frame = feature_timestamps
        else:
            ts_frame = pd.DataFrame(dict(feature_timestamps))
        if not ts_frame.index.equals(X.index):
            try:
                ts_frame = ts_frame.reindex(X.index)
            except Exception as exc:
                raise ValueError(
                    "feature_timestamps could not be aligned to X.index. "
                    "Pass an index that matches X."
                ) from exc
        for feature in ts_frame.columns:
            if feature not in X.columns:
                continue
            ts = _coerce_to_timestamp(ts_frame[feature], f"feature_timestamps[{feature!r}]")
            ts.index = X.index
            offending_mask = ts > (pred_ts + tolerance)
            n_off = int(offending_mask.sum())
            if n_off > 0:
                lag = (ts[offending_mask] - pred_ts[offending_mask]).max()
                leaks.append(
                    TemporalLeak(
                        feature=feature,
                        n_offending=n_off,
                        max_lag=lag,
                        detail=(
                            f"{n_off} of {len(X)} rows have {feature} as-of "
                            f"timestamp later than the prediction timestamp "
                            f"(max lag: {lag})."
                        ),
                    )
                )
            checked.add(feature)

    return TemporalIntegrityReport(
        n_samples=len(X),
        n_features_checked=len(checked),
        leaks=leaks,
    )
