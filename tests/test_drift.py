"""Tests for drift detection module."""

import numpy as np
import pandas as pd
import pytest

from finreg.drift import detect_drift


@pytest.fixture
def stable_data():
    """Two datasets from the same distribution."""
    rng = np.random.RandomState(42)
    ref = pd.DataFrame({
        "income": rng.normal(50000, 10000, 1000),
        "debt_ratio": rng.uniform(0, 1, 1000),
        "score": rng.normal(700, 50, 1000),
    })
    cur = pd.DataFrame({
        "income": rng.normal(50000, 10000, 1000),
        "debt_ratio": rng.uniform(0, 1, 1000),
        "score": rng.normal(700, 50, 1000),
    })
    return ref, cur


@pytest.fixture
def drifted_data():
    """Two datasets where current has shifted distributions."""
    rng = np.random.RandomState(42)
    ref = pd.DataFrame({
        "income": rng.normal(50000, 10000, 1000),
        "debt_ratio": rng.uniform(0, 0.5, 1000),
        "score": rng.normal(700, 50, 1000),
    })
    cur = pd.DataFrame({
        "income": rng.normal(70000, 10000, 1000),  # shifted +20k
        "debt_ratio": rng.uniform(0.3, 1.0, 1000),  # shifted up
        "score": rng.normal(700, 50, 1000),  # stable
    })
    return ref, cur


class TestKSDrift:
    def test_no_drift_detected(self, stable_data):
        ref, cur = stable_data
        report = detect_drift(ref, cur, method="ks")
        assert not report.has_drift
        assert report.n_drifted == 0

    def test_drift_detected(self, drifted_data):
        ref, cur = drifted_data
        report = detect_drift(ref, cur, method="ks")
        assert report.has_drift
        assert report.n_drifted >= 2
        assert "income" in report.drifted_features()
        assert "debt_ratio" in report.drifted_features()
        assert "score" not in report.drifted_features()

    def test_specific_features(self, drifted_data):
        ref, cur = drifted_data
        report = detect_drift(ref, cur, method="ks", features=["income"])
        assert report.n_features == 1
        assert report.results[0].feature == "income"
        assert report.results[0].drifted


class TestPSIDrift:
    def test_no_drift_psi(self, stable_data):
        ref, cur = stable_data
        report = detect_drift(ref, cur, method="psi")
        assert not report.has_drift

    def test_drift_psi(self, drifted_data):
        ref, cur = drifted_data
        report = detect_drift(ref, cur, method="psi")
        assert report.has_drift
        assert "income" in report.drifted_features()


class TestDriftReport:
    def test_report_to_dict(self, drifted_data):
        ref, cur = drifted_data
        report = detect_drift(ref, cur, method="ks")
        d = report.to_dict()
        assert "has_drift" in d
        assert "drifted_features" in d
        assert "details" in d
        assert len(d["details"]) == 3

    def test_drift_fraction(self, drifted_data):
        ref, cur = drifted_data
        report = detect_drift(ref, cur, method="ks")
        assert 0 < report.drift_fraction <= 1.0

    def test_shift_direction(self, drifted_data):
        ref, cur = drifted_data
        report = detect_drift(ref, cur, method="ks")
        income_result = next(r for r in report.results if r.feature == "income")
        assert income_result.shift > 0  # income shifted upward

    def test_invalid_method(self, stable_data):
        ref, cur = stable_data
        with pytest.raises(ValueError, match="Unknown method"):
            detect_drift(ref, cur, method="invalid")
