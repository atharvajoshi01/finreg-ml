"""Tests for data validation module."""

import numpy as np
import pandas as pd

from finreg.validators import validate_training_data


class TestValidation:
    def test_clean_data_passes(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
        y = pd.Series(rng.randint(0, 2, 200))
        report = validate_training_data(X, y)
        assert report.passed
        assert report.n_errors == 0

    def test_too_few_samples(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        report = validate_training_data(X, y, min_samples=50)
        assert not report.passed
        assert any(i.check == "min_samples" for i in report.issues)

    def test_missing_values(self):
        X = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan, 5] * 20})
        y = pd.Series([0, 1] * 50)
        report = validate_training_data(X, y)
        assert any(i.check == "missing_values" for i in report.issues)

    def test_constant_feature(self):
        X = pd.DataFrame({"a": [1] * 100, "b": np.random.randn(100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        report = validate_training_data(X, y)
        assert any(i.check == "constant_feature" for i in report.issues)

    def test_infinite_values(self):
        X = pd.DataFrame({"a": [1, 2, np.inf, 4, 5] * 20})
        y = pd.Series([0, 1] * 50)
        report = validate_training_data(X, y)
        assert any(i.check == "infinite_values" for i in report.issues)

    def test_class_imbalance(self):
        X = pd.DataFrame({"a": np.random.randn(200)})
        y = pd.Series([0] * 195 + [1] * 5)
        report = validate_training_data(X, y)
        assert any(i.check == "class_imbalance" for i in report.issues)

    def test_target_leakage(self):
        rng = np.random.RandomState(42)
        y = pd.Series(rng.randint(0, 2, 200))
        X = pd.DataFrame({"leaky": y.astype(float) + rng.normal(0, 0.001, 200)})
        report = validate_training_data(X, y)
        assert any(i.check == "target_leakage" for i in report.issues)

    def test_report_to_dict(self):
        X = pd.DataFrame({"a": np.random.randn(100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        report = validate_training_data(X, y)
        d = report.to_dict()
        assert "passed" in d
        assert "issues" in d
