"""Tests for data validation module."""

import numpy as np
import pandas as pd
import pytest

from finreg.validators import (
    validate_temporal_integrity,
    validate_training_data,
)


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

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_constant_feature(self):
        X = pd.DataFrame({"a": [1] * 100, "b": np.random.randn(100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        report = validate_training_data(X, y)
        assert any(i.check == "constant_feature" for i in report.issues)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
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


class TestTemporalIntegrity:
    """Point-in-time correctness is one of the most common silent failures in
    credit-style ML. These tests pin the contract that catches it."""

    def _build_panel(self, n=100, leak_rows=0, prediction_date="2024-06-01"):
        pred_ts = pd.Timestamp(prediction_date)
        rng = np.random.RandomState(42)
        balance = rng.normal(50000, 10000, n)
        # Most rows: balance as-of yesterday (clean).
        # Last `leak_rows` rows: balance as-of tomorrow (leak).
        as_of = [pred_ts - pd.Timedelta(days=1)] * (n - leak_rows) + (
            [pred_ts + pd.Timedelta(days=1)] * leak_rows
        )
        X = pd.DataFrame(
            {
                "annual_income": rng.normal(60000, 15000, n),
                "account_balance": balance,
                "balance_as_of": pd.to_datetime(as_of),
            }
        )
        y = pd.Series(rng.randint(0, 2, n))
        prediction_timestamps = pd.Series([pred_ts] * n, index=X.index)
        return X, y, prediction_timestamps

    def test_clean_panel_passes(self):
        X, _, pred_ts = self._build_panel(leak_rows=0)
        report = validate_temporal_integrity(
            X=X,
            prediction_timestamps=pred_ts,
            feature_timestamp_columns={"account_balance": "balance_as_of"},
        )
        assert report.passed
        assert report.n_features_checked == 1
        assert report.n_leaking_features == 0

    def test_leak_is_caught(self):
        X, _, pred_ts = self._build_panel(leak_rows=5)
        report = validate_temporal_integrity(
            X=X,
            prediction_timestamps=pred_ts,
            feature_timestamp_columns={"account_balance": "balance_as_of"},
        )
        assert not report.passed
        assert report.n_leaking_features == 1
        leak = report.leaks[0]
        assert leak.feature == "account_balance"
        assert leak.n_offending == 5
        assert leak.max_lag == pd.Timedelta(days=1)

    def test_tolerance_absorbs_small_clock_skew(self):
        X, _, pred_ts = self._build_panel(leak_rows=5)
        # Tolerate 2 days. The leaks are 1 day late, so they should pass.
        report = validate_temporal_integrity(
            X=X,
            prediction_timestamps=pred_ts,
            feature_timestamp_columns={"account_balance": "balance_as_of"},
            tolerance=pd.Timedelta(days=2),
        )
        assert report.passed

    def test_per_cell_timestamps(self):
        pred_ts = pd.Timestamp("2024-06-01")
        n = 50
        rng = np.random.RandomState(7)
        X = pd.DataFrame({"score": rng.normal(0, 1, n)})
        # Half the rows are clean, half leak by 3 days.
        ts_series = pd.Series(
            [pred_ts - pd.Timedelta(days=1)] * 25
            + [pred_ts + pd.Timedelta(days=3)] * 25
        )
        report = validate_temporal_integrity(
            X=X,
            prediction_timestamps=pd.Series([pred_ts] * n),
            feature_timestamps=pd.DataFrame({"score": ts_series}),
        )
        assert not report.passed
        assert report.leaks[0].n_offending == 25
        assert report.leaks[0].max_lag == pd.Timedelta(days=3)

    def test_missing_timestamp_inputs_raises(self):
        X = pd.DataFrame({"a": [1.0, 2.0]})
        pred = pd.Series([pd.Timestamp("2024-01-01")] * 2)
        with pytest.raises(ValueError, match="feature_timestamps"):
            validate_temporal_integrity(X=X, prediction_timestamps=pred)

    def test_mismatched_lengths_raises(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        pred = pd.Series([pd.Timestamp("2024-01-01")] * 2)  # wrong length
        with pytest.raises(ValueError, match="length"):
            validate_temporal_integrity(
                X=X,
                prediction_timestamps=pred,
                feature_timestamp_columns={"a": "a"},
            )

    def test_unknown_timestamp_column_raises(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        pred = pd.Series([pd.Timestamp("2024-01-01")] * 3)
        with pytest.raises(ValueError, match="not present"):
            validate_temporal_integrity(
                X=X,
                prediction_timestamps=pred,
                feature_timestamp_columns={"a": "does_not_exist"},
            )

    def test_report_repr_and_dict(self):
        X, _, pred_ts = self._build_panel(leak_rows=3)
        report = validate_temporal_integrity(
            X=X,
            prediction_timestamps=pred_ts,
            feature_timestamp_columns={"account_balance": "balance_as_of"},
        )
        rendered = repr(report)
        assert rendered.startswith("TemporalIntegrityReport(")
        assert "FAIL" in rendered
        d = report.to_dict()
        assert d["passed"] is False
        assert d["leaks"][0]["feature"] == "account_balance"
        assert d["leaks"][0]["max_lag_seconds"] == 86400


class TestTemporalIntegrityInGovernedModel:
    """The temporal check should compose with GovernedModel as a fit-time hook."""

    def _make_credit_panel(self, n=200, leak_rows=0, seed=0):
        rng = np.random.RandomState(seed)
        pred_ts = pd.Timestamp("2024-06-01")
        as_of_dates = [pred_ts - pd.Timedelta(days=1)] * (n - leak_rows) + (
            [pred_ts + pd.Timedelta(days=2)] * leak_rows
        )
        X = pd.DataFrame(
            {
                "income": rng.normal(60000, 15000, n),
                "balance": rng.normal(50000, 10000, n),
                "balance_as_of": pd.to_datetime(as_of_dates),
            }
        )
        y = pd.Series(rng.randint(0, 2, n))
        pred = pd.Series([pred_ts] * n)
        return X, y, pred

    def test_clean_panel_does_not_block_fit(self):
        from sklearn.linear_model import LogisticRegression

        from finreg import GovernedModel

        X, y, pred = self._make_credit_panel(leak_rows=0)
        model = GovernedModel(estimator=LogisticRegression(), risk_tier="high")
        model.fit(
            X,
            y,
            prediction_timestamps=pred,
            feature_timestamp_columns={"balance": "balance_as_of"},
        )
        assert model.temporal_report is not None
        assert model.temporal_report.passed
        # The timestamp column was stripped before fitting.
        assert "balance_as_of" not in model._feature_names

    def test_strict_mode_blocks_fit_on_leak(self):
        from sklearn.linear_model import LogisticRegression

        from finreg import GovernedModel

        X, y, pred = self._make_credit_panel(leak_rows=10)
        model = GovernedModel(estimator=LogisticRegression(), risk_tier="high")
        with pytest.raises(ValueError, match="Temporal integrity"):
            model.fit(
                X,
                y,
                prediction_timestamps=pred,
                feature_timestamp_columns={"balance": "balance_as_of"},
                strict_temporal=True,
            )

    def test_compliance_report_surfaces_temporal_check(self):
        from sklearn.linear_model import LogisticRegression

        from finreg import GovernedModel

        X, y, pred = self._make_credit_panel(leak_rows=0)
        model = GovernedModel(estimator=LogisticRegression(), risk_tier="high")
        model.fit(
            X,
            y,
            prediction_timestamps=pred,
            feature_timestamp_columns={"balance": "balance_as_of"},
        )
        report = model.compliance_report(has_data_governance=True)
        temporal_check = [
            c for c in report.checks if c.requirement == "Point-in-time correctness"
        ]
        assert len(temporal_check) == 1
        assert temporal_check[0].status == "pass"

    def test_compliance_warns_when_no_temporal_check(self):
        from sklearn.linear_model import LogisticRegression

        from finreg import GovernedModel

        X, y, _ = self._make_credit_panel(leak_rows=0)
        # No temporal check is run, so caller is responsible for not passing
        # the timestamp column to the estimator.
        X_features_only = X.drop(columns=["balance_as_of"])
        model = GovernedModel(estimator=LogisticRegression(), risk_tier="high")
        model.fit(X_features_only, y)
        report = model.compliance_report(has_data_governance=True)
        temporal_check = [
            c for c in report.checks if c.requirement == "Point-in-time correctness"
        ]
        assert len(temporal_check) == 1
        assert temporal_check[0].status == "warning"


class TestVersionConsistency:
    def test_init_matches_pyproject(self):
        """Ensure __version__ in __init__.py matches pyproject.toml."""
        import tomllib
        from pathlib import Path

        import finreg

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        assert finreg.__version__ == data["project"]["version"]
