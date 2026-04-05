"""Tests for GovernedModel pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from finreg import GovernedModel
from finreg.compliance import RiskTier

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

needs_shap = pytest.mark.skipif(not HAS_SHAP, reason="shap not importable")


@pytest.fixture
def credit_data():
    """Synthetic credit scoring dataset."""
    rng = np.random.RandomState(42)
    n = 500
    X = pd.DataFrame({
        "income": rng.normal(50000, 15000, n),
        "debt_ratio": rng.uniform(0, 1, n),
        "credit_history_months": rng.randint(6, 240, n),
        "age_group": rng.choice(["18-30", "31-50", "51+"], n),
        "gender": rng.choice(["M", "F"], n),
    })
    y = pd.Series(
        (X["income"] > 45000).astype(int) & (X["debt_ratio"] < 0.5).astype(int),
        name="approved",
    )
    return X, y


@pytest.fixture
def model(credit_data):
    X, y = credit_data
    m = GovernedModel(
        estimator=LogisticRegression(max_iter=1000),
        protected_attributes=["age_group", "gender"],
        risk_tier="high",
        model_name="TestCreditScorer",
    )
    # Drop non-numeric columns for logistic regression
    X_numeric = X.drop(columns=["age_group", "gender"])
    m.fit(X_numeric, y)
    return m, X, y


class TestFit:
    def test_fit_logs_training(self, credit_data):
        X, y = credit_data
        X_numeric = X.drop(columns=["age_group", "gender"])
        m = GovernedModel(
            estimator=LogisticRegression(max_iter=1000),
            risk_tier="high",
        )
        m.fit(X_numeric, y)
        assert m._is_fitted
        assert len(m.audit_log) >= 2  # training_started + training_completed

    def test_fit_requires_dataframe(self, credit_data):
        X, y = credit_data
        m = GovernedModel(estimator=LogisticRegression())
        with pytest.raises(TypeError, match="DataFrame"):
            m.fit(X.values, y)

    def test_fit_captures_hyperparameters(self, credit_data):
        X, y = credit_data
        X_numeric = X.drop(columns=["age_group", "gender"])
        m = GovernedModel(estimator=LogisticRegression(C=0.5, max_iter=200))
        m.fit(X_numeric, y)
        card = m.model_card()
        assert card.hyperparameters["C"] == 0.5


class TestPredict:
    def test_predict_returns_array(self, model):
        m, X, y = model
        X_numeric = X.drop(columns=["age_group", "gender"])
        preds = m.predict(X_numeric[:10])
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 10

    def test_predict_before_fit_raises(self):
        m = GovernedModel(estimator=LogisticRegression())
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict(pd.DataFrame({"x": [1, 2]}))


class TestEvaluate:
    def test_evaluate_returns_metrics(self, model):
        m, X, y = model
        X_numeric = X.drop(columns=["age_group", "gender"])
        metrics = m.evaluate(X_numeric, y)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())


class TestExplain:
    @needs_shap
    def test_explain_returns_report(self, model):
        m, X, y = model
        X_numeric = X.drop(columns=["age_group", "gender"])
        report = m.explain(X_numeric[:50])
        assert report.shap_values.shape[0] == 50
        assert len(report.feature_importance) > 0
        assert report.base_value is not None


class TestFairness:
    def test_fairness_report_returns_list(self, model):
        m, X, y = model
        reports = m.fairness_report(X, y)
        assert len(reports) == 2  # age_group and gender
        for r in reports:
            assert r.demographic_parity_diff >= 0
            assert len(r.groups) > 0

    def test_fairness_no_attributes_raises(self, credit_data):
        X, y = credit_data
        X_numeric = X.drop(columns=["age_group", "gender"])
        m = GovernedModel(estimator=LogisticRegression(max_iter=1000))
        m.fit(X_numeric, y)
        with pytest.raises(ValueError, match="No protected attributes"):
            m.fairness_report(X_numeric, y)


class TestCompliance:
    def test_compliance_high_risk_no_docs(self, credit_data):
        X, y = credit_data
        X_numeric = X.drop(columns=["age_group", "gender"])
        m = GovernedModel(
            estimator=LogisticRegression(max_iter=1000),
            risk_tier="high",
        )
        m.fit(X_numeric, y)
        report = m.compliance_report()
        assert report.risk_tier == RiskTier.HIGH
        assert report.failed > 0
        assert not report.compliant

    @needs_shap
    def test_compliance_after_full_pipeline(self, model):
        m, X, y = model
        X_numeric = X.drop(columns=["age_group", "gender"])
        m.evaluate(X_numeric, y)
        m.explain(X_numeric[:50])
        m.fairness_report(X, y)
        report = m.compliance_report(has_human_oversight=True, has_data_governance=True)
        assert report.compliant
        assert report.failed == 0

    def test_compliance_minimal_risk(self, credit_data):
        X, y = credit_data
        X_numeric = X.drop(columns=["age_group", "gender"])
        m = GovernedModel(
            estimator=LogisticRegression(max_iter=1000),
            risk_tier="minimal",
        )
        m.fit(X_numeric, y)
        report = m.compliance_report()
        # Minimal risk has no hard requirements
        assert report.failed == 0


class TestModelCard:
    def test_model_card_generated_on_fit(self, model):
        m, X, y = model
        card = m.model_card()
        assert card.model_name == "TestCreditScorer"
        assert card.n_training_samples == 500
        assert len(card.features_used) > 0
        assert card.training_metrics["accuracy"] > 0

    def test_model_card_to_json(self, model):
        m, X, y = model
        card = m.model_card()
        json_str = card.to_json()
        assert "TestCreditScorer" in json_str

    def test_model_card_to_markdown(self, model):
        m, X, y = model
        card = m.model_card()
        md = card.to_markdown()
        assert "# Model Card: TestCreditScorer" in md


class TestAuditLog:
    def test_audit_log_integrity(self, model):
        m, X, y = model
        for entry in m.audit_log.entries:
            assert entry.checksum is not None
            assert len(entry.checksum) == 16

    def test_audit_log_to_json(self, model):
        m, X, y = model
        json_str = m.audit_log.to_json()
        assert "training_started" in json_str
        assert "training_completed" in json_str
