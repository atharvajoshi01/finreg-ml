"""Tests for consolidated report generation."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from finreg import GovernedModel
from finreg.drift import detect_drift
from finreg.report import generate_report


@pytest.fixture
def trained_model():
    rng = np.random.RandomState(42)
    n = 300
    X = pd.DataFrame({
        "income": rng.normal(50000, 10000, n),
        "debt": rng.uniform(0, 1, n),
        "history": rng.randint(6, 200, n),
        "age_group": rng.choice(["young", "old"], n),
    })
    y = pd.Series((X["income"] > 45000).astype(int), name="approved")

    model = GovernedModel(
        estimator=LogisticRegression(max_iter=1000),
        protected_attributes=["age_group"],
        risk_tier="high",
        model_name="ReportTestModel",
    )
    X_num = X.drop(columns=["age_group"])
    model.fit(X_num, y)
    model.evaluate(X_num, y)
    fairness = model.fairness_report(X, y)
    compliance = model.compliance_report()

    ref = X_num
    cur = pd.DataFrame({
        "income": rng.normal(60000, 10000, 200),
        "debt": rng.uniform(0.2, 1, 200),
        "history": rng.randint(6, 200, 200),
    })
    drift = detect_drift(ref, cur, method="ks")

    return model, fairness, compliance, drift


class TestReportJSON:
    def test_json_output(self, trained_model):
        model, fairness, compliance, drift = trained_model
        output = generate_report(
            model_card=model.model_card(),
            compliance=compliance,
            fairness=fairness,
            drift=drift,
            format="json",
        )
        data = json.loads(output)
        assert "model_card" in data
        assert "compliance" in data
        assert "fairness" in data
        assert "drift" in data
        assert "generated_at" in data

    def test_json_to_file(self, trained_model, tmp_path):
        model, fairness, compliance, drift = trained_model
        path = str(tmp_path / "report.json")
        generate_report(
            model_card=model.model_card(),
            compliance=compliance,
            format="json",
            path=path,
        )
        with open(path) as f:
            data = json.load(f)
        assert data["compliance"]["risk_tier"] == "high"


class TestReportMarkdown:
    def test_markdown_output(self, trained_model):
        model, fairness, compliance, drift = trained_model
        output = generate_report(
            model_card=model.model_card(),
            compliance=compliance,
            fairness=fairness,
            drift=drift,
            format="markdown",
        )
        assert "# Compliance Report" in output
        assert "ReportTestModel" in output
        assert "EU AI Act Compliance" in output
        assert "Fairness Audit" in output
        assert "Data Drift" in output

    def test_partial_report(self, trained_model):
        model, fairness, compliance, drift = trained_model
        output = generate_report(compliance=compliance, format="markdown")
        assert "EU AI Act Compliance" in output
        assert "Fairness Audit" not in output


class TestReportEdgeCases:
    def test_empty_report(self):
        output = generate_report(format="json")
        data = json.loads(output)
        assert "generated_at" in data
        assert "model_card" not in data

    def test_invalid_format(self, trained_model):
        model, fairness, compliance, drift = trained_model
        with pytest.raises(ValueError, match="Unknown format"):
            generate_report(compliance=compliance, format="html")
