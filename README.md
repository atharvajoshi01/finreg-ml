# finreg-ml

[![PyPI](https://img.shields.io/pypi/v/finreg-ml)](https://pypi.org/project/finreg-ml/)
[![CI](https://github.com/atharvajoshi01/finreg-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvajoshi01/finreg-ml/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Regulation-aware ML pipeline for finance. Train a model, get a compliance report.

Built for teams shipping ML in regulated industries (credit scoring, fraud detection, insurance pricing) who need explainability, fairness audits, and EU AI Act documentation as part of their workflow — not as an afterthought.

## Installation

```bash
pip install finreg-ml
```

## Quick Start

```python
from finreg import GovernedModel
from sklearn.ensemble import GradientBoostingClassifier

# Wrap any sklearn-compatible estimator
model = GovernedModel(
    estimator=GradientBoostingClassifier(),
    protected_attributes=["age_group", "gender"],
    risk_tier="high",  # EU AI Act classification
    model_name="CreditScorer",
)

# Train — metadata captured automatically
model.fit(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)

# Explain — SHAP values auto-generated
explanations = model.explain(X_test)
print(explanations.top_features(5))

# Fairness audit — bias metrics across protected attributes
fairness = model.fairness_report(X_test, y_test)
for report in fairness:
    print(f"{report.attribute}: disparate impact = {report.disparate_impact_ratio:.3f}")

# Compliance check — EU AI Act Article 13 assessment
compliance = model.compliance_report()
print(f"Compliant: {compliance.compliant}")
print(f"Passed: {compliance.passed}, Failed: {compliance.failed}")

# Export everything
model.model_card().to_json("model_card.json")
model.audit_log.to_json("audit_log.json")
compliance.to_json("compliance_report.json")
```

## What It Does

| Module | Purpose |
|--------|---------|
| `GovernedModel` | Sklearn-compatible wrapper that captures training metadata automatically |
| `finreg.explain` | SHAP-based explanations and feature importance |
| `finreg.fairness` | Demographic parity, equalized odds, disparate impact (four-fifths rule) |
| `finreg.model_card` | Auto-generated model cards (Google Model Card spec) |
| `finreg.compliance` | EU AI Act compliance assessment (Articles 5, 10, 12, 13, 14, 15) |
| `finreg.audit` | Append-only audit log with SHA-256 integrity checksums |

## EU AI Act Coverage

Checks against these articles for HIGH-risk systems:

- **Article 5** — Prohibited AI practices detection
- **Article 10** — Data governance requirements
- **Article 12** — Automatic logging and traceability
- **Article 13** — Transparency and explainability
- **Article 14** — Human oversight mechanisms
- **Article 15** — Accuracy metrics and bias testing

## Fairness Metrics

For each protected attribute, computes:

- **Demographic parity difference** — gap in positive prediction rates across groups
- **Equalized odds difference** — gap in true positive rates
- **Disparate impact ratio** — ratio of lowest to highest positive rate
- **Four-fifths rule** — passes if disparate impact ratio >= 0.8

## Development

```bash
git clone https://github.com/atharvajoshi01/finreg-ml.git
cd finreg-ml
pip install -e ".[dev]"
pytest
```

## License

MIT
