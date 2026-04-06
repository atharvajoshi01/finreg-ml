"""
Credit Scoring Pipeline with finreg-ml
=======================================

End-to-end example: train a credit scoring model with automatic
explainability, fairness auditing, and EU AI Act compliance checks.

Usage:
    python examples/credit_scoring.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from finreg import GovernedModel

# ---------------------------------------------------------------------------
# 1. Generate synthetic credit data
# ---------------------------------------------------------------------------

rng = np.random.RandomState(42)
n = 2000

data = pd.DataFrame({
    "annual_income": rng.normal(55000, 20000, n).clip(15000),
    "debt_to_income": rng.uniform(0.05, 0.95, n),
    "credit_history_months": rng.randint(6, 300, n),
    "num_open_accounts": rng.randint(1, 15, n),
    "recent_inquiries": rng.poisson(2, n),
    "age_group": rng.choice(["18-30", "31-45", "46-60", "60+"], n),
    "gender": rng.choice(["M", "F"], n),
})

# Target: loan approval (influenced by income and debt ratio)
approval_prob = (
    0.3
    + 0.4 * (data["annual_income"] > 40000).astype(float)
    + 0.2 * (data["debt_to_income"] < 0.4).astype(float)
    - 0.1 * (data["recent_inquiries"] > 3).astype(float)
)
data["approved"] = (rng.random(n) < approval_prob).astype(int)

print(f"Dataset: {n} samples, approval rate: {data['approved'].mean():.1%}")
print()

# ---------------------------------------------------------------------------
# 2. Split data
# ---------------------------------------------------------------------------

feature_cols = [
    "annual_income", "debt_to_income", "credit_history_months",
    "num_open_accounts", "recent_inquiries",
]

X = data[feature_cols + ["age_group", "gender"]]
y = data["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Train with GovernedModel
# ---------------------------------------------------------------------------

model = GovernedModel(
    estimator=GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    ),
    protected_attributes=["age_group", "gender"],
    risk_tier="high",  # Credit scoring = HIGH risk under EU AI Act
    model_name="CreditApprovalModel",
    intended_use="Automated credit approval decisions for consumer lending",
)

# fit() only uses numeric features — protected attributes are kept for fairness audit
X_train_numeric = X_train[feature_cols]
X_test_numeric = X_test[feature_cols]

model.fit(X_train_numeric, y_train)
print("Model trained.")
print()

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------

metrics = model.evaluate(X_test_numeric, y_test)
print("Evaluation Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
print()

# ---------------------------------------------------------------------------
# 5. Explain (requires shap to be importable)
# ---------------------------------------------------------------------------

try:
    explanations = model.explain(X_test_numeric[:100])
    print("Top 5 Features by SHAP Importance:")
    for feat, val in explanations.top_features(5).items():
        print(f"  {feat}: {val:.4f}")
    print()
except ImportError:
    print("SHAP not available — skipping explanations.")
    print()

# ---------------------------------------------------------------------------
# 6. Fairness audit
# ---------------------------------------------------------------------------

fairness = model.fairness_report(X_test, y_test)
print("Fairness Audit:")
for report in fairness:
    print(f"\n  Attribute: {report.attribute}")
    print(f"  Demographic parity diff: {report.demographic_parity_diff:.4f}")
    if report.disparate_impact_ratio is not None:
        print(f"  Disparate impact ratio: {report.disparate_impact_ratio:.4f}")
        print(f"  Passes 4/5 rule: {report.passes_four_fifths_rule}")
    for g in report.groups:
        print(f"    {g.group_name}: n={g.size}, positive_rate={g.positive_rate:.3f}")
print()

# ---------------------------------------------------------------------------
# 7. Compliance report
# ---------------------------------------------------------------------------

compliance = model.compliance_report(
    has_human_oversight=True,
    has_data_governance=True,
)

print(f"EU AI Act Compliance Report:")
print(f"  Risk tier: {compliance.risk_tier.value.upper()}")
print(f"  Compliant: {compliance.compliant}")
print(f"  Passed: {compliance.passed}, Failed: {compliance.failed}")

if compliance.recommendations:
    print(f"\n  Recommendations:")
    for rec in compliance.recommendations:
        print(f"    - {rec}")
print()

# ---------------------------------------------------------------------------
# 8. Export artifacts
# ---------------------------------------------------------------------------

model.model_card().to_json("model_card.json")
model.audit_log.to_json("audit_log.json")
compliance.to_json("compliance_report.json")

print("Exported: model_card.json, audit_log.json, compliance_report.json")
