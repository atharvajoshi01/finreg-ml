"""EU AI Act compliance report generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class RiskTier(str, Enum):
    """EU AI Act risk classification (Article 6)."""

    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"


@dataclass
class ComplianceCheck:
    """A single compliance requirement check."""

    article: str
    requirement: str
    status: str  # "pass", "fail", "warning"
    detail: str


@dataclass
class ComplianceReport:
    """EU AI Act compliance assessment for a trained model."""

    risk_tier: RiskTier
    checks: List[ComplianceCheck] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_name: str = ""
    recommendations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == "pass")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def compliant(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "risk_tier": self.risk_tier.value,
            "generated_at": self.generated_at,
            "compliant": self.compliant,
            "passed": self.passed,
            "failed": self.failed,
            "checks": [
                {
                    "article": c.article,
                    "requirement": c.requirement,
                    "status": c.status,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
            "recommendations": self.recommendations,
        }

    def to_json(self, path: Optional[str] = None) -> str:
        output = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(output)
        return output


# High-risk use cases per Annex III
HIGH_RISK_DOMAINS = {
    "credit_scoring",
    "insurance_pricing",
    "employment_recruitment",
    "education_assessment",
    "law_enforcement",
    "migration_border",
    "biometric_identification",
    "critical_infrastructure",
    "medical_diagnosis",
    "justice_democracy",
}


def assess_compliance(
    risk_tier: RiskTier,
    has_explanations: bool = False,
    has_fairness_audit: bool = False,
    has_model_card: bool = False,
    has_audit_log: bool = False,
    has_human_oversight: bool = False,
    has_accuracy_metrics: bool = False,
    has_data_governance: bool = False,
    model_name: str = "",
) -> ComplianceReport:
    """Assess EU AI Act compliance based on available documentation.

    Args:
        risk_tier: The risk classification of the AI system.
        has_explanations: Whether SHAP/LIME explanations are available.
        has_fairness_audit: Whether bias testing has been performed.
        has_model_card: Whether model documentation exists.
        has_audit_log: Whether decision logging is enabled.
        has_human_oversight: Whether human override capability exists.
        has_accuracy_metrics: Whether performance metrics are documented.
        has_data_governance: Whether data governance is documented.
        model_name: Name of the model being assessed.

    Returns:
        ComplianceReport with pass/fail checks and recommendations.
    """
    checks: List[ComplianceCheck] = []
    recommendations: List[str] = []

    if risk_tier == RiskTier.UNACCEPTABLE:
        checks.append(ComplianceCheck(
            article="Article 5",
            requirement="Prohibited AI practices",
            status="fail",
            detail="System classified as UNACCEPTABLE risk. Deployment is prohibited.",
        ))
        return ComplianceReport(
            risk_tier=risk_tier,
            checks=checks,
            model_name=model_name,
            recommendations=["Do not deploy this system in the EU."],
        )

    # Article 13 — Transparency
    if has_model_card:
        checks.append(ComplianceCheck(
            article="Article 13",
            requirement="Technical documentation",
            status="pass",
            detail="Model card with capabilities, limitations, and intended purpose is available.",
        ))
    else:
        checks.append(ComplianceCheck(
            article="Article 13",
            requirement="Technical documentation",
            status="fail" if risk_tier == RiskTier.HIGH else "warning",
            detail="No model documentation found.",
        ))
        recommendations.append(
            "Generate a model card describing capabilities, limitations, and intended use."
        )

    # Article 13 — Explainability
    if has_explanations:
        checks.append(ComplianceCheck(
            article="Article 13",
            requirement="Explainability",
            status="pass",
            detail="SHAP explanations are available for model decisions.",
        ))
    else:
        checks.append(ComplianceCheck(
            article="Article 13",
            requirement="Explainability",
            status="fail" if risk_tier == RiskTier.HIGH else "warning",
            detail="No explainability method applied.",
        ))
        recommendations.append("Run .explain() to generate SHAP-based explanations.")

    # Article 10 — Data governance
    if has_data_governance:
        checks.append(ComplianceCheck(
            article="Article 10",
            requirement="Data governance",
            status="pass",
            detail="Data governance practices are documented.",
        ))
    elif risk_tier == RiskTier.HIGH:
        checks.append(ComplianceCheck(
            article="Article 10",
            requirement="Data governance",
            status="fail",
            detail="No data governance documentation for high-risk system.",
        ))
        recommendations.append("Document data sources, preprocessing, and quality controls.")

    # Article 12 — Record keeping
    if has_audit_log:
        checks.append(ComplianceCheck(
            article="Article 12",
            requirement="Automatic logging",
            status="pass",
            detail="Audit log with integrity checksums is active.",
        ))
    else:
        checks.append(ComplianceCheck(
            article="Article 12",
            requirement="Automatic logging",
            status="fail" if risk_tier == RiskTier.HIGH else "warning",
            detail="No audit logging enabled.",
        ))
        recommendations.append("Enable audit logging for traceability.")

    # Article 14 — Human oversight
    if has_human_oversight:
        checks.append(ComplianceCheck(
            article="Article 14",
            requirement="Human oversight",
            status="pass",
            detail="Human oversight mechanism is documented.",
        ))
    elif risk_tier == RiskTier.HIGH:
        checks.append(ComplianceCheck(
            article="Article 14",
            requirement="Human oversight",
            status="fail",
            detail="No human oversight mechanism for high-risk system.",
        ))
        recommendations.append(
            "Implement human override capability (interrupt, override, shut down)."
        )

    # Article 15 — Accuracy and bias testing
    if has_accuracy_metrics:
        checks.append(ComplianceCheck(
            article="Article 15",
            requirement="Accuracy metrics",
            status="pass",
            detail="Performance metrics are documented.",
        ))
    elif risk_tier == RiskTier.HIGH:
        checks.append(ComplianceCheck(
            article="Article 15",
            requirement="Accuracy metrics",
            status="fail",
            detail="No accuracy metrics documented for high-risk system.",
        ))
        recommendations.append("Document model accuracy, precision, recall, and AUC.")

    if has_fairness_audit:
        checks.append(ComplianceCheck(
            article="Article 15",
            requirement="Bias testing",
            status="pass",
            detail="Fairness audit across protected attributes has been performed.",
        ))
    else:
        checks.append(ComplianceCheck(
            article="Article 15",
            requirement="Bias testing",
            status="fail" if risk_tier == RiskTier.HIGH else "warning",
            detail="No bias testing performed.",
        ))
        recommendations.append(
            "Run .fairness_report() to assess bias across protected attributes."
        )

    return ComplianceReport(
        risk_tier=risk_tier,
        checks=checks,
        model_name=model_name,
        recommendations=recommendations,
    )
