"""finreg-ml: Regulation-aware ML pipeline for finance."""

__version__ = "0.1.0"

from finreg.pipeline import GovernedModel
from finreg.audit import AuditLog, AuditEntry
from finreg.fairness import FairnessReport, fairness_metrics
from finreg.explain import ExplanationReport
from finreg.model_card import ModelCard
from finreg.compliance import ComplianceReport, RiskTier

__all__ = [
    "GovernedModel",
    "AuditLog",
    "AuditEntry",
    "FairnessReport",
    "fairness_metrics",
    "ExplanationReport",
    "ModelCard",
    "ComplianceReport",
    "RiskTier",
]
