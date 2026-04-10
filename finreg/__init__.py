"""finreg-ml: Regulation-aware ML pipeline for finance."""

__version__ = "0.1.0"

from finreg.pipeline import GovernedModel
from finreg.audit import AuditLog, AuditEntry
from finreg.fairness import FairnessReport, fairness_metrics
from finreg.explain import ExplanationReport
from finreg.model_card import ModelCard
from finreg.compliance import ComplianceReport, RiskTier
from finreg.drift import DriftReport, detect_drift
from finreg.report import generate_report

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
    "DriftReport",
    "detect_drift",
    "generate_report",
]
