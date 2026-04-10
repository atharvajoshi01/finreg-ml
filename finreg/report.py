"""Generate consolidated compliance reports in multiple formats."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from finreg.compliance import ComplianceReport
from finreg.drift import DriftReport
from finreg.fairness import FairnessReport
from finreg.model_card import ModelCard


def generate_report(
    model_card: Optional[ModelCard] = None,
    compliance: Optional[ComplianceReport] = None,
    fairness: Optional[List[FairnessReport]] = None,
    drift: Optional[DriftReport] = None,
    format: str = "json",
    path: Optional[str] = None,
) -> str:
    """Generate a consolidated compliance report.

    Combines model card, compliance checks, fairness audit, and drift
    detection into a single report.

    Args:
        model_card: Auto-generated model card.
        compliance: EU AI Act compliance report.
        fairness: Fairness audit results.
        drift: Data drift detection results.
        format: Output format — "json" or "markdown".
        path: Optional file path to write the report.

    Returns:
        Report as a string in the specified format.
    """
    data: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framework": "finreg-ml",
    }

    if model_card:
        data["model_card"] = model_card.to_dict()

    if compliance:
        data["compliance"] = compliance.to_dict()

    if fairness:
        data["fairness"] = [r.to_dict() for r in fairness]

    if drift:
        data["drift"] = drift.to_dict()

    if format == "json":
        output = json.dumps(data, indent=2)
    elif format == "markdown":
        output = _to_markdown(data, model_card, compliance, fairness, drift)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'markdown'.")

    if path:
        with open(path, "w") as f:
            f.write(output)

    return output


def _to_markdown(
    data: Dict[str, Any],
    model_card: Optional[ModelCard],
    compliance: Optional[ComplianceReport],
    fairness: Optional[List[FairnessReport]],
    drift: Optional[DriftReport],
) -> str:
    lines = [
        "# Compliance Report",
        "",
        f"Generated: {data['generated_at']}",
        "",
    ]

    if model_card:
        lines.append("## Model Details")
        lines.append(f"- **Name:** {model_card.model_name}")
        lines.append(f"- **Type:** {model_card.model_type}")
        lines.append(f"- **Training samples:** {model_card.n_training_samples}")
        lines.append(f"- **Features:** {model_card.n_features}")
        if model_card.training_metrics:
            lines.append("")
            lines.append("### Training Metrics")
            for k, v in model_card.training_metrics.items():
                lines.append(f"- {k}: {v:.4f}")
        if model_card.evaluation_metrics:
            lines.append("")
            lines.append("### Evaluation Metrics")
            for k, v in model_card.evaluation_metrics.items():
                lines.append(f"- {k}: {v:.4f}")
        lines.append("")

    if compliance:
        lines.append("## EU AI Act Compliance")
        lines.append(f"- **Risk tier:** {compliance.risk_tier.value.upper()}")
        lines.append(f"- **Compliant:** {'YES' if compliance.compliant else 'NO'}")
        lines.append(f"- **Passed:** {compliance.passed} | **Failed:** {compliance.failed}")
        lines.append("")
        for c in compliance.checks:
            icon = {"pass": "PASS", "fail": "FAIL", "warning": "WARN"}[c.status]
            lines.append(f"- [{icon}] **{c.article}** {c.requirement}: {c.detail}")
        if compliance.recommendations:
            lines.append("")
            lines.append("### Recommendations")
            for rec in compliance.recommendations:
                lines.append(f"- {rec}")
        lines.append("")

    if fairness:
        lines.append("## Fairness Audit")
        for r in fairness:
            lines.append(f"### {r.attribute}")
            lines.append(f"- Demographic parity diff: {r.demographic_parity_diff:.4f}")
            if r.disparate_impact_ratio is not None:
                lines.append(f"- Disparate impact ratio: {r.disparate_impact_ratio:.4f}")
                status = "PASS" if r.passes_four_fifths_rule else "FAIL"
                lines.append(f"- Four-fifths rule: {status}")
            for g in r.groups:
                lines.append(f"  - {g.group_name}: n={g.size}, rate={g.positive_rate:.3f}")
        lines.append("")

    if drift:
        lines.append("## Data Drift")
        lines.append(f"- **Drift detected:** {'YES' if drift.has_drift else 'NO'}")
        lines.append(f"- **Features drifted:** {drift.n_drifted}/{drift.n_features}")
        if drift.drifted_features():
            lines.append(f"- **Affected:** {', '.join(drift.drifted_features())}")
        lines.append("")
        for r in drift.results:
            status = "DRIFTED" if r.drifted else "OK"
            lines.append(f"- {r.feature}: stat={r.statistic:.4f}, p={r.p_value:.4f} [{status}]")

    return "\n".join(lines)
