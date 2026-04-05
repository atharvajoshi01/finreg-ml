"""Auto-generated model cards following Google's Model Card spec."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ModelCard:
    """Auto-generated model card from training metadata.

    Based on Google's Model Cards for Model Reporting:
    https://arxiv.org/abs/1810.03993
    """

    model_name: str = ""
    model_type: str = ""
    version: str = "1.0.0"
    description: str = ""
    intended_use: str = ""
    out_of_scope_use: str = ""
    training_data_description: str = ""
    evaluation_data_description: str = ""

    # Populated automatically
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    framework: str = "scikit-learn"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_metrics: Dict[str, float] = field(default_factory=dict)
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    features_used: List[str] = field(default_factory=list)
    n_training_samples: int = 0
    n_features: int = 0
    target_column: Optional[str] = None
    fairness_summary: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_details": {
                "name": self.model_name,
                "type": self.model_type,
                "version": self.version,
                "description": self.description,
                "framework": self.framework,
                "created_at": self.created_at,
            },
            "intended_use": {
                "primary": self.intended_use,
                "out_of_scope": self.out_of_scope_use,
            },
            "training_data": {
                "description": self.training_data_description,
                "n_samples": self.n_training_samples,
                "n_features": self.n_features,
                "features": self.features_used,
                "target": self.target_column,
            },
            "hyperparameters": self.hyperparameters,
            "performance": {
                "training": self.training_metrics,
                "evaluation": self.evaluation_metrics,
            },
            "fairness": self.fairness_summary,
            "limitations": self.limitations,
            "ethical_considerations": self.ethical_considerations,
        }

    def to_json(self, path: Optional[str] = None) -> str:
        output = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(output)
        return output

    def to_markdown(self) -> str:
        lines = [
            f"# Model Card: {self.model_name}",
            "",
            f"**Type:** {self.model_type}",
            f"**Version:** {self.version}",
            f"**Created:** {self.created_at}",
            f"**Framework:** {self.framework}",
            "",
            "## Description",
            self.description or "Not provided.",
            "",
            "## Intended Use",
            self.intended_use or "Not provided.",
            "",
            "## Training Data",
            f"- Samples: {self.n_training_samples}",
            f"- Features: {self.n_features}",
            f"- Target: {self.target_column or 'Not specified'}",
            "",
            "## Performance",
        ]

        if self.training_metrics:
            lines.append("### Training Metrics")
            for k, v in self.training_metrics.items():
                lines.append(f"- {k}: {v:.4f}")

        if self.evaluation_metrics:
            lines.append("")
            lines.append("### Evaluation Metrics")
            for k, v in self.evaluation_metrics.items():
                lines.append(f"- {k}: {v:.4f}")

        if self.fairness_summary:
            lines.append("")
            lines.append("## Fairness Assessment")
            for report in self.fairness_summary:
                attr = report.get("attribute", "unknown")
                dp = report.get("demographic_parity_diff", "N/A")
                di = report.get("disparate_impact_ratio", "N/A")
                passes = report.get("passes_four_fifths_rule", "N/A")
                lines.append(f"### {attr}")
                lines.append(f"- Demographic parity diff: {dp}")
                lines.append(f"- Disparate impact ratio: {di}")
                lines.append(f"- Passes 4/5 rule: {passes}")

        if self.limitations:
            lines.append("")
            lines.append("## Limitations")
            for lim in self.limitations:
                lines.append(f"- {lim}")

        return "\n".join(lines)
