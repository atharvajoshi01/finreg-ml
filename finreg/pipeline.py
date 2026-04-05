"""GovernedModel — sklearn-compatible wrapper with built-in compliance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from finreg.audit import AuditLog
from finreg.compliance import ComplianceReport, RiskTier, assess_compliance
from finreg.explain import ExplanationReport, compute_explanations
from finreg.fairness import FairnessReport, fairness_metrics
from finreg.model_card import ModelCard


class GovernedModel(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible model wrapper with regulation-aware governance.

    Wraps any scikit-learn compatible classifier and automatically captures
    training metadata, generates explanations, computes fairness metrics,
    and produces compliance documentation.

    Parameters:
        estimator: Any scikit-learn compatible classifier.
        protected_attributes: Column names of sensitive/protected features.
        risk_tier: EU AI Act risk classification for this use case.
        model_name: Human-readable name for the model.
        intended_use: Description of the model's intended use case.

    Example::

        from finreg import GovernedModel
        from sklearn.ensemble import GradientBoostingClassifier

        model = GovernedModel(
            estimator=GradientBoostingClassifier(),
            protected_attributes=["age_group", "gender"],
            risk_tier="high",
            model_name="CreditScorer",
        )
        model.fit(X_train, y_train)
        model.explain(X_test)
        model.fairness_report(X_test, y_test)
        report = model.compliance_report()
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        protected_attributes: Optional[List[str]] = None,
        risk_tier: Union[str, RiskTier] = RiskTier.HIGH,
        model_name: str = "",
        intended_use: str = "",
    ) -> None:
        self.estimator = estimator
        self.protected_attributes = protected_attributes or []
        self.risk_tier = RiskTier(risk_tier) if isinstance(risk_tier, str) else risk_tier
        self.model_name = model_name or type(estimator).__name__
        self.intended_use = intended_use

        # Internal state
        self.audit_log = AuditLog()
        self._model_card: Optional[ModelCard] = None
        self._explanation: Optional[ExplanationReport] = None
        self._fairness_reports: List[FairnessReport] = []
        self._is_fitted = False
        self._train_metrics: Dict[str, float] = {}
        self._eval_metrics: Dict[str, float] = {}
        self._feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **fit_params: Any,
    ) -> GovernedModel:
        """Fit the estimator and log training metadata.

        Args:
            X: Training feature matrix (must be a DataFrame).
            y: Training target.
            **fit_params: Additional parameters passed to estimator.fit().

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame for auditability.")

        self._feature_names = list(X.columns)

        self.audit_log.log(
            "training_started",
            model_type=type(self.estimator).__name__,
            n_samples=len(X),
            n_features=len(self._feature_names),
            features=self._feature_names,
            risk_tier=self.risk_tier.value,
            hyperparameters=self.estimator.get_params(),
        )

        self.estimator.fit(X, y, **fit_params)
        self._is_fitted = True

        # Compute training metrics
        y_pred = self.estimator.predict(X)
        self._train_metrics = self._compute_metrics(y, y_pred)

        self.audit_log.log(
            "training_completed",
            training_metrics=self._train_metrics,
        )

        # Build initial model card
        self._model_card = ModelCard(
            model_name=self.model_name,
            model_type=type(self.estimator).__name__,
            intended_use=self.intended_use,
            framework="scikit-learn",
            hyperparameters=self.estimator.get_params(),
            training_metrics=self._train_metrics,
            features_used=self._feature_names,
            n_training_samples=len(X),
            n_features=len(self._feature_names),
            target_column=y.name if hasattr(y, "name") else None,
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict and log the prediction event."""
        self._check_fitted()
        result = self.estimator.predict(X)
        self.audit_log.log("prediction", n_samples=len(X))
        return result

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities and log the event."""
        self._check_fitted()
        result = self.estimator.predict_proba(X)
        self.audit_log.log("prediction_proba", n_samples=len(X))
        return result

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate the model on held-out data and log metrics.

        Args:
            X: Evaluation feature matrix.
            y: True labels.

        Returns:
            Dictionary of evaluation metrics.
        """
        self._check_fitted()
        y_pred = self.predict(X)
        self._eval_metrics = self._compute_metrics(y, y_pred)

        if self._model_card:
            self._model_card.evaluation_metrics = self._eval_metrics

        self.audit_log.log("evaluation", metrics=self._eval_metrics)
        return self._eval_metrics

    def explain(self, X: pd.DataFrame, max_samples: int = 500) -> ExplanationReport:
        """Generate SHAP explanations for the model.

        Args:
            X: Feature matrix to explain.
            max_samples: Max samples for SHAP computation.

        Returns:
            ExplanationReport with SHAP values and feature importance.
        """
        self._check_fitted()
        self._explanation = compute_explanations(self.estimator, X, max_samples)

        self.audit_log.log(
            "explanations_generated",
            n_samples=len(X),
            top_features=self._explanation.top_features(5),
        )
        return self._explanation

    def fairness_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        attributes: Optional[List[str]] = None,
    ) -> List[FairnessReport]:
        """Compute fairness metrics across protected attributes.

        Args:
            X: Feature matrix containing protected attribute columns.
            y: True labels.
            attributes: Override which attributes to assess.
                Defaults to self.protected_attributes.

        Returns:
            List of FairnessReport, one per protected attribute.
        """
        self._check_fitted()
        attrs = attributes or self.protected_attributes
        if not attrs:
            raise ValueError(
                "No protected attributes specified. Pass them to GovernedModel() "
                "or to fairness_report(attributes=[...])."
            )

        # Predict using only the features the model was trained on
        predict_cols = [c for c in self._feature_names if c in X.columns]
        y_pred = self.estimator.predict(X[predict_cols])
        self._fairness_reports = []

        for attr in attrs:
            if attr not in X.columns:
                raise KeyError(f"Protected attribute '{attr}' not found in X.")
            report = fairness_metrics(y, y_pred, X[attr])
            self._fairness_reports.append(report)

            self.audit_log.log(
                "fairness_audit",
                attribute=attr,
                demographic_parity_diff=report.demographic_parity_diff,
                disparate_impact_ratio=report.disparate_impact_ratio,
                passes_four_fifths_rule=report.passes_four_fifths_rule,
            )

        if self._model_card:
            self._model_card.fairness_summary = [r.to_dict() for r in self._fairness_reports]

        return self._fairness_reports

    def compliance_report(
        self,
        has_human_oversight: bool = False,
        has_data_governance: bool = False,
    ) -> ComplianceReport:
        """Generate EU AI Act compliance report.

        Automatically checks what documentation has been generated
        (explanations, fairness audit, model card, audit log) and
        assesses compliance against the relevant articles.

        Args:
            has_human_oversight: Whether human override exists (cannot be auto-detected).
            has_data_governance: Whether data governance is documented.

        Returns:
            ComplianceReport with pass/fail checks per article.
        """
        report = assess_compliance(
            risk_tier=self.risk_tier,
            has_explanations=self._explanation is not None,
            has_fairness_audit=len(self._fairness_reports) > 0,
            has_model_card=self._model_card is not None,
            has_audit_log=len(self.audit_log) > 0,
            has_human_oversight=has_human_oversight,
            has_accuracy_metrics=bool(self._eval_metrics),
            has_data_governance=has_data_governance,
            model_name=self.model_name,
        )

        self.audit_log.log(
            "compliance_assessed",
            risk_tier=self.risk_tier.value,
            compliant=report.compliant,
            passed=report.passed,
            failed=report.failed,
        )

        return report

    def model_card(self) -> ModelCard:
        """Return the auto-generated model card."""
        self._check_fitted()
        if self._model_card is None:
            raise RuntimeError("Model card not yet generated. Call fit() first.")
        return self._model_card

    # --- private helpers ---

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred))
        except ValueError:
            pass
        return metrics
