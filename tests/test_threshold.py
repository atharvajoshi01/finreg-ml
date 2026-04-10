"""Tests for threshold tuning module."""

import numpy as np
import pandas as pd

from finreg.threshold import find_optimal_threshold


class TestThresholdBasic:
    def test_finds_threshold(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 500)
        y_proba = y_true * 0.7 + rng.uniform(0, 0.3, 500)
        y_proba = np.clip(y_proba, 0, 1)

        result = find_optimal_threshold(y_true, y_proba, metric="f1")
        assert 0 < result.optimal_threshold < 1
        assert result.metric_value > 0.5
        assert result.metric_name == "f1"

    def test_all_metrics(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 300)
        y_proba = y_true * 0.8 + rng.uniform(0, 0.2, 300)
        y_proba = np.clip(y_proba, 0, 1)

        for metric in ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]:
            result = find_optimal_threshold(y_true, y_proba, metric=metric)
            assert result.metric_value >= 0

    def test_returns_all_results(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_proba = rng.uniform(0, 1, 200)
        result = find_optimal_threshold(y_true, y_proba, n_thresholds=50)
        assert result.thresholds_evaluated == 50
        assert len(result.all_results) == 50


class TestThresholdFairness:
    def test_fairness_constraint(self):
        rng = np.random.RandomState(42)
        n = 1000
        y_true = rng.randint(0, 2, n)
        group = pd.Series(np.where(rng.random(n) > 0.5, "A", "B"), name="group")
        y_proba = y_true * 0.6 + (group == "A").astype(float) * 0.2 + rng.uniform(0, 0.2, n)
        y_proba = np.clip(y_proba, 0, 1)

        result = find_optimal_threshold(
            y_true, y_proba, metric="f1",
            sensitive=group, min_disparate_impact=0.8,
        )
        assert result.fairness_constraint is not None
        passing = [r for r in result.all_results if r["passes_fairness"]]
        assert len(passing) > 0

    def test_without_fairness(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 200)
        y_proba = rng.uniform(0, 1, 200)
        result = find_optimal_threshold(y_true, y_proba, metric="f1")
        assert result.fairness_constraint is None
