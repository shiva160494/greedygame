from __future__ import annotations

from collections import Counter
from .base import BaseModel
from .context import _normalise


class AdaptiveEnsemble:
    """Soft-voting ensemble; weights can be supplied from a walk-forward backtest."""
    def __init__(self, models: list[BaseModel], weights: dict[str, float] | None = None):
        self.models = models
        self.weights = weights or {model.name: 1.0 for model in models}

    def predict_proba(self, history: list[str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        final: Counter[str] = Counter()
        individual: dict[str, dict[str, float]] = {}
        for model in self.models:
            probabilities = model.predict_proba(history)
            individual[model.name] = probabilities
            for outcome, probability in probabilities.items():
                final[outcome] += probability * self.weights.get(model.name, 0.0)
        return _normalise(final), individual
