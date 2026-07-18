from __future__ import annotations

import json
from pathlib import Path
from . import OUTCOMES
from .models.context import MarkovModel, SequenceMatcher, SimilarityMatcher
from .models.ensemble import AdaptiveEnsemble
from .models.streak import StreakModel


class PredictionEngine:
    def __init__(self, weights: dict[str, float] | None = None, weights_file: str | Path | None = None):
        if weights is None and weights_file and Path(weights_file).exists():
            weights = json.loads(Path(weights_file).read_text(encoding="utf-8"))
        self.models = [SequenceMatcher(), SimilarityMatcher(), MarkovModel(), StreakModel()]
        self.ensemble = AdaptiveEnsemble(self.models, weights)

    def predict(self, history: list[str]) -> dict:
        probabilities, model_probabilities = self.ensemble.predict_proba(history)
        # A transparent fallback when pattern models have no support.
        if not probabilities and history:
            counts = {outcome: history.count(outcome) / len(history) for outcome in OUTCOMES}
            probabilities = counts
        ordered = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
        return {"ranking": ordered, "models": model_probabilities, "support": sum(bool(v) for v in model_probabilities.values())}
