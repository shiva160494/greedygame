"""Main prediction engine combining all models."""

from __future__ import annotations

import json
from pathlib import Path

from . import OUTCOMES
from .models.ensemble import AdaptiveEnsemble
from .models.frequency_bias import FrequencyBiasModel
from .models.streak_momentum import StreakMomentumModel
from .models.inverse_frequency import InverseFrequencyModel
from .models.cycle_detection import CycleDetectionModel
from .models.confidence import ConfidenceFilter


class PredictionEngine:
    """Advanced game prediction engine with bias detection and confidence scoring."""

    def __init__(self, weights: dict[str, float] | None = None, weights_file: str | Path | None = None):
        """Initialize prediction engine.
        
        Args:
            weights: Optional manual weights for models
            weights_file: Optional path to JSON file with saved weights from backtest
        """
        # Load weights from file if provided
        if weights is None and weights_file and Path(weights_file).exists():
            weights = json.loads(Path(weights_file).read_text(encoding="utf-8"))
        
        # Game-specific models focused on bias detection
        self.models = [
            FrequencyBiasModel(min_samples=50, deviation_threshold=0.01),
            StreakMomentumModel(min_history=20),
            InverseFrequencyModel(window=10, overuse_threshold=3),
            CycleDetectionModel(min_cycle_len=4, max_cycle_len=12, min_repeats=2),
        ]
        
        self.ensemble = AdaptiveEnsemble(self.models, weights)
        self.confidence_filter = ConfidenceFilter()

    def predict(self, history: list[str]) -> dict:
        """Generate predictions for the next outcome.
        
        Args:
            history: List of past outcomes
            
        Returns:
            Dictionary with:
            - ranking: Sorted list of (outcome, probability) tuples
            - models: Individual model predictions
            - support: Number of models with signal
            - confidence: How much better than random (0-1)
            - statistical_significance: Whether any bias detected
        """
        # Get ensemble predictions
        probabilities, model_probabilities = self.ensemble.predict_proba(history)
        
        # Check if we have statistical signal
        has_signal = self.confidence_filter.has_statistical_signal(probabilities)
        
        # If no signal detected and we have enough history, fall back to uniform
        if not probabilities and history:
            probabilities = {outcome: 1 / 8 for outcome in OUTCOMES}
        
        # Calculate confidence score
        confidence = self.confidence_filter.calculate_confidence(history, probabilities)
        
        # Check for statistical significance in any outcome
        has_significance = any(
            self.confidence_filter.chi_square_test(history, outcome)
            for outcome in OUTCOMES
        )
        
        # Sort by probability (descending), then by outcome name
        ordered = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
        
        return {
            "ranking": ordered,
            "models": model_probabilities,
            "support": sum(bool(v) for v in model_probabilities.values()),
            "confidence": round(confidence, 3),
            "statistical_significance": has_significance,
        }
