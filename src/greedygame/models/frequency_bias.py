"""Frequency bias detection model.

Detects if certain outcomes appear more/less frequently than expected
by random chance (1/8 = 12.5% for 8 outcomes).
"""

from __future__ import annotations

from collections import Counter
from .base import BaseModel


class FrequencyBiasModel(BaseModel):
    """Detects statistical bias in outcome frequencies."""
    
    name = "frequency_bias"

    def __init__(self, min_samples: int = 50, deviation_threshold: float = 0.01):
        """Initialize frequency bias model.
        
        Args:
            min_samples: Minimum history length to make predictions
            deviation_threshold: Minimum frequency deviation (default 1%)
        """
        self.min_samples = min_samples
        self.deviation_threshold = deviation_threshold

    def predict_proba(self, history: list[str]) -> dict[str, float]:
        """Predict based on frequency bias.
        
        Looks for outcomes that appear significantly more or less
        frequently than the 12.5% baseline (1/8).
        """
        if len(history) < self.min_samples:
            return {}
        
        counts = Counter(history)
        total = len(history)
        expected_freq = 1 / 8  # 12.5% for 8 outcomes
        
        predictions = {}
        
        for outcome, count in counts.items():
            actual_freq = count / total
            deviation = actual_freq - expected_freq
            
            # Only predict outcomes that deviate significantly
            if abs(deviation) > self.deviation_threshold:
                # Weight by how strong the bias is
                # Positive bias (more frequent) gets boosted probability
                bias_strength = deviation / expected_freq  # -1 to +inf
                probability = 0.5 * (1 + min(bias_strength / 2, 1.0))
                predictions[outcome] = probability
        
        # Normalize to probability distribution
        total_weight = sum(predictions.values())
        if total_weight > 0:
            return {k: v / total_weight for k, v in predictions.items()}
        
        return {}
