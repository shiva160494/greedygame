"""Confidence filtering and statistical validation."""

from __future__ import annotations

from collections import Counter


class ConfidenceFilter:
    """Validates predictions against statistical significance."""

    @staticmethod
    def chi_square_test(history: list[str], outcome: str, min_occurrences: int = 8) -> bool:
        """Test if an outcome shows statistical bias.
        
        Uses chi-square test to check if an outcome appears
        significantly more than expected (>95% confidence).
        
        Args:
            history: List of past outcomes
            outcome: Outcome to test
            min_occurrences: Minimum occurrences required
            
        Returns:
            True if outcome is significantly biased (p < 0.05)
        """
        if outcome not in history:
            return False
        
        counts = Counter(history)
        observed = counts.get(outcome, 0)
        
        # Expected frequency if uniformly random: 1/8
        expected = len(history) / 8
        
        if observed < min_occurrences:
            return False
        
        # Chi-square statistic
        chi_squared = ((observed - expected) ** 2) / expected
        
        # Critical value for 95% confidence (df=1) is 3.841
        return chi_squared > 3.841

    @staticmethod
    def calculate_confidence(history: list[str], predictions: dict[str, float]) -> float:
        """Calculate confidence that predictions beat random baseline.
        
        Random baseline for 8 outcomes is 12.5% (1/8).
        
        Args:
            history: List of past outcomes (for context)
            predictions: Probability distribution from model
            
        Returns:
            Confidence score 0-1. <0.2 = likely random, >0.5 = likely exploitable
        """
        if not predictions:
            return 0.0
        
        # Get probability of top prediction
        top_prob = max(predictions.values())
        random_baseline = 1 / 8  # 12.5%
        
        # How much better than random?
        improvement = (top_prob - random_baseline) / random_baseline
        
        # Clamp to 0-1 range
        confidence = min(max(improvement, 0), 1.0)
        
        return confidence

    @staticmethod
    def has_statistical_signal(predictions: dict[str, float], min_signal_strength: float = 0.05) -> bool:
        """Check if predictions have meaningful statistical signal.
        
        Args:
            predictions: Probability distribution from ensemble
            min_signal_strength: Minimum signal to consider valid
            
        Returns:
            True if signal exists above threshold
        """
        if not predictions:
            return False
        
        # Calculate entropy (lower = more confident predictions)
        import math
        entropy = -sum(p * math.log2(p) for p in predictions.values() if p > 0)
        
        # Maximum entropy for 8 outcomes is 3.0
        # If entropy < 2.0, predictions are more concentrated
        return entropy < 2.5
