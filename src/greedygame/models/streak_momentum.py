"""Streak momentum model.

Analyzes what happens after consecutive identical outcomes.
Some games may increase/decrease probability of certain outcomes
after streaks.
"""

from __future__ import annotations

from collections import Counter
from .base import BaseModel


def _normalise(counts: Counter[str] | dict[str, float]) -> dict[str, float]:
    """Normalize counts to probability distribution."""
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


class StreakMomentumModel(BaseModel):
    """Detects patterns after streaks of identical outcomes."""
    
    name = "streak_momentum"

    def __init__(self, min_history: int = 20):
        """Initialize streak momentum model.
        
        Args:
            min_history: Minimum history length to make predictions
        """
        self.min_history = min_history

    @staticmethod
    def _current_streak(history: list[str]) -> tuple[str | None, int]:
        """Get the current streak (how many of the same outcome in a row)."""
        if not history:
            return None, 0
        
        item, size = history[-1], 1
        for previous in reversed(history[:-1]):
            if previous != item:
                break
            size += 1
        
        return item, size

    def predict_proba(self, history: list[str]) -> dict[str, float]:
        """Predict based on streak momentum.
        
        Analyzes what typically happens after streaks of various lengths.
        """
        if len(history) < self.min_history:
            return {}
        
        current_item, current_streak_length = self._current_streak(history)
        if not current_item:
            return {}
        
        votes = Counter()
        
        # Analyze what happens after streaks of different lengths
        for min_streak_len in [2, 3, 5, 8]:
            for i in range(len(history) - min_streak_len):
                # Check if we have a streak of at least min_streak_len
                if history[i] == history[i + min_streak_len - 1]:
                    is_streak = all(
                        history[i] == history[i + j]
                        for j in range(min_streak_len)
                    )
                    
                    if is_streak and i + min_streak_len < len(history):
                        next_val = history[i + min_streak_len]
                        # Weight: longer streaks are less predictive
                        weight = 1 / (2 ** (min_streak_len - 2))
                        votes[next_val] += weight
        
        return _normalise(votes)
