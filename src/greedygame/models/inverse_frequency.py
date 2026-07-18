"""Inverse frequency model.

Detects if the game punishes outcomes that have appeared too
frequently in recent history.
"""

from __future__ import annotations

from collections import Counter
from .base import BaseModel


def _normalise(counts: Counter[str] | dict[str, float]) -> dict[str, float]:
    """Normalize counts to probability distribution."""
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


class InverseFrequencyModel(BaseModel):
    """Detects punishment patterns for frequently occurring outcomes."""
    
    name = "inverse_frequency"

    def __init__(self, window: int = 10, overuse_threshold: int = 3):
        """Initialize inverse frequency model.
        
        Args:
            window: Recent window to check for overuse
            overuse_threshold: Minimum count in window to mark as overused
        """
        self.window = window
        self.overuse_threshold = overuse_threshold

    def predict_proba(self, history: list[str]) -> dict[str, float]:
        """Predict based on inverse frequency pattern.
        
        If outcomes appear too frequently recently, what typically
        replaces them in the next round?
        """
        if len(history) < self.window * 2:
            return {}
        
        # What's been appearing most recently?
        recent_window = history[-self.window:]
        recent_counts = Counter(recent_window)
        
        votes = Counter()
        
        # Find outcomes that appeared after "overused" ones
        for outcome, count in recent_counts.items():
            if count > self.overuse_threshold:
                # Find what came after this overused outcome
                for i in range(len(history) - 1):
                    if history[i] == outcome and history[i + 1] != outcome:
                        next_val = history[i + 1]
                        # Weight inversely by how overused it was
                        weight = 1 / (count ** 0.5)
                        votes[next_val] += weight
        
        return _normalise(votes)
