"""Cycle detection model.

Detects if outcomes repeat in predictable cycles.
"""

from __future__ import annotations

from collections import Counter
from .base import BaseModel


def _normalise(counts: Counter[str] | dict[str, float]) -> dict[str, float]:
    """Normalize counts to probability distribution."""
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


class CycleDetectionModel(BaseModel):
    """Detects repeating patterns in outcomes."""
    
    name = "cycle"

    def __init__(self, min_cycle_len: int = 4, max_cycle_len: int = 12, min_repeats: int = 2):
        """Initialize cycle detection model.
        
        Args:
            min_cycle_len: Minimum cycle length to detect
            max_cycle_len: Maximum cycle length to detect
            min_repeats: Minimum number of repeats to consider valid
        """
        self.min_cycle_len = min_cycle_len
        self.max_cycle_len = max_cycle_len
        self.min_repeats = min_repeats

    def predict_proba(self, history: list[str]) -> dict[str, float]:
        """Predict based on cycle detection.
        
        Looks for repeating patterns and extrapolates the next value.
        """
        if len(history) < self.max_cycle_len * 2:
            return {}
        
        votes = Counter()
        
        # Try to detect repeating patterns of various lengths
        for cycle_len in range(self.min_cycle_len, self.max_cycle_len + 1):
            repeat_count = 0
            
            # Count how many times this cycle repeats
            for i in range(len(history) - 2 * cycle_len):
                cycle1 = tuple(history[i:i + cycle_len])
                cycle2 = tuple(history[i + cycle_len:i + 2 * cycle_len])
                
                if cycle1 == cycle2:
                    repeat_count += 1
            
            # If cycle repeats enough times, use it for prediction
            if repeat_count >= self.min_repeats:
                # Where are we in the current cycle?
                current_pos = len(history) % cycle_len
                
                # What comes next in the cycle?
                if current_pos < cycle_len - 1:
                    # Get the expected next value from a recent cycle
                    cycle_start = len(history) - (len(history) % cycle_len) - cycle_len
                    if cycle_start >= 0 and current_pos + 1 < len(history):
                        predicted = history[cycle_start + current_pos + 1]
                        # Weight by how strongly the cycle repeats
                        weight = repeat_count
                        votes[predicted] += weight
        
        return _normalise(votes)
