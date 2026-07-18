from __future__ import annotations

from collections import Counter
from .base import BaseModel


def _normalise(counts: Counter[str] | dict[str, float]) -> dict[str, float]:
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


class SequenceMatcher(BaseModel):
    name = "sequence"

    def __init__(self, min_order: int = 3, max_order: int = 8):
        self.min_order, self.max_order = min_order, max_order

    def predict_proba(self, history: list[str]) -> dict[str, float]:
        votes: Counter[str] = Counter()
        for order in range(self.min_order, min(self.max_order, len(history) - 1) + 1):
            context = history[-order:]
            for start in range(0, len(history) - order):
                if history[start : start + order] == context:
                    votes[history[start + order]] += 2 ** (order - self.min_order)
        return _normalise(votes)


class MarkovModel(BaseModel):
    name = "markov"

    def __init__(self, max_order: int = 5):
        self.max_order = max_order

    def predict_proba(self, history: list[str]) -> dict[str, float]:
        votes: Counter[str] = Counter()
        for order in range(1, min(self.max_order, len(history) - 1) + 1):
            context, local = history[-order:], Counter()
            for start in range(len(history) - order):
                if history[start : start + order] == context:
                    local[history[start + order]] += 1
            for outcome, count in _normalise(local).items():
                votes[outcome] += count * order
        return _normalise(votes)


class SimilarityMatcher(BaseModel):
    name = "similarity"

    def __init__(self, window: int = 6, minimum_similarity: float = 0.5):
        self.window, self.minimum_similarity = window, minimum_similarity

    def predict_proba(self, history: list[str]) -> dict[str, float]:
        if len(history) <= self.window:
            return {}
        target, votes = history[-self.window:], Counter()
        for start in range(len(history) - self.window):
            candidate = history[start : start + self.window]
            similarity = sum(a == b for a, b in zip(candidate, target)) / self.window
            if similarity >= self.minimum_similarity:
                votes[history[start + self.window]] += similarity ** 3
        return _normalise(votes)
