from __future__ import annotations

from collections import Counter
from .base import BaseModel
from .context import _normalise


class StreakModel(BaseModel):
    name = "streak"

    @staticmethod
    def current_streak(history: list[str]) -> tuple[str | None, int]:
        if not history:
            return None, 0
        item, size = history[-1], 1
        for previous in reversed(history[:-1]):
            if previous != item:
                break
            size += 1
        return item, size

    def predict_proba(self, history: list[str]) -> dict[str, float]:
        item, target_size = self.current_streak(history)
        if not item:
            return {}
        votes: Counter[str] = Counter()
        index = 0
        while index < len(history):
            if history[index] != item:
                index += 1
                continue
            end = index
            while end < len(history) and history[end] == item:
                end += 1
            if end - index == target_size and end < len(history):
                votes[history[end]] += 1
            index = end
        return _normalise(votes)
