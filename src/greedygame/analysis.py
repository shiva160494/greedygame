from __future__ import annotations

from collections import Counter, defaultdict
from math import log2
from . import OUTCOMES
from .models.streak import StreakModel


def frequencies(history: list[str]) -> list[dict]:
    counts = Counter(history)
    total = len(history)
    return [{"outcome": item, "count": counts[item], "percent": round(100 * counts[item] / total, 2) if total else 0}
            for item in OUTCOMES]


def transitions(history: list[str]) -> dict[str, dict[str, int]]:
    matrix = {source: {target: 0 for target in OUTCOMES} for source in OUTCOMES}
    for source, target in zip(history, history[1:]):
        matrix[source][target] += 1
    return matrix


def top_sequences(history: list[str], length: int, limit: int = 20) -> list[tuple[tuple[str, ...], int]]:
    counts = Counter(tuple(history[index:index + length]) for index in range(max(0, len(history) - length + 1)))
    return counts.most_common(limit)


def streak_summary(history: list[str]) -> dict[str, dict[str, int]]:
    result = {item: {"double_or_more": 0, "triple_or_more": 0, "longest": 0} for item in OUTCOMES}
    index = 0
    while index < len(history):
        end = index + 1
        while end < len(history) and history[end] == history[index]:
            end += 1
        item, size = history[index], end - index
        result[item]["longest"] = max(result[item]["longest"], size)
        result[item]["double_or_more"] += int(size >= 2)
        result[item]["triple_or_more"] += int(size >= 3)
        index = end
    current_item, current_size = StreakModel.current_streak(history)
    return {"by_outcome": result, "current_outcome": current_item, "current_length": current_size}


def entropy(history: list[str]) -> float:
    if not history:
        return 0.0
    total, counts = len(history), Counter(history)
    return round(-sum((count / total) * log2(count / total) for count in counts.values()), 4)


def summary(history: list[str]) -> dict:
    return {"total_outcomes": len(history), "entropy_bits": entropy(history), "frequency": frequencies(history),
            "streaks": streak_summary(history), "top_3_sequences": top_sequences(history, 3)}
