from __future__ import annotations

from collections import Counter
from .engine import PredictionEngine


def _rank_of(ranking: list[tuple[str, float]], actual: str) -> int | None:
    for position, (outcome, _) in enumerate(ranking, 1):
        if outcome == actual:
            return position
    return None


def walk_forward(history: list[str], warmup: int = 30) -> dict:
    """Evaluate only on future outcomes; no prediction sees its actual result."""
    rows, model_hits = [], Counter()
    model_trials = Counter()
    for index in range(warmup, len(history)):
        train, actual = history[:index], history[index]
        prediction = PredictionEngine().predict(train)
        rank = _rank_of(prediction["ranking"], actual)
        for name, probabilities in prediction["models"].items():
            if probabilities:
                model_trials[name] += 1
                if max(probabilities, key=probabilities.get) == actual:
                    model_hits[name] += 1
        rows.append({"spin": index + 1, "actual": actual, "rank": rank or 0,
                     "top1": rank == 1, "top3": bool(rank and rank <= 3), "top5": bool(rank and rank <= 5)})
    total = len(rows)
    rate = lambda key: round(100 * sum(row[key] for row in rows) / total, 2) if total else 0.0
    # Smooth weights prevent a model with a few lucky hits from dominating.
    weights = {name: (model_hits[name] + 1) / (model_trials[name] + 8)
               for name in model_trials}
    return {"predictions": total, "top1_accuracy": rate("top1"), "top3_accuracy": rate("top3"),
            "top5_accuracy": rate("top5"), "chance_top1": 12.5, "model_top1": {name: round(100 * model_hits[name] / model_trials[name], 2)
            for name in model_trials}, "suggested_weights": weights, "rows": rows}
