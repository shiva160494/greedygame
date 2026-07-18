"""Walk-forward backtesting for model evaluation."""

from __future__ import annotations

from collections import Counter

from .engine import PredictionEngine


def _rank_of(ranking: list[tuple[str, float]], actual: str) -> int | None:
    """Get rank of actual outcome in prediction ranking."""
    for position, (outcome, _) in enumerate(ranking, 1):
        if outcome == actual:
            return position
    return None


def walk_forward(history: list[str], warmup: int = 50) -> dict:
    """Run walk-forward backtesting.
    
    Evaluates predictions on future outcomes that the model hasn't seen.
    No data leakage - each prediction is made before seeing the result.
    
    Args:
        history: List of past outcomes
        warmup: Minimum history to collect before testing starts
        
    Returns:
        Dictionary with accuracy metrics and model performance
    """
    rows = []
    model_hits = Counter()
    model_trials = Counter()
    
    for index in range(warmup, len(history)):
        # Train on everything up to current index
        train, actual = history[:index], history[index]
        
        # Make prediction
        prediction = PredictionEngine().predict(train)
        rank = _rank_of(prediction["ranking"], actual)
        
        # Track model-specific performance
        for name, probabilities in prediction["models"].items():
            if probabilities:
                model_trials[name] += 1
                if max(probabilities, key=probabilities.get) == actual:
                    model_hits[name] += 1
        
        rows.append(
            {
                "spin": index + 1,
                "actual": actual,
                "rank": rank or 0,
                "top1": rank == 1,
                "top3": bool(rank and rank <= 3),
                "top5": bool(rank and rank <= 5),
                "confidence": prediction.get("confidence", 0),
            }
        )
    
    total = len(rows)
    rate = lambda key: (
        round(100 * sum(row[key] for row in rows) / total, 2) if total else 0.0
    )
    
    # Calculate model accuracy rates
    model_accuracy = {}
    for name in model_trials:
        if model_trials[name] > 0:
            model_accuracy[name] = round(
                100 * model_hits[name] / model_trials[name], 2
            )
    
    # Smooth weights prevent a model with a few lucky hits from dominating
    weights = {
        name: (model_hits[name] + 1) / (model_trials[name] + 8)
        for name in model_trials
    }
    
    return {
        "predictions": total,
        "top1_accuracy": rate("top1"),
        "top3_accuracy": rate("top3"),
        "top5_accuracy": rate("top5"),
        "chance_top1": 12.5,
        "model_accuracy": model_accuracy,
        "suggested_weights": weights,
        "rows": rows,
    }
