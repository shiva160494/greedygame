"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import summary
from .backtest import walk_forward
from .engine import PredictionEngine
from .storage import HistoryStore


def _default_data() -> Path:
    """Get default data file path."""
    return Path.cwd() / "data" / "outcomes.csv"


def _show_prediction(history: list[str], weights_file: Path) -> None:
    """Display top predictions with confidence scores."""
    prediction = PredictionEngine(weights_file=weights_file).predict(history)
    
    if not prediction["ranking"]:
        print("Need at least one saved outcome.")
        return
    
    print("\n=== Historical Ranking (Not a Gambling Forecast) ===")
    for rank, (outcome, probability) in enumerate(prediction["ranking"][:5], 1):
        print(f"  {rank}. {outcome:<12} {probability:6.2%}")
    
    print(f"\nPattern models with signal: {prediction['support']}/4")
    print(f"Confidence vs random: {prediction['confidence']:.1%}")
    
    if prediction["statistical_significance"]:
        print("✓ Statistical significance detected (95% confidence)")
    else:
        print("✗ No statistical significance - results may be random")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Greedy Game - Game outcome analysis and prediction engine"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_default_data(),
        help="path to outcomes.csv",
    )
    
    commands = parser.add_subparsers(dest="command", required=True)
    
    # Add outcomes
    add = commands.add_parser("add", help="append one or more outcomes")
    add.add_argument("outcomes", nargs="+", help="e.g. Tomato Corn Cabbage")
    
    # Import from file
    imp = commands.add_parser(
        "import", help="append line-separated outcomes from a text file"
    )
    imp.add_argument("file", type=Path)
    
    # Show predictions
    commands.add_parser("predict", help="show predictions with confidence scores")
    
    # Show statistics
    commands.add_parser("stats", help="show descriptive statistics")
    
    # Open desktop app
    commands.add_parser("app", help="open the desktop app")
    
    # Run backtest
    backtest = commands.add_parser(
        "backtest", help="run leakage-free walk-forward validation"
    )
    backtest.add_argument(
        "--warmup", type=int, default=50, help="minimum history before testing"
    )
    backtest.add_argument(
        "--save-weights",
        action="store_true",
        help="save backtest-derived soft-voting weights",
    )
    
    args = parser.parse_args()
    store = HistoryStore(args.data)
    weights_path = args.data.parent / "model_weights.json"
    
    if args.command == "app":
        from .gui import launch
        launch(args.data)
    
    elif args.command == "add":
        added = store.append(args.outcomes)
        print(f"Saved {len(added)} outcome(s). Total: {len(store.load())}.")
        _show_prediction(store.load(), weights_path)
    
    elif args.command == "import":
        added = store.import_lines(args.file)
        print(f"Imported {len(added)} outcome(s). Total: {len(store.load())}.")
    
    elif args.command == "predict":
        _show_prediction(store.load(), weights_path)
    
    elif args.command == "stats":
        print(json.dumps(summary(store.load()), indent=2, default=list))
    
    elif args.command == "backtest":
        result = walk_forward(store.load(), args.warmup)
        rows = result.pop("rows")  # Remove rows from output
        
        if args.save_weights:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            weights_path.write_text(
                json.dumps(result["suggested_weights"], indent=2),
                encoding="utf-8",
            )
            result["weights_saved_to"] = str(weights_path)
        
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
