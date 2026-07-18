from __future__ import annotations

import argparse
import json
from pathlib import Path
from .analysis import summary
from .backtest import walk_forward
from .engine import PredictionEngine
from .storage import HistoryStore


def _default_data() -> Path:
    return Path.cwd() / "data" / "outcomes.csv"


def _show_prediction(history: list[str], weights_file: Path) -> None:
    prediction = PredictionEngine(weights_file=weights_file).predict(history)
    if not prediction["ranking"]:
        print("Need at least one saved outcome.")
        return
    print("Historical ranking (not a reliable gambling forecast):")
    for rank, (outcome, probability) in enumerate(prediction["ranking"][:5], 1):
        print(f"  {rank}. {outcome:<8} {probability:6.2%}")
    print(f"Pattern models with support: {prediction['support']}/4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy Game historical-analysis engine")
    parser.add_argument("--data", type=Path, default=_default_data(), help="path to outcomes.csv")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="append one or more outcomes")
    add.add_argument("outcomes", nargs="+", help="e.g. Tomato Corn Cabbage")
    imp = commands.add_parser("import", help="append line-separated outcomes from a text file")
    imp.add_argument("file", type=Path)
    commands.add_parser("predict", help="show a historical, model-based ranking")
    commands.add_parser("stats", help="show descriptive statistics")
    commands.add_parser("app", help="open the desktop app")
    backtest = commands.add_parser("backtest", help="run leakage-free walk-forward validation")
    backtest.add_argument("--warmup", type=int, default=30)
    backtest.add_argument("--save-weights", action="store_true", help="save backtest-derived soft-voting weights")
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
        rows = result.pop("rows")
        if args.save_weights:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            weights_path.write_text(json.dumps(result["suggested_weights"], indent=2), encoding="utf-8")
            result["weights_saved_to"] = str(weights_path)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
