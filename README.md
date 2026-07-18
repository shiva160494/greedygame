# Greedy Game - Analysis & Prediction Toolkit

A comprehensive Python toolkit for analyzing and predicting game outcomes with advanced statistical models and bias detection.

## Features

### V2 - Advanced Game-Specific Models

- **Frequency Bias Detection**: Identifies statistically significant outcome biases (>1% deviation)
- **Streak Momentum Analysis**: Detects if streaks influence subsequent outcomes
- **Inverse Frequency Modeling**: Identifies punishment patterns for repetitive outcomes
- **Cycle Detection**: Discovers repeating sequences in outcomes (4-12 length)
- **Statistical Significance Testing**: Chi-square validation for >95% confidence
- **Confidence Scoring**: Measures how much better predictions are vs random baseline (12.5%)

### Desktop Application

- Real-time outcome recording
- Live predictions with confidence scores
- Historical analysis and statistics
- Walk-forward backtesting
- Model weight optimization

## Models Included

1. **FrequencyBiasModel** - Detects if certain outcomes appear significantly more/less often
2. **StreakMomentumModel** - Analyzes what typically happens after consecutive identical outcomes
3. **InverseFrequencyModel** - Detects if overused outcomes are punished with specific alternatives
4. **CycleDetectionModel** - Identifies repeating cycle patterns in the sequence
5. **ConfidenceFilter** - Statistical validation using chi-square testing

## Installation

```bash
git clone https://github.com/shiva160494/greedygame.git
cd greedygame
python -m pip install -r requirements.txt
```

## Quick Start

### Desktop Application

**Windows:**
```bash
start_app.bat
```

**PowerShell:**
```bash
.\run.ps1 app
```

**Python:**
```bash
python -m greedygame.cli app
```

### Command Line

```bash
# Add outcomes
python -m greedygame.cli add Tomato Chicken Corn

# Get predictions with confidence scores
python -m greedygame.cli predict

# Run walk-forward backtesting
python -m greedygame.cli backtest --warmup 50 --save-weights

# Show detailed statistics
python -m greedygame.cli stats
```

### Python API

```python
from src.greedygame.engine import PredictionEngine
from src.greedygame.storage import HistoryStore

# Load game history
store = HistoryStore("data/outcomes.csv")
history = store.load()

# Get advanced predictions
engine = PredictionEngine()
predictions = engine.predict(history)

print(f"Top prediction: {predictions['ranking'][0][0]}")
print(f"Confidence: {predictions['confidence']:.1%}")
print(f"Statistical significance: {predictions['statistical_significance']}")
print(f"Models with signal: {predictions['support']}/4")
```

## Key Improvements (V2 vs V1)

| Feature | V1 | V2 |
|---------|----|----|  
| Pattern Matching | Rigid exact sequences | Recency-weighted bias detection |
| Game-Specific | Generic patterns | Frequency bias, momentum, cycles |
| Confidence Metric | None | 0-1 score vs random baseline |
| Random Detection | Always predicts | Returns uniform if no signal |
| Statistical Testing | None | Chi-square validation (95% CI) |
| Accuracy Baseline | 12.5% | Detects when beating baseline |

## Understanding the Output

### Prediction Response

```json
{
  "ranking": [
    ["Tomato", 0.18],
    ["Chicken", 0.15],
    ["Corn", 0.14],
    ["Shrimp", 0.13]
  ],
  "confidence": 0.45,
  "models": {
    "frequency_bias": {"Tomato": 0.28, "Chicken": 0.12},
    "streak_momentum": {"Corn": 0.19},
    "inverse_frequency": {},
    "cycle": {}
  },
  "support": 2,
  "statistical_significance": true
}
```

### Key Metrics

- **ranking**: Sorted list of (outcome, probability) tuples
- **confidence**: Improvement over random baseline (0-1). 
  - < 0.2 = likely random
  - 0.2-0.5 = weak patterns
  - > 0.5 = strong patterns
- **support**: Number of models detecting signal (0-4)
- **statistical_significance**: Whether any outcome shows >95% confidence bias
- **models**: Individual predictions from each model

## Backtest Results

### Walk-Forward Validation

Tests predictions on future outcomes (no data leakage):

```bash
python -m greedygame.cli backtest --warmup 50 --save-weights
```

**Metrics:**
- **Top-1 Accuracy**: % correct in position 1
- **Top-3 Accuracy**: % correct in top 3
- **Top-5 Accuracy**: % correct in top 5
- **Baseline**: 12.5% for truly random 8-outcome game
- **Model Accuracy**: Individual performance per model

### Example Output

```json
{
  "predictions": 150,
  "top1_accuracy": 15.3,
  "top3_accuracy": 32.7,
  "top5_accuracy": 48.0,
  "chance_top1": 12.5,
  "model_accuracy": {
    "frequency_bias": 18.5,
    "streak_momentum": 16.2,
    "inverse_frequency": 14.1,
    "cycle": 13.8
  },
  "suggested_weights": {
    "frequency_bias": 0.32,
    "streak_momentum": 0.28,
    "inverse_frequency": 0.22,
    "cycle": 0.18
  }
}
```

## Statistical Concepts

### Confidence Score

Measures improvement over random baseline:

```
confidence = min((top_prob - 0.125) / 0.125, 1.0)

12.5% = 0% improvement (random)
25.0% = 100% improvement (2x random)
```

### Chi-Square Test

Validates if outcome is statistically biased:

```
χ² = (observed - expected)² / expected

If χ² > 3.841 → 95% confidence outcome is biased
```

### Entropy

Measures randomness of outcomes:

```
Entropy Range: 0 to 3.0
0.0 = perfectly predictable
3.0 = perfectly random (8 equally likely outcomes)
```

## Data Format

Outcomes stored in `data/outcomes.csv`:

```csv
spin,outcome
1,Tomato
2,Chicken
3,Tomato
4,Corn
5,Chicken
```

## Troubleshooting

### Q: Accuracy stays at 12.5%?

**A:** The game is likely truly random. Check:
- `confidence` score - if < 0.2, no exploitable patterns
- `statistical_significance` - if false, no bias detected
- Try collecting 200+ spins for more reliable patterns

### Q: Some models show signal but low accuracy?

**A:** Weak patterns detected but not yet exploitable:
1. Collect more data (100+ minimum, 500+ recommended)
2. Run backtest with `--save-weights` to optimize ensemble
3. Check which models work best in backtest results

### Q: How do I optimize model weights?

**A:** Use walk-forward backtesting:

```bash
python -m greedygame.cli backtest --warmup 50 --save-weights
```

This saves optimal weights to `data/model_weights.json` based on historical performance.

## When to Use This Toolkit

✅ **Good for:**
- Analyzing game outcome distributions
- Finding biased or exploitable patterns
- Validating if patterns are statistically real
- Testing prediction strategies
- Human-controlled or semi-random games

❌ **Not for:**
- Gambling (highly risky, never guaranteed)
- Perfectly fair/cryptographic RNG
- Real money without extensive validation
- Predicting truly random events

## Architecture

```
src/greedygame/
├── __init__.py              # Package definition
├── engine.py                # Main prediction engine
├── analysis.py              # Historical analysis functions
├── backtest.py              # Walk-forward validation
├── storage.py               # CSV data management
├── cli.py                   # Command-line interface
├── gui.py                   # Desktop application (Tkinter)
└── models/
    ├── base.py              # Abstract base model
    ├── frequency_bias.py    # Bias detection (NEW)
    ├── streak_momentum.py   # Streak analysis (NEW)
    ├── inverse_frequency.py # Punishment patterns (NEW)
    ├── cycle_detection.py   # Cycle finding (NEW)
    ├── confidence.py        # Statistical validation (NEW)
    └── ensemble.py          # Weighted voting
```

## Performance Tuning

### Model Parameters

You can customize model behavior in `engine.py`:

```python
FrequencyBiasModel(min_samples=50, deviation_threshold=0.01)
StreakMomentumModel(min_history=20)
InverseFrequencyModel(window=10, overuse_threshold=3)
CycleDetectionModel(min_cycle_len=4, max_cycle_len=12, min_repeats=2)
```

### Ensemble Weighting

Optimal weights are learned from backtesting or can be manually set:

```python
weights = {
    "frequency_bias": 0.32,
    "streak_momentum": 0.28,
    "inverse_frequency": 0.22,
    "cycle": 0.18
}
engine = PredictionEngine(weights=weights)
```

## License

MIT

## Contributing

Contributions welcome! Focus areas:
- Additional prediction models
- Performance optimization
- Better statistical tests
- UI/UX improvements
