# GitHub Copilot Instructions

## Purpose
This repository builds an MVP in Python for a machine learning pipeline that trains models to trade a target cryptocurrency.

The default trading target is **ETH**, but this must be configurable.
The default signal/input assets are **ETH, BNB, and SOL**, but these must also be configurable and easy to replace with other assets later.

The MVP should focus on **offline training and evaluation first**, not live trading execution.

---

## Current scope: MVP only
Copilot should optimize for the smallest clean implementation that can realistically grow into the full system later.

The MVP includes these steps:

1. **Data Collection and Preprocessing**
2. **Feature Engineering**
3. **Data Splitting**
4. **Model Selection and Training**
5. **Model Evaluation and Basic Optimization**
6. **Model Serialization for later deployment**
7. **Basic Monitoring hooks for future continuous learning**

The MVP should **not** try to fully implement:

- live order execution against an exchange
- websocket streaming
- reinforcement learning
- deep learning unless explicitly requested
- a complex microservice architecture
- full MLOps infrastructure

Keep the MVP pragmatic, modular, and testable.

---

## Product goal
Build a system that can:

- collect historical market data for multiple assets
- use one asset as the trading target, default `ETH`
- use one or more configurable assets as signal sources, default `ETH`, `BNB`, `SOL`
- generate features from all configured signal sources
- train a model to predict a future move or trading signal for the target asset
- evaluate the model with both ML metrics and simple trading metrics
- save the trained model and metadata to disk

---

## Design principles
When generating code, follow these principles:

### 1. Configuration first
Do not hardcode `ETH`, `BNB`, and `SOL` deep inside business logic.
Use a configuration object or YAML/TOML/JSON config so the following can be changed centrally:

- target trading symbol
- signal symbols
- timeframe
- training period
- prediction horizon
- target threshold
- feature set
- model type

Good default:

```yaml
trading_symbol: ETH
signal_symbols:
  - ETH
  - BNB
  - SOL
timeframe: 1h
prediction_horizon_bars: 12
return_threshold: 0.01
```

### 2. Modular pipeline
Organize code so each step in the ML pipeline is a separate module.
Avoid giant scripts.

### 3. Time-series correctness
This is financial time-series data.
Never use random train/test split.
Always preserve temporal order.
Avoid target leakage.

### 4. Simple abstractions
Use lightweight classes or functions with clear responsibilities.
Avoid over-engineering.
The MVP should be easy to read and extend.

### 5. Reproducibility
Training runs should be reproducible.
Use deterministic seeds where applicable.
Persist model artifacts and training metadata.

### 6. Local-first developer experience
Assume the first version runs locally from CLI.
A command like this should be possible:

```bash
python -m src.pipeline.train_pipeline --config config/training.yaml
```

---

## Preferred tech stack for MVP
Prefer the following unless a later requirement changes it:

- Python 3.11+
- pandas
- numpy
- scikit-learn
- pydantic for config models if useful
- PyYAML for config loading
- joblib for model persistence
- pathlib for file paths
- logging from stdlib
- pytest for tests

Use LightGBM/XGBoost only if explicitly added later.
For the MVP, start with a strong baseline such as:

- LogisticRegression
- RandomForestClassifier
- HistGradientBoostingClassifier

Prefer one baseline model first, then make extension easy.

---

## Recommended repository structure
Copilot should generate code that fits this structure:

```text
.github/
  copilot-instructions.md
config/
  training.yaml
  data_sources.yaml
src/
  ingestion/
    base.py
    market_data.py
  preprocessing/
    cleaner.py
    merger.py
  features/
    builder.py
    technical.py
  labels/
    target_builder.py
  split/
    timeseries.py
  models/
    registry.py
    train.py
    predict.py
  evaluation/
    metrics.py
    backtest.py
  pipeline/
    train_pipeline.py
  utils/
    io.py
    logging_utils.py
tests/
artifacts/
```

Keep files small and responsibility-focused.

---

## Data model expectations
Use pandas DataFrames with a time index in UTC.

Expected minimal schema for raw OHLCV data per symbol:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `symbol`

After merging signal sources, columns should be clearly prefixed by symbol when needed, for example:

- `ETH_close`
- `ETH_volume`
- `BNB_close`
- `SOL_close`

Do not allow ambiguous duplicate column names.

---

## Data ingestion instructions
For the MVP, implement data ingestion in a way that can later support multiple providers.

### Requirements
- Introduce a base interface or protocol for market data sources.
- Start with one simple historical OHLCV implementation.
- The data source should accept:
  - symbol
  - start datetime
  - end datetime
  - timeframe
- Returned data must be normalized into the repository's standard schema.

### Guidance
Even if only one provider is implemented first, structure it so other providers can be added later without rewriting the pipeline.

Example concepts:

- `MarketDataSource`
- `fetch_ohlcv(symbol, start, end, timeframe)`

Do not bind the whole system directly to one provider's raw response shape.

---

## Preprocessing instructions
The preprocessing stage should:

- parse timestamps
- sort by time ascending
- remove duplicates
- standardize timezone to UTC
- handle missing values carefully
- merge multiple signal assets on timestamp
- keep only rows valid for feature generation and target creation

Avoid aggressive forward-filling unless justified.
Document any fill strategy explicitly.

---

## Feature engineering instructions
This is important for the MVP.
Generate features from all configured signal symbols, not only the trading symbol.

### MVP feature set
Start with simple, explainable features such as:

For each configured signal symbol:

- 1-bar return
- 5-bar return
- 20-bar return
- rolling mean over short window
- rolling mean over longer window
- ratio between short and long moving average
- rolling volatility
- rolling volume mean
- rolling volume z-score when feasible

Also allow optional cross-asset features such as:

- relative strength of target asset vs another signal asset
- rolling correlation between target and another signal asset

### Important rules
- Features must use only current and past data.
- No future information may leak into features.
- Drop rows with insufficient lookback after feature construction.
- Feature creation should be implemented as pure, testable transformations.

---

## Target labeling instructions
For the MVP, use a supervised learning target.
Prefer a simple target that predicts future return direction for the trading symbol.

### Default target
Create a 3-class target for the configured trading symbol:

- `1` for upward move above threshold over the prediction horizon
- `-1` for downward move below negative threshold over the prediction horizon
- `0` otherwise

Example default logic:

- horizon: 12 bars
- threshold: 1%

This target must be configurable.

### Important rules
- Labels are created only from the target trading symbol, default `ETH`.
- Signal symbols can include the target symbol itself.
- Shift logic must be carefully implemented and tested.

---

## Splitting instructions
Use time-series splitting only.

### MVP requirement
Implement one deterministic chronological split into:

- train
- validation
- test

For example:

- 70% train
- 15% validation
- 15% test

### Preferred extension path
Design code so walk-forward validation can be added later.

Do not use `train_test_split(..., shuffle=True)`.
Never shuffle financial time-series data during evaluation.

---

## Model training instructions
For the MVP, start with one baseline classifier.
A good first choice is `RandomForestClassifier` or `LogisticRegression` depending on feature shape and simplicity.

### Requirements
- Use config-driven model creation.
- Separate model registry from training orchestration.
- Fit on train set.
- Optionally evaluate on validation set before final test evaluation.
- Persist model artifact to disk.

### Output artifacts
Save at minimum:

- trained model file
- config used for the run
- feature column list
- metrics JSON or YAML
- timestamped run directory

---

## Evaluation instructions
Evaluate with both ML and trading-oriented metrics.

### Minimum ML metrics
- accuracy
- precision
- recall
- F1 score
- confusion matrix

### Minimum trading-oriented evaluation
Use the model predictions to create a very simple strategy simulation.
This can be naive in the MVP, but should still be implemented clearly.

At minimum compute:

- strategy returns
- cumulative returns
- hit rate
- max drawdown

Assume transaction fee is configurable and included in the backtest.

### Important rules
- Evaluation must be done on validation/test periods not used for training.
- Keep backtest logic simple and transparent.
- Prefer readability over fake sophistication.

---

## Deployment instructions for MVP
Full deployment is out of scope for now, but the code should prepare for it.

### MVP requirement
Implement model serialization and a small prediction interface that can:

- load a trained model from disk
- accept a feature DataFrame or a single feature row
- return predictions

This should be sufficient to support later batch inference.

---

## Monitoring instructions for MVP
Do not build a full monitoring platform yet.
Instead, add simple hooks and placeholders for:

- logging training metrics
- logging run metadata
- detecting missing columns or empty datasets
- detecting feature drift in a later phase

Basic structured logging is enough.

---

## Coding style instructions
When generating code:

- prefer clear names over short names
- add docstrings to public functions/classes
- use type hints
- keep functions small
- isolate I/O from transformation logic where practical
- raise meaningful exceptions
- avoid notebook-style hidden state
- prefer `pathlib.Path` over raw strings for paths

Keep code professional and production-friendly, but not ceremonially enterprise-heavy.

---

## Testing instructions
The MVP should include focused unit tests for critical correctness.
At minimum, generate tests for:

- target creation
- time-series splitting
- feature generation without leakage
- multi-symbol merge behavior
- config loading

If there is a bug risk around shifting or timestamp alignment, add tests first.

---

## What Copilot should prioritize implementing first
When asked to generate or modify code in this repository, prioritize work in this order:

1. configuration model and config files
2. historical data ingestion interface and one provider implementation
3. preprocessing and multi-symbol merge
4. feature engineering for configurable signal symbols
5. target labeling for configurable trading symbol
6. time-series split
7. baseline model training
8. evaluation and simple backtest
9. model save/load support
10. tests and cleanup

---

## Default assumptions for the MVP
Unless the user explicitly changes them, assume:

- trading symbol = `ETH`
- signal symbols = `['ETH', 'BNB', 'SOL']`
- timeframe = `1h`
- task type = multiclass classification
- target = future return direction over configurable horizon
- offline historical training only
- local artifact storage under `artifacts/`

---

## Anti-patterns to avoid
Copilot should avoid generating code that:

- hardcodes symbols in multiple places
- mixes data fetching, feature engineering, training, and evaluation in one script
- uses random data split for time series
- leaks future data into features
- couples the model to one specific data provider schema
- introduces a heavy framework without clear need
- builds live trading before offline evaluation works
- optimizes prematurely for a distributed architecture

---

## Definition of done for the MVP
The MVP is considered done when the repository can:

1. load config with target symbol and signal symbols
2. fetch historical OHLCV for all configured symbols
3. preprocess and merge the datasets
4. create features for all configured signal symbols
5. create labels for the configured trading symbol
6. split data chronologically into train/validation/test
7. train a baseline model
8. evaluate it with ML and simple trading metrics
9. save model and run artifacts to disk
10. run from a single CLI entry point without manual notebook steps

---

## If uncertain
If requirements are ambiguous, choose the simpler MVP-friendly implementation that preserves clean extension paths.
Favor clarity, correctness, and time-series safety over cleverness.
