# AGENTS.md

## Purpose
This repository builds an MVP in Python for a machine learning pipeline that trains models to trade a configurable target cryptocurrency.

Default assumptions for the MVP:
- target trading symbol: `ETH`
- signal symbols: `ETH`, `BNB`, `SOL`
- timeframe: `1h`
- task: supervised multiclass classification
- mode: offline historical training and evaluation only

These defaults must be easy to change through configuration. Do not hardcode them across the codebase.

## Environment and secrets

- Känsliga värden (API-nycklar etc.) ska **aldrig** finnas i YAML-filer som commitas.
- Kopiera `.env.example` → `.env` för lokal körning. `.env` är gitignorerad.
- `config/local.yaml` kan användas för lokala icke-känsliga overrides och är gitignorerad.
  Kopiera `config/local.yaml.example` → `config/local.yaml` för lokal dev.
- `config/training.yaml` innehåller bara ML-parametrar och är alltid säker att committa.
- Om ett krav-env-var (`CRYPTAN_DATA_API_KEY`, `CRYPTAN_DATA_API_SECRET`) saknas eller
  fortfarande har värdet `changeme` kastas `EnvironmentError` med ett tydligt meddelande.

## Work style for agents
When implementing work in this repository:
1. Make a short plan.
2. Keep scope narrow and finish one pipeline step at a time.
3. Prefer small, reviewable changes over broad speculative refactors.
4. Update or add tests with every meaningful behavior change.
5. Explain assumptions when they matter.

## MVP boundaries
Implement the minimum clean system that supports:
1. historical multi-symbol OHLCV ingestion
2. preprocessing and multi-symbol merge
3. feature engineering
4. target generation for the target symbol
5. chronological train/validation/test split
6. baseline model training
7. evaluation with ML metrics and simple trading metrics
8. model and metadata persistence

Out of scope unless explicitly requested:
- live order execution
- websocket streaming
- reinforcement learning
- deep learning architectures
- distributed microservices
- elaborate MLOps platforms

## Architecture rules
- Use a `src/` layout.
- Keep modules small and responsibility-focused.
- Separate ingestion, preprocessing, features, labels, split, models, evaluation, and pipeline orchestration.
- Prefer pure transformations for feature generation and labeling.
- Keep I/O at the edges.
- Use configuration to control symbols, timeframe, thresholds, model type, and artifact paths.

## Time-series safety rules
This is financial time-series work. Be strict.

- Never use random shuffling for train/test evaluation.
- Never leak future information into features.
- Build labels carefully with explicit shifting.
- Keep timestamps in UTC.
- Sort ascending by timestamp before merge, feature creation, and split.
- Add tests around alignment, shifting, and lookback windows.

## Data rules
Normalized OHLCV schema should be easy to work with.
Preferred minimal columns:
- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `symbol`

After merging multiple symbols, use clear symbol-prefixed column names such as:
- `ETH_close`
- `BNB_volume`
- `SOL_return_5`

Do not allow ambiguous duplicate columns.

## Model rules
For the MVP, prefer a strong but simple baseline.
Recommended first models:
- `LogisticRegression`
- `RandomForestClassifier`
- `HistGradientBoostingClassifier`

Start with one baseline model and keep registry-driven extension easy.

## Evaluation rules
Do not stop at raw classifier metrics.
Compute at least:
- accuracy
- precision
- recall
- F1
- confusion matrix
- simple strategy returns
- cumulative returns
- hit rate
- max drawdown

Backtest logic should remain transparent and configurable, including transaction fees.

## Artifacts
Each training run should save:
- model artifact
- config snapshot
- feature column list
- metrics file
- run metadata

Use timestamped run directories under `artifacts/`.

## Coding style
- Python 3.11+
- type hints on public interfaces
- meaningful exceptions
- `pathlib.Path` for file paths
- stdlib `logging`
- `pytest` for tests
- prefer clarity over cleverness

## Definition of done for a change
A change is not done until:
- code is coherent with MVP scope
- config drives behavior instead of hidden constants
- time-series correctness is preserved
- tests cover the risky logic
- the change is understandable by a new contributor
