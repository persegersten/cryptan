---
applyTo: "src/ingestion/**/*.py,src/preprocessing/**/*.py"
---

# Ingestion and preprocessing instructions

These files handle historical market data ingestion and normalization for the MVP.

## Scope
Implement only what the MVP needs:
- historical OHLCV ingestion
- schema normalization
- timestamp normalization to UTC
- duplicate handling
- chronological sorting
- multi-symbol merge

Do not implement live trading, streaming, or exchange-specific execution code here.

## Design rules
- Introduce a base interface or protocol for market data providers.
- Provider-specific response parsing must stay inside ingestion adapters.
- The rest of the system should work with a normalized schema, not provider-native payloads.
- Keep adapters easy to swap later.

## Preferred normalized schema
Each symbol dataset should normalize to columns:
- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `symbol`

Return a pandas DataFrame with predictable dtypes.

## Timestamp rules
- Parse timestamps explicitly.
- Convert to UTC.
- Sort ascending.
- Remove duplicates by timestamp using a documented strategy.
- Do not rely on provider ordering.

## Missing data rules
- Do not forward-fill blindly.
- Make fill behavior explicit and minimal.
- If rows are dropped because of invalid or missing critical fields, do so clearly and predictably.
- Prefer transparent preprocessing over aggressive data repair.

## Multi-symbol merge rules
- Merge on timestamp only after each symbol dataset is cleaned.
- Prefix symbol columns after merge so there are no ambiguous duplicate names.
- Ensure the target symbol and signal symbols can be changed via config.
- Assume defaults are `ETH` as target and `ETH`, `BNB`, `SOL` as signal symbols, but do not hardcode that behavior.

## Function design
Prefer functions or classes like:
- `MarketDataSource`
- `fetch_ohlcv(symbol, start, end, timeframe)`
- `clean_market_data(df)`
- `merge_symbol_frames(frames)`

Keep return contracts explicit and easy to test.

## Testing requirements
Whenever ingestion or preprocessing logic is added or changed, add tests for:
- UTC timestamp handling
- duplicate removal
- merge alignment across symbols
- schema normalization
- missing data behavior
