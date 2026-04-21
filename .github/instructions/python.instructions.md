---
applyTo: "src/**/*.py,tests/**/*.py"
---

# Python instructions

Generate Python that is clean, explicit, and easy to extend for the crypto ML MVP.

## General rules
- Use Python 3.11+ syntax.
- Add type hints to public functions, methods, and dataclasses/models.
- Prefer `dataclass` or `pydantic` models for structured config and outputs when helpful.
- Keep functions small and focused.
- Use descriptive names instead of abbreviations.
- Prefer pure functions for transformations.
- Keep I/O, network calls, and filesystem access separate from business logic.
- Use `pathlib.Path` instead of raw string paths.
- Use stdlib `logging` rather than `print`.

## Project-specific rules
- Symbols such as `ETH`, `BNB`, and `SOL` are defaults only. They must come from config, not be hardcoded across modules.
- Preserve time-series ordering.
- Never introduce random train/test splits for time-series data.
- Never use future information in features.
- Make all important windows, thresholds, and model choices configurable.

## Error handling
- Raise meaningful exceptions with actionable messages.
- Validate required columns before processing DataFrames.
- Fail fast on empty DataFrames, duplicate timestamps where invalid, or missing feature columns.
- Avoid silent coercion of malformed input.

## DataFrame conventions
- Use UTC timestamps.
- Sort by timestamp ascending before transformations that depend on order.
- Avoid chained assignment patterns that make behavior unclear.
- Prefer explicit column creation and return a new or clearly mutated DataFrame.
- Use symbol-prefixed columns after multi-symbol merge.

## Model and pipeline conventions
- Separate model registry from model training orchestration.
- Save artifacts with clear filenames.
- Keep CLI entry points thin and delegate to pipeline functions.
- Make it possible to run the MVP locally end-to-end.

## Testing expectations
- Add pytest tests for risky logic.
- Prefer deterministic synthetic fixtures over external data.
- Test for time-series correctness, shifting behavior, merge behavior, and config loading.
