---
applyTo: "tests/**/*.py"
---

# Test instructions

Use pytest and write tests that protect the risky parts of this crypto ML MVP.

## Primary goals
- Catch time-series leakage.
- Catch off-by-one errors in target creation.
- Catch bad timestamp alignment across symbols.
- Catch config regressions.

## Test style
- Prefer small, deterministic, synthetic datasets.
- Do not depend on external APIs or live market data.
- Avoid flaky timing-based tests.
- Keep each test focused on one behavior.
- Use fixtures when they improve readability, not by reflex.

## Highest-priority coverage
Add or maintain tests for:
- config loading with configurable target symbol and signal symbols
- chronological splitting into train, validation, and test sets
- target generation for the configured target symbol
- feature generation with no future leakage
- multi-symbol merge with symbol-prefixed columns
- handling of missing rows and duplicate timestamps

## Assertion guidance
- Assert exact timestamps where alignment matters.
- Assert exact labels where shift logic matters.
- Assert column names after merge and feature generation.
- Assert that training/test boundaries preserve temporal order.
- Assert that insufficient lookback rows are dropped when expected.

## Anti-patterns
Do not write tests that:
- only check that a function returns without error
- rely on network access
- silently accept reordered time-series data
- hide assumptions inside massive fixtures
