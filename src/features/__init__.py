"""Feature engineering module.

Public API
----------
- :func:`src.features.builder.build_features` — orchestrate all features from
  a merged multi-symbol OHLCV DataFrame using a :class:`TrainingConfig`.
- :mod:`src.features.technical` — pure, testable per-symbol transformations.
"""
