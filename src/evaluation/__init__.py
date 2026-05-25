"""Evaluation and simple backtest helpers."""

__all__ = ["evaluate_and_save_report", "evaluate_model"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from src.evaluation import report

        return getattr(report, name)
    raise AttributeError(f"module 'src.evaluation' has no attribute {name!r}")
