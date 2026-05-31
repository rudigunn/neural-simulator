"""Metric helpers for KPI and decision-oriented evaluation."""

from __future__ import annotations


def regression_metrics(
    y_true: list[dict[str, float]],
    y_pred: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("metric inputs cannot be empty")

    names = sorted(y_true[0])
    metrics = {}
    for name in names:
        true_values = [item[name] for item in y_true]
        pred_values = [item[name] for item in y_pred]
        errors = [pred - true for true, pred in zip(true_values, pred_values)]
        mae = sum(abs(error) for error in errors) / len(errors)
        rmse = (sum(error**2 for error in errors) / len(errors)) ** 0.5
        mean_true = sum(true_values) / len(true_values)
        total_sum_squares = sum((true - mean_true) ** 2 for true in true_values)
        residual_sum_squares = sum(error**2 for error in errors)
        r2 = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares else 0.0
        metrics[name] = {"mae": mae, "rmse": rmse, "r2": r2}
    return metrics


def bottleneck_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("metric inputs cannot be empty")
    return sum(true == pred for true, pred in zip(y_true, y_pred)) / len(y_true)
