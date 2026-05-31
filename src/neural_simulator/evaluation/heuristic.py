"""Evaluate the fast heuristic surrogate on saved dataset splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neural_simulator.data.dataset import load_dataset_jsonl
from neural_simulator.evaluation.metrics import bottleneck_accuracy, regression_metrics
from neural_simulator.models.heuristic import predict_scenario_kpis
from neural_simulator.simulation.scenario import SupplyChainScenario


def evaluate_heuristic_dataset(path: str | Path) -> dict[str, Any]:
    items = load_dataset_jsonl(path)
    if not items:
        raise ValueError(f"dataset split is empty: {path}")

    true_kpis = []
    predicted_kpis = []
    true_bottlenecks = []
    predicted_bottlenecks = []

    for item in items:
        scenario = SupplyChainScenario.from_dict(_row_scenario(item))
        prediction = predict_scenario_kpis(scenario)

        true_kpis.append(_row_kpis(item))
        predicted_kpis.append(prediction["kpis"])
        true_bottlenecks.append(_row_bottleneck(item))
        predicted_bottlenecks.append(prediction["bottleneck_machine"])

    return {
        "kpi_regression": regression_metrics(true_kpis, predicted_kpis),
        "bottleneck_accuracy": bottleneck_accuracy(true_bottlenecks, predicted_bottlenecks),
        "examples": len(items),
    }


def _row_scenario(row: dict[str, Any]) -> dict[str, Any]:
    return row["scenario"]


def _row_kpis(row: dict[str, Any]) -> dict[str, float]:
    return row["kpis"]


def _row_bottleneck(row: dict[str, Any]) -> str:
    return str(row["bottleneck_machine"])
