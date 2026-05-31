"""Train and compare classical baselines on absolute KPI prediction."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neural_simulator.data.dataset import MODEL_KPI_NAMES, load_dataset_jsonl
from neural_simulator.evaluation.metrics import regression_metrics
from neural_simulator.models.baselines import engineer_features
from neural_simulator.models.heuristic import predict_scenario_kpis
from neural_simulator.simulation.scenario import SupplyChainScenario
from neural_simulator.simulation.simulator import run_simulation


def evaluate_baseline_scorecard(
    *,
    train_path: str | Path,
    test_path: str | Path,
    random_state: int = 0,
    random_forest_estimators: int = 200,
) -> dict[str, Any]:
    """Evaluate mean, linear, random forest baselines on absolute KPI prediction."""

    train_rows = load_dataset_jsonl(train_path)
    test_rows = load_dataset_jsonl(test_path)
    if not train_rows:
        raise ValueError(f"training split is empty: {train_path}")
    if not test_rows:
        raise ValueError(f"test split is empty: {test_path}")

    feature_names = sorted(engineer_features(
        SupplyChainScenario.from_dict(_row_scenario(train_rows[0]))
    ))
    x_train = _feature_matrix(train_rows, feature_names)
    x_test = _feature_matrix(test_rows, feature_names)
    y_train = _target_matrix(train_rows)
    y_test = _target_matrix(test_rows)

    # --- Mean predictor ---
    mean_prediction = np.repeat(y_train.mean(axis=0, keepdims=True), len(test_rows), axis=0)

    # --- Linear Ridge ---
    linear_regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    linear_regressor.fit(x_train, y_train)

    started_at = perf_counter()
    linear_pred = linear_regressor.predict(x_test)
    linear_elapsed = perf_counter() - started_at

    # --- Random Forest ---
    forest_regressor = RandomForestRegressor(
        n_estimators=random_forest_estimators,
        random_state=random_state,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    forest_regressor.fit(x_train, y_train)

    started_at = perf_counter()
    forest_pred = forest_regressor.predict(x_test)
    forest_elapsed = perf_counter() - started_at

    # --- Gradient Boosting ---
    gradient_boosting_regressor = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=200,
            min_samples_leaf=2,
            learning_rate=0.05,
            random_state=random_state,
        )
    )
    gradient_boosting_regressor.fit(x_train, y_train)

    started_at = perf_counter()
    gb_pred = gradient_boosting_regressor.predict(x_test)
    gb_elapsed = perf_counter() - started_at

    # --- Physics-style workload heuristic ---
    started_at = perf_counter()
    physics_pred = _physics_prediction_matrix(test_rows)
    physics_elapsed = perf_counter() - started_at

    # --- SimPy ground truth timing ---
    simpy_elapsed = _time_simpy(test_rows)

    scorecard = {
        "dataset": {
            "train_examples": len(train_rows),
            "test_examples": len(test_rows),
            "feature_count": len(feature_names),
        },
        "models": {
            "mean": _model_result(y_test, mean_prediction, elapsed=0.0),
            "linear_ridge": _model_result(y_test, linear_pred, elapsed=linear_elapsed),
            "random_forest": _model_result(y_test, forest_pred, elapsed=forest_elapsed),
            "hist_gradient_boosting": _model_result(y_test, gb_pred, elapsed=gb_elapsed),
            "physics_heuristic": _model_result(
                y_test,
                physics_pred,
                elapsed=physics_elapsed,
            ),
        },
        "simpy_benchmark": {
            "test_examples": len(test_rows),
            "elapsed_seconds": simpy_elapsed,
            "examples_per_second": len(test_rows) / simpy_elapsed
            if simpy_elapsed > 0
            else 0.0,
        },
    }
    return scorecard


def _feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            [
                engineer_features(
                    SupplyChainScenario.from_dict(_row_scenario(row))
                )[name]
                for name in feature_names
            ]
            for row in rows
        ],
        dtype=float,
    )


def _target_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[_row_kpis(row)[name] for name in MODEL_KPI_NAMES] for row in rows],
        dtype=float,
    )


def _physics_prediction_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                predict_scenario_kpis(
                    SupplyChainScenario.from_dict(_row_scenario(row))
                )["kpis"][name]
                for name in MODEL_KPI_NAMES
            ]
            for row in rows
        ],
        dtype=float,
    )


def _model_result(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "kpi_regression": regression_metrics(
            _matrix_to_dicts(y_true),
            _matrix_to_dicts(y_pred),
        ),
        "elapsed_seconds": elapsed,
    }


def _matrix_to_dicts(values: np.ndarray) -> list[dict[str, float]]:
    return [
        {name: float(value) for name, value in zip(MODEL_KPI_NAMES, row, strict=True)}
        for row in values
    ]


def _time_simpy(rows: list[dict[str, Any]]) -> float:
    started_at = perf_counter()
    for row in rows:
        scenario = SupplyChainScenario.from_dict(_row_scenario(row))
        run_simulation(scenario)
    return perf_counter() - started_at


def _row_scenario(row: dict[str, Any]) -> dict[str, Any]:
    return row["scenario"]


def _row_kpis(row: dict[str, Any]) -> dict[str, float]:
    return row["kpis"]
