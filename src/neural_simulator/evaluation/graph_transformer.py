"""Evaluate saved Graph Transformer surrogate checkpoints."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from neural_simulator.data.dataset import MODEL_KPI_NAMES, load_dataset_jsonl
from neural_simulator.evaluation.metrics import bottleneck_accuracy
from neural_simulator.evaluation.metrics import regression_metrics
from neural_simulator.graphs.pyg import graph_dict_to_pyg_data
from neural_simulator.models.graph_transformer import build_graph_transformer_surrogate
from neural_simulator.models.heuristic import predict_scenario_kpis
from neural_simulator.simulation.scenario import SupplyChainScenario

RESIDUAL_KPI_NAMES = {"makespan", "throughput"}


def evaluate_graph_transformer_checkpoint(
    *,
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    batch_size: int = 64,
    device: str = "auto",
) -> dict[str, Any]:
    """Load a checkpoint and evaluate it on a saved JSONL split."""

    try:
        import torch
        from torch_geometric.data import Batch
    except ImportError as exc:
        raise ImportError("Install ML dependencies with `uv sync --group ml`.") from exc

    torch_device = _resolve_device(torch, device)
    rows = load_dataset_jsonl(dataset_path)
    if not rows:
        raise ValueError(f"dataset split is empty: {dataset_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    kpi_names = list(checkpoint.get("kpi_names", MODEL_KPI_NAMES))
    model = build_graph_transformer_surrogate(
        node_feature_dim=int(checkpoint["node_feature_dim"]),
        edge_feature_dim=int(checkpoint["edge_feature_dim"]),
        output_dim=len(kpi_names),
        hidden_dim=int(checkpoint["hidden_dim"]),
        layers=int(checkpoint["layers"]),
        heads=int(checkpoint["heads"]),
        dropout=float(checkpoint.get("dropout", 0.1)),
        node_encoder_type=checkpoint.get("node_encoder_type", "linear"),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(torch_device)
    model.eval()

    true_kpis: list[dict[str, float]] = []
    pred_kpis: list[dict[str, float]] = []
    true_bottlenecks: list[str] = []
    pred_bottlenecks: list[str] = []
    utilization_absolute_errors: list[float] = []

    started_at = perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch = Batch.from_data_list(
                [graph_dict_to_pyg_data(_row_graph(row)) for row in batch_rows]
            ).to(torch_device)
            feature_stats = checkpoint.get("feature_stats")
            if feature_stats is not None:
                _normalize_graph_batch(batch, feature_stats)
            if hasattr(model, "forward_with_aux"):
                output, utilization_output = model.forward_with_aux(batch)
                _collect_machine_utilization_metrics(
                    batch_rows,
                    batch,
                    utilization_output,
                    true_bottlenecks,
                    pred_bottlenecks,
                    utilization_absolute_errors,
                )
            else:
                output = model(batch)

            pred_kpis.extend(
                _tensor_to_dicts(
                    output,
                    kpi_names,
                    checkpoint.get("target_stats"),
                    checkpoint.get("target_mode", "absolute"),
                    batch_rows,
                )
            )
            true_kpis.extend(
                [
                    {name: float(_row_kpis(row)[name]) for name in kpi_names}
                    for row in batch_rows
                ]
            )

    elapsed_seconds = perf_counter() - started_at
    result = {
        "checkpoint": str(checkpoint_path),
        "dataset": {
            "path": str(dataset_path),
            "examples": len(rows),
            "batch_size": batch_size,
            "elapsed_seconds": elapsed_seconds,
            "examples_per_second": len(rows) / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0,
            "device": str(torch_device),
        },
        "kpi_regression": regression_metrics(true_kpis, pred_kpis),
    }
    if utilization_absolute_errors:
        result["machine_utilization"] = {
            "mae": sum(utilization_absolute_errors) / len(utilization_absolute_errors),
            "bottleneck_accuracy": bottleneck_accuracy(true_bottlenecks, pred_bottlenecks),
        }
    return result


def _tensor_to_dicts(
    tensor,
    kpi_names: list[str],
    target_stats: dict[str, dict[str, list[float]]] | None,
    target_mode: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, float]]:
    if target_stats is not None:
        stats = target_stats["kpis"]
        mean = tensor.new_tensor(stats["mean"])
        std = tensor.new_tensor(stats["std"])
        tensor = tensor * std + mean
        if target_mode == "heuristic_residual_log1p":
            tensor = (tensor + _heuristic_log_tensor(rows, kpi_names, tensor)).expm1()
            tensor = tensor.clamp_min(0.0)
        elif target_mode == "hybrid_log1p":
            tensor = _decode_hybrid_log_tensor(tensor, rows, kpi_names)
        elif target_stats.get("transform") == "log1p":
            tensor = tensor.expm1().clamp_min(0.0)
    values = tensor.detach().cpu().tolist()
    return [
        {name: float(value) for name, value in zip(kpi_names, row, strict=True)}
        for row in values
    ]


def _normalize_graph_batch(batch, feature_stats: dict[str, dict[str, list[float]]]) -> None:
    node_mean = batch.x.new_tensor(feature_stats["node"]["mean"])
    node_std = batch.x.new_tensor(feature_stats["node"]["std"])
    batch.x = (batch.x - node_mean) / node_std
    if batch.edge_attr.numel() > 0:
        edge_mean = batch.edge_attr.new_tensor(feature_stats["edge"]["mean"])
        edge_std = batch.edge_attr.new_tensor(feature_stats["edge"]["std"])
        batch.edge_attr = (batch.edge_attr - edge_mean) / edge_std
    if hasattr(batch, "graph_features") and "graph" in feature_stats:
        graph_mean = batch.graph_features.new_tensor(feature_stats["graph"]["mean"])
        graph_std = batch.graph_features.new_tensor(feature_stats["graph"]["std"])
        batch.graph_features = (batch.graph_features - graph_mean) / graph_std


def _resolve_device(torch, device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(device)


def _row_graph(row: dict[str, Any]) -> dict[str, Any]:
    return row["graph"]


def _row_kpis(row: dict[str, Any]) -> dict[str, float]:
    return row["kpis"]


def _row_scenario(row: dict[str, Any]) -> dict[str, Any]:
    return row["scenario"]


def _heuristic_log_tensor(
    rows: list[dict[str, Any]],
    kpi_names: list[str],
    like,
):
    import torch

    values = []
    for row in rows:
        scenario = SupplyChainScenario.from_dict(_row_scenario(row))
        heuristic_kpis = predict_scenario_kpis(scenario)["kpis"]
        values.append([max(0.0, heuristic_kpis[name]) for name in kpi_names])
    return torch.log1p(like.new_tensor(values))


def _decode_hybrid_log_tensor(tensor, rows: list[dict[str, Any]], kpi_names: list[str]):
    heuristic_log = _heuristic_log_tensor(rows, kpi_names, tensor)
    decoded_columns = []
    for idx, name in enumerate(kpi_names):
        column = tensor[:, idx]
        if name in RESIDUAL_KPI_NAMES:
            column = column + heuristic_log[:, idx]
        decoded_columns.append(column.expm1().clamp_min(0.0))
    import torch

    return torch.stack(decoded_columns, dim=1)


def _collect_machine_utilization_metrics(
    rows: list[dict[str, Any]],
    batch,
    utilization_output,
    true_bottlenecks: list[str],
    pred_bottlenecks: list[str],
    absolute_errors: list[float],
) -> None:
    offset = 0
    predictions = utilization_output.detach().cpu().tolist()
    for row, graph_data in zip(rows, batch.to_data_list(), strict=True):
        machine_predictions: dict[str, float] = {}
        for local_idx, node_id in enumerate(graph_data.node_id):
            if not node_id.startswith("machine:"):
                continue
            machine_id = node_id.split(":", maxsplit=1)[1]
            predicted = float(predictions[offset + local_idx])
            truth = float(row["machine_utilization"][machine_id])
            machine_predictions[machine_id] = predicted
            absolute_errors.append(abs(predicted - truth))
        offset += graph_data.x.shape[0]
        if machine_predictions:
            true_bottlenecks.append(str(row["bottleneck_machine"]))
            pred_bottlenecks.append(
                max(
                    machine_predictions,
                    key=lambda machine_id: (machine_predictions[machine_id], machine_id),
                )
            )
