"""Graph Transformer surrogate training loop."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
import random
from typing import Any

from neural_simulator.data.dataset import MODEL_KPI_NAMES, load_dataset_jsonl
from neural_simulator.graphs.pyg import graph_dict_to_pyg_data
from neural_simulator.models.heuristic import predict_scenario_kpis
from neural_simulator.models.graph_transformer import build_graph_transformer_surrogate
from neural_simulator.simulation.scenario import SupplyChainScenario

RESIDUAL_KPI_NAMES = {"makespan", "throughput"}


@dataclass(frozen=True)
class TrainingSummary:
    checkpoint_path: Path
    train_loss: float
    validation_loss: float | None
    epochs: int
    train_examples: int
    validation_examples: int
    device: str
    best_epoch: int
    best_validation_loss: float | None
    stopped_early: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "epochs": self.epochs,
            "train_examples": self.train_examples,
            "validation_examples": self.validation_examples,
            "device": self.device,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "stopped_early": self.stopped_early,
        }


def train_graph_transformer(
    *,
    train_path: str | Path,
    checkpoint_path: str | Path,
    validation_path: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    hidden_dim: int = 64,
    layers: int = 3,
    heads: int = 4,
    dropout: float = 0.1,
    device: str = "auto",
    early_stopping_patience: int | None = 10,
    min_delta: float = 1e-4,
    node_encoder_type: str = "schema_attention",
    max_grad_norm: float = 1.0,
    target_transform: str = "log1p",
    machine_loss_weight: float = 0.2,
    target_mode: str = "hybrid_log1p",
) -> TrainingSummary:
    try:
        import torch
        from torch_geometric.data import Batch
    except ImportError as exc:
        raise ImportError("Install ML dependencies with `uv sync --group ml`.") from exc

    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be >= 1 or None")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")
    if target_transform not in {"identity", "log1p"}:
        raise ValueError("target_transform must be one of: identity, log1p")
    if target_mode not in {"absolute", "heuristic_residual_log1p", "hybrid_log1p"}:
        raise ValueError(
            "target_mode must be one of: absolute, heuristic_residual_log1p, hybrid_log1p"
        )
    if machine_loss_weight < 0:
        raise ValueError("machine_loss_weight must be non-negative")

    torch_device = _resolve_device(torch, device)
    train_examples = _load_examples(train_path, target_mode)
    validation_examples = (
        _load_examples(validation_path, target_mode) if validation_path else []
    )
    if not train_examples:
        raise ValueError("training split is empty")

    first_graph = train_examples[0]["graph"]
    effective_target_transform = (
        "identity"
        if target_mode in {"heuristic_residual_log1p", "hybrid_log1p"}
        else target_transform
    )
    target_stats = _compute_target_stats(train_examples, effective_target_transform)
    feature_stats = _compute_feature_stats(train_examples)
    model = build_graph_transformer_surrogate(
        node_feature_dim=first_graph.x.shape[-1],
        edge_feature_dim=first_graph.edge_attr.shape[-1],
        output_dim=len(MODEL_KPI_NAMES),
        hidden_dim=hidden_dim,
        layers=layers,
        heads=heads,
        dropout=dropout,
        node_encoder_type=node_encoder_type,
    )
    model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.MSELoss()
    rng = random.Random(0)

    train_loss = 0.0
    validation_loss = None
    best_epoch = 0
    best_validation_loss: float | None = None
    best_state_dict = copy.deepcopy(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )
    epochs_without_improvement = 0
    completed_epochs = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train_examples)
        train_loss = _run_epoch(
            model=model,
            examples=train_examples,
            batch_size=batch_size,
            loss_fn=loss_fn,
            batch_factory=Batch,
            target_stats=target_stats,
            feature_stats=feature_stats,
            device=torch_device,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
            machine_loss_weight=machine_loss_weight,
        )
        scheduler.step()
        completed_epochs = epoch

        if validation_examples:
            model.eval()
            with torch.no_grad():
                validation_loss = _run_epoch(
                    model=model,
                    examples=validation_examples,
                    batch_size=batch_size,
                    loss_fn=loss_fn,
                    batch_factory=Batch,
                    target_stats=target_stats,
                    feature_stats=feature_stats,
                    device=torch_device,
                    optimizer=None,
                    max_grad_norm=max_grad_norm,
                    machine_loss_weight=machine_loss_weight,
                )
            if (
                best_validation_loss is None
                or validation_loss < best_validation_loss - min_delta
            ):
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(
                    {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    }
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                stopped_early = True
                break

    if validation_examples:
        model.load_state_dict(best_state_dict)
        validation_loss = best_validation_loss
    else:
        best_epoch = completed_epochs
        best_state_dict = {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        }
        validation_loss = None

    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "kpi_names": MODEL_KPI_NAMES,
            "node_feature_dim": first_graph.x.shape[-1],
            "edge_feature_dim": first_graph.edge_attr.shape[-1],
            "hidden_dim": hidden_dim,
            "layers": layers,
            "heads": heads,
            "dropout": dropout,
            "node_encoder_type": node_encoder_type,
            "target_stats": target_stats,
            "feature_stats": feature_stats,
            "training_device": str(torch_device),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "completed_epochs": completed_epochs,
            "early_stopping_patience": early_stopping_patience,
            "min_delta": min_delta,
            "target_transform": target_transform,
            "effective_target_transform": effective_target_transform,
            "target_mode": target_mode,
            "machine_loss_weight": machine_loss_weight,
        },
        target,
    )

    return TrainingSummary(
        checkpoint_path=target,
        train_loss=train_loss,
        validation_loss=validation_loss,
        epochs=completed_epochs,
        train_examples=len(train_examples),
        validation_examples=len(validation_examples),
        device=str(torch_device),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=stopped_early,
    )


def _load_examples(path: str | Path | None, target_mode: str) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = load_dataset_jsonl(path)
    return [_build_example(row, target_mode) for row in rows]


def _build_example(row: dict[str, Any], target_mode: str) -> dict[str, Any]:
    graph = graph_dict_to_pyg_data(_row_graph(row))
    _add_machine_utilization_targets(graph, row["machine_utilization"])
    return {
        "graph": graph,
        "y": _target_values(row, target_mode),
    }


def _target_values(row: dict[str, Any], target_mode: str) -> list[float]:
    import math

    kpis = _row_kpis(row)
    if target_mode == "absolute":
        return [kpis[name] for name in MODEL_KPI_NAMES]
    if target_mode in {"heuristic_residual_log1p", "hybrid_log1p"}:
        scenario = SupplyChainScenario.from_dict(_row_scenario(row))
        heuristic_kpis = predict_scenario_kpis(scenario)["kpis"]
        values = []
        for name in MODEL_KPI_NAMES:
            truth_log = math.log1p(max(0.0, kpis[name]))
            if target_mode == "heuristic_residual_log1p" or name in RESIDUAL_KPI_NAMES:
                values.append(truth_log - math.log1p(max(0.0, heuristic_kpis[name])))
            else:
                values.append(truth_log)
        return values
    raise ValueError(f"unsupported target_mode: {target_mode}")


def _add_machine_utilization_targets(
    graph,
    machine_utilization: dict[str, float],
) -> None:
    import torch

    values = torch.zeros(graph.x.shape[0], dtype=torch.float32)
    mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
    for idx, node_id in enumerate(graph.node_id):
        if not node_id.startswith("machine:"):
            continue
        machine_id = node_id.split(":", maxsplit=1)[1]
        if machine_id not in machine_utilization:
            continue
        values[idx] = float(machine_utilization[machine_id])
        mask[idx] = True
    graph.machine_utilization_y = values
    graph.machine_utilization_mask = mask


def _row_graph(row: dict[str, Any]) -> dict[str, Any]:
    return row["graph"]


def _row_kpis(row: dict[str, Any]) -> dict[str, float]:
    return row["kpis"]


def _row_scenario(row: dict[str, Any]) -> dict[str, Any]:
    return row["scenario"]


def _run_epoch(
    *,
    model,
    examples: list[dict[str, Any]],
    batch_size: int,
    loss_fn,
    batch_factory,
    target_stats: dict[str, Any],
    feature_stats: dict[str, dict[str, list[float]]],
    device,
    optimizer,
    max_grad_norm: float,
    machine_loss_weight: float,
) -> float:
    import torch

    total_loss = 0.0
    total_examples = 0

    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        batch = batch_factory.from_data_list(
            [example["graph"] for example in batch_examples]
        ).to(device)
        _normalize_graph_batch(batch, feature_stats)
        y = _target_tensor(
            [example["y"] for example in batch_examples],
            target_stats["kpis"],
            target_stats["transform"],
            device,
        )

        pred, machine_utilization_pred = model.forward_with_aux(batch)
        loss = loss_fn(pred, y)
        if machine_loss_weight > 0 and batch.machine_utilization_mask.any():
            machine_loss = loss_fn(
                machine_utilization_pred[batch.machine_utilization_mask],
                batch.machine_utilization_y[batch.machine_utilization_mask],
            )
            loss = loss + machine_loss_weight * machine_loss

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += float(loss.detach()) * len(batch_examples)
        total_examples += len(batch_examples)

    return total_loss / total_examples


def _compute_target_stats(
    examples: list[dict[str, Any]],
    target_transform: str,
) -> dict[str, Any]:
    import torch

    values = torch.tensor([example["y"] for example in examples], dtype=torch.float32)
    transformed = _transform_targets(values, target_transform)
    return {
        "kpis": _stats_tensor(transformed),
        "transform": target_transform,
    }


def _compute_feature_stats(
    examples: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    import torch

    node_features = torch.cat(
        [example["graph"].x for example in examples],
        dim=0,
    )
    edge_features = torch.cat(
        [
            example["graph"].edge_attr
            for example in examples
            if example["graph"].edge_attr.numel() > 0
        ],
        dim=0,
    )
    graph_features = torch.cat(
        [example["graph"].graph_features for example in examples],
        dim=0,
    )
    return {
        "node": _stats_tensor(node_features),
        "edge": _stats_tensor(edge_features),
        "graph": _stats_tensor(graph_features),
    }


def _stats_tensor(tensor) -> dict[str, list[float]]:
    std = tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {
        "mean": tensor.mean(dim=0).tolist(),
        "std": std.tolist(),
    }


def _normalize_graph_batch(batch, feature_stats: dict[str, dict[str, list[float]]]) -> None:
    node_mean = batch.x.new_tensor(feature_stats["node"]["mean"])
    node_std = batch.x.new_tensor(feature_stats["node"]["std"])
    batch.x = (batch.x - node_mean) / node_std
    if batch.edge_attr.numel() > 0:
        edge_mean = batch.edge_attr.new_tensor(feature_stats["edge"]["mean"])
        edge_std = batch.edge_attr.new_tensor(feature_stats["edge"]["std"])
        batch.edge_attr = (batch.edge_attr - edge_mean) / edge_std
    if hasattr(batch, "graph_features"):
        graph_mean = batch.graph_features.new_tensor(feature_stats["graph"]["mean"])
        graph_std = batch.graph_features.new_tensor(feature_stats["graph"]["std"])
        batch.graph_features = (batch.graph_features - graph_mean) / graph_std


def _target_tensor(
    values: list[list[float]],
    stats: dict[str, list[float]],
    transform: str,
    device,
):
    import torch

    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    tensor = _transform_targets(tensor, transform)
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
    return (tensor - mean) / std


def _transform_targets(tensor, transform: str):
    import torch

    if transform == "identity":
        return tensor
    if transform == "log1p":
        return torch.log1p(tensor.clamp_min(0.0))
    raise ValueError(f"unsupported target_transform: {transform}")


def _resolve_device(torch, device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(device)
