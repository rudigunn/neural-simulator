"""Train and evaluate the GraphGPS surrogate with RWSE."""

from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neural_simulator.data.dataset import MODEL_KPI_NAMES, load_dataset_jsonl
from neural_simulator.graphs.pyg import graph_dict_to_pyg_data
from neural_simulator.models.graph_gps import build_graph_gps_surrogate
from neural_simulator.models.heuristic import predict_scenario_kpis
from neural_simulator.simulation.scenario import SupplyChainScenario
from neural_simulator.evaluation.metrics import regression_metrics

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "ood-topology-pilot"
CHECKPOINT_PATH = ROOT / "checkpoints" / "graph_gps.pt"
RESULTS_PATH = ROOT / "results" / "gps_experiment.json"

RESIDUAL_KPI_NAMES = {"makespan", "throughput"}
TARGET_MODE = "hybrid_log1p"

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
HIDDEN_DIM = 64
GPS_LAYERS = 4
HEADS = 4
DROPOUT = 0.1
RWSE_WALK_LENGTH = 20
MAX_GRAD_NORM = 1.0
MACHINE_LOSS_WEIGHT = 0.2
EARLY_STOPPING_PATIENCE = 15
MIN_DELTA = 1e-4
DEVICE = "cuda"


def main() -> None:
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.transforms import AddRandomWalkPE

    torch_device = torch.device(DEVICE)
    print(f"Using device: {torch_device} ({torch.cuda.get_device_name(0)})")

    # ------------------------------------------------------------------
    # Load data & apply RWSE
    # ------------------------------------------------------------------
    rwse_transform = AddRandomWalkPE(walk_length=RWSE_WALK_LENGTH)

    print("Loading training data...")
    train_examples = _load_examples(DATA_DIR / "train.jsonl", rwse_transform)
    print(f"  {len(train_examples)} training examples")

    print("Loading validation data...")
    val_examples = _load_examples(DATA_DIR / "validation.jsonl", rwse_transform)
    print(f"  {len(val_examples)} validation examples")

    print("Loading test data...")
    test_examples = _load_examples(DATA_DIR / "test.jsonl", rwse_transform)
    print(f"  {len(test_examples)} test examples")

    # ------------------------------------------------------------------
    # Compute normalization stats from training set
    # ------------------------------------------------------------------
    target_stats = _compute_target_stats(train_examples)
    feature_stats = _compute_feature_stats(train_examples)

    first_graph = train_examples[0]["graph"]
    node_feature_dim = first_graph.x.shape[-1]
    edge_feature_dim = first_graph.edge_attr.shape[-1]
    print(f"Node features: {node_feature_dim}, Edge features: {edge_feature_dim}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = build_graph_gps_surrogate(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        output_dim=len(MODEL_KPI_NAMES),
        rwse_dim=RWSE_WALK_LENGTH,
        hidden_dim=HIDDEN_DIM,
        gps_layers=GPS_LAYERS,
        heads=HEADS,
        dropout=DROPOUT,
    )
    model.to(torch_device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = torch.nn.MSELoss()
    rng = random.Random(42)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    print(f"\nTraining for {EPOCHS} epochs...")
    print("-" * 80)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        rng.shuffle(train_examples)

        train_loss = _run_epoch(
            model=model,
            examples=train_examples,
            batch_size=BATCH_SIZE,
            loss_fn=loss_fn,
            batch_factory=Batch,
            target_stats=target_stats,
            feature_stats=feature_stats,
            device=torch_device,
            optimizer=optimizer,
            max_grad_norm=MAX_GRAD_NORM,
            machine_loss_weight=MACHINE_LOSS_WEIGHT,
        )
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(
                model=model,
                examples=val_examples,
                batch_size=BATCH_SIZE,
                loss_fn=loss_fn,
                batch_factory=Batch,
                target_stats=target_stats,
                feature_stats=feature_stats,
                device=torch_device,
                optimizer=None,
                max_grad_norm=MAX_GRAD_NORM,
                machine_loss_weight=MACHINE_LOSS_WEIGHT,
            )

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        improved = ""
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            epochs_without_improvement = 0
            improved = " *"
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={lr:.2e} | "
            f"{elapsed:.1f}s{improved}"
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience {EARLY_STOPPING_PATIENCE})")
            break

    # ------------------------------------------------------------------
    # Restore best checkpoint
    # ------------------------------------------------------------------
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    print(f"\nBest validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            }
            if best_state_dict is None
            else best_state_dict,
            "kpi_names": MODEL_KPI_NAMES,
            "node_feature_dim": node_feature_dim,
            "edge_feature_dim": edge_feature_dim,
            "hidden_dim": HIDDEN_DIM,
            "gps_layers": GPS_LAYERS,
            "heads": HEADS,
            "dropout": DROPOUT,
            "rwse_walk_length": RWSE_WALK_LENGTH,
            "target_stats": target_stats,
            "feature_stats": feature_stats,
            "target_mode": TARGET_MODE,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
        },
        CHECKPOINT_PATH,
    )
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")

    # ------------------------------------------------------------------
    # Evaluate on all splits
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    results = {}
    for split_name, examples, path in [
        ("train", train_examples, DATA_DIR / "train.jsonl"),
        ("validation", val_examples, DATA_DIR / "validation.jsonl"),
        ("test", test_examples, DATA_DIR / "test.jsonl"),
    ]:
        split_metrics = _evaluate_split(
            model=model,
            examples=examples,
            rows=load_dataset_jsonl(path),
            target_stats=target_stats,
            feature_stats=feature_stats,
            device=torch_device,
            batch_factory=Batch,
        )
        results[split_name] = split_metrics
        print(f"\n--- {split_name.upper()} ({len(examples)} examples) ---")
        for kpi_name in MODEL_KPI_NAMES:
            m = split_metrics["kpi_regression"][kpi_name]
            print(f"  {kpi_name:25s}  R2={m['r2']:+.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    experiment = {
        "model": "GraphGPS",
        "config": {
            "hidden_dim": HIDDEN_DIM,
            "gps_layers": GPS_LAYERS,
            "heads": HEADS,
            "dropout": DROPOUT,
            "rwse_walk_length": RWSE_WALK_LENGTH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "target_mode": TARGET_MODE,
            "machine_loss_weight": MACHINE_LOSS_WEIGHT,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        "training": {
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "device": str(torch_device),
            "parameters": n_params,
        },
        "results": results,
        "baseline_comparison": {
            "note": "OOD test R2 targets to beat from TransformerConv model",
            "baseline": {
                "makespan": 0.80,
                "mean_flow_time": 0.12,
                "mean_queue_wait_time": 0.37,
                "mean_tardiness": 0.21,
                "throughput": 0.77,
            },
            "gps_test": {
                kpi: results["test"]["kpi_regression"][kpi]["r2"]
                for kpi in MODEL_KPI_NAMES
            },
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(experiment, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")

    # Print comparison summary
    print("\n" + "=" * 80)
    print("OOD TEST COMPARISON: TransformerConv baseline vs GraphGPS")
    print("=" * 80)
    baseline = experiment["baseline_comparison"]["baseline"]
    gps = experiment["baseline_comparison"]["gps_test"]
    for kpi in MODEL_KPI_NAMES:
        b = baseline[kpi]
        g = gps[kpi]
        delta = g - b
        marker = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
        print(f"  {kpi:25s}  baseline={b:+.4f}  gps={g:+.4f}  delta={delta:+.4f}  [{marker}]")


# ===========================================================================
# Data loading
# ===========================================================================

def _load_examples(path: Path, rwse_transform) -> list[dict[str, Any]]:
    rows = load_dataset_jsonl(path)
    examples = []
    for row in rows:
        graph = graph_dict_to_pyg_data(row["graph"])
        _add_machine_utilization_targets(graph, row["machine_utilization"])
        # Apply RWSE transform
        graph = rwse_transform(graph)
        y = _target_values(row)
        examples.append({"graph": graph, "y": y})
    return examples


def _target_values(row: dict[str, Any]) -> list[float]:
    kpis = row["kpis"]
    scenario = SupplyChainScenario.from_dict(row["scenario"])
    heuristic_kpis = predict_scenario_kpis(scenario)["kpis"]
    values = []
    for name in MODEL_KPI_NAMES:
        truth_log = math.log1p(max(0.0, kpis[name]))
        if name in RESIDUAL_KPI_NAMES:
            values.append(truth_log - math.log1p(max(0.0, heuristic_kpis[name])))
        else:
            values.append(truth_log)
    return values


def _add_machine_utilization_targets(graph, machine_utilization: dict[str, float]) -> None:
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


# ===========================================================================
# Normalization
# ===========================================================================

def _compute_target_stats(examples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    values = torch.tensor([ex["y"] for ex in examples], dtype=torch.float32)
    # target_mode is hybrid_log1p, so transform is identity
    return {
        "kpis": _stats_tensor(values),
        "transform": "identity",
    }


def _compute_feature_stats(examples: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    import torch

    node_features = torch.cat([ex["graph"].x for ex in examples], dim=0)
    edge_features = torch.cat(
        [ex["graph"].edge_attr for ex in examples if ex["graph"].edge_attr.numel() > 0],
        dim=0,
    )
    graph_features = torch.cat([ex["graph"].graph_features for ex in examples], dim=0)
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


# ===========================================================================
# Training epoch
# ===========================================================================

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
            [ex["graph"] for ex in batch_examples]
        ).to(device)
        _normalize_graph_batch(batch, feature_stats)
        y = _target_tensor(
            [ex["y"] for ex in batch_examples],
            target_stats["kpis"],
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


def _target_tensor(values: list[list[float]], stats: dict[str, list[float]], device):
    import torch

    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
    return (tensor - mean) / std


# ===========================================================================
# Evaluation
# ===========================================================================

def _evaluate_split(
    *,
    model,
    examples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    target_stats: dict[str, Any],
    feature_stats: dict[str, dict[str, list[float]]],
    device,
    batch_factory,
) -> dict[str, Any]:
    import torch

    model.eval()
    true_kpis: list[dict[str, float]] = []
    pred_kpis: list[dict[str, float]] = []

    with torch.no_grad():
        for start in range(0, len(examples), 64):
            batch_examples = examples[start : start + 64]
            batch_rows = rows[start : start + 64]
            batch = batch_factory.from_data_list(
                [ex["graph"] for ex in batch_examples]
            ).to(device)
            _normalize_graph_batch(batch, feature_stats)

            output = model(batch)
            pred_kpis.extend(
                _decode_predictions(output, target_stats, batch_rows)
            )
            true_kpis.extend(
                [
                    {name: float(row["kpis"][name]) for name in MODEL_KPI_NAMES}
                    for row in batch_rows
                ]
            )

    return {
        "kpi_regression": regression_metrics(true_kpis, pred_kpis),
    }


def _decode_predictions(
    tensor,
    target_stats: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, float]]:
    """Undo z-score normalization and hybrid_log1p encoding."""
    import torch

    stats = target_stats["kpis"]
    mean = tensor.new_tensor(stats["mean"])
    std = tensor.new_tensor(stats["std"])
    tensor = tensor * std + mean

    # hybrid_log1p decode
    heuristic_log = _heuristic_log_tensor(rows, tensor)
    decoded_columns = []
    for idx, name in enumerate(MODEL_KPI_NAMES):
        column = tensor[:, idx]
        if name in RESIDUAL_KPI_NAMES:
            column = column + heuristic_log[:, idx]
        decoded_columns.append(column.expm1().clamp_min(0.0))
    tensor = torch.stack(decoded_columns, dim=1)

    values = tensor.detach().cpu().tolist()
    return [
        {name: float(v) for name, v in zip(MODEL_KPI_NAMES, row, strict=True)}
        for row in values
    ]


def _heuristic_log_tensor(rows: list[dict[str, Any]], like):
    import torch

    values = []
    for row in rows:
        scenario = SupplyChainScenario.from_dict(row["scenario"])
        heuristic_kpis = predict_scenario_kpis(scenario)["kpis"]
        values.append([max(0.0, heuristic_kpis[name]) for name in MODEL_KPI_NAMES])
    return torch.log1p(like.new_tensor(values))


if __name__ == "__main__":
    main()
