"""Train the multi-scale type-aware pooling surrogate with Set2Set readout.

Usage:
    uv run python scripts/train_multiscale.py

Hypothesis: separately pooling each node type preserves structural info
that global pooling destroys, improving flow_time and tardiness predictions
on out-of-distribution topologies.
"""

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
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/ood-topology-pilot")
TRAIN_PATH = DATA_DIR / "train.jsonl"
VALIDATION_PATH = DATA_DIR / "validation.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"
CHECKPOINT_PATH = Path("checkpoints/multiscale_pool.pt")
RESULTS_PATH = Path("results/multiscale_experiment.json")

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
HIDDEN_DIM = 96
LAYERS = 3
HEADS = 4
DROPOUT = 0.15
SET2SET_STEPS = 6
WEIGHT_DECAY = 5e-2
MAX_GRAD_NORM = 1.0
MACHINE_LOSS_WEIGHT = 0.3
EARLY_STOPPING_PATIENCE = 40
MIN_DELTA = 1e-5
TARGET_MODE = "hybrid_log1p"
DEVICE = "cuda"
WARMUP_EPOCHS = 5

RESIDUAL_KPI_NAMES = {"makespan", "throughput"}

# ---------------------------------------------------------------------------
# Imports (after config so we fail early on missing deps)
# ---------------------------------------------------------------------------
import torch
from torch_geometric.data import Batch

from neural_simulator.data.dataset import MODEL_KPI_NAMES, load_dataset_jsonl
from neural_simulator.evaluation.metrics import regression_metrics
from neural_simulator.graphs.pyg import graph_dict_to_pyg_data
from neural_simulator.models.heuristic import predict_scenario_kpis
from neural_simulator.models.multiscale_pool import build_multiscale_pool_surrogate
from neural_simulator.simulation.scenario import SupplyChainScenario


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _build_example(row: dict[str, Any]) -> dict[str, Any]:
    graph = graph_dict_to_pyg_data(row["graph"])
    # Add order_mask and operation_mask (machine_mask is already set)
    node_types = graph.node_type
    graph.order_mask = torch.tensor(
        [nt == "order" for nt in node_types], dtype=torch.bool
    )
    graph.operation_mask = torch.tensor(
        [nt == "operation" for nt in node_types], dtype=torch.bool
    )
    # Add machine utilization targets
    _add_machine_utilization_targets(graph, row["machine_utilization"])
    return {
        "graph": graph,
        "y": _target_values(row),
    }


def _add_machine_utilization_targets(
    graph, machine_utilization: dict[str, float]
) -> None:
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


def _load_examples(path: Path) -> list[dict[str, Any]]:
    rows = load_dataset_jsonl(path)
    return [_build_example(row) for row in rows]


# ---------------------------------------------------------------------------
# Feature / target normalization
# ---------------------------------------------------------------------------
def _stats_tensor(tensor: torch.Tensor) -> dict[str, list[float]]:
    std = tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {
        "mean": tensor.mean(dim=0).tolist(),
        "std": std.tolist(),
    }


def compute_target_stats(examples: list[dict[str, Any]]) -> dict[str, Any]:
    values = torch.tensor(
        [ex["y"] for ex in examples], dtype=torch.float32
    )
    return {
        "kpis": _stats_tensor(values),
        "transform": "identity",  # already log1p-transformed in _target_values
    }


def compute_feature_stats(examples: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    node_features = torch.cat([ex["graph"].x for ex in examples], dim=0)
    edge_features = torch.cat(
        [
            ex["graph"].edge_attr
            for ex in examples
            if ex["graph"].edge_attr.numel() > 0
        ],
        dim=0,
    )
    graph_features = torch.cat(
        [ex["graph"].graph_features for ex in examples], dim=0
    )
    return {
        "node": _stats_tensor(node_features),
        "edge": _stats_tensor(edge_features),
        "graph": _stats_tensor(graph_features),
    }


def normalize_graph_batch(
    batch, feature_stats: dict[str, dict[str, list[float]]]
) -> None:
    node_mean = batch.x.new_tensor(feature_stats["node"]["mean"])
    node_std = batch.x.new_tensor(feature_stats["node"]["std"])
    batch.x = (batch.x - node_mean) / node_std
    if batch.edge_attr.numel() > 0:
        edge_mean = batch.edge_attr.new_tensor(feature_stats["edge"]["mean"])
        edge_std = batch.edge_attr.new_tensor(feature_stats["edge"]["std"])
        batch.edge_attr = (batch.edge_attr - edge_mean) / edge_std
    if hasattr(batch, "graph_features") and "graph" in feature_stats:
        graph_mean = batch.graph_features.new_tensor(
            feature_stats["graph"]["mean"]
        )
        graph_std = batch.graph_features.new_tensor(
            feature_stats["graph"]["std"]
        )
        batch.graph_features = (batch.graph_features - graph_mean) / graph_std


def target_tensor(
    values: list[list[float]],
    stats: dict[str, list[float]],
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
    return (tensor - mean) / std


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_epoch(
    *,
    model: torch.nn.Module,
    examples: list[dict[str, Any]],
    target_stats: dict[str, Any],
    feature_stats: dict[str, dict[str, list[float]]],
    device: torch.device,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    total_loss = 0.0
    total_examples = 0

    for start in range(0, len(examples), BATCH_SIZE):
        batch_examples = examples[start : start + BATCH_SIZE]
        batch = Batch.from_data_list(
            [ex["graph"] for ex in batch_examples]
        ).to(device)
        normalize_graph_batch(batch, feature_stats)
        y = target_tensor(
            [ex["y"] for ex in batch_examples],
            target_stats["kpis"],
            device,
        )

        pred, machine_util_pred = model.forward_with_aux(batch)
        loss = loss_fn(pred, y)

        if (
            MACHINE_LOSS_WEIGHT > 0
            and batch.machine_utilization_mask.any()
        ):
            machine_loss = loss_fn(
                machine_util_pred[batch.machine_utilization_mask],
                batch.machine_utilization_y[batch.machine_utilization_mask],
            )
            loss = loss + MACHINE_LOSS_WEIGHT * machine_loss

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        total_loss += float(loss.detach()) * len(batch_examples)
        total_examples += len(batch_examples)

    return total_loss / total_examples


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def decode_predictions(
    tensor: torch.Tensor,
    target_stats: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, float]]:
    """Decode z-score normalized model output back to original KPI scale."""
    stats = target_stats["kpis"]
    mean = tensor.new_tensor(stats["mean"])
    std = tensor.new_tensor(stats["std"])
    tensor = tensor * std + mean

    # Decode hybrid_log1p: residual KPIs add heuristic log, rest are direct log1p
    heuristic_logs = _heuristic_log_tensor(rows, tensor)
    decoded_columns = []
    for idx, name in enumerate(MODEL_KPI_NAMES):
        column = tensor[:, idx]
        if name in RESIDUAL_KPI_NAMES:
            column = column + heuristic_logs[:, idx]
        decoded_columns.append(column.expm1().clamp_min(0.0))
    decoded = torch.stack(decoded_columns, dim=1)

    values = decoded.detach().cpu().tolist()
    return [
        {name: float(v) for name, v in zip(MODEL_KPI_NAMES, row_vals, strict=True)}
        for row_vals in values
    ]


def _heuristic_log_tensor(
    rows: list[dict[str, Any]], like: torch.Tensor
) -> torch.Tensor:
    values = []
    for row in rows:
        scenario = SupplyChainScenario.from_dict(row["scenario"])
        heuristic_kpis = predict_scenario_kpis(scenario)["kpis"]
        values.append(
            [max(0.0, heuristic_kpis[name]) for name in MODEL_KPI_NAMES]
        )
    return torch.log1p(like.new_tensor(values))


def evaluate_split(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    target_stats: dict[str, Any],
    feature_stats: dict[str, dict[str, list[float]]],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Evaluate model on a split and return per-KPI regression metrics."""
    model.eval()
    all_preds: list[dict[str, float]] = []
    all_true: list[dict[str, float]] = []

    with torch.no_grad():
        for start in range(0, len(examples), BATCH_SIZE):
            batch_examples = examples[start : start + BATCH_SIZE]
            batch_rows = rows[start : start + BATCH_SIZE]
            batch = Batch.from_data_list(
                [ex["graph"] for ex in batch_examples]
            ).to(device)
            normalize_graph_batch(batch, feature_stats)

            pred, _ = model.forward_with_aux(batch)
            batch_preds = decode_predictions(pred, target_stats, batch_rows)
            all_preds.extend(batch_preds)

            all_true.extend(
                [
                    {
                        name: float(row["kpis"][name])
                        for name in MODEL_KPI_NAMES
                    }
                    for row in batch_rows
                ]
            )

    return regression_metrics(all_true, all_preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device(DEVICE)
    print(f"Device: {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # ---- Load data --------------------------------------------------------
    print("Loading training data...")
    train_rows = load_dataset_jsonl(TRAIN_PATH)
    train_examples = [_build_example(row) for row in train_rows]
    print(f"  Train: {len(train_examples)} examples")

    print("Loading validation data...")
    val_rows = load_dataset_jsonl(VALIDATION_PATH)
    val_examples = [_build_example(row) for row in val_rows]
    print(f"  Validation: {len(val_examples)} examples")

    print("Loading test data...")
    test_rows = load_dataset_jsonl(TEST_PATH)
    test_examples = [_build_example(row) for row in test_rows]
    print(f"  Test: {len(test_examples)} examples")

    # ---- Compute stats ----------------------------------------------------
    target_stats = compute_target_stats(train_examples)
    feature_stats = compute_feature_stats(train_examples)

    # ---- Build model ------------------------------------------------------
    first_graph = train_examples[0]["graph"]
    model = build_multiscale_pool_surrogate(
        node_feature_dim=first_graph.x.shape[-1],
        edge_feature_dim=first_graph.edge_attr.shape[-1],
        output_dim=len(MODEL_KPI_NAMES),
        hidden_dim=HIDDEN_DIM,
        layers=LAYERS,
        heads=HEADS,
        dropout=DROPOUT,
        set2set_processing_steps=SET2SET_STEPS,
        node_encoder_type="schema_attention",
    )
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # ---- Optimizer --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    rng = random.Random(0)

    # ---- Training ---------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0
    stopped_early = False
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(train_examples)
        train_loss = run_epoch(
            model=model,
            examples=train_examples,
            target_stats=target_stats,
            feature_stats=feature_stats,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(
                model=model,
                examples=val_examples,
                target_stats=target_stats,
                feature_stats=feature_stats,
                device=device,
                loss_fn=loss_fn,
                optimizer=None,
            )

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={lr:.2e} | "
            f"best={best_epoch}"
        )

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
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

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)"
            )
            stopped_early = True
            break

    train_elapsed = time.time() - train_start
    completed_epochs = epoch

    # ---- Restore best checkpoint ------------------------------------------
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)

    # ---- Save checkpoint --------------------------------------------------
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state_dict or {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "kpi_names": MODEL_KPI_NAMES,
            "node_feature_dim": first_graph.x.shape[-1],
            "edge_feature_dim": first_graph.edge_attr.shape[-1],
            "hidden_dim": HIDDEN_DIM,
            "layers": LAYERS,
            "heads": HEADS,
            "dropout": DROPOUT,
            "set2set_processing_steps": SET2SET_STEPS,
            "node_encoder_type": "schema_attention",
            "target_stats": target_stats,
            "feature_stats": feature_stats,
            "target_mode": TARGET_MODE,
            "training_device": str(device),
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "completed_epochs": completed_epochs,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "machine_loss_weight": MACHINE_LOSS_WEIGHT,
        },
        CHECKPOINT_PATH,
    )
    print(f"\nCheckpoint saved to {CHECKPOINT_PATH}")

    # ---- Evaluate on test set ---------------------------------------------
    print("\nEvaluating on test set...")
    test_metrics = evaluate_split(
        model, test_rows, test_examples, target_stats, feature_stats, device
    )

    print("\n--- Test Results ---")
    for kpi_name in MODEL_KPI_NAMES:
        m = test_metrics[kpi_name]
        print(f"  {kpi_name:25s}  R2={m['r2']:+.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

    # ---- Also evaluate on validation set ----------------------------------
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_split(
        model, val_rows, val_examples, target_stats, feature_stats, device
    )

    print("\n--- Validation Results ---")
    for kpi_name in MODEL_KPI_NAMES:
        m = val_metrics[kpi_name]
        print(f"  {kpi_name:25s}  R2={m['r2']:+.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")

    # ---- Save results -----------------------------------------------------
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    baseline_r2 = {
        "makespan": 0.80,
        "mean_flow_time": 0.12,
        "mean_queue_wait_time": 0.37,
        "mean_tardiness": 0.21,
        "throughput": 0.77,
    }

    results = {
        "model": "multiscale_pool_set2set",
        "architecture": {
            "message_passing": "GATv2Conv",
            "pooling": f"Set2Set (processing_steps={SET2SET_STEPS})",
            "readout": "multi-scale type-aware + intermediate layer pooling",
            "hidden_dim": HIDDEN_DIM,
            "layers": LAYERS,
            "heads": HEADS,
            "dropout": DROPOUT,
        },
        "training": {
            "epochs_completed": completed_epochs,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "stopped_early": stopped_early,
            "training_seconds": train_elapsed,
            "device": str(device),
            "target_mode": TARGET_MODE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
        },
        "test_results": {
            kpi: test_metrics[kpi] for kpi in MODEL_KPI_NAMES
        },
        "validation_results": {
            kpi: val_metrics[kpi] for kpi in MODEL_KPI_NAMES
        },
        "baseline_comparison": {
            kpi: {
                "baseline_r2": baseline_r2[kpi],
                "multiscale_r2": test_metrics[kpi]["r2"],
                "improvement": test_metrics[kpi]["r2"] - baseline_r2[kpi],
            }
            for kpi in MODEL_KPI_NAMES
        },
        "checkpoint_path": str(CHECKPOINT_PATH),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ---- Summary ----------------------------------------------------------
    print("\n=== Baseline Comparison ===")
    for kpi in MODEL_KPI_NAMES:
        old = baseline_r2[kpi]
        new = test_metrics[kpi]["r2"]
        delta = new - old
        marker = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
        print(
            f"  {kpi:25s}  baseline={old:+.4f}  ours={new:+.4f}  "
            f"delta={delta:+.4f}  [{marker}]"
        )


if __name__ == "__main__":
    main()
