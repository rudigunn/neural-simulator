"""Command line interface for the simulator POC."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from neural_simulator.data.dataset import (
    DatasetItem,
    generate_dataset_items,
    generate_topology_holdout_splits,
    save_dataset_split_map,
    save_dataset_splits,
)
from neural_simulator.evaluation.baseline_scorecard import evaluate_baseline_scorecard
from neural_simulator.evaluation.graph_transformer import (
    evaluate_graph_transformer_checkpoint,
)
from neural_simulator.evaluation.heuristic import evaluate_heuristic_dataset
from neural_simulator.models.heuristic import predict_scenario_kpis
from neural_simulator.simulation.generator import generate_scenario
from neural_simulator.simulation.scenario import (
    SupplyChainScenario,
    load_scenario_json,
    save_scenario_json,
)
from neural_simulator.simulation.simulator import run_simulation
from neural_simulator.training.graph_transformer import train_graph_transformer

app = typer.Typer(help="Graph surrogate POC for supply-chain DES.")
console = Console()


@app.command()
def simulate(
    seed: int = typer.Option(7, help="Scenario seed."),
    machines: int = typer.Option(4, min=1, help="Number of machines."),
    orders: int = typer.Option(8, min=1, help="Number of orders."),
    output: Path | None = typer.Option(None, help="Optional path to save scenario JSON."),
) -> None:
    """Generate and simulate one scenario."""

    scenario = generate_scenario(seed, n_machines=machines, n_orders=orders)
    if output is not None:
        save_scenario_json(scenario, output)

    result = run_simulation(scenario)
    _print_kpis("SimPy KPI truth", result.kpis)
    console.print(f"bottleneck_machine: [bold]{result.bottleneck_machine}[/bold]")


@app.command("generate-dataset")
def generate_dataset(
    count: int = typer.Option(100, min=1, help="Number of examples."),
    seed: int = typer.Option(0, help="Dataset seed."),
    output_dir: Path = typer.Option(Path("data/poc"), help="Output directory."),
) -> None:
    """Generate scenario KPI JSONL splits."""

    items = generate_dataset_items(count=count, seed=seed)
    paths = save_dataset_splits(items, output_dir)
    for split_name, path in paths.items():
        console.print(f"{split_name}: {path}")


@app.command("generate-topology-holdout")
def generate_topology_holdout(
    train_count: int = typer.Option(1000, min=1, help="Training examples."),
    validation_count: int = typer.Option(150, min=1, help="Validation examples."),
    test_count: int = typer.Option(150, min=1, help="Test examples."),
    seed: int = typer.Option(0, help="Dataset seed."),
    output_dir: Path = typer.Option(
        Path("data/topology-holdout"),
        help="Output directory.",
    ),
    min_train_machines: int = typer.Option(5, min=1, help="Minimum train machines."),
    max_train_machines: int = typer.Option(10, min=1, help="Maximum train machines."),
    min_validation_machines: int = typer.Option(
        12,
        min=1,
        help="Minimum OOD validation machines.",
    ),
    max_validation_machines: int = typer.Option(
        15,
        min=1,
        help="Maximum OOD validation machines.",
    ),
    test_machines: int = typer.Option(20, min=1, help="Held-out test machine count."),
    orders_per_machine: int = typer.Option(
        4,
        min=0,
        help="Orders generated per machine; use 0 for generator defaults.",
    ),
    min_route_length: int = typer.Option(2, min=1, help="Minimum operations per route."),
    max_route_length: int | None = typer.Option(
        8,
        min=1,
        help="Maximum operations per route; may exceed machine count for repeats.",
    ),
    reentrant_route_probability: float = typer.Option(
        0.35,
        min=0.0,
        max=1.0,
        help="Probability that an eligible route revisits at least one machine.",
    ),
    route_shuffle_probability: float = typer.Option(
        0.5,
        min=0.0,
        max=1.0,
        help="Probability that a non-reentrant route uses random machine order.",
    ),
    wide_input_buffers: bool = typer.Option(
        True,
        help="Use order-count-sized buffers to reduce generated deadlocks.",
    ),
) -> None:
    """Generate a harder topology holdout split."""

    if min_train_machines > max_train_machines:
        raise typer.BadParameter("min_train_machines must be <= max_train_machines")
    if min_validation_machines > max_validation_machines:
        raise typer.BadParameter(
            "min_validation_machines must be <= max_validation_machines"
        )
    if max_route_length is not None and min_route_length > max_route_length:
        raise typer.BadParameter("min_route_length must be <= max_route_length")

    train_machine_counts = tuple(range(min_train_machines, max_train_machines + 1))
    validation_machine_counts = tuple(
        range(min_validation_machines, max_validation_machines + 1)
    )
    splits = generate_topology_holdout_splits(
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
        seed=seed,
        train_machine_counts=train_machine_counts,
        validation_machine_counts=validation_machine_counts,
        test_machine_count=test_machines,
        orders_per_machine=None if orders_per_machine == 0 else orders_per_machine,
        min_route_length=min_route_length,
        max_route_length=max_route_length,
        reentrant_route_probability=reentrant_route_probability,
        route_shuffle_probability=route_shuffle_probability,
        wide_input_buffers=wide_input_buffers,
    )
    metadata = {
        "kind": "hard_topology_holdout",
        "seed": seed,
        "train_machine_counts": list(train_machine_counts),
        "validation_machine_counts": list(validation_machine_counts),
        "test_machine_count": test_machines,
        "orders_per_machine": None if orders_per_machine == 0 else orders_per_machine,
        "route_complexity": {
            "min_route_length": min_route_length,
            "max_route_length": max_route_length,
            "reentrant_route_probability": reentrant_route_probability,
            "route_shuffle_probability": route_shuffle_probability,
            "wide_input_buffers": wide_input_buffers,
        },
        "split_counts": {name: len(items) for name, items in splits.items()},
        "observed_machine_counts": {
            name: _observed_machine_counts(items) for name, items in splits.items()
        },
        "observed_route_stats": {
            name: _observed_route_stats(items) for name, items in splits.items()
        },
    }
    paths = save_dataset_split_map(splits, output_dir, metadata=metadata)
    for split_name, path in paths.items():
        console.print(f"{split_name}: {path}")


@app.command("evaluate-heuristic")
def evaluate_heuristic(
    dataset: Path = typer.Option(..., help="Dataset split JSONL path."),
) -> None:
    """Evaluate the fast heuristic surrogate on a saved split."""

    metrics = evaluate_heuristic_dataset(dataset)
    console.print(json.dumps(metrics, indent=2, sort_keys=True))


@app.command("evaluate-baselines")
def evaluate_baselines(
    train: Path = typer.Option(..., help="Training split JSONL path."),
    test: Path = typer.Option(..., help="Test split JSONL path."),
    random_state: int = typer.Option(0, help="Random seed for sklearn baselines."),
    random_forest_estimators: int = typer.Option(
        200,
        min=1,
        help="Number of trees for random forest baselines.",
    ),
) -> None:
    """Compare classical baselines and the heuristic surrogate."""

    scorecard = evaluate_baseline_scorecard(
        train_path=train,
        test_path=test,
        random_state=random_state,
        random_forest_estimators=random_forest_estimators,
    )
    console.print(json.dumps(scorecard, indent=2, sort_keys=True))


@app.command("train-graph-transformer")
def train_graph_transformer_command(
    train: Path = typer.Option(..., help="Training split JSONL path."),
    validation: Path | None = typer.Option(None, help="Validation split JSONL path."),
    checkpoint: Path = typer.Option(
        Path("checkpoints/graph_transformer.pt"),
        help="Output checkpoint path.",
    ),
    epochs: int = typer.Option(10, min=1, help="Training epochs."),
    batch_size: int = typer.Option(16, min=1, help="Batch size."),
    learning_rate: float = typer.Option(3e-4, min=0.0, help="Learning rate."),
    dropout: float = typer.Option(0.1, min=0.0, max=1.0, help="Dropout rate."),
    device: str = typer.Option("auto", help="Training device: auto, cpu, or cuda."),
    node_encoder: str = typer.Option(
        "schema_attention",
        help="Node encoder: schema_attention or linear.",
    ),
    early_stopping_patience: int = typer.Option(
        10,
        min=0,
        help="Validation epochs without improvement before stopping; 0 disables.",
    ),
    min_delta: float = typer.Option(
        1e-4,
        min=0.0,
        help="Minimum validation-loss improvement required to reset patience.",
    ),
    target_transform: str = typer.Option(
        "log1p",
        help="Absolute-target transform: log1p or identity.",
    ),
    target_mode: str = typer.Option(
        "hybrid_log1p",
        help="Target mode: hybrid_log1p, heuristic_residual_log1p, or absolute.",
    ),
    machine_loss_weight: float = typer.Option(
        0.2,
        min=0.0,
        help="Auxiliary machine-utilization loss weight.",
    ),
) -> None:
    """Train the PyG Graph Transformer surrogate."""

    summary = train_graph_transformer(
        train_path=train,
        validation_path=validation,
        checkpoint_path=checkpoint,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout=dropout,
        device=device,
        early_stopping_patience=(
            None if early_stopping_patience == 0 else early_stopping_patience
        ),
        min_delta=min_delta,
        node_encoder_type=node_encoder,
        target_transform=target_transform,
        target_mode=target_mode,
        machine_loss_weight=machine_loss_weight,
    )
    console.print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


@app.command("evaluate-graph-transformer")
def evaluate_graph_transformer_command(
    checkpoint: Path = typer.Option(..., help="Graph Transformer checkpoint path."),
    dataset: Path = typer.Option(..., help="Dataset split JSONL path."),
    batch_size: int = typer.Option(64, min=1, help="Evaluation batch size."),
    device: str = typer.Option("auto", help="Evaluation device: auto, cpu, or cuda."),
) -> None:
    """Evaluate a saved Graph Transformer surrogate checkpoint."""

    metrics = evaluate_graph_transformer_checkpoint(
        checkpoint_path=checkpoint,
        dataset_path=dataset,
        batch_size=batch_size,
        device=device,
    )
    console.print(json.dumps(metrics, indent=2, sort_keys=True))


@app.command()
def demo(
    scenario: Path | None = typer.Option(None, help="Scenario JSON path."),
    seed: int = typer.Option(7, help="Fallback generated scenario seed."),
    validate: bool = typer.Option(True, help="Compare heuristic surrogate to SimPy truth."),
) -> None:
    """Run a demo with the fast heuristic surrogate."""

    loaded_scenario = _load_or_generate_scenario(scenario, seed)
    prediction = predict_scenario_kpis(loaded_scenario)

    _print_kpis("Heuristic KPI prediction", prediction["kpis"])
    console.print(f"predicted_bottleneck_machine: {prediction['bottleneck_machine']}")

    if validate:
        truth = run_simulation(loaded_scenario)
        _print_kpis("SimPy KPI truth", truth.kpis)
        console.print(f"truth_bottleneck_machine: {truth.bottleneck_machine}")


def main() -> None:
    app()


def _load_or_generate_scenario(path: Path | None, seed: int) -> SupplyChainScenario:
    if path is None:
        return generate_scenario(seed, n_machines=5, n_orders=10)
    return load_scenario_json(path)


def _print_kpis(title: str, kpis: dict[str, float]) -> None:
    table = Table(title=title)
    table.add_column("KPI")
    table.add_column("Value", justify="right")
    for name, value in kpis.items():
        table.add_row(name, f"{value:.4f}")
    console.print(table)


def _observed_machine_counts(items: list[DatasetItem]) -> list[int]:
    return sorted({len(item.scenario["machines"]) for item in items})


def _observed_route_stats(items: list[DatasetItem]) -> dict[str, float]:
    routes = [
        order["route"]
        for item in items
        for order in item.scenario["orders"]
    ]
    if not routes:
        return {
            "max_route_length": 0.0,
            "mean_route_length": 0.0,
            "reentrant_route_share": 0.0,
        }
    route_lengths = [len(route) for route in routes]
    reentrant_routes = [route for route in routes if len(set(route)) < len(route)]
    return {
        "max_route_length": float(max(route_lengths)),
        "mean_route_length": sum(route_lengths) / len(route_lengths),
        "reentrant_route_share": len(reentrant_routes) / len(routes),
    }
