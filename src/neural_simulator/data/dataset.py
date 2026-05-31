"""Generate and persist scenario KPI examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

from neural_simulator.graphs import scenario_to_graph_dict
from neural_simulator.simulation.generator import generate_scenario
from neural_simulator.simulation.simulator import SimulationResult, run_simulation

KPI_NAMES = [
    "makespan",
    "throughput",
    "mean_flow_time",
    "mean_tardiness",
    "mean_queue_wait_time",
    "mean_blocked_time",
    "completed_orders",
]

MODEL_KPI_NAMES = [
    "makespan",
    "throughput",
    "mean_flow_time",
    "mean_tardiness",
    "mean_queue_wait_time",
]


@dataclass(frozen=True)
class DatasetItem:
    scenario_seed: int
    scenario: dict[str, Any]
    graph: dict[str, Any]
    kpis: dict[str, float]
    bottleneck_machine: str
    machine_utilization: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_dataset_item(
    seed: int,
    *,
    n_machines: int | None = None,
    n_orders: int | None = None,
    min_route_length: int = 1,
    max_route_length: int | None = None,
    reentrant_route_probability: float = 0.0,
    route_shuffle_probability: float = 0.0,
    wide_input_buffers: bool = False,
) -> DatasetItem:
    scenario = generate_scenario(
        seed,
        n_machines=n_machines,
        n_orders=n_orders,
        min_route_length=min_route_length,
        max_route_length=max_route_length,
        reentrant_route_probability=reentrant_route_probability,
        route_shuffle_probability=route_shuffle_probability,
        wide_input_buffers=wide_input_buffers,
    )
    result = run_simulation(scenario)
    return _build_item(
        scenario_seed=seed,
        scenario=scenario.to_dict(),
        graph=scenario_to_graph_dict(scenario),
        result=result,
    )


def generate_dataset_items(
    count: int,
    *,
    seed: int = 0,
    n_machines: int | None = None,
    n_orders: int | None = None,
    min_route_length: int = 1,
    max_route_length: int | None = None,
    reentrant_route_probability: float = 0.0,
    route_shuffle_probability: float = 0.0,
    wide_input_buffers: bool = False,
) -> list[DatasetItem]:
    return [
        make_dataset_item(
            seed + idx,
            n_machines=n_machines,
            n_orders=n_orders,
            min_route_length=min_route_length,
            max_route_length=max_route_length,
            reentrant_route_probability=reentrant_route_probability,
            route_shuffle_probability=route_shuffle_probability,
            wide_input_buffers=wide_input_buffers,
        )
        for idx in range(count)
    ]


def generate_topology_holdout_splits(
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int = 0,
    train_machine_counts: tuple[int, ...] = (5, 6, 7, 8, 9, 10),
    validation_machine_counts: tuple[int, ...] = (12, 13, 14, 15),
    test_machine_count: int = 20,
    orders_per_machine: int | None = 4,
    min_route_length: int = 2,
    max_route_length: int | None = 8,
    reentrant_route_probability: float = 0.35,
    route_shuffle_probability: float = 0.5,
    wide_input_buffers: bool = True,
) -> dict[str, list[DatasetItem]]:
    """Generate train/validation/test splits for topology generalization."""

    if train_count < 1 or validation_count < 1 or test_count < 1:
        raise ValueError("split counts must all be positive")
    if not train_machine_counts:
        raise ValueError("train_machine_counts cannot be empty")
    if not validation_machine_counts:
        raise ValueError("validation_machine_counts cannot be empty")
    if any(count < 1 for count in train_machine_counts):
        raise ValueError("train_machine_counts must contain positive values")
    if any(count < 1 for count in validation_machine_counts):
        raise ValueError("validation_machine_counts must contain positive values")
    if test_machine_count < 1:
        raise ValueError("test_machine_count must be positive")
    if orders_per_machine is not None and orders_per_machine < 1:
        raise ValueError("orders_per_machine must be positive when provided")

    rng = random.Random(seed)
    next_seed = seed

    def build_split(
        count: int,
        machine_counts: tuple[int, ...],
    ) -> list[DatasetItem]:
        nonlocal next_seed
        items = []
        for _ in range(count):
            n_machines = rng.choice(machine_counts)
            n_orders = (
                None
                if orders_per_machine is None
                else n_machines * orders_per_machine
            )
            items.append(
                make_dataset_item(
                    next_seed,
                    n_machines=n_machines,
                    n_orders=n_orders,
                    min_route_length=min_route_length,
                    max_route_length=max_route_length,
                    reentrant_route_probability=reentrant_route_probability,
                    route_shuffle_probability=route_shuffle_probability,
                    wide_input_buffers=wide_input_buffers,
                )
            )
            next_seed += 1
        return items

    return {
        "train": build_split(train_count, train_machine_counts),
        "validation": build_split(validation_count, validation_machine_counts),
        "test": build_split(test_count, (test_machine_count,)),
    }


def save_dataset_splits(
    items: list[DatasetItem],
    output_dir: str | Path,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> dict[str, Path]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_end = int(len(items) * train_fraction)
    validation_end = train_end + int(len(items) * validation_fraction)
    splits = {
        "train": items[:train_end],
        "validation": items[train_end:validation_end],
        "test": items[validation_end:],
    }

    return save_dataset_split_map(splits, output_path)


def save_dataset_split_map(
    splits: dict[str, list[DatasetItem]],
    output_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written = {}
    for split_name, split_items in splits.items():
        split_path = output_path / f"{split_name}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for item in split_items:
                json.dump(item.to_dict(), handle, sort_keys=True)
                handle.write("\n")
        written[split_name] = split_path
    if metadata is not None:
        metadata_path = output_path / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        written["metadata"] = metadata_path
    return written


def load_dataset_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def _build_item(
    *,
    scenario_seed: int,
    scenario: dict[str, Any],
    graph: dict[str, Any],
    result: SimulationResult,
) -> DatasetItem:
    return DatasetItem(
        scenario_seed=scenario_seed,
        scenario=scenario,
        graph=graph,
        kpis={name: result.kpis[name] for name in KPI_NAMES},
        bottleneck_machine=result.bottleneck_machine,
        machine_utilization=result.machine_utilization,
    )
