"""Serializable scenario types for the supply-chain simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Machine:
    machine_id: str
    speed: float
    capacity: int
    processing_time_by_product_type: dict[str, float]
    setup_time_by_product_transition: dict[str, dict[str, float]] | None = None
    input_buffer_capacity: int | None = None

    def setup_time(self, previous_product_type: str | None, next_product_type: str) -> float:
        if previous_product_type is None:
            return 0.0
        if previous_product_type == next_product_type:
            return 0.0
        if self.setup_time_by_product_transition is None:
            return 0.0
        return float(
            self.setup_time_by_product_transition.get(previous_product_type, {}).get(
                next_product_type,
                0.0,
            )
        )

    def mean_setup_time(self) -> float:
        if self.setup_time_by_product_transition is None:
            return 0.0
        values = [
            float(value)
            for targets in self.setup_time_by_product_transition.values()
            for value in targets.values()
        ]
        return sum(values) / len(values) if values else 0.0

    def mean_setup_time_to_product(self, next_product_type: str) -> float:
        if self.setup_time_by_product_transition is None:
            return 0.0
        values = [
            float(targets[next_product_type])
            for previous_product_type, targets in self.setup_time_by_product_transition.items()
            if previous_product_type != next_product_type and next_product_type in targets
        ]
        return sum(values) / len(values) if values else 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Machine":
        return cls(
            machine_id=str(value["machine_id"]),
            speed=float(value["speed"]),
            capacity=int(value["capacity"]),
            processing_time_by_product_type={
                str(product): float(duration)
                for product, duration in value["processing_time_by_product_type"].items()
            },
            setup_time_by_product_transition=value.get("setup_time_by_product_transition"),
            input_buffer_capacity=(
                None
                if value.get("input_buffer_capacity") is None
                else int(value["input_buffer_capacity"])
            ),
        )


@dataclass(frozen=True)
class Order:
    order_id: str
    quantity: float
    release_time: float
    due_date: float
    product_type: str
    route: list[str]


@dataclass(frozen=True)
class SupplyChainScenario:
    scenario_id: str
    seed: int
    machines: list[Machine]
    orders: list[Order]
    dispatch_rule: str = "fifo"

    def machine_by_id(self) -> dict[str, Machine]:
        return {machine.machine_id: machine for machine in self.machines}

    def order_by_id(self) -> dict[str, Order]:
        return {order.order_id: order for order in self.orders}

    def product_types(self) -> list[str]:
        products = set()
        for machine in self.machines:
            products.update(machine.processing_time_by_product_type)
        for order in self.orders:
            products.add(order.product_type)
        return sorted(products)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SupplyChainScenario":
        return cls(
            scenario_id=str(value["scenario_id"]),
            seed=int(value["seed"]),
            machines=[Machine.from_dict(machine) for machine in value["machines"]],
            orders=[Order(**order) for order in value["orders"]],
            dispatch_rule=str(value.get("dispatch_rule", "fifo")),
        )


def load_scenario_json(path: str | Path) -> SupplyChainScenario:
    with Path(path).open("r", encoding="utf-8") as handle:
        return SupplyChainScenario.from_dict(json.load(handle))


def save_scenario_json(scenario: SupplyChainScenario, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(scenario.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
