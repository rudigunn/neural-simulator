"""Small non-neural baselines for early evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from neural_simulator.simulation.scenario import Order, SupplyChainScenario

MACHINE_RANK_FEATURES = 6


def engineer_features(scenario: SupplyChainScenario) -> dict[str, float]:
    machines = scenario.machine_by_id()
    machine_processing_load = {machine.machine_id: 0.0 for machine in scenario.machines}
    machine_operation_count = {machine.machine_id: 0.0 for machine in scenario.machines}
    operations_by_machine: dict[str, list[tuple[Order, int, float]]] = {
        machine.machine_id: [] for machine in scenario.machines
    }
    route_lengths = []
    due_slacks = []
    total_quantity = 0.0

    for order in scenario.orders:
        total_quantity += order.quantity
        route_lengths.append(float(len(order.route)))
        route_work = 0.0
        for route_index, machine_id in enumerate(order.route):
            machine = machines[machine_id]
            work = (
                order.quantity
                * machine.processing_time_by_product_type[order.product_type]
                / machine.speed
            )
            estimated_setup = machine.mean_setup_time_to_product(order.product_type)
            machine_processing_load[machine_id] += work / machine.capacity
            machine_operation_count[machine_id] += 1.0
            operations_by_machine[machine_id].append((order, route_index, work))
            route_work += work + estimated_setup
        due_slacks.append(order.due_date - order.release_time - route_work)

    machine_setup_load = _estimate_setup_loads(scenario, operations_by_machine)
    machine_load = {
        machine.machine_id: (
            machine_processing_load[machine.machine_id]
            + machine_setup_load[machine.machine_id]
        )
        for machine in scenario.machines
    }
    loads = list(machine_load.values())
    setup_loads = list(machine_setup_load.values())
    finite_buffer_capacities = [
        float(machine.input_buffer_capacity)
        for machine in scenario.machines
        if machine.input_buffer_capacity is not None
    ]
    buffer_pressure = {
        machine.machine_id: (
            machine_operation_count[machine.machine_id] / machine.input_buffer_capacity
            if machine.input_buffer_capacity
            else 0.0
        )
        for machine in scenario.machines
    }
    buffer_pressures = list(buffer_pressure.values())
    total_workload = sum(loads)
    total_processing_workload = sum(machine_processing_load.values())
    total_setup_workload = sum(setup_loads)
    mean_load = _mean(loads)
    load_variance = _mean([(load - mean_load) ** 2 for load in loads])

    features = {
        **_dispatch_features(scenario.dispatch_rule),
        "n_machines": float(len(scenario.machines)),
        "n_orders": float(len(scenario.orders)),
        "n_operations": float(sum(len(order.route) for order in scenario.orders)),
        "total_quantity": total_quantity,
        "total_workload": total_workload,
        "total_processing_workload": total_processing_workload,
        "total_setup_workload": total_setup_workload,
        "setup_workload_share": (
            total_setup_workload / total_workload if total_workload else 0.0
        ),
        "mean_machine_load": mean_load,
        "max_machine_load": max(loads) if loads else 0.0,
        "mean_machine_setup_load": _mean(setup_loads),
        "max_machine_setup_load": max(setup_loads) if setup_loads else 0.0,
        "finite_buffer_machine_share": (
            len(finite_buffer_capacities) / len(scenario.machines)
            if scenario.machines
            else 0.0
        ),
        "mean_input_buffer_capacity": _mean(finite_buffer_capacities),
        "min_input_buffer_capacity": (
            min(finite_buffer_capacities) if finite_buffer_capacities else 0.0
        ),
        "max_buffer_pressure": max(buffer_pressures) if buffer_pressures else 0.0,
        "mean_buffer_pressure": _mean(buffer_pressures),
        "load_cv": (load_variance**0.5 / mean_load) if mean_load else 0.0,
        "mean_route_length": _mean(route_lengths),
        "mean_due_slack": _mean(due_slacks),
        "min_due_slack": min(due_slacks) if due_slacks else 0.0,
    }
    ranked_machines = sorted(
        scenario.machines,
        key=lambda machine: (
            machine_load[machine.machine_id],
            machine_operation_count[machine.machine_id],
            machine.machine_id,
        ),
        reverse=True,
    )
    for rank in range(MACHINE_RANK_FEATURES):
        if rank < len(ranked_machines):
            machine = ranked_machines[rank]
            load = machine_load[machine.machine_id]
            processing_load = machine_processing_load[machine.machine_id]
            setup_load = machine_setup_load[machine.machine_id]
            operation_count = machine_operation_count[machine.machine_id]
            features[f"rank_{rank}_machine_load"] = load
            features[f"rank_{rank}_machine_load_share"] = (
                load / total_workload if total_workload else 0.0
            )
            features[f"rank_{rank}_machine_processing_load"] = processing_load
            features[f"rank_{rank}_machine_setup_load"] = setup_load
            features[f"rank_{rank}_machine_has_finite_input_buffer"] = float(
                machine.input_buffer_capacity is not None
            )
            features[f"rank_{rank}_machine_input_buffer_capacity"] = (
                0.0
                if machine.input_buffer_capacity is None
                else float(machine.input_buffer_capacity)
            )
            features[f"rank_{rank}_machine_buffer_pressure"] = buffer_pressure[
                machine.machine_id
            ]
            features[f"rank_{rank}_machine_speed"] = machine.speed
            features[f"rank_{rank}_machine_capacity"] = float(machine.capacity)
            features[f"rank_{rank}_machine_operation_count"] = operation_count
        else:
            features[f"rank_{rank}_machine_load"] = 0.0
            features[f"rank_{rank}_machine_load_share"] = 0.0
            features[f"rank_{rank}_machine_processing_load"] = 0.0
            features[f"rank_{rank}_machine_setup_load"] = 0.0
            features[f"rank_{rank}_machine_has_finite_input_buffer"] = 0.0
            features[f"rank_{rank}_machine_input_buffer_capacity"] = 0.0
            features[f"rank_{rank}_machine_buffer_pressure"] = 0.0
            features[f"rank_{rank}_machine_speed"] = 0.0
            features[f"rank_{rank}_machine_capacity"] = 0.0
            features[f"rank_{rank}_machine_operation_count"] = 0.0
    if len(ranked_machines) >= 2:
        features["top_load_gap"] = (
            machine_load[ranked_machines[0].machine_id]
            - machine_load[ranked_machines[1].machine_id]
        )
    else:
        features["top_load_gap"] = features["max_machine_load"]
    return features


def _estimate_setup_loads(
    scenario: SupplyChainScenario,
    operations_by_machine: dict[str, list[tuple[Order, int, float]]],
) -> dict[str, float]:
    machines = scenario.machine_by_id()
    setup_loads = {machine.machine_id: 0.0 for machine in scenario.machines}
    for machine_id, operations in operations_by_machine.items():
        machine = machines[machine_id]
        previous_product: str | None = None
        for order, _route_index, _work in sorted(
            operations,
            key=lambda operation: _dispatch_sort_key(scenario, operation),
        ):
            setup_loads[machine_id] += (
                machine.setup_time(previous_product, order.product_type) / machine.capacity
            )
            previous_product = order.product_type
    return setup_loads


def _dispatch_sort_key(
    scenario: SupplyChainScenario,
    operation: tuple[Order, int, float],
) -> tuple[float, float, str, int]:
    order, route_index, work = operation
    if scenario.dispatch_rule == "earliest_due_date":
        return (order.due_date, order.release_time, order.order_id, route_index)
    if scenario.dispatch_rule == "shortest_processing_time":
        return (work, order.release_time, order.order_id, route_index)
    return (order.release_time, float(route_index), order.order_id, route_index)


def _dispatch_features(dispatch_rule: str) -> dict[str, float]:
    return {
        "dispatch_fifo": float(dispatch_rule == "fifo"),
        "dispatch_earliest_due_date": float(dispatch_rule == "earliest_due_date"),
        "dispatch_shortest_processing_time": float(
            dispatch_rule == "shortest_processing_time"
        ),
    }


@dataclass
class MeanKpiPredictor:
    target_means: dict[str, float] | None = None

    def fit(self, targets: list[dict[str, float]]) -> "MeanKpiPredictor":
        if not targets:
            raise ValueError("targets cannot be empty")
        names = sorted(targets[0])
        self.target_means = {
            name: sum(target[name] for target in targets) / len(targets) for name in names
        }
        return self

    def predict(self, count: int = 1) -> list[dict[str, float]]:
        if self.target_means is None:
            raise RuntimeError("fit must be called before predict")
        return [dict(self.target_means) for _ in range(count)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
