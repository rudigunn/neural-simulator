"""Fast heuristic surrogate used before the neural model is trained."""

from __future__ import annotations

from typing import Any

from neural_simulator.simulation.scenario import Order, SupplyChainScenario


def predict_scenario_kpis(scenario: SupplyChainScenario) -> dict[str, Any]:
    """Estimate KPIs without running SimPy.

    This is intentionally simple. It gives the CLI a fast surrogate-like path
    before a trained Graph Transformer checkpoint exists.
    """

    machines = scenario.machine_by_id()
    load_by_machine = {machine.machine_id: 0.0 for machine in scenario.machines}
    operations_by_machine: dict[str, list[tuple[Order, int, float]]] = {
        machine.machine_id: [] for machine in scenario.machines
    }
    route_durations: list[float] = []
    queue_estimates: list[float] = []

    for order in scenario.orders:
        route_duration = 0.0
        for route_index, machine_id in enumerate(order.route):
            machine = machines[machine_id]
            duration = (
                order.quantity
                * machine.processing_time_by_product_type[order.product_type]
                / machine.speed
            )
            estimated_setup = machine.mean_setup_time_to_product(order.product_type)
            route_duration += duration + estimated_setup
            load_by_machine[machine_id] += duration / machine.capacity
            operations_by_machine[machine_id].append((order, route_index, duration))
        route_durations.append(route_duration)

    setup_load_by_machine = _estimate_setup_loads(scenario, operations_by_machine)
    for machine_id, setup_load in setup_load_by_machine.items():
        load_by_machine[machine_id] += setup_load

    blocked_time_estimates = _estimate_blocked_times(scenario, load_by_machine)
    bottleneck_machine = max(load_by_machine, key=lambda key: (load_by_machine[key], key))
    release_span = max((order.release_time for order in scenario.orders), default=0.0)
    makespan = (
        release_span
        + max(load_by_machine.values(), default=0.0)
        + _mean(route_durations)
        + _mean(blocked_time_estimates)
    )

    machine_utilization = {}
    for machine in scenario.machines:
        denominator = makespan if makespan else 1.0
        utilization = min(0.999, load_by_machine[machine.machine_id] / denominator)
        machine_utilization[machine.machine_id] = utilization

    for order, route_duration in zip(scenario.orders, route_durations):
        route_utilization = _mean([machine_utilization[machine_id] for machine_id in order.route])
        congestion_multiplier = route_utilization / max(0.1, 1.0 - route_utilization)
        queue_estimates.append(route_duration * 0.25 * congestion_multiplier)

    flow_estimates = [
        route_duration + queue_estimate
        for route_duration, queue_estimate in zip(route_durations, queue_estimates)
    ]
    tardiness_estimates = [
        max(0.0, order.release_time + flow_time - order.due_date)
        for order, flow_time in zip(scenario.orders, flow_estimates)
    ]

    kpis = {
        "makespan": makespan,
        "throughput": len(scenario.orders) / makespan if makespan else 0.0,
        "mean_flow_time": _mean(flow_estimates),
        "mean_tardiness": _mean(tardiness_estimates),
        "mean_queue_wait_time": _mean(queue_estimates),
        "mean_blocked_time": _mean(blocked_time_estimates),
        "completed_orders": float(len(scenario.orders)),
    }
    return {
        "kpis": kpis,
        "machine_utilization": machine_utilization,
        "bottleneck_machine": bottleneck_machine,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _estimate_setup_loads(
    scenario: SupplyChainScenario,
    operations_by_machine: dict[str, list[tuple[Order, int, float]]],
) -> dict[str, float]:
    machines = scenario.machine_by_id()
    setup_loads = {machine.machine_id: 0.0 for machine in scenario.machines}
    for machine_id, operations in operations_by_machine.items():
        machine = machines[machine_id]
        previous_product: str | None = None
        for order, _route_index, _duration in sorted(
            operations,
            key=lambda operation: _dispatch_sort_key(scenario, operation),
        ):
            setup_loads[machine_id] += (
                machine.setup_time(previous_product, order.product_type) / machine.capacity
            )
            previous_product = order.product_type
    return setup_loads


def _estimate_blocked_times(
    scenario: SupplyChainScenario,
    load_by_machine: dict[str, float],
) -> list[float]:
    estimates = []
    for order in scenario.orders:
        for current_machine_id, next_machine_id in zip(
            order.route,
            order.route[1:],
            strict=False,
        ):
            next_machine = scenario.machine_by_id()[next_machine_id]
            if next_machine.input_buffer_capacity is None:
                estimates.append(0.0)
                continue
            downstream_load = load_by_machine[next_machine_id]
            buffer_pressure = downstream_load / max(1.0, next_machine.input_buffer_capacity)
            current_machine_load = load_by_machine[current_machine_id]
            estimates.append(max(0.0, buffer_pressure - current_machine_load) * 0.05)
    return estimates


def _dispatch_sort_key(
    scenario: SupplyChainScenario,
    operation: tuple[Order, int, float],
) -> tuple[float, float, str, int]:
    order, route_index, duration = operation
    if scenario.dispatch_rule == "earliest_due_date":
        return (order.due_date, order.release_time, order.order_id, route_index)
    if scenario.dispatch_rule == "shortest_processing_time":
        return (duration, order.release_time, order.order_id, route_index)
    return (order.release_time, float(route_index), order.order_id, route_index)
