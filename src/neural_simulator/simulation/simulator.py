"""SimPy discrete-event supply-chain simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import simpy

from neural_simulator.simulation.scenario import Order, SupplyChainScenario


@dataclass(frozen=True)
class SimulationResult:
    scenario_id: str
    kpis: dict[str, float]
    machine_utilization: dict[str, float]
    machine_queue_wait: dict[str, float]
    machine_blocked_time: dict[str, float]
    bottleneck_machine: str
    order_completion_times: dict[str, float]
    operation_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "kpis": self.kpis,
            "machine_utilization": self.machine_utilization,
            "machine_queue_wait": self.machine_queue_wait,
            "machine_blocked_time": self.machine_blocked_time,
            "bottleneck_machine": self.bottleneck_machine,
            "order_completion_times": self.order_completion_times,
            "operation_records": self.operation_records,
        }


def processing_duration(scenario: SupplyChainScenario, order: Order, machine_id: str) -> float:
    machine = scenario.machine_by_id()[machine_id]
    base_time = machine.processing_time_by_product_type[order.product_type]
    return order.quantity * base_time / machine.speed


def setup_duration(
    scenario: SupplyChainScenario,
    machine_id: str,
    previous_product_type: str | None,
    next_product_type: str,
) -> float:
    machine = scenario.machine_by_id()[machine_id]
    return machine.setup_time(previous_product_type, next_product_type)


def run_simulation(scenario: SupplyChainScenario) -> SimulationResult:
    """Run the deterministic DES and return KPI outputs."""

    _validate_dispatch_rule(scenario.dispatch_rule)
    _validate_input_buffers(scenario)

    env = simpy.Environment()
    machines = scenario.machine_by_id()
    resources = {
        machine.machine_id: simpy.PriorityResource(env, capacity=machine.capacity)
        for machine in scenario.machines
    }
    input_buffers = {
        machine.machine_id: (
            None
            if machine.input_buffer_capacity is None
            else simpy.Container(
                env,
                capacity=machine.input_buffer_capacity,
                init=machine.input_buffer_capacity,
            )
        )
        for machine in scenario.machines
    }
    busy_time = {machine.machine_id: 0.0 for machine in scenario.machines}
    queue_waits = {machine.machine_id: [] for machine in scenario.machines}
    machine_blocked_time = {machine.machine_id: 0.0 for machine in scenario.machines}
    last_product_by_machine: dict[str, str | None] = {
        machine.machine_id: None for machine in scenario.machines
    }
    operation_records: list[dict[str, Any]] = []
    completion_times: dict[str, float] = {}

    def order_process(order: Order):
        yield env.timeout(order.release_time)
        reserved_buffer_machine_id: str | None = None
        reserved_buffer_wait = 0.0
        for route_index, machine_id in enumerate(order.route):
            resource = resources[machine_id]
            if reserved_buffer_machine_id == machine_id:
                input_buffer_wait = reserved_buffer_wait
                reserved_buffer_machine_id = None
                reserved_buffer_wait = 0.0
            else:
                input_buffer_wait = yield from _reserve_input_buffer(
                    env,
                    input_buffers,
                    machine_id,
                )
            queue_entered_at = env.now
            priority = _dispatch_priority(scenario, order, machine_id)
            with resource.request(priority=priority) as request:
                yield request
                start_time = env.now
                yield from _release_input_buffer(env, input_buffers, machine_id)
                queue_wait = start_time - queue_entered_at
                queue_waits[machine_id].append(queue_wait)

                previous_product_type = last_product_by_machine[machine_id]
                current_setup_duration = setup_duration(
                    scenario,
                    machine_id,
                    previous_product_type,
                    order.product_type,
                )
                current_processing_duration = processing_duration(
                    scenario,
                    order,
                    machine_id,
                )
                duration = current_setup_duration + current_processing_duration
                last_product_by_machine[machine_id] = order.product_type
                yield env.timeout(duration)
                processing_end_time = env.now

                blocked_time = 0.0
                if route_index + 1 < len(order.route):
                    next_machine_id = order.route[route_index + 1]
                    blocked_started_at = env.now
                    reserved_buffer_wait = yield from _reserve_input_buffer(
                        env,
                        input_buffers,
                        next_machine_id,
                    )
                    blocked_time = env.now - blocked_started_at
                    reserved_buffer_machine_id = next_machine_id

                busy_time[machine_id] += duration + blocked_time
                machine_blocked_time[machine_id] += blocked_time
                end_time = env.now

                operation_records.append(
                    {
                        "order_id": order.order_id,
                        "machine_id": machine_id,
                        "route_index": route_index,
                        "input_buffer_wait": input_buffer_wait,
                        "queue_wait": queue_wait,
                        "start_time": start_time,
                        "processing_end_time": processing_end_time,
                        "end_time": end_time,
                        "duration": duration,
                        "processing_duration": current_processing_duration,
                        "setup_duration": current_setup_duration,
                        "blocked_time": blocked_time,
                        "previous_product_type": previous_product_type,
                        "product_type": order.product_type,
                        "dispatch_priority": priority,
                    }
                )
        completion_times[order.order_id] = env.now

    for order in scenario.orders:
        env.process(order_process(order))

    env.run()

    if not scenario.orders:
        raise ValueError("scenario must contain at least one order")
    if len(completion_times) != len(scenario.orders):
        incomplete = sorted(
            order.order_id
            for order in scenario.orders
            if order.order_id not in completion_times
        )
        raise RuntimeError(
            "simulation ended before all orders completed; "
            f"possible finite-buffer deadlock for orders: {incomplete}"
        )

    makespan = max(completion_times.values(), default=0.0)
    throughput = len(completion_times) / makespan if makespan > 0 else 0.0
    order_by_id = scenario.order_by_id()
    flow_times = [
        completion_time - order_by_id[order_id].release_time
        for order_id, completion_time in completion_times.items()
    ]
    tardiness = [
        max(0.0, completion_time - order_by_id[order_id].due_date)
        for order_id, completion_time in completion_times.items()
    ]
    all_waits = [wait for waits in queue_waits.values() for wait in waits]
    all_blocked_times = [record["blocked_time"] for record in operation_records]

    machine_utilization = {}
    machine_queue_wait = {}
    for machine_id, machine in machines.items():
        denominator = makespan * machine.capacity
        machine_utilization[machine_id] = busy_time[machine_id] / denominator if denominator else 0.0
        waits = queue_waits[machine_id]
        machine_queue_wait[machine_id] = sum(waits) / len(waits) if waits else 0.0

    bottleneck_machine = max(
        machine_utilization,
        key=lambda machine_id: (machine_utilization[machine_id], machine_id),
    )

    kpis = {
        "makespan": makespan,
        "throughput": throughput,
        "mean_flow_time": _mean(flow_times),
        "mean_tardiness": _mean(tardiness),
        "mean_queue_wait_time": _mean(all_waits),
        "mean_blocked_time": _mean(all_blocked_times),
        "completed_orders": float(len(completion_times)),
    }

    return SimulationResult(
        scenario_id=scenario.scenario_id,
        kpis=kpis,
        machine_utilization=machine_utilization,
        machine_queue_wait=machine_queue_wait,
        machine_blocked_time=machine_blocked_time,
        bottleneck_machine=bottleneck_machine,
        order_completion_times=completion_times,
        operation_records=sorted(
            operation_records,
            key=lambda item: (item["start_time"], item["order_id"], item["route_index"]),
        ),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _reserve_input_buffer(
    env: simpy.Environment,
    input_buffers: dict[str, simpy.Container | None],
    machine_id: str,
):
    wait_started_at = env.now
    input_buffer = input_buffers[machine_id]
    if input_buffer is not None:
        yield input_buffer.get(1)
    return env.now - wait_started_at


def _release_input_buffer(
    env: simpy.Environment,
    input_buffers: dict[str, simpy.Container | None],
    machine_id: str,
):
    input_buffer = input_buffers[machine_id]
    if input_buffer is not None:
        yield input_buffer.put(1)


def _validate_dispatch_rule(dispatch_rule: str) -> None:
    supported = {"fifo", "earliest_due_date", "shortest_processing_time"}
    if dispatch_rule not in supported:
        raise ValueError(
            f"unsupported dispatch_rule {dispatch_rule!r}; "
            f"expected one of {sorted(supported)}"
        )


def _validate_input_buffers(scenario: SupplyChainScenario) -> None:
    for machine in scenario.machines:
        if machine.input_buffer_capacity is not None and machine.input_buffer_capacity < 1:
            raise ValueError(
                "input_buffer_capacity must be positive when finite: "
                f"{machine.machine_id}={machine.input_buffer_capacity}"
            )


def _dispatch_priority(
    scenario: SupplyChainScenario,
    order: Order,
    machine_id: str,
) -> float:
    if scenario.dispatch_rule == "fifo":
        return 0.0
    if scenario.dispatch_rule == "earliest_due_date":
        return order.due_date
    if scenario.dispatch_rule == "shortest_processing_time":
        return processing_duration(scenario, order, machine_id)
    _validate_dispatch_rule(scenario.dispatch_rule)
    return 0.0
