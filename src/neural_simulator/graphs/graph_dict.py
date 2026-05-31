"""Serializable graph representation for scenarios."""

from __future__ import annotations

from typing import Any

from neural_simulator.simulation.scenario import SupplyChainScenario

KNOWN_PRODUCT_TYPES = ("A", "B", "C")

NODE_FEATURE_NAMES = [
    "is_order",
    "is_operation",
    "is_machine",
    "product_A",
    "product_B",
    "product_C",
    "product_unknown",
    "dispatch_fifo",
    "dispatch_earliest_due_date",
    "dispatch_shortest_processing_time",
    "quantity",
    "release_time",
    "due_date",
    "product_type_index",
    "route_index",
    "base_processing_time",
    "estimated_processing_time",
    "estimated_setup_time",
    "machine_speed",
    "machine_capacity",
    "has_finite_input_buffer",
    "machine_input_buffer_capacity",
    "machine_processing_time_mean",
    "machine_setup_time_mean",
    "machine_processing_time_A",
    "machine_processing_time_B",
    "machine_processing_time_C",
]

EDGE_FEATURE_NAMES = [
    "edge_type_index",
    "is_order_to_operation",
    "is_operation_to_order",
    "is_operation_to_machine",
    "is_machine_to_operation",
    "is_precedes",
    "is_follows",
    "product_A",
    "product_B",
    "product_C",
    "product_unknown",
    "dispatch_fifo",
    "dispatch_earliest_due_date",
    "dispatch_shortest_processing_time",
    "route_index",
    "quantity",
    "product_type_index",
    "base_processing_time",
    "estimated_processing_time",
    "estimated_setup_time",
    "has_finite_input_buffer",
    "machine_input_buffer_capacity",
]

EDGE_TYPES = {
    "order_to_operation": 0.0,
    "operation_to_order": 1.0,
    "operation_to_machine": 2.0,
    "machine_to_operation": 3.0,
    "precedes": 4.0,
    "follows": 5.0,
}


def scenario_to_graph_dict(scenario: SupplyChainScenario) -> dict[str, Any]:
    """Convert a scenario into a JSON-serializable homogeneous graph."""

    product_index = {product: idx for idx, product in enumerate(scenario.product_types())}
    machines = scenario.machine_by_id()
    dispatch_features = _dispatch_features(scenario.dispatch_rule)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for machine in scenario.machines:
        node_id = f"machine:{machine.machine_id}"
        mean_processing = sum(machine.processing_time_by_product_type.values()) / len(
            machine.processing_time_by_product_type
        )
        nodes.append(
            {
                "node_id": node_id,
                "entity_id": machine.machine_id,
                "node_type": "machine",
                "features": _features(
                    is_machine=1.0,
                    **dispatch_features,
                    machine_speed=machine.speed,
                    machine_capacity=float(machine.capacity),
                    **_input_buffer_features(machine.input_buffer_capacity),
                    machine_processing_time_mean=mean_processing,
                    machine_setup_time_mean=machine.mean_setup_time(),
                    **_machine_product_features(machine.processing_time_by_product_type),
                ),
            }
        )

    for order in scenario.orders:
        order_node_id = f"order:{order.order_id}"
        nodes.append(
            {
                "node_id": order_node_id,
                "entity_id": order.order_id,
                "node_type": "order",
                "features": _features(
                    is_order=1.0,
                    **dispatch_features,
                    **_product_features(order.product_type),
                    quantity=order.quantity,
                    release_time=order.release_time,
                    due_date=order.due_date,
                    product_type_index=float(product_index[order.product_type]),
                ),
            }
        )

        previous_operation_node_id: str | None = None
        for route_index, machine_id in enumerate(order.route):
            machine = machines[machine_id]
            base_processing_time = machine.processing_time_by_product_type[order.product_type]
            estimated_processing_time = order.quantity * base_processing_time / machine.speed
            estimated_setup_time = machine.mean_setup_time_to_product(order.product_type)
            operation_node_id = f"operation:{order.order_id}:{route_index}"
            machine_node_id = f"machine:{machine_id}"

            nodes.append(
                {
                    "node_id": operation_node_id,
                    "entity_id": f"{order.order_id}:{route_index}",
                    "node_type": "operation",
                    "features": _features(
                        is_operation=1.0,
                        **dispatch_features,
                        **_product_features(order.product_type),
                        quantity=order.quantity,
                        product_type_index=float(product_index[order.product_type]),
                        route_index=float(route_index),
                        base_processing_time=base_processing_time,
                        estimated_processing_time=estimated_processing_time,
                        estimated_setup_time=estimated_setup_time,
                        machine_speed=machine.speed,
                        machine_capacity=float(machine.capacity),
                        **_input_buffer_features(machine.input_buffer_capacity),
                        machine_setup_time_mean=machine.mean_setup_time(),
                    ),
                }
            )

            edge_features = _edge_features(
                **dispatch_features,
                **_product_features(order.product_type),
                route_index=float(route_index),
                quantity=order.quantity,
                product_type_index=float(product_index[order.product_type]),
                base_processing_time=base_processing_time,
                estimated_processing_time=estimated_processing_time,
                estimated_setup_time=estimated_setup_time,
                **_input_buffer_features(machine.input_buffer_capacity),
            )
            _add_edge(edges, order_node_id, operation_node_id, "order_to_operation", edge_features)
            _add_edge(edges, operation_node_id, order_node_id, "operation_to_order", edge_features)
            _add_edge(edges, operation_node_id, machine_node_id, "operation_to_machine", edge_features)
            _add_edge(edges, machine_node_id, operation_node_id, "machine_to_operation", edge_features)

            if previous_operation_node_id is not None:
                _add_edge(edges, previous_operation_node_id, operation_node_id, "precedes", edge_features)
                _add_edge(edges, operation_node_id, previous_operation_node_id, "follows", edge_features)
            previous_operation_node_id = operation_node_id

    return {
        "scenario_id": scenario.scenario_id,
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "nodes": nodes,
        "edges": edges,
    }


def _features(**values: float) -> list[float]:
    raw = {feature_name: 0.0 for feature_name in NODE_FEATURE_NAMES}
    raw.update(values)
    return [float(raw[name]) for name in NODE_FEATURE_NAMES]


def _edge_features(**values: float) -> dict[str, float]:
    raw = {feature_name: 0.0 for feature_name in EDGE_FEATURE_NAMES}
    raw.update(values)
    return raw


def _product_features(product_type: str) -> dict[str, float]:
    values = {f"product_{product}": 0.0 for product in KNOWN_PRODUCT_TYPES}
    values["product_unknown"] = 0.0
    if product_type in KNOWN_PRODUCT_TYPES:
        values[f"product_{product_type}"] = 1.0
    else:
        values["product_unknown"] = 1.0
    return values


def _dispatch_features(dispatch_rule: str) -> dict[str, float]:
    return {
        "dispatch_fifo": float(dispatch_rule == "fifo"),
        "dispatch_earliest_due_date": float(dispatch_rule == "earliest_due_date"),
        "dispatch_shortest_processing_time": float(
            dispatch_rule == "shortest_processing_time"
        ),
    }


def _input_buffer_features(input_buffer_capacity: int | None) -> dict[str, float]:
    return {
        "has_finite_input_buffer": float(input_buffer_capacity is not None),
        "machine_input_buffer_capacity": (
            0.0 if input_buffer_capacity is None else float(input_buffer_capacity)
        ),
    }


def _machine_product_features(
    processing_time_by_product_type: dict[str, float],
) -> dict[str, float]:
    return {
        f"machine_processing_time_{product}": float(
            processing_time_by_product_type.get(product, 0.0)
        )
        for product in KNOWN_PRODUCT_TYPES
    }


def _add_edge(
    edges: list[dict[str, Any]],
    source: str,
    target: str,
    edge_type: str,
    features: dict[str, float],
) -> None:
    edge_values = dict(features)
    edge_values["edge_type_index"] = EDGE_TYPES[edge_type]
    edge_values[f"is_{edge_type}"] = 1.0
    edges.append(
        {
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "features": [float(edge_values[name]) for name in EDGE_FEATURE_NAMES],
        }
    )
