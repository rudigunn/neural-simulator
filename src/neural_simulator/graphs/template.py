"""Cached graph topology for fast variant conversion.

In the optimization search use case, thousands of scenario variants share the
same graph topology (same machines, orders, routes) but differ in feature values
(machine speeds, order quantities).  ``build_template`` captures the fixed
topology once, and ``template_to_pyg_data`` rebuilds only the feature tensors
for each variant, skipping all dict/list allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neural_simulator.graphs.graph_dict import (
    EDGE_FEATURE_NAMES,
    EDGE_TYPES,
    KNOWN_PRODUCT_TYPES,
    NODE_FEATURE_NAMES,
)
from neural_simulator.simulation.scenario import SupplyChainScenario


@dataclass(frozen=True)
class _NodeMeta:
    """Per-node metadata needed to recompute features from a variant."""

    node_type: str  # "machine", "order", "operation"
    node_id: str
    # machine nodes
    machine_id: str | None = None
    # order nodes
    order_id: str | None = None
    product_type: str | None = None
    product_type_index: float = 0.0
    release_time: float = 0.0
    due_date: float = 0.0
    # operation nodes
    route_index: int = 0
    op_machine_id: str | None = None
    op_order_id: str | None = None


@dataclass(frozen=True)
class _EdgeMeta:
    """Per-edge metadata needed to recompute features from a variant."""

    edge_type: str
    order_id: str
    machine_id: str
    product_type: str
    product_type_index: float
    route_index: int


@dataclass(frozen=True)
class GraphTemplate:
    """Cached graph topology for a scenario."""

    node_metas: tuple[_NodeMeta, ...]
    edge_metas: tuple[_EdgeMeta, ...]
    edge_sources: tuple[int, ...]
    edge_targets: tuple[int, ...]
    dispatch_rule: str
    node_feature_names: list[str]
    edge_feature_names: list[str]
    # Pre-computed lists reused across all variants
    node_ids: list[str] | None = None
    node_types: list[str] | None = None


# Pre-compute feature name -> index for fast writes
_NODE_FEAT_IDX = {name: idx for idx, name in enumerate(NODE_FEATURE_NAMES)}
_EDGE_FEAT_IDX = {name: idx for idx, name in enumerate(EDGE_FEATURE_NAMES)}


def build_template(scenario: SupplyChainScenario) -> GraphTemplate:
    """Build a reusable topology template from a base scenario."""

    product_index = {p: idx for idx, p in enumerate(scenario.product_types())}
    node_metas: list[_NodeMeta] = []
    edge_metas: list[_EdgeMeta] = []
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    node_id_to_idx: dict[str, int] = {}

    def _add_node(meta: _NodeMeta) -> int:
        idx = len(node_metas)
        node_metas.append(meta)
        node_id_to_idx[meta.node_id] = idx
        return idx

    def _add_edge(src_idx: int, tgt_idx: int, meta: _EdgeMeta) -> None:
        edge_sources.append(src_idx)
        edge_targets.append(tgt_idx)
        edge_metas.append(meta)

    # Machine nodes
    for machine in scenario.machines:
        _add_node(_NodeMeta(
            node_type="machine",
            node_id=f"machine:{machine.machine_id}",
            machine_id=machine.machine_id,
        ))

    # Order + operation nodes + edges
    for order in scenario.orders:
        order_node_id = f"order:{order.order_id}"
        _add_node(_NodeMeta(
            node_type="order",
            node_id=order_node_id,
            order_id=order.order_id,
            product_type=order.product_type,
            product_type_index=float(product_index[order.product_type]),
            release_time=order.release_time,
            due_date=order.due_date,
        ))

        prev_op_idx: int | None = None
        for route_index, machine_id in enumerate(order.route):
            op_node_id = f"operation:{order.order_id}:{route_index}"
            machine_node_id = f"machine:{machine_id}"

            op_idx = _add_node(_NodeMeta(
                node_type="operation",
                node_id=op_node_id,
                op_order_id=order.order_id,
                op_machine_id=machine_id,
                product_type=order.product_type,
                product_type_index=float(product_index[order.product_type]),
                route_index=route_index,
            ))

            order_idx = node_id_to_idx[order_node_id]
            machine_idx = node_id_to_idx[machine_node_id]

            edge_meta = _EdgeMeta(
                edge_type="",  # filled per-edge below
                order_id=order.order_id,
                machine_id=machine_id,
                product_type=order.product_type,
                product_type_index=float(product_index[order.product_type]),
                route_index=route_index,
            )

            for edge_type, src, tgt in [
                ("order_to_operation", order_idx, op_idx),
                ("operation_to_order", op_idx, order_idx),
                ("operation_to_machine", op_idx, machine_idx),
                ("machine_to_operation", machine_idx, op_idx),
            ]:
                _add_edge(src, tgt, _EdgeMeta(
                    edge_type=edge_type,
                    order_id=edge_meta.order_id,
                    machine_id=edge_meta.machine_id,
                    product_type=edge_meta.product_type,
                    product_type_index=edge_meta.product_type_index,
                    route_index=edge_meta.route_index,
                ))

            if prev_op_idx is not None:
                for edge_type, src, tgt in [
                    ("precedes", prev_op_idx, op_idx),
                    ("follows", op_idx, prev_op_idx),
                ]:
                    _add_edge(src, tgt, _EdgeMeta(
                        edge_type=edge_type,
                        order_id=edge_meta.order_id,
                        machine_id=edge_meta.machine_id,
                        product_type=edge_meta.product_type,
                        product_type_index=edge_meta.product_type_index,
                        route_index=edge_meta.route_index,
                    ))
            prev_op_idx = op_idx

    node_metas_tuple = tuple(node_metas)
    return GraphTemplate(
        node_metas=node_metas_tuple,
        edge_metas=tuple(edge_metas),
        edge_sources=tuple(edge_sources),
        edge_targets=tuple(edge_targets),
        dispatch_rule=scenario.dispatch_rule,
        node_feature_names=NODE_FEATURE_NAMES,
        edge_feature_names=EDGE_FEATURE_NAMES,
        node_ids=[m.node_id for m in node_metas_tuple],
        node_types=[m.node_type for m in node_metas_tuple],
    )


def template_to_pyg_data(template: GraphTemplate, scenario: SupplyChainScenario):
    """Build a PyG Data object from a cached template and a variant scenario.

    Only recomputes feature tensors; topology is reused from the template.
    Uses numpy arrays for fast element-wise writes, then converts to torch once.
    """

    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "Install ML dependencies with `uv sync --group ml` to use PyG conversion."
        ) from exc

    import numpy as np

    machines = scenario.machine_by_id()
    orders = scenario.order_by_id()
    dispatch_feats = _dispatch_vec(template.dispatch_rule)

    # --- Node features (numpy for fast element writes) ---
    num_nodes = len(template.node_metas)
    num_node_feats = len(NODE_FEATURE_NAMES)
    x = np.zeros((num_nodes, num_node_feats), dtype=np.float32)

    for i, meta in enumerate(template.node_metas):
        row = x[i]
        if meta.node_type == "machine":
            machine = machines[meta.machine_id]
            pt_values = machine.processing_time_by_product_type
            mean_pt = sum(pt_values.values()) / len(pt_values)
            row[_NODE_FEAT_IDX["is_machine"]] = 1.0
            row[_NODE_FEAT_IDX["machine_speed"]] = machine.speed
            row[_NODE_FEAT_IDX["machine_capacity"]] = float(machine.capacity)
            row[_NODE_FEAT_IDX["machine_processing_time_mean"]] = mean_pt
            row[_NODE_FEAT_IDX["machine_setup_time_mean"]] = machine.mean_setup_time()
            _write_dispatch(row, dispatch_feats)
            _write_input_buffer(row, machine.input_buffer_capacity)
            for product in KNOWN_PRODUCT_TYPES:
                row[_NODE_FEAT_IDX[f"machine_processing_time_{product}"]] = float(
                    pt_values.get(product, 0.0)
                )

        elif meta.node_type == "order":
            order = orders[meta.order_id]
            row[_NODE_FEAT_IDX["is_order"]] = 1.0
            row[_NODE_FEAT_IDX["quantity"]] = order.quantity
            row[_NODE_FEAT_IDX["release_time"]] = order.release_time
            row[_NODE_FEAT_IDX["due_date"]] = order.due_date
            row[_NODE_FEAT_IDX["product_type_index"]] = meta.product_type_index
            _write_dispatch(row, dispatch_feats)
            _write_product(row, meta.product_type)

        else:  # operation
            order = orders[meta.op_order_id]
            machine = machines[meta.op_machine_id]
            base_pt = machine.processing_time_by_product_type[meta.product_type]
            est_pt = order.quantity * base_pt / machine.speed
            est_setup = machine.mean_setup_time_to_product(meta.product_type)
            row[_NODE_FEAT_IDX["is_operation"]] = 1.0
            row[_NODE_FEAT_IDX["quantity"]] = order.quantity
            row[_NODE_FEAT_IDX["product_type_index"]] = meta.product_type_index
            row[_NODE_FEAT_IDX["route_index"]] = float(meta.route_index)
            row[_NODE_FEAT_IDX["base_processing_time"]] = base_pt
            row[_NODE_FEAT_IDX["estimated_processing_time"]] = est_pt
            row[_NODE_FEAT_IDX["estimated_setup_time"]] = est_setup
            row[_NODE_FEAT_IDX["machine_speed"]] = machine.speed
            row[_NODE_FEAT_IDX["machine_capacity"]] = float(machine.capacity)
            row[_NODE_FEAT_IDX["machine_setup_time_mean"]] = machine.mean_setup_time()
            _write_dispatch(row, dispatch_feats)
            _write_product(row, meta.product_type)
            _write_input_buffer(row, machine.input_buffer_capacity)

    x_tensor = torch.from_numpy(x)

    # --- Edge features (numpy for fast element writes) ---
    num_edges = len(template.edge_metas)
    num_edge_feats = len(EDGE_FEATURE_NAMES)

    # Cache edge_index tensor on first call (topology never changes)
    edge_index = _get_cached_edge_index(template)

    if num_edges > 0:
        edge_attr = np.zeros((num_edges, num_edge_feats), dtype=np.float32)

        for i, emeta in enumerate(template.edge_metas):
            order = orders[emeta.order_id]
            machine = machines[emeta.machine_id]
            base_pt = machine.processing_time_by_product_type[emeta.product_type]
            est_pt = order.quantity * base_pt / machine.speed
            est_setup = machine.mean_setup_time_to_product(emeta.product_type)

            row = edge_attr[i]
            row[_EDGE_FEAT_IDX["edge_type_index"]] = EDGE_TYPES[emeta.edge_type]
            row[_EDGE_FEAT_IDX[f"is_{emeta.edge_type}"]] = 1.0
            row[_EDGE_FEAT_IDX["route_index"]] = float(emeta.route_index)
            row[_EDGE_FEAT_IDX["quantity"]] = order.quantity
            row[_EDGE_FEAT_IDX["product_type_index"]] = emeta.product_type_index
            row[_EDGE_FEAT_IDX["base_processing_time"]] = base_pt
            row[_EDGE_FEAT_IDX["estimated_processing_time"]] = est_pt
            row[_EDGE_FEAT_IDX["estimated_setup_time"]] = est_setup
            _write_edge_dispatch(row, dispatch_feats)
            _write_edge_product(row, emeta.product_type)
            _write_edge_input_buffer(row, machine.input_buffer_capacity)

        edge_attr_tensor = torch.from_numpy(edge_attr)
    else:
        edge_attr_tensor = torch.empty((0, num_edge_feats), dtype=torch.float32)

    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr_tensor)
    data.node_id = template.node_ids
    data.node_type = template.node_types
    data.machine_mask = torch.tensor(
        [meta.node_type == "machine" for meta in template.node_metas],
        dtype=torch.bool,
    )
    data.graph_features = torch.tensor([_graph_features(template, scenario)], dtype=torch.float32)
    data.scenario_id = scenario.scenario_id
    return data


def _graph_features(
    template: GraphTemplate,
    scenario: SupplyChainScenario,
) -> list[float]:
    machines = scenario.machine_by_id()
    order_count = len(scenario.orders)
    machine_count = len(scenario.machines)
    operation_count = sum(len(order.route) for order in scenario.orders)
    total_estimated_processing = 0.0
    total_estimated_setup = 0.0
    total_operation_quantity = 0.0

    for order in scenario.orders:
        total_operation_quantity += order.quantity * len(order.route)
        for machine_id in order.route:
            machine = machines[machine_id]
            total_estimated_processing += (
                order.quantity
                * machine.processing_time_by_product_type[order.product_type]
                / machine.speed
            )
            total_estimated_setup += machine.mean_setup_time_to_product(order.product_type)

    return [
        float(len(template.node_metas)),
        float(order_count),
        float(operation_count),
        float(machine_count),
        float(sum(order.quantity for order in scenario.orders)),
        float(total_operation_quantity),
        float(total_estimated_processing),
        float(total_estimated_setup),
        float(operation_count / order_count if order_count else 0.0),
        float(sum(machine.capacity for machine in scenario.machines)),
        float(
            sum(machine.speed for machine in scenario.machines) / machine_count
            if machine_count
            else 0.0
        ),
        float(
            sum(
                machine.input_buffer_capacity is not None
                for machine in scenario.machines
            )
            / machine_count
            if machine_count
            else 0.0
        ),
    ]


# ---- Cached edge_index tensor (topology is fixed) ----

_edge_index_cache: dict[int, Any] = {}


def _get_cached_edge_index(template: GraphTemplate):
    """Return a cached edge_index tensor for this template."""
    import torch

    key = id(template)
    cached = _edge_index_cache.get(key)
    if cached is not None:
        return cached
    if len(template.edge_metas) > 0:
        edge_index = torch.tensor(
            [list(template.edge_sources), list(template.edge_targets)],
            dtype=torch.long,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    _edge_index_cache[key] = edge_index
    return edge_index


# ---- Feature writers (numpy row access) ----

def _dispatch_vec(dispatch_rule: str) -> tuple[float, float, float]:
    return (
        float(dispatch_rule == "fifo"),
        float(dispatch_rule == "earliest_due_date"),
        float(dispatch_rule == "shortest_processing_time"),
    )


def _write_dispatch(row, dispatch_feats: tuple[float, float, float]) -> None:
    row[_NODE_FEAT_IDX["dispatch_fifo"]] = dispatch_feats[0]
    row[_NODE_FEAT_IDX["dispatch_earliest_due_date"]] = dispatch_feats[1]
    row[_NODE_FEAT_IDX["dispatch_shortest_processing_time"]] = dispatch_feats[2]


def _write_product(row, product_type: str) -> None:
    if product_type in KNOWN_PRODUCT_TYPES:
        row[_NODE_FEAT_IDX[f"product_{product_type}"]] = 1.0
    else:
        row[_NODE_FEAT_IDX["product_unknown"]] = 1.0


def _write_input_buffer(row, capacity: int | None) -> None:
    if capacity is not None:
        row[_NODE_FEAT_IDX["has_finite_input_buffer"]] = 1.0
        row[_NODE_FEAT_IDX["machine_input_buffer_capacity"]] = float(capacity)


def _write_edge_dispatch(row, dispatch_feats: tuple[float, float, float]) -> None:
    row[_EDGE_FEAT_IDX["dispatch_fifo"]] = dispatch_feats[0]
    row[_EDGE_FEAT_IDX["dispatch_earliest_due_date"]] = dispatch_feats[1]
    row[_EDGE_FEAT_IDX["dispatch_shortest_processing_time"]] = dispatch_feats[2]


def _write_edge_product(row, product_type: str) -> None:
    if product_type in KNOWN_PRODUCT_TYPES:
        row[_EDGE_FEAT_IDX[f"product_{product_type}"]] = 1.0
    else:
        row[_EDGE_FEAT_IDX["product_unknown"]] = 1.0


def _write_edge_input_buffer(row, capacity: int | None) -> None:
    if capacity is not None:
        row[_EDGE_FEAT_IDX["has_finite_input_buffer"]] = 1.0
        row[_EDGE_FEAT_IDX["machine_input_buffer_capacity"]] = float(capacity)
