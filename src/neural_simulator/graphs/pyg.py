"""Optional PyTorch Geometric conversion helpers."""

from __future__ import annotations

from typing import Any


def graph_dict_to_pyg_data(graph: dict[str, Any], *, y: list[float] | None = None):
    """Convert a graph dict to a PyG Data object.

    PyTorch and PyTorch Geometric are optional `ml` dependencies. Importing them
    here keeps the simulator and dataset tools lightweight.
    """

    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "Install ML dependencies with `uv sync --group ml` to use PyG conversion."
        ) from exc

    node_index = {node["node_id"]: idx for idx, node in enumerate(graph["nodes"])}
    x = torch.tensor([node["features"] for node in graph["nodes"]], dtype=torch.float32)

    if graph["edges"]:
        edge_index = torch.tensor(
            [
                [node_index[edge["source"]] for edge in graph["edges"]],
                [node_index[edge["target"]] for edge in graph["edges"]],
            ],
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            [edge["features"] for edge in graph["edges"]],
            dtype=torch.float32,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(graph["edge_feature_names"])), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_id = [node["node_id"] for node in graph["nodes"]]
    data.node_type = [node["node_type"] for node in graph["nodes"]]
    data.machine_mask = torch.tensor(
        [node["node_type"] == "machine" for node in graph["nodes"]],
        dtype=torch.bool,
    )
    data.graph_features = torch.tensor([_graph_features(graph)], dtype=torch.float32)
    data.scenario_id = graph["scenario_id"]
    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float32)
    return data


def _graph_features(graph: dict[str, Any]) -> list[float]:
    feature_index = {
        name: idx for idx, name in enumerate(graph["node_feature_names"])
    }
    nodes = graph["nodes"]
    order_nodes = [node for node in nodes if node["node_type"] == "order"]
    operation_nodes = [node for node in nodes if node["node_type"] == "operation"]
    machine_nodes = [node for node in nodes if node["node_type"] == "machine"]

    quantity_idx = feature_index["quantity"]
    estimated_processing_idx = feature_index["estimated_processing_time"]
    estimated_setup_idx = feature_index["estimated_setup_time"]
    machine_capacity_idx = feature_index["machine_capacity"]
    machine_speed_idx = feature_index["machine_speed"]
    finite_buffer_idx = feature_index["has_finite_input_buffer"]

    order_count = len(order_nodes)
    machine_count = len(machine_nodes)
    total_machine_speed = sum(
        node["features"][machine_speed_idx] for node in machine_nodes
    )
    finite_buffer_count = sum(
        node["features"][finite_buffer_idx] for node in machine_nodes
    )
    return [
        float(len(nodes)),
        float(order_count),
        float(len(operation_nodes)),
        float(machine_count),
        float(sum(node["features"][quantity_idx] for node in order_nodes)),
        float(sum(node["features"][quantity_idx] for node in operation_nodes)),
        float(
            sum(
                node["features"][estimated_processing_idx]
                for node in operation_nodes
            )
        ),
        float(sum(node["features"][estimated_setup_idx] for node in operation_nodes)),
        float(len(operation_nodes) / order_count if order_count else 0.0),
        float(sum(node["features"][machine_capacity_idx] for node in machine_nodes)),
        float(total_machine_speed / machine_count if machine_count else 0.0),
        float(finite_buffer_count / machine_count if machine_count else 0.0),
    ]
