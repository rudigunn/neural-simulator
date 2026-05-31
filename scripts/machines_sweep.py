"""Latency sweep: fixed 1000-order workload, machines 20 to 100.

Compares SimPy against surrogate inference only. Latency does not depend on the
trained weights, so the script builds a fresh Performer-GPS model with the
correct dimensions rather than loading a checkpoint.

Run:  uv run python scripts/machines_sweep.py
Out:  results/machines_1000orders.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import frontier_experiment as fe
from neural_simulator.data.dataset import MODEL_KPI_NAMES
from neural_simulator.graphs import scenario_to_graph_dict
from neural_simulator.graphs.pyg import graph_dict_to_pyg_data
from neural_simulator.models.graph_gps import build_graph_gps_surrogate
from neural_simulator.simulation.generator import generate_scenario

OUT = Path(__file__).resolve().parents[1] / "results" / "machines_1000orders.json"

SCALES = [(20, 1000), (40, 1000), (60, 1000), (80, 1000), (100, 1000)]
N_PER_SCALE = 8
BATCH_SIZE = 16


def sample_graphs(k=24):
    graphs = []
    for i in range(k):
        s = generate_scenario(
            950_000 + i, n_machines=10, n_orders=40,
            min_route_length=2, max_route_length=6,
            reentrant_route_probability=0.3, route_shuffle_probability=0.3,
            wide_input_buffers=True,
        )
        g = graph_dict_to_pyg_data(scenario_to_graph_dict(s))
        g.y = torch.zeros(len(MODEL_KPI_NAMES))
        graphs.append(g)
    return graphs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    sample = sample_graphs()
    fs = fe.feature_stats(sample)
    model = build_graph_gps_surrogate(
        node_feature_dim=sample[0].x.shape[1],
        edge_feature_dim=sample[0].edge_attr.shape[1],
        output_dim=len(MODEL_KPI_NAMES),
        rwse_dim=0, hidden_dim=64, gps_layers=4,
        heads=4, dropout=0.1, attn_type="performer",
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    speed = fe.speed_benchmark(model, fs, device, walk_length=0,
                               scales=SCALES, n=N_PER_SCALE, batch_size=BATCH_SIZE)
    OUT.write_text(json.dumps(
        {"config": {"fixed_orders": 1000, "n_per_scale": N_PER_SCALE,
                    "batch_size": BATCH_SIZE, "device": str(device)},
         "speed": speed}, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
