"""Benchmark: template-based graph conversion vs current pipeline vs SimPy."""

from __future__ import annotations

import time
from dataclasses import replace

import torch
from torch_geometric.data import Batch

from neural_simulator.graphs import build_template, scenario_to_graph_dict, template_to_pyg_data
from neural_simulator.graphs.pyg import graph_dict_to_pyg_data
from neural_simulator.models.graph_transformer import build_graph_transformer_surrogate
from neural_simulator.data.dataset import MODEL_KPI_NAMES
from neural_simulator.simulation.generator import generate_scenario
from neural_simulator.simulation.simulator import run_simulation


BATCH_SIZE = 64


def make_variants(base_scenario, n_variants: int, rng_seed: int = 0):
    """Create n_variants by perturbing machine speeds and order quantities."""
    import random
    rng = random.Random(rng_seed)
    variants = []
    for _ in range(n_variants):
        machines = [
            replace(m, speed=m.speed * rng.uniform(0.5, 2.0))
            for m in base_scenario.machines
        ]
        orders = [
            replace(o, quantity=max(1, int(o.quantity * rng.uniform(0.5, 2.0))))
            for o in base_scenario.orders
        ]
        variants.append(replace(base_scenario, machines=machines, orders=orders))
    return variants


def benchmark_scale(n_machines: int, n_orders: int, n_variants: int):
    print(f"\n{'='*60}")
    print(f"Scale: {n_machines} machines, {n_orders} orders, {n_variants} variants")
    print(f"{'='*60}")

    base = generate_scenario(42, n_machines=n_machines, n_orders=n_orders)
    variants = make_variants(base, n_variants)

    # --- SimPy ---
    t0 = time.perf_counter()
    for v in variants:
        run_simulation(v)
    simpy_time = time.perf_counter() - t0
    print(f"SimPy:                    {simpy_time:.3f}s  ({simpy_time/n_variants*1000:.2f} ms/variant)")

    # --- Current pipeline (dict -> PyG) ---
    t0 = time.perf_counter()
    for v in variants:
        gd = scenario_to_graph_dict(v)
        graph_dict_to_pyg_data(gd)
    current_time = time.perf_counter() - t0
    print(f"Current pipeline:         {current_time:.3f}s  ({current_time/n_variants*1000:.2f} ms/variant)")

    # --- Template pipeline (conversion only) ---
    t_build = time.perf_counter()
    template = build_template(base)
    build_time = time.perf_counter() - t_build

    t0 = time.perf_counter()
    for v in variants:
        template_to_pyg_data(template, v)
    template_time = time.perf_counter() - t0
    print(f"Template conversion:      {template_time:.3f}s  ({template_time/n_variants*1000:.2f} ms/variant)  (build: {build_time*1000:.2f} ms)")

    # --- Template + batched GNN inference (GPU if available) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GNN device: {device}")

    sample_data = template_to_pyg_data(template, base)
    model = build_graph_transformer_surrogate(
        node_feature_dim=sample_data.x.shape[-1],
        edge_feature_dim=sample_data.edge_attr.shape[-1],
        output_dim=len(MODEL_KPI_NAMES),
        hidden_dim=64,
        layers=3,
        heads=4,
        dropout=0.0,
    )
    model = model.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        model(sample_data.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Convert all, then batch-infer on device
    t0 = time.perf_counter()
    all_data = [template_to_pyg_data(template, v) for v in variants]
    convert_time = time.perf_counter() - t0

    t_infer = time.perf_counter()
    with torch.no_grad():
        for batch_start in range(0, n_variants, BATCH_SIZE):
            batch = Batch.from_data_list(all_data[batch_start:batch_start + BATCH_SIZE])
            batch = batch.to(device)
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_time = time.perf_counter() - t_infer

    total_gnn_time = convert_time + infer_time
    print(f"Template + batched GNN:   {total_gnn_time:.3f}s  ({total_gnn_time/n_variants*1000:.2f} ms/variant)  [conv: {convert_time:.3f}s, infer: {infer_time:.3f}s]")

    # --- Speedups ---
    print("\nSpeedups vs SimPy:")
    print(f"  Current pipeline (conv only):      {simpy_time/current_time:.1f}x {'faster' if current_time < simpy_time else 'SLOWER'}")
    print(f"  Template (conv only):              {simpy_time/template_time:.1f}x {'faster' if template_time < simpy_time else 'SLOWER'}")
    print(f"  Template + batched GNN (e2e):      {simpy_time/total_gnn_time:.1f}x {'faster' if total_gnn_time < simpy_time else 'SLOWER'}")
    print(f"  Template vs current pipeline:      {current_time/template_time:.1f}x faster conversion")


if __name__ == "__main__":
    benchmark_scale(6, 20, 1000)
    benchmark_scale(15, 100, 500)
    benchmark_scale(30, 500, 100)
