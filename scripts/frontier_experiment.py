"""Train one GraphGPS surrogate and measure OOD accuracy and speed.

The script reports two things:

1. Generalization. Train once, then measure per-KPI R2 on held-out topology
   scales (20 / 30 / 45 machines). Vary --train (train_wide vs train_narrow) and
   --attn-type (performer vs multihead) to fill the ablation table.

2. Speed. Wall-clock per scenario for the simulator vs the surrogate, across
   scales. The surrogate cost is split into graph conversion, RWSE positional
   encoding, and model inference, so the cost of linear vs dense attention is
   visible separately from the RWSE preprocessing cost.

The data pipeline matches scripts/train_gps.py: graph-dict to PyG, log1p plus
z-score targets, feature z-scoring from training-set statistics.

Usage:
    uv run python scripts/frontier_experiment.py \
        --train data/frontier/train_wide.jsonl \
        --val data/frontier/val.jsonl \
        --test data/frontier/test_20.jsonl data/frontier/test_30.jsonl data/frontier/test_45.jsonl \
        --attn-type performer --rwse-dim 20 \
        --output results/frontier_wide_performer.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import AddRandomWalkPE

from neural_simulator.data.dataset import MODEL_KPI_NAMES, load_dataset_jsonl
from neural_simulator.graphs import scenario_to_graph_dict
from neural_simulator.graphs.pyg import graph_dict_to_pyg_data
from neural_simulator.models.graph_gps import build_graph_gps_surrogate
from neural_simulator.simulation.generator import generate_scenario
from neural_simulator.simulation.simulator import run_simulation


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_graphs(path, walk_length):
    """graph-dict rows -> PyG Data with raw KPI targets and optional RWSE."""
    rows = load_dataset_jsonl(path)
    rwse = AddRandomWalkPE(walk_length=walk_length) if walk_length > 0 else None
    graphs = []
    for row in rows:
        g = graph_dict_to_pyg_data(row["graph"])
        if rwse is not None:
            g = rwse(g)
        g.y = torch.tensor(
            [float(row["kpis"][name]) for name in MODEL_KPI_NAMES],
            dtype=torch.float32,
        )
        graphs.append(g)
    return graphs


def stats_tensor(t):
    return t.mean(dim=0), t.std(dim=0, unbiased=False).clamp_min(1e-6)


def feature_stats(graphs):
    x = torch.cat([g.x for g in graphs], dim=0)
    ea = torch.cat([g.edge_attr for g in graphs if g.edge_attr.numel() > 0], dim=0)
    gf = torch.cat([g.graph_features for g in graphs], dim=0)
    return {"node": stats_tensor(x), "edge": stats_tensor(ea), "graph": stats_tensor(gf)}


def normalize_batch(batch, fs):
    nm, ns = fs["node"]
    batch.x = (batch.x - nm.to(batch.x)) / ns.to(batch.x)
    if batch.edge_attr.numel() > 0:
        em, es = fs["edge"]
        batch.edge_attr = (batch.edge_attr - em.to(batch.edge_attr)) / es.to(batch.edge_attr)
    if hasattr(batch, "graph_features"):
        gm, gs = fs["graph"]
        batch.graph_features = (batch.graph_features - gm.to(batch.graph_features)) / gs.to(batch.graph_features)


# log1p + z-score target normalization (KPIs positive, heavy-tailed)
class TargetNorm:
    def __init__(self, graphs):
        log = torch.log1p(torch.stack([g.y for g in graphs]).clamp_min(0))
        self.mean, self.std = log.mean(dim=0), log.std(dim=0).clamp_min(1e-6)

    def encode(self, y):
        return (torch.log1p(y.clamp_min(0)) - self.mean.to(y)) / self.std.to(y)

    def decode(self, y):
        return torch.expm1(y * self.std.to(y) + self.mean.to(y)).clamp_min(0)


def batches(graphs, batch_size, shuffle=False, rng=None):
    idx = list(range(len(graphs)))
    if shuffle and rng is not None:
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield Batch.from_data_list([graphs[i] for i in idx[start:start + batch_size]])


# ----------------------------------------------------------------------------
# Train / eval
# ----------------------------------------------------------------------------
def run_epoch(model, graphs, fs, tnorm, device, batch_size, optimizer, loss_fn, rng):
    total, n = 0.0, 0
    for batch in batches(graphs, batch_size, shuffle=optimizer is not None, rng=rng):
        batch = batch.to(device)
        normalize_batch(batch, fs)
        y = tnorm.encode(batch.y.view(-1, len(MODEL_KPI_NAMES)).to(device))
        pred = model(batch)
        loss = loss_fn(pred, y)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


def evaluate(model, graphs, fs, tnorm, device, batch_size):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for batch in batches(graphs, batch_size):
            batch = batch.to(device)
            actual = batch.y.view(-1, len(MODEL_KPI_NAMES)).clone()
            normalize_batch(batch, fs)
            preds.append(tnorm.decode(model(batch)).cpu())
            actuals.append(actual.cpu())
    preds, actuals = torch.cat(preds), torch.cat(actuals)
    out = {}
    for i, name in enumerate(MODEL_KPI_NAMES):
        p, a = preds[:, i], actuals[:, i]
        ss_res = float(((p - a) ** 2).sum())
        ss_tot = float(((a - a.mean()) ** 2).sum())
        out[name] = {
            "r2": round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0,
            "mae": round(float((p - a).abs().mean()), 4),
        }
    return out


def train(model, train_g, val_g, fs, tnorm, device, epochs, lr, batch_size, patience=20):
    import random
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.SmoothL1Loss()
    rng = random.Random(42)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        run_epoch(model, train_g, fs, tnorm, device, batch_size, opt, loss_fn, rng)
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = run_epoch(model, val_g, fs, tnorm, device, batch_size, None, loss_fn, rng)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  val {vloss:.4f}")
        if bad >= patience:
            print(f"  early stop @ {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ----------------------------------------------------------------------------
# Speed benchmark
# ----------------------------------------------------------------------------
def speed_benchmark(model, fs, device, walk_length, scales, n=40, batch_size=64):
    print("\n=== Speed benchmark (per scenario) ===")
    rwse = AddRandomWalkPE(walk_length=walk_length) if walk_length > 0 else None
    results = {}
    for n_machines, n_orders in scales:
        scenarios = [
            generate_scenario(
                700_000 + n_machines * 1000 + i,
                n_machines=n_machines, n_orders=n_orders,
                min_route_length=2, max_route_length=6,
                reentrant_route_probability=0.3, route_shuffle_probability=0.3,
                wide_input_buffers=True,
            )
            for i in range(n)
        ]

        t0 = time.perf_counter()
        for s in scenarios:
            run_simulation(s)
        sim_ms = (time.perf_counter() - t0) / n * 1000

        t0 = time.perf_counter()
        graphs = [graph_dict_to_pyg_data(scenario_to_graph_dict(s)) for s in scenarios]
        conv_ms = (time.perf_counter() - t0) / n * 1000

        rwse_ms = 0.0
        if rwse is not None:
            t0 = time.perf_counter()
            graphs = [rwse(g) for g in graphs]
            rwse_ms = (time.perf_counter() - t0) / n * 1000

        model.eval()
        with torch.no_grad():
            for batch in batches(graphs, batch_size):  # warm up
                b = batch.to(device); normalize_batch(b, fs); model(b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for batch in batches(graphs, batch_size):
                b = batch.to(device); normalize_batch(b, fs); model(b)
            if device.type == "cuda":
                torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) / n * 1000

        total = conv_ms + rwse_ms + infer_ms
        results[f"{n_machines}m/{n_orders}o"] = {
            "sim_ms": round(sim_ms, 3), "convert_ms": round(conv_ms, 3),
            "rwse_ms": round(rwse_ms, 3), "infer_ms": round(infer_ms, 3),
            "surrogate_total_ms": round(total, 3),
            "speedup_total": round(sim_ms / total, 2) if total > 0 else 0,
            "speedup_infer_only": round(sim_ms / infer_ms, 2) if infer_ms > 0 else 0,
        }
        print(f"  {n_machines:3d}m/{n_orders:3d}o | sim {sim_ms:8.2f}ms | "
              f"conv {conv_ms:6.2f} rwse {rwse_ms:7.2f} infer {infer_ms:6.2f} | "
              f"total {total:8.2f}ms | x{sim_ms/total if total>0 else 0:.2f} "
              f"(infer-only x{sim_ms/infer_ms if infer_ms>0 else 0:.2f})")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, nargs="+", required=True)
    p.add_argument("--attn-type", choices=["multihead", "performer"], default="performer")
    p.add_argument("--rwse-dim", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--gps-layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-speed", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    device = torch.device(
        "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"Device: {device} | attn={args.attn_type} | rwse={args.rwse_dim} | train={args.train.name}")

    print("Loading graphs...")
    train_g = load_graphs(args.train, args.rwse_dim)
    val_g = load_graphs(args.val, args.rwse_dim)
    print(f"  {len(train_g)} train, {len(val_g)} val | targets={MODEL_KPI_NAMES}")

    fs = feature_stats(train_g)
    tnorm = TargetNorm(train_g)

    model = build_graph_gps_surrogate(
        node_feature_dim=train_g[0].x.shape[1],
        edge_feature_dim=train_g[0].edge_attr.shape[1],
        output_dim=len(MODEL_KPI_NAMES),
        rwse_dim=args.rwse_dim,
        hidden_dim=args.hidden_dim,
        gps_layers=args.gps_layers,
        heads=4, dropout=0.1,
        attn_type=args.attn_type,
    ).to(device)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"Parameters: {n_params:,}")

    print("Training...")
    t0 = time.perf_counter()
    model = train(model, train_g, val_g, fs, tnorm, device, args.epochs, args.lr, args.batch_size)
    train_time = time.perf_counter() - t0
    print(f"Trained in {train_time:.1f}s")

    print("\n=== OOD generalization across scales ===")
    ood = {}
    for tp in args.test:
        tg = load_graphs(tp, args.rwse_dim)
        metrics = evaluate(model, tg, fs, tnorm, device, 64)
        ood[tp.stem] = metrics
        print(f"  {tp.stem:10s} | " + " ".join(f"{k}={v['r2']:.3f}" for k, v in metrics.items()))

    speed = {}
    if not args.no_speed:
        speed = speed_benchmark(model, fs, device, args.rwse_dim,
                                scales=[(20, 80), (30, 120), (45, 180), (60, 240), (100, 400)])

    results = {
        "config": {
            "attn_type": args.attn_type, "rwse_dim": args.rwse_dim,
            "hidden_dim": args.hidden_dim, "gps_layers": args.gps_layers,
            "train_file": str(args.train), "params": n_params,
            "train_time_s": round(train_time, 1),
        },
        "ood_generalization": ood,
        "speed": speed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
