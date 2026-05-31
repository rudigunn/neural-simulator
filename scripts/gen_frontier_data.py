"""Generate the datasets used by scripts/frontier_experiment.py.

Produces two training distributions and several held-out test scales:

1. A wide training distribution (3-16 machines) and held-out larger scales
   (20, 30, 45 machines), so generalization can be plotted as a curve across
   scale rather than a single out-of-distribution point.

2. A narrow training distribution (5-10 machines) for a narrow-vs-wide
   comparison, isolating the effect of the training distribution from the
   effect of the architecture.

All splits use re-entrant and shuffled routes, so the held-out scales differ in
topology and not only in node count.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neural_simulator.data.dataset import make_dataset_item, save_dataset_split_map


def _gen(name, count, out_dir, *, machine_counts, orders_per_machine, seed):
    rng = random.Random(seed)
    items = []
    next_seed = seed
    t0 = time.perf_counter()
    for _ in range(count):
        n_machines = rng.choice(machine_counts)
        items.append(
            make_dataset_item(
                next_seed,
                n_machines=n_machines,
                n_orders=n_machines * orders_per_machine,
                min_route_length=2,
                max_route_length=6,
                reentrant_route_probability=0.3,
                route_shuffle_probability=0.3,
                wide_input_buffers=True,
            )
        )
        next_seed += 1
    save_dataset_split_map({name: items}, out_dir)
    dt = time.perf_counter() - t0
    lo, hi = min(machine_counts), max(machine_counts)
    print(f"  {name:14s} {count:5d} scenarios  {lo}-{hi} machines  "
          f"{orders_per_machine} orders/machine  ({dt:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/frontier"))
    parser.add_argument("--train-count", type=int, default=1500)
    parser.add_argument("--narrow-count", type=int, default=600)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=120)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("Generating WIDE training distribution (3-16 machines)...")
    _gen("train_wide", args.train_count, out,
         machine_counts=tuple(range(3, 17)), orders_per_machine=4, seed=1)

    print("Generating NARROW training distribution (5-10 machines, old pilot)...")
    _gen("train_narrow", args.narrow_count, out,
         machine_counts=tuple(range(5, 11)), orders_per_machine=4, seed=900_001)

    print("Generating validation (8-16 machines)...")
    _gen("val", args.val_count, out,
         machine_counts=tuple(range(8, 17)), orders_per_machine=4, seed=200_001)

    print("Generating multi-scale OOD test sets...")
    _gen("test_20", args.test_count, out, machine_counts=(20,), orders_per_machine=4, seed=300_001)
    _gen("test_30", args.test_count, out, machine_counts=(30,), orders_per_machine=4, seed=400_001)
    _gen("test_45", max(args.test_count // 2, 60), out, machine_counts=(45,), orders_per_machine=4, seed=500_001)

    print(f"\nDone. Datasets written to {out}/")


if __name__ == "__main__":
    main()
