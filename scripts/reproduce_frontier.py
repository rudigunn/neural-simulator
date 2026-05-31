"""Reproduce the frontier results end to end.

Runs the full pipeline that produces the frontier charts:

  1. Generate the frontier datasets            (scripts/gen_frontier_data.py)
  2. Train and evaluate three configurations    (scripts/frontier_experiment.py)
       a. wide   + performer + RWSE 20   -> results/frontier_wide_performer.json
       b. wide   + performer + no RWSE   -> results/frontier_wide_performer_nopos.json
       c. narrow + performer + no RWSE   -> results/frontier_narrow_performer.json
  3. Run the fixed-workload latency sweep        (scripts/machines_sweep.py)
       -> results/machines_1000orders.json
  4. Render the charts                           (scripts/results_chart.py,
                                                   scripts/machines_sweep_chart.py)
       -> assets/frontier_results_chart.png, assets/latency_1000orders.png

Configuration (b) is the one read by the charts and quoted as the best config.
Steps 2a and 2c are ablations and skip the speed benchmark to save time.

All steps (training and charts) need the ML dependency group, which includes
torch, torch_geometric, and matplotlib:
    uv sync --group dev --group ml
If matplotlib lives in a different interpreter than torch, pass --chart-python
to point the chart steps at it.

Usage:
    uv run python scripts/reproduce_frontier.py
    uv run python scripts/reproduce_frontier.py --skip-data --skip-charts
    uv run python scripts/reproduce_frontier.py --chart-python python
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/frontier"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--python", default=sys.executable,
        help="Interpreter for the data and training steps (needs torch).",
    )
    parser.add_argument(
        "--chart-python", default=sys.executable,
        help="Interpreter for the chart steps (needs matplotlib).",
    )
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-speed", action="store_true",
                        help="Skip the machines_sweep latency run.")
    parser.add_argument("--skip-charts", action="store_true")
    args = parser.parse_args()

    py = args.python
    data = args.data_dir
    results = args.results_dir

    def split(name: str) -> str:
        return str(data / f"{name}.jsonl")

    test_files = [split("test_20"), split("test_30"), split("test_45")]

    if not args.skip_data:
        run([py, str(SCRIPTS / "gen_frontier_data.py"), "--output-dir", str(data)])

    if not args.skip_train:
        # (a) wide + performer + RWSE 20 (ablation, no speed benchmark)
        run([py, str(SCRIPTS / "frontier_experiment.py"),
             "--train", split("train_wide"), "--val", split("val"),
             "--test", *test_files,
             "--attn-type", "performer", "--rwse-dim", "20", "--no-speed",
             "--output", str(results / "frontier_wide_performer.json")])

        # (b) wide + performer + no RWSE (best config; feeds the charts)
        run([py, str(SCRIPTS / "frontier_experiment.py"),
             "--train", split("train_wide"), "--val", split("val"),
             "--test", *test_files,
             "--attn-type", "performer", "--rwse-dim", "0",
             "--output", str(results / "frontier_wide_performer_nopos.json")])

        # (c) narrow + performer + no RWSE (ablation, no speed benchmark)
        run([py, str(SCRIPTS / "frontier_experiment.py"),
             "--train", split("train_narrow"), "--val", split("val"),
             "--test", *test_files,
             "--attn-type", "performer", "--rwse-dim", "0", "--no-speed",
             "--output", str(results / "frontier_narrow_performer.json")])

    if not args.skip_speed:
        run([py, str(SCRIPTS / "machines_sweep.py")])

    if not args.skip_charts:
        run([args.chart_python, str(SCRIPTS / "results_chart.py")])
        run([args.chart_python, str(SCRIPTS / "machines_sweep_chart.py")])

    print("\nDone.")


if __name__ == "__main__":
    main()
