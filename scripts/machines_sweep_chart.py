"""Single-panel latency chart: surrogate inference vs SimPy.

Fixed 1000-order workload, machines 20 to 100. Surrogate inference only (the
cached-search case). Numbers are read from results/machines_1000orders.json.

Run:  uv run python scripts/machines_sweep_chart.py
Out:  assets/latency_1000orders.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "machines_1000orders.json"
OUT = ROOT / "assets" / "latency_1000orders.png"

plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": "#9e9e9e", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

C_SIM = "#1565c0"
C_CACHE = "#2e7d32"


def main():
    d = json.loads(SRC.read_text())["speed"]
    rows = sorted((int(k.split("m/")[0]), v) for k, v in d.items())
    x = [m for m, _ in rows]
    sim = [v["sim_ms"] for _, v in rows]
    infer = [v["infer_ms"] for _, v in rows]
    sp = [v["speedup_infer_only"] for _, v in rows]

    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=150)

    ax.fill_between(x, infer, sim, color=C_CACHE, alpha=0.06, zorder=1)
    ax.plot(x, sim, "-o", color=C_SIM, lw=2.6, ms=8, zorder=3,
            label="SimPy discrete-event simulation")
    ax.plot(x, infer, "-o", color=C_CACHE, lw=2.6, ms=8, zorder=3,
            label="Neural surrogate (cached search)")

    ax.set_xticks(x)
    ax.set_ylim(0, max(sim) * 1.18)
    ax.set_xlabel("Number of machines  (1000-order workload)")
    ax.set_ylabel("Latency per scenario  (ms)")
    ax.set_title("Surrogate is about 4x faster than SimPy across plant sizes",
                 fontweight="bold", pad=12)

    # midpoint speedup callout
    mid = len(x) // 2
    ax.annotate(f"{min(sp):.1f}-{max(sp):.1f}x faster",
                xy=(x[mid], (sim[mid] + infer[mid]) / 2),
                ha="center", va="center", fontsize=12.5,
                color=C_CACHE, fontweight="bold")

    ax.legend(loc="lower right", fontsize=10.5, frameon=True, framealpha=0.96)
    fig.text(0.5, 0.005,
             "Measured on RTX 3070, 417,927-param Performer-GPS. "
             "Source: results/machines_1000orders.json",
             ha="center", fontsize=8, color="#9e9e9e", style="italic")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
