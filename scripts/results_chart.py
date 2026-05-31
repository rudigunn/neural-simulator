"""Two-panel results chart, read from results/frontier_wide_performer_nopos.json.

That JSON holds the wide-training, Performer-attention, no-RWSE configuration.

  (A) Inference latency per scenario vs machine count: SimPy simulation vs
      surrogate inference, measured from 20 to 100 machines.

  (B) Out-of-distribution accuracy (R2) per KPI vs machine count. Trained on
      3-16 machines, tested on unseen 20 / 30 / 45-machine re-entrant layouts.

All numbers are read directly from the JSON.

Run:  uv run python scripts/results_chart.py
Out:  assets/frontier_results_chart.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "frontier_wide_performer_nopos.json"
OUT = ROOT / "assets" / "frontier_results_chart.png"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": "#9e9e9e",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

C_SIM = "#1565c0"     # SimPy
C_CACHE = "#2e7d32"   # surrogate cached (the win)
C_E2E = "#e65100"     # surrogate end-to-end


def load():
    d = json.loads(SRC.read_text())
    sp = d["speed"]
    scales = [(int(k.split("m/")[0]), v) for k, v in sp.items()]
    scales.sort()
    x = [m for m, _ in scales]
    sim = [v["sim_ms"] for _, v in scales]
    infer = [v["infer_ms"] for _, v in scales]
    total = [v["surrogate_total_ms"] for _, v in scales]

    ood = d["ood_generalization"]
    acc_scales = sorted(int(k.split("_")[1]) for k in ood)
    kpis = ["makespan", "throughput", "mean_flow_time",
            "mean_tardiness", "mean_queue_wait_time"]
    acc = {kpi: [ood[f"test_{s}"][kpi]["r2"] for s in acc_scales] for kpi in kpis}
    return x, sim, infer, total, acc_scales, acc, d["config"]


def main():
    x, sim, infer, total, acc_scales, acc, cfg = load()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=150)

    # ---- Panel A: latency vs scale (log y) ----
    axL.fill_between(x, infer, sim, color=C_CACHE, alpha=0.06)
    axL.plot(x, sim, "-o", color=C_SIM, lw=2.6, ms=7, label="SimPy")
    axL.plot(x, infer, "-o", color=C_CACHE, lw=2.6, ms=7, label="Surrogate")
    axL.set_yscale("log")
    axL.set_yticks([1, 2, 5, 10, 20, 50, 100])
    axL.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axL.yaxis.set_minor_locator(plt.NullLocator())
    axL.set_xlabel("Number of machines")
    axL.set_ylabel("Latency per scenario  (ms, log)")
    axL.set_title("Inference latency", fontweight="bold")

    sp = sim[-1] / infer[-1]
    axL.annotate(f"{sp:.1f}x faster", xy=(x[-1], infer[-1]),
                 xytext=(x[-1] - 30, infer[-1] * 0.45),
                 fontsize=11, color=C_CACHE, fontweight="bold", ha="center", va="top")
    axL.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.95)

    # ---- Panel B: OOD accuracy per KPI ----
    holds = {"mean_flow_time": ("flow time", "#00897b"),
             "mean_tardiness": ("tardiness", "#3949ab"),
             "mean_queue_wait_time": ("queue wait", "#5e35b1")}
    breaks = {"makespan": ("makespan", "#e65100"),
              "throughput": ("throughput", "#c62828")}

    for kpi, (lab, col) in holds.items():
        axR.plot(acc_scales, acc[kpi], "-o", color=col, lw=2.2, ms=6, label=lab)
    for kpi, (lab, col) in breaks.items():
        axR.plot(acc_scales, acc[kpi], "--s", color=col, lw=2.0, ms=6, label=lab)

    axR.axhline(0, color="#9e9e9e", lw=1.0, ls=":")
    axR.axhspan(-0.6, 0, color="#c62828", alpha=0.05)
    axR.set_ylim(-0.5, 1.0)
    axR.set_xticks(acc_scales)
    axR.set_xlabel("Number of machines  (trained on 3-16)")
    axR.set_ylabel("OOD accuracy  (R²)")
    axR.set_title("Generalization to unseen scales", fontweight="bold")
    axR.legend(loc="lower left", fontsize=9, ncol=2, frameon=True, framealpha=0.95)

    fig.suptitle("Neural surrogate vs SimPy, measured on RTX 3070",
                 fontsize=14, fontweight="bold", y=0.99)

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
