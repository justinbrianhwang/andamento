"""Generate the paper's result figures from results/*.json.

Outputs vector PDFs into paper/figures/ (gitignored):
  fig-landscapes.pdf   spread + failure structure per device (Sec. 5)
  fig-regret.pdf       within-5% vs budget, 3 strategies, 4 devices (Sec. 6)
  fig-time.pdf         within-5% vs wall-clock budget (Sec. 6)
  fig-warmstart.pdf    cross-device warm-start heatmap (Sec. 6)
  fig-grid.pdf         grid execution semantics schematic (Sec. 2)

Usage (from repo root):
  python analysis/plot_paper.py
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_replay import load as load_axes          # noqa: E402
from warm_start import run as warm_run, WARM_K       # noqa: E402

OUT = "paper/figures"
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",          # Times-compatible math glyphs
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

STRATEGY_STYLE = {
    "random":    dict(color="#888888", marker="o", label="random"),
    "tpe":       dict(color="#1f77b4", marker="s", label="TPE (per-axis)"),
    "surrogate": dict(color="#d62728", marker="^", label="feature surrogate"),
}

TYPE_COLOR = {
    "loud":    "#4c72b0",   # compile-time rejection
    "refusal": "#c44e52",   # launch refusal
    "hybrid":  "#8172b3",   # refusal + cliff
    "cliff":   "#dd8452",   # silent cliff
    "none":    "#55a868",   # no failures, tame
}

# (sweep file, display name, failure type)
DEVICES = [
    ("results/sweep_v5e_bf16.json",  "TPU v5e",                     "loud"),
    ("results/sweep_v6e_bf16.json",  "TPU v6e",                     "loud"),
    ("results/sweep_a100_bf16.json", "A100 (DC Ampere)",            "refusal"),
    ("results/sweep_result_NVIDIA_GeForce_RTX_4090.json",
     "RTX 4090 (consumer Ada)",     "refusal"),
    ("results/sweep_result_NVIDIA_GeForce_RTX_5090.json",
     "RTX 5090 (consumer Blackwell)", "refusal"),
    ("results/sweep_result_NVIDIA_GeForce_RTX_3090.json",
     "RTX 3090 (consumer Ampere)",  "hybrid"),
    ("results/sweep_result_NVIDIA_GeForce_RTX_4060_Laptop_GPU.json",
     "RTX 4060 Laptop (Ada)",       "hybrid"),
    ("results/sweep_l4_bf16.json",   "L4 (DC Ada)",                 "cliff"),
    ("results/sweep_g4_bf16.json",   "RTX PRO 6000 (wkst. Blackwell)", "cliff"),
    ("results/sweep_result_h100_nvl.json", "H100 NVL (DC Hopper)",  "none"),
    ("results/sweep_result_b200.json", "B200 (DC Blackwell)",       "none"),
]


def sweep_stats(path):
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    rs = blob["sweeps"][0]["results"]
    ok = [r["median_ms"] for r in rs if r["status"] == "ok"]
    return {"n": len(rs), "n_fail": len(rs) - len(ok),
            "spread": max(ok) / min(ok)}


def fig_landscapes():
    stats = [(name, typ, sweep_stats(path)) for path, name, typ in DEVICES]
    fig, ax = plt.subplots(figsize=(6.9, 3.1))
    ys = range(len(stats))[::-1]
    for y, (name, typ, s) in zip(ys, stats):
        ax.barh(y, s["spread"], color=TYPE_COLOR[typ], height=0.62)
        note = f"{s['spread']:.1f}$\\times$"
        if s["n_fail"]:
            kind = "compile-rejected" if typ == "loud" else "refused"
            note += f"   ({s['n_fail']}/{s['n']} {kind})"
        ax.text(s["spread"] * 1.12, y, note, va="center", fontsize=7)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([name for name, _, _ in stats])
    ax.set_xscale("log")
    ax.set_xlim(1, 6000)
    ax.set_xlabel("latency spread among surviving configurations "
                  "(worst / best, log scale)")
    handles = [Rectangle((0, 0), 1, 1, color=TYPE_COLOR[t]) for t in
               ("loud", "refusal", "hybrid", "cliff", "none")]
    ax.legend(handles, ["compile-rejected (TPU)", "launch-refused",
                        "hybrid (refusal + slow tail)",
                        "feasible, heavy tail",
                        "feasible, moderate tail"],
              loc="lower right", frameon=False, ncol=1,
              title="outcome class", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig-landscapes.pdf")
    plt.close(fig)


def fig_regret():
    with open("results/surrogate_replay.json", encoding="utf-8") as f:
        data = json.load(f)["devices"]
    panels = [
        ("NVIDIA GeForce RTX 4090", "RTX 4090 — interaction-coupled refusals"),
        ("NVIDIA A100-SXM4-40GB", "A100 — warp-dependent refusals"),
        ("TPU v6 lite", "TPU v6e — compile-time VMEM boundary"),
        ("NVIDIA L4", "L4 — smooth, failure-free"),
    ]
    budgets = [5, 10, 20, 40, 60]
    def wilson(p, n=300, z=1.96):
        den = 1 + z * z / n
        center = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return center - half, center + half

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.6), sharex=True,
                             sharey=True)
    for ax, (dev, title) in zip(axes.flat, panels):
        for strat, style in STRATEGY_STYLE.items():
            ps = [data[dev][f"{strat}@{b}"]["frac_within_5pct"]
                  for b in budgets]
            ys = [100 * p for p in ps]
            los = [100 * wilson(p)[0] for p in ps]
            his = [100 * wilson(p)[1] for p in ps]
            ax.plot(budgets, ys, lw=1.4, ms=3.5, **style)
            ax.fill_between(budgets, los, his, color=style["color"],
                            alpha=0.15, lw=0)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels(budgets)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0, 104)
        ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    for ax in axes[1]:
        ax.set_xlabel("measurement budget $B$")
    for ax in axes[:, 0]:
        ax.set_ylabel("runs within 5% of oracle (%)")
    axes[0, 0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig-regret.pdf")
    plt.close(fig)


def fig_time():
    with open("results/time_regret.json", encoding="utf-8") as f:
        blob = json.load(f)
    budgets = blob["budgets_s"]
    panels = [
        ("NVIDIA GeForce RTX 4090",
         "RTX 4090 (exhaustive sweep: 21 min)"),
        ("NVIDIA GeForce RTX 3090",
         "RTX 3090 (exhaustive sweep: 46 min)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=True)
    for ax, (dev, title) in zip(axes, panels):
        summ = blob["devices"][dev]["summary"]
        for strat, style in STRATEGY_STYLE.items():
            pts = [(t, 100 * summ[f"{strat}@{t}s"]["frac_within_5pct"])
                   for t in budgets if f"{strat}@{t}s" in summ]
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    lw=1.4, ms=3.5, **style)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{t}" for t in budgets])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("wall-clock budget (s, log scale)")
        ax.set_ylim(0, 104)
        ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    axes[0].set_ylabel("runs within 5% of oracle (%)")
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig-time.pdf")
    plt.close(fig)


WARM_FILES = [
    ("results/sweep_a100_bf16.json", "A100"),
    ("results/sweep_result_NVIDIA_GeForce_RTX_3090.json", "3090"),
    ("results/sweep_l4_bf16.json", "L4"),
    ("results/sweep_result_NVIDIA_GeForce_RTX_4090.json", "4090"),
    ("results/sweep_g4_bf16.json", "RTX PRO"),
    ("results/sweep_result_NVIDIA_GeForce_RTX_5090.json", "5090"),
]
WARM_SEEDS = 300
WARM_BUDGET = 20


def fig_warmstart():
    lands = []
    for path, label in WARM_FILES:
        _, rows = load_axes(path)
        lands.append((label, rows))

    def mean_regret(trows, oracle, warm_order):
        rs = []
        for s in range(WARM_SEEDS):
            best = warm_run(trows, WARM_BUDGET, s, warm_order)
            if best is not None:
                rs.append(best / oracle)
        return sum(rs) / len(rs)

    n = len(lands)
    mat = [[math.nan] * n for _ in range(n)]
    cold = []
    for i, (tgt, trows) in enumerate(lands):
        oracle = min(r["ms"] for r in trows if r["ms"] is not None)
        cold.append(mean_regret(trows, oracle, None))
        for j, (src, srows) in enumerate(lands):
            if i == j:
                continue
            ranked = sorted((r for r in srows if r["ms"] is not None),
                            key=lambda r: r["ms"])
            mat[i][j] = mean_regret(trows, oracle,
                                    [r["axes"] for r in ranked])

    fig, ax = plt.subplots(figsize=(3.7, 3.1))
    lim = 0.15
    delta = [[(mat[i][j] - cold[i]) if not math.isnan(mat[i][j]) else 0.0
              for j in range(n)] for i in range(n)]
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       color="#e8e8e8"))
                ax.text(j, i, f"cold\n{cold[i]:.3f}", ha="center",
                        va="center", fontsize=6, color="#555555")
            else:
                v = delta[i][j]
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=6.5,
                        fontweight="bold" if v > 1e-9 else "normal")
    labels = [l for l, _ in lands]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45,
                                                ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("warm-start source (its top-5 configs)")
    ax.set_ylabel("target device")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("warm $-$ cold mean regret at $B{=}20$", fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig-warmstart.pdf")
    plt.close(fig)


def fig_grid():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.4))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10); ax.set_ylim(0, 5)
        ax.axis("off")

    # --- Mosaic / TPU: sequential grid --------------------------------
    ax1.set_title("Mosaic (TPU): sequential grid", fontsize=9)
    for k in range(4):
        x = 0.6 + 2.3 * k
        ax1.add_patch(Rectangle((x, 3.1), 1.6, 1.3, facecolor="#cfe3f5",
                                edgecolor="#4c72b0", lw=1.0))
        ax1.text(x + 0.8, 3.75, f"$k={k}$", ha="center", va="center",
                 fontsize=8)
        if k < 3:
            ax1.add_patch(FancyArrowPatch((x + 1.65, 3.75),
                                          (x + 2.25, 3.75),
                                          arrowstyle="-|>",
                                          mutation_scale=9,
                                          color="#4c72b0"))
        ax1.add_patch(FancyArrowPatch((x + 0.8, 3.05), (4.9, 1.75),
                                      arrowstyle="-|>", mutation_scale=7,
                                      color="#999999", lw=0.7))
    ax1.add_patch(Rectangle((2.5, 0.9), 4.8, 0.9, facecolor="#fff3cc",
                            edgecolor="#c9a227", lw=1.0))
    ax1.text(4.9, 1.35, "VMEM scratch accumulator", ha="center",
             va="center", fontsize=7)
    ax1.text(4.9, 0.35, "grid steps run in order on one core;\n"
             "the $K$ reduction spans steps ($+\\!=$)",
             ha="center", va="center", fontsize=7, color="#444444")

    # --- Triton / GPU: concurrent grid --------------------------------
    ax2.set_title("Triton (GPU): concurrent grid", fontsize=9)
    for i in range(3):
        for j in range(3):
            x, y = 1.1 + 2.0 * j, 3.9 - 1.25 * i
            ax2.add_patch(Rectangle((x, y), 1.55, 1.0,
                                    facecolor="#f5d5d5",
                                    edgecolor="#c44e52", lw=1.0))
            ax2.text(x + 0.775, y + 0.5, "for $k$: $+\\!=$",
                     ha="center", va="center", fontsize=6.5)
    ax2.text(7.6, 3.1, "CTAs launch\nconcurrently", ha="left",
             va="center", fontsize=7.5)
    ax2.text(7.6, 2.0, "knobs:\nnum_warps\nnum_stages", ha="left",
             va="center", fontsize=7.5, color="#444444")
    ax2.text(4.05, 0.35, "no cross-step state: each block reduces $K$\n"
             "inside the kernel, in registers",
             ha="center", va="center", fontsize=7, color="#444444")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig-grid.pdf")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_landscapes(); print("fig-landscapes.pdf")
    fig_regret();     print("fig-regret.pdf")
    fig_time();       print("fig-time.pdf")
    fig_grid();       print("fig-grid.pdf")
    fig_warmstart();  print("fig-warmstart.pdf")
