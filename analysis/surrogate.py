"""Feature-based surrogate search: fixing TPE's axis-interaction blindness.

TPE-lite models each axis independently, which breaks on landscapes where
the same axis value is simultaneously optimal and fatal depending on the
other axes (RTX 4090: num_warps=4 is the optimum on small tiles AND the
refusal zone on big tiles, so per-axis TPE steers away from the best
region and loses to random).

The surrogate instead works in the static feature space every config
carries (smem estimate, threads, tile elems, arithmetic intensity, ...):

  - ridge regression on log-latency of the successes observed so far,
  - a k-NN failure-probability estimate over observed outcomes,
  - acquisition = predicted log-ms + penalty * P(fail), argmin over the
    unmeasured configs, with a small epsilon of random exploration.

Features couple the axes (smem estimate is a product of tile dims and
stages), so "big tile + w4 fails but big tile + w8 runs" is learnable
from a handful of observations — precisely what per-axis counts cannot
express. Pure stdlib.

Usage:
  python analysis/surrogate.py results/sweep_*.json
"""

import json
import math
import random
import sys

from search_replay import load as load_axes, tpe_pick, INIT_RANDOM, GAMMA

BUDGETS = [5, 10, 20, 40, 60]
SEEDS = 300
RIDGE_LAMBDA = 1.0
KNN_K = 3
FAIL_PENALTY = 2.0   # nats added per unit failure probability (~7.4x ms)
EPSILON = 0.1        # exploration probability after bootstrap


def feature_vector(row):
    """Static features known before running the config. Uses the recorded
    feature dict when present, plus log2 axis values; TPU rows without a
    feature dict fall back to derived products of the tile dims."""
    axes = row["axes"]
    f = row.get("features") or {}
    v = [math.log2(a) for a in axes]
    dims = [a for a in axes if a >= 32]           # tile dims (bm, bn, bk...)
    if len(dims) >= 2:
        prod = 1
        for d in dims:
            prod *= d
        v.append(math.log2(prod))                  # total tile volume
        v.append(math.log2(dims[0] * dims[1]))     # output tile elems
    for key in ("smem_ratio", "arithmetic_intensity", "elems_per_thread",
                "threads", "vmem_ratio", "vmem_estimate_bytes"):
        if key in f:
            val = f[key]
            v.append(math.log2(val) if val > 0 else 0.0)
    return v


def load(path):
    """Like search_replay.load but keeps the per-config feature dict."""
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    sweep = blob["sweeps"][0] if "sweeps" in blob else blob
    rows = []
    for r in sweep["results"]:
        axes = tuple(r["config"]) + (r.get("num_warps"), r.get("num_stages"))
        axes = tuple(a for a in axes if a is not None)
        rows.append({"axes": axes,
                     "ms": r["median_ms"] if r["status"] == "ok" else None,
                     "features": r.get("features")})
    return blob.get("device", "?"), rows


def standardize(rows):
    vecs = [feature_vector(r) for r in rows]
    d = len(vecs[0])
    means = [sum(v[i] for v in vecs) / len(vecs) for i in range(d)]
    sds = []
    for i in range(d):
        var = sum((v[i] - means[i]) ** 2 for v in vecs) / len(vecs)
        sds.append(math.sqrt(var) or 1.0)
    for r, v in zip(rows, vecs):
        r["x"] = [(v[i] - means[i]) / sds[i] for i in range(d)]
    return rows


def solve(a, b):
    """Gaussian elimination with partial pivoting for the (small) normal
    equations."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        if abs(m[col][col]) < 1e-12:
            continue
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    w = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][c] * w[c] for c in range(r + 1, n))
        w[r] = s / m[r][r] if abs(m[r][r]) > 1e-12 else 0.0
    return w


def ridge_fit(xs, ys, lam=RIDGE_LAMBDA):
    d = len(xs[0]) + 1                              # + intercept
    xtx = [[lam if i == j else 0.0 for j in range(d)] for i in range(d)]
    xty = [0.0] * d
    for x, y in zip(xs, ys):
        xe = [1.0] + x
        for i in range(d):
            xty[i] += xe[i] * y
            for j in range(d):
                xtx[i][j] += xe[i] * xe[j]
    return solve(xtx, xty)


def predict(w, x):
    xe = [1.0] + x
    return sum(wi * xi for wi, xi in zip(w, xe))


def surrogate_pick(observed, remaining):
    succ = [r for r in observed if r["ms"] is not None]
    if len(succ) < 3 or random.random() < EPSILON:
        return random.choice(remaining)
    w = ridge_fit([r["x"] for r in succ],
                  [math.log(r["ms"]) for r in succ])

    def p_fail(x):
        dists = sorted((sum((a - b) ** 2 for a, b in zip(x, r["x"])),
                        r["ms"] is None) for r in observed)
        top = dists[:KNN_K]
        return sum(1 for _, failed in top if failed) / len(top)

    def acquisition(r):
        return predict(w, r["x"]) + FAIL_PENALTY * p_fail(r["x"])

    return min(remaining, key=acquisition)


def run_strategy(rows, budget, strategy, rng_seed):
    random.seed(rng_seed)
    remaining = list(rows)
    random.shuffle(remaining)
    observed = []
    for _ in range(min(budget, len(rows))):
        if strategy == "random" or len(observed) < INIT_RANDOM:
            pick = remaining.pop()
        elif strategy == "tpe":
            pick = tpe_pick(observed, remaining)
            remaining.remove(pick)
        else:
            pick = surrogate_pick(observed, remaining)
            remaining.remove(pick)
        observed.append(pick)
    found = [r["ms"] for r in observed if r["ms"] is not None]
    wasted = sum(1 for r in observed if r["ms"] is None)
    return (min(found) if found else None), wasted


def main():
    paths = sys.argv[1:]
    all_out = {}
    for path in paths:
        device, rows = load(path)
        standardize(rows)
        oracle = min(r["ms"] for r in rows if r["ms"] is not None)
        n_fail = sum(1 for r in rows if r["ms"] is None)
        print(f"\n{device}: {len(rows)} configs ({n_fail} failures), "
              f"oracle {oracle:.3f} ms")
        print(f"{'budget':>7} | {'strategy':>9} | {'mean regret':>11} | "
              f"{'within 5%':>9} | {'wasted':>6}")
        summary = {}
        for budget in BUDGETS:
            for strategy in ("random", "tpe", "surrogate"):
                regrets, wastes = [], []
                for seed in range(SEEDS):
                    best, wasted = run_strategy(rows, budget, strategy, seed)
                    if best is None:
                        continue
                    regrets.append(best / oracle)
                    wastes.append(wasted)
                mean_r = sum(regrets) / len(regrets)
                within = sum(1 for r in regrets if r <= 1.05) / len(regrets)
                mean_w = sum(wastes) / len(wastes)
                summary[f"{strategy}@{budget}"] = {
                    "mean_regret": mean_r, "frac_within_5pct": within,
                    "mean_wasted": mean_w,
                }
                print(f"{budget:>7} | {strategy:>9} | {mean_r:>10.3f}x | "
                      f"{within:>8.0%} | {mean_w:>6.1f}")
        all_out[device] = summary

    with open("results/surrogate_replay.json", "w", encoding="utf-8") as fh:
        json.dump({"seeds": SEEDS, "ridge_lambda": RIDGE_LAMBDA,
                   "knn_k": KNN_K, "fail_penalty": FAIL_PENALTY,
                   "epsilon": EPSILON, "devices": all_out}, fh, indent=1)
    print("\nwrote results/surrogate_replay.json")


if __name__ == "__main__":
    main()
