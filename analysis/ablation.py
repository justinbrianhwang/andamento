"""Ablation: which ingredient of the feature surrogate does the work?

The full surrogate changes several things at once relative to per-axis
TPE (multivariate model, engineered resource features, ridge latency
model, k-NN failure probability). This replay separates them:

  random       uniform without replacement
  tpe          per-axis TPE-lite (baseline)
  full         engineered features + ridge(log ms) + kNN failure penalty
  lat_only     engineered features + ridge only (failures ignored)
  fail_only    engineered features + kNN failure model only
  onehot       one-hot axis encoding + ridge + kNN (multivariate,
               no engineered features)
  rawaxes      log2 raw axis values + ridge + kNN (numeric axes,
               no resource features)
  worst_impute engineered features + ridge, failures imputed as
               10x the worst observed latency (no failure model)
  hardfilter   capacity estimate as a static filter (smem_ratio > 1
               pruned unprobed), then random --- the design Andamento
               rejects; regret is measured against the TRUE oracle,
               so pruning the optimum shows up as a regret floor

All model variants bootstrap from INIT_RANDOM random probes and use
epsilon-greedy exploration like the full surrogate. Pure stdlib.

Usage:
  python analysis/ablation.py results/sweep_*.json
"""

import json
import math
import random
import sys

from search_replay import tpe_pick, INIT_RANDOM
from surrogate import (load, standardize, ridge_fit, predict,
                       FAIL_PENALTY, KNN_K, EPSILON)

BUDGETS = [10, 20, 40]
SEEDS = 200


def add_encodings(rows):
    """Attach one-hot and raw-axis encodings next to the engineered x."""
    n_axes = len(rows[0]["axes"])
    values = [sorted({r["axes"][i] for r in rows}) for i in range(n_axes)]
    for r in rows:
        onehot = []
        for i, v in enumerate(r["axes"]):
            onehot.extend(1.0 if v == u else 0.0 for u in values[i])
        r["x_onehot"] = onehot
    vecs = [[math.log2(a) for a in r["axes"]] for r in rows]
    d = len(vecs[0])
    means = [sum(v[i] for v in vecs) / len(vecs) for i in range(d)]
    sds = []
    for i in range(d):
        var = sum((v[i] - means[i]) ** 2 for v in vecs) / len(vecs)
        sds.append(math.sqrt(var) or 1.0)
    for r, v in zip(rows, vecs):
        r["x_axes"] = [(v[i] - means[i]) / sds[i] for i in range(d)]
    return rows


def p_fail(observed, x, key):
    dists = sorted((sum((a - b) ** 2 for a, b in zip(x, r[key])),
                    r["ms"] is None) for r in observed)
    top = dists[:KNN_K]
    return sum(1 for _, failed in top if failed) / len(top)


def model_pick(observed, remaining, key, use_lat, use_fail,
               impute_worst=False):
    succ = [r for r in observed if r["ms"] is not None]
    if len(succ) < 3 or random.random() < EPSILON:
        return random.choice(remaining)

    w = None
    if use_lat:
        xs = [r[key] for r in succ]
        ys = [math.log(r["ms"]) for r in succ]
        if impute_worst:
            worst = math.log(10 * max(r["ms"] for r in succ))
            for r in observed:
                if r["ms"] is None:
                    xs.append(r[key])
                    ys.append(worst)
        w = ridge_fit(xs, ys)

    def acq(r):
        s = predict(w, r[key]) if w is not None else 0.0
        if use_fail:
            s += FAIL_PENALTY * p_fail(observed, r[key], key)
        return s

    best = min(acq(r) for r in remaining)
    ties = [r for r in remaining if acq(r) <= best + 1e-12]
    return random.choice(ties) if len(ties) > 1 else ties[0]


VARIANTS = {
    "full":         dict(key="x",        use_lat=True,  use_fail=True),
    "lat_only":     dict(key="x",        use_lat=True,  use_fail=False),
    "fail_only":    dict(key="x",        use_lat=False, use_fail=True),
    "onehot":       dict(key="x_onehot", use_lat=True,  use_fail=True),
    "rawaxes":      dict(key="x_axes",   use_lat=True,  use_fail=True),
    "worst_impute": dict(key="x",        use_lat=True,  use_fail=False,
                         impute_worst=True),
}


def pred_ratio(row):
    f = row.get("features") or {}
    return f.get("smem_ratio") or f.get("vmem_ratio") or 0.0


def run(rows, budget, strategy, seed):
    random.seed(seed)
    pool = list(rows)
    if strategy == "hardfilter":
        pool = [r for r in pool if pred_ratio(r) <= 1.0]
    random.shuffle(pool)
    observed = []
    for _ in range(min(budget, len(pool))):
        if strategy in ("random", "hardfilter") or len(observed) < INIT_RANDOM:
            pick = pool.pop()
        elif strategy == "tpe":
            pick = tpe_pick(observed, pool)
            pool.remove(pick)
        else:
            pick = model_pick(observed, pool, **VARIANTS[strategy])
            pool.remove(pick)
        observed.append(pick)
    found = [r["ms"] for r in observed if r["ms"] is not None]
    return min(found) if found else None


def main():
    out = {}
    for path in sys.argv[1:]:
        device, rows = load(path)
        standardize(rows)
        add_encodings(rows)
        oracle = min(r["ms"] for r in rows if r["ms"] is not None)
        pruned_opt = any(r["ms"] == oracle and pred_ratio(r) > 1.0
                         for r in rows)
        n_pruned = sum(1 for r in rows if pred_ratio(r) > 1.0)
        print(f"\n{device}: hard filter would prune {n_pruned} configs"
              f"{' INCLUDING THE ORACLE' if pruned_opt else ''}")
        header = f"{'variant':>12} |" + "".join(
            f"  B={b}: reg / <5%  |" for b in BUDGETS)
        print(header)
        dev_out = {"hardfilter_prunes": n_pruned,
                   "hardfilter_prunes_oracle": pruned_opt}
        for strat in (["random", "tpe"] + list(VARIANTS) + ["hardfilter"]):
            line = f"{strat:>12} |"
            for b in BUDGETS:
                regs = []
                for seed in range(SEEDS):
                    best = run(rows, b, strat, seed)
                    if best is not None:
                        regs.append(best / oracle)
                mean_r = sum(regs) / len(regs)
                within = sum(1 for r in regs if r <= 1.05) / len(regs)
                dev_out[f"{strat}@{b}"] = {
                    "mean_regret": mean_r, "frac_within_5pct": within}
                line += f" {mean_r:6.3f} / {within:4.0%} |"
            print(line)
        out[device] = dev_out

    with open("results/ablation_replay.json", "w", encoding="utf-8") as fh:
        json.dump({"seeds": SEEDS, "budgets": BUDGETS,
                   "devices": out}, fh, indent=1)
    print("\nwrote results/ablation_replay.json")


if __name__ == "__main__":
    main()
