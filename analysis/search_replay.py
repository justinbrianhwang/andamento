"""Replay budgeted-search strategies over an exhaustively measured landscape.

Given a sweep JSON whose configuration space was measured to completion,
simulate what a search strategy would have found after B real measurements.
Because the landscape is fully known, the exhaustive optimum is the oracle
and regret is exact — and one TPU/GPU session's worth of data supports
thousands of simulated searches at zero hardware cost.

Failures matter: selecting a config that refuses to launch consumes budget
and returns no latency, exactly as it would on hardware. A strategy that
learns to avoid the failure region converges faster; that difference is the
point of measuring it.

Strategies:
  random    — uniform without replacement.
  tpe       — TPE-style categorical Parzen: split observations into good
              (top-gamma successes) and bad (rest + failures), score every
              unmeasured config by per-axis likelihood ratio, measure the
              argmax. Pure stdlib.

Usage:
  python analysis/search_replay.py results/sweep_a100_bf16.json
"""

import json
import math
import random
import sys
from collections import Counter

BUDGETS = [5, 10, 20, 40, 60]
SEEDS = 300
GAMMA = 0.3          # fraction of successes considered "good"
INIT_RANDOM = 5      # TPE bootstraps with this many random draws
LAPLACE = 1.0


def load(path):
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    sweep = blob["sweeps"][0] if "sweeps" in blob else blob
    rows = []
    for r in sweep["results"]:
        axes = tuple(r["config"]) + (r.get("num_warps"), r.get("num_stages"))
        axes = tuple(a for a in axes if a is not None)
        rows.append({"axes": axes,
                     "ms": r["median_ms"] if r["status"] == "ok" else None})
    return blob.get("device", "?"), rows


def tpe_pick(observed, remaining):
    """Score unmeasured configs by per-axis good/bad likelihood ratio."""
    succ = sorted((r for r in observed if r["ms"] is not None),
                  key=lambda r: r["ms"])
    n_good = max(1, int(len(succ) * GAMMA))
    good = succ[:n_good]
    bad = succ[n_good:] + [r for r in observed if r["ms"] is None]
    if not good or not bad:
        return random.choice(remaining)

    n_axes = len(remaining[0]["axes"])
    gc = [Counter(r["axes"][i] for r in good) for i in range(n_axes)]
    bc = [Counter(r["axes"][i] for r in bad) for i in range(n_axes)]

    def score(cfg):
        s = 0.0
        for i, v in enumerate(cfg["axes"]):
            pg = (gc[i][v] + LAPLACE) / (len(good) + LAPLACE * len(gc[i]))
            pb = (bc[i][v] + LAPLACE) / (len(bad) + LAPLACE * len(bc[i]))
            s += math.log(pg / pb)
        return s

    best = max(remaining, key=score)
    return best


def run_strategy(rows, budget, strategy, rng_seed):
    random.seed(rng_seed)
    remaining = list(rows)
    random.shuffle(remaining)
    observed = []
    for step in range(min(budget, len(rows))):
        if strategy == "random" or len(observed) < INIT_RANDOM:
            pick = remaining.pop()
        else:
            pick = tpe_pick(observed, remaining)
            remaining.remove(pick)
        observed.append(pick)
    found = [r["ms"] for r in observed if r["ms"] is not None]
    wasted = sum(1 for r in observed if r["ms"] is None)
    return (min(found) if found else None), wasted


def main():
    path = sys.argv[1]
    device, rows = load(path)
    oracle = min(r["ms"] for r in rows if r["ms"] is not None)
    n_fail = sum(1 for r in rows if r["ms"] is None)
    print(f"{device}: {len(rows)} configs ({n_fail} failures), "
          f"oracle {oracle:.3f} ms")
    print(f"\n{'budget':>7} | {'strategy':>8} | {'mean regret':>11} | "
          f"{'p90 regret':>10} | {'within 5%':>9} | {'wasted on fail':>14}")

    summary = {}
    for budget in BUDGETS:
        for strategy in ("random", "tpe"):
            regrets, wastes, misses = [], [], 0
            for seed in range(SEEDS):
                best, wasted = run_strategy(rows, budget, strategy, seed)
                if best is None:
                    misses += 1
                    continue
                regrets.append(best / oracle)
                wastes.append(wasted)
            regrets.sort()
            mean_r = sum(regrets) / len(regrets)
            p90 = regrets[int(0.9 * len(regrets))]
            within = sum(1 for r in regrets if r <= 1.05) / len(regrets)
            mean_w = sum(wastes) / len(wastes)
            summary[f"{strategy}@{budget}"] = {
                "mean_regret": mean_r, "p90_regret": p90,
                "frac_within_5pct": within, "mean_wasted": mean_w,
                "all_failed_runs": misses,
            }
            print(f"{budget:>7} | {strategy:>8} | {mean_r:>10.3f}x | "
                  f"{p90:>9.3f}x | {within:>8.0%} | {mean_w:>13.1f}")

    out = path.replace(".json", "_replay.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"device": device, "oracle_ms": oracle,
                   "n_configs": len(rows), "n_failures": n_fail,
                   "seeds": SEEDS, "gamma": GAMMA, "summary": summary}, f,
                  indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
