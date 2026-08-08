"""Time-weighted regret: budget in wall-clock seconds, not probe counts.

Counting probes hides the real cost structure of a sweep: compile time
dominates (3-228 s per config in our data) and, on cliff devices, the
slow configs are also the expensive-to-compile ones (RTX 3090's 82 ms
cliffs take 115 s to compile vs 6-15 s for the fast region). A strategy
that avoids the cliff region therefore saves far more wall-clock time
than probe counting suggests.

Cost model per probe, from the recorded measurements:
  cost = compile_s + REPS * median_ms / 1000        (successful configs)
  cost = median compile_s of the device's successes  (failed configs;
         their compile completed before the launch refusal, but the
         sweep JSON only records compile_s for successes)

Reports best-found-so-far regret at fixed wall-clock budgets for the
random / tpe / surrogate strategies, replayed over the exhaustive
landscapes. Pure stdlib.

Usage:
  python analysis/time_regret.py results/sweep_*.json
"""

import json
import random
import sys

from search_replay import tpe_pick, INIT_RANDOM
from surrogate import load, standardize, surrogate_pick

TIME_BUDGETS = [120, 300, 600, 1200, 2400]   # seconds
SEEDS = 200
REPS = 20                                    # measurement repetitions


def load_costed(path):
    device, rows = load(path)
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    sweep = blob["sweeps"][0] if "sweeps" in blob else blob
    compiles = [r.get("compile_s") for r in sweep["results"]
                if r["status"] == "ok" and r.get("compile_s")]
    compiles.sort()
    fallback = compiles[len(compiles) // 2] if compiles else 30.0
    for row, r in zip(rows, sweep["results"]):
        if r["status"] == "ok":
            row["cost"] = (r.get("compile_s") or fallback) \
                + REPS * r["median_ms"] / 1000.0
        else:
            row["cost"] = fallback
    return device, rows


def run_strategy(rows, strategy, rng_seed, horizon):
    """Returns [(elapsed_seconds, best_ms_so_far), ...] per probe."""
    random.seed(rng_seed)
    remaining = list(rows)
    random.shuffle(remaining)
    observed, trace = [], []
    elapsed, best = 0.0, None
    while remaining and elapsed < horizon:
        if strategy == "random" or len(observed) < INIT_RANDOM:
            pick = remaining.pop()
        elif strategy == "tpe":
            pick = tpe_pick(observed, remaining)
            remaining.remove(pick)
        else:
            pick = surrogate_pick(observed, remaining)
            remaining.remove(pick)
        observed.append(pick)
        elapsed += pick["cost"]
        if pick["ms"] is not None and (best is None or pick["ms"] < best):
            best = pick["ms"]
        trace.append((elapsed, best))
    return trace


def best_at(trace, t):
    best = None
    for elapsed, b in trace:
        if elapsed > t:
            break
        best = b
    return best


def main():
    paths = sys.argv[1:]
    all_out = {}
    for path in paths:
        device, rows = load_costed(path)
        standardize(rows)
        oracle = min(r["ms"] for r in rows if r["ms"] is not None)
        total_cost = sum(r["cost"] for r in rows)
        print(f"\n{device}: oracle {oracle:.3f} ms, exhaustive sweep "
              f"costs {total_cost/60:.0f} min")
        print(f"{'seconds':>8} | {'strategy':>9} | {'mean regret':>11} | "
              f"{'within 5%':>9} | {'no result':>9}")

        traces = {s: [run_strategy(rows, s, seed, max(TIME_BUDGETS))
                      for seed in range(SEEDS)]
                  for s in ("random", "tpe", "surrogate")}
        summary = {}
        for t in TIME_BUDGETS:
            for strategy, runs in traces.items():
                regrets, missing = [], 0
                for trace in runs:
                    b = best_at(trace, t)
                    if b is None:
                        missing += 1
                    else:
                        regrets.append(b / oracle)
                if not regrets:
                    print(f"{t:>8} | {strategy:>9} | {'-':>11} | "
                          f"{'-':>9} | {missing:>8.0%}")
                    continue
                mean_r = sum(regrets) / len(regrets)
                within = sum(1 for r in regrets if r <= 1.05) / len(regrets)
                summary[f"{strategy}@{t}s"] = {
                    "mean_regret": mean_r, "frac_within_5pct": within,
                    "no_result_runs": missing,
                }
                print(f"{t:>8} | {strategy:>9} | {mean_r:>10.3f}x | "
                      f"{within:>8.0%} | {missing/SEEDS:>8.0%}")
        all_out[device] = {"oracle_ms": oracle,
                           "exhaustive_cost_s": total_cost,
                           "summary": summary}

    with open("results/time_regret.json", "w", encoding="utf-8") as fh:
        json.dump({"seeds": SEEDS, "reps": REPS,
                   "budgets_s": TIME_BUDGETS, "devices": all_out}, fh,
                  indent=1)
    print("\nwrote results/time_regret.json")


if __name__ == "__main__":
    main()
