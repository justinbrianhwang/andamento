"""Cross-device warm start: does one device's ranking buy search on another?

For every ordered pair of fully-measured landscapes sharing a config
space, spend the first probes of the budget on the source device's
top-ranked configs, then continue with TPE. Compared against cold random
and cold TPE at the same budget, over many seeds.

The interesting outcome is not only the mean regret but where transfer
HURTS: a source whose optimum sits on the target's cliff would poison
the budget, and we want to see whether that happens between real devices.

Usage:
  python analysis/warm_start.py results/sweep_l4_bf16.json results/sweep_g4_bf16.json ...
"""

import json
import random
import sys

from search_replay import load, tpe_pick, INIT_RANDOM

BUDGET = 20
SEEDS = 300
WARM_K = 5  # probes seeded from the source ranking


def run(rows, budget, seed, warm_order=None):
    random.seed(seed)
    remaining = list(rows)
    random.shuffle(remaining)
    observed = []

    if warm_order:
        for cfg_axes in warm_order:
            if len(observed) >= WARM_K:
                break
            pick = next((r for r in remaining if r["axes"] == cfg_axes), None)
            if pick:
                remaining.remove(pick)
                observed.append(pick)

    while len(observed) < min(budget, len(rows)):
        if warm_order is None and len(observed) < INIT_RANDOM:
            pick = remaining.pop()
        elif warm_order is None or len(observed) >= WARM_K:
            pick = tpe_pick(observed, remaining) if len(observed) >= INIT_RANDOM \
                else remaining.pop()
            if pick in remaining:
                remaining.remove(pick)
        else:
            pick = remaining.pop()
        observed.append(pick)

    found = [r["ms"] for r in observed if r["ms"] is not None]
    return min(found) if found else None


def main():
    paths = sys.argv[1:]
    lands = {}
    for p in paths:
        device, rows = load(p)
        lands[device] = rows

    print(f"budget {BUDGET}, warm-start with source top-{WARM_K}, {SEEDS} seeds\n")
    print(f"{'target':<28} {'cold rand':>9} {'cold tpe':>9} | warm from ...")

    for tgt, trows in lands.items():
        oracle = min(r["ms"] for r in trows if r["ms"] is not None)

        def mean_regret(warm_order=None):
            rs = []
            for s in range(SEEDS):
                best = run(trows, BUDGET, s, warm_order)
                if best is not None:
                    rs.append(best / oracle)
            return sum(rs) / len(rs)

        cold_r = mean_regret()
        # cold TPE == run with warm_order=None (same path); distinguish by
        # replaying pure-random for the baseline instead
        random.seed(0)

        def mean_regret_random():
            rs = []
            for s in range(SEEDS):
                random.seed(s)
                remaining = list(trows)
                random.shuffle(remaining)
                obs = remaining[:BUDGET]
                found = [r["ms"] for r in obs if r["ms"] is not None]
                if found:
                    rs.append(min(found) / oracle)
            return sum(rs) / len(rs)

        cold_random = mean_regret_random()
        line = f"{tgt:<28} {cold_random:>8.3f}x {cold_r:>8.3f}x |"
        for src, srows in lands.items():
            if src == tgt:
                continue
            ranked = sorted((r for r in srows if r["ms"] is not None),
                            key=lambda r: r["ms"])
            order = [r["axes"] for r in ranked]
            wr = mean_regret(order)
            line += f" {src.split()[1] if len(src.split())>1 else src}:{wr:.3f}x"
        print(line)


if __name__ == "__main__":
    main()
