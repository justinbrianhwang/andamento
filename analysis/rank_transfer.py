"""Do good Pallas configurations stay good across TPU generations?

Reads the pilot result files and reports, per device pair, how well the
configuration ranking is preserved (Spearman rho, Kendall tau) and what a
practitioner would actually lose by tuning on one device and deploying on
the other.
"""

import glob
import json
import os
from itertools import combinations

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load():
    """device -> {config_tuple: median_ms} for configs that ran successfully."""
    devices = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        # GPU files nest per-dtype sweeps; TPU files are flat.
        sweeps = blob.get("sweeps") or [{"dtype": blob.get("dtype", "bf16"), "results": blob["results"]}]
        for sweep in sweeps:
            name = blob["device"]
            if blob.get("sweeps"):
                name = f"{name} [{sweep['dtype']}]"
            ok = {tuple(r["config"]): r["median_ms"]
                  for r in sweep.get("results", []) if r["status"] == "ok"}
            if ok:
                devices[name] = ok
    return devices


def spearman(xs, ys):
    n = len(xs)
    rx, ry = rank(xs), rank(ys)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - 6 * d2 / (n * (n * n - 1))


def kendall(xs, ys):
    con = dis = 0
    for i, j in combinations(range(len(xs)), 2):
        s = (xs[i] - xs[j]) * (ys[i] - ys[j])
        con += s > 0
        dis += s < 0
    return (con - dis) / (con + dis) if con + dis else float("nan")


def rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0] * len(vals)
    for pos, i in enumerate(order):
        r[i] = pos + 1
    return r


def main():
    devices = load()
    print("Devices with successful measurements:")
    for name, cfgs in devices.items():
        best = min(cfgs, key=cfgs.get)
        print(f"  {name:28s} {len(cfgs)} configs | best {best} @ {cfgs[best]:.3f} ms")

    names = list(devices)
    for a, b in combinations(names, 2):
        shared = sorted(set(devices[a]) & set(devices[b]))
        if len(shared) < 3:
            print(f"\n{a} vs {b}: only {len(shared)} shared configs, skipping")
            continue
        xs = [devices[a][c] for c in shared]
        ys = [devices[b][c] for c in shared]

        best_a = min(shared, key=lambda c: devices[a][c])
        best_b = min(shared, key=lambda c: devices[b][c])
        # What you lose by tuning on A and deploying on B.
        transfer_loss = devices[b][best_a] / devices[b][best_b]

        print(f"\n{a} vs {b}  ({len(shared)} shared configs)")
        print(f"  Spearman rho : {spearman(xs, ys):+.3f}")
        print(f"  Kendall tau  : {kendall(xs, ys):+.3f}")
        print(f"  best on {a}: {best_a}")
        print(f"  best on {b}: {best_b}")
        print(f"  tuning on {a} and deploying on {b} costs {(transfer_loss - 1) * 100:.1f}% "
              f"vs tuning natively")

        spread_a = max(xs) / min(xs)
        spread_b = max(ys) / min(ys)
        print(f"  spread: {spread_a:.1f}x on {a}, {spread_b:.1f}x on {b}")


if __name__ == "__main__":
    main()
