"""Round-2 quantification: transfer safety, capacity confusion, imputation.

1. Cross-device transfer metrics over the six GPU landscapes:
     J_fail(s,t)      Jaccard similarity of failure sets
     survival_5(s->t) fraction of source top-5 feasible on target
     top5_regret      best target latency among source top-5 / oracle
     spearman(s,t)    rank correlation over common successes
   plus the warm/cold regret matrix (recomputed, 300 seeds), and the
   correlation of each metric with the warm-start gain.

2. Capacity-model confusion per device: predicted-feasible
   (estimate ratio <= 1.0) vs measured outcome, and whether a hard
   filter would prune the device's true optimum.

3. Wall-clock imputation sensitivity: rerun the time-budget replay on
   the 4090 and 3090 charging refusals the p25 / p50 / p75 of the
   device's measured compile times, to check the strategy ordering is
   not an artifact of the imputed value.

Usage (from repo root):  python analysis/transfer_metrics.py
"""

import json
import math
import random

from search_replay import load as load_axes, tpe_pick, INIT_RANDOM
from warm_start import run as warm_run
from surrogate import load as load_feat, standardize, surrogate_pick

GPU6 = [
    ("results/sweep_a100_bf16.json", "A100"),
    ("results/sweep_result_NVIDIA_GeForce_RTX_3090.json", "3090"),
    ("results/sweep_l4_bf16.json", "L4"),
    ("results/sweep_result_NVIDIA_GeForce_RTX_4090.json", "4090"),
    ("results/sweep_g4_bf16.json", "RTX PRO"),
    ("results/sweep_result_NVIDIA_GeForce_RTX_5090.json", "5090"),
]
ALL11 = GPU6 + [
    ("results/sweep_result_h100_nvl.json", "H100"),
    ("results/sweep_result_b200.json", "B200"),
    ("results/sweep_result_NVIDIA_GeForce_RTX_4060_Laptop_GPU.json",
     "4060L"),
    ("results/sweep_v5e_bf16.json", "v5e"),
    ("results/sweep_v6e_bf16.json", "v6e"),
]
SEEDS = 300
BUDGET = 20
TOPK = 5


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else 0.0


def pearson(xs, ys):
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else 0.0


def transfer_metrics():
    lands = []
    for path, label in GPU6:
        _, rows = load_axes(path)
        lands.append((label, rows,
                      {r["axes"]: r["ms"] for r in rows}))

    out = {"pairs": []}
    gains, survs, rhos, jacs, t5regs = [], [], [], [], []
    for ti, (tgt, trows, tmap) in enumerate(lands):
        oracle = min(m for m in tmap.values() if m is not None)

        def mean_regret(order):
            rs = []
            for s in range(SEEDS):
                best = warm_run(trows, BUDGET, s, order)
                if best is not None:
                    rs.append(best / oracle)
            return sum(rs) / len(rs)

        cold = mean_regret(None)
        for si, (src, srows, smap) in enumerate(lands):
            if si == ti:
                continue
            ranked = sorted((r for r in srows if r["ms"] is not None),
                            key=lambda r: r["ms"])
            top = [r["axes"] for r in ranked[:TOPK]]
            warm = mean_regret([r["axes"] for r in ranked])

            f_s = {a for a, m in smap.items() if m is None}
            f_t = {a for a, m in tmap.items() if m is None}
            union = f_s | f_t
            jac = (len(f_s & f_t) / len(union)) if union else 1.0
            surv = sum(1 for a in top if tmap.get(a) is not None) / TOPK
            alive = [tmap[a] for a in top if tmap.get(a) is not None]
            t5 = (min(alive) / oracle) if alive else None
            common = [a for a in smap if smap[a] is not None
                      and tmap.get(a) is not None]
            rho = spearman([smap[a] for a in common],
                           [tmap[a] for a in common])
            gain = cold - warm
            out["pairs"].append({
                "source": src, "target": tgt, "warm": warm, "cold": cold,
                "gain": gain, "jaccard_fail": jac, "survival_5": surv,
                "top5_target_regret": t5, "spearman_common": rho,
            })
            gains.append(gain)
            survs.append(surv)
            rhos.append(rho)
            jacs.append(jac)
            t5regs.append(t5 if t5 is not None else 10.0)

    out["corr_with_gain"] = {
        "survival_5": pearson(survs, gains),
        "spearman_common": pearson(rhos, gains),
        "jaccard_fail": pearson(jacs, gains),
        "neg_top5_regret": pearson([-x for x in t5regs], gains),
    }
    print("correlation of transfer gain with:", out["corr_with_gain"])
    return out


def confusion():
    out = {}
    print(f"\n{'device':>8} | pred-ok&ok | pred-ok&fail | pred-bad&ran |"
          f" pred-bad&fail | oracle pruned?")
    for path, label in ALL11:
        _, rows = load_feat(path)
        oracle = min(r["ms"] for r in rows if r["ms"] is not None)
        a = b = c = d = 0
        opt_pruned = False
        for r in rows:
            f = r.get("features") or {}
            ratio = f.get("smem_ratio") or f.get("vmem_ratio") or 0.0
            pred_ok = ratio <= 1.0
            ran = r["ms"] is not None
            if pred_ok and ran:
                a += 1
            elif pred_ok and not ran:
                b += 1
            elif not pred_ok and ran:
                c += 1
            else:
                d += 1
            if ran and r["ms"] == oracle and not pred_ok:
                opt_pruned = True
        out[label] = {"pred_ok_ran": a, "pred_ok_fail": b,
                      "pred_bad_ran": c, "pred_bad_fail": d,
                      "oracle_pruned": opt_pruned}
        print(f"{label:>8} | {a:10d} | {b:12d} | {c:12d} | {d:13d} |"
              f" {'YES' if opt_pruned else 'no':>5}")
    return out


def imputation_sensitivity():
    REPS = 20
    T_BUDGETS = [300, 600, 1200]
    SEEDS_T = 150
    out = {}
    for path, label in [
        ("results/sweep_result_NVIDIA_GeForce_RTX_4090.json", "4090"),
        ("results/sweep_result_NVIDIA_GeForce_RTX_3090.json", "3090"),
    ]:
        _, rows = load_feat(path)
        standardize(rows)
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        res = blob["sweeps"][0]["results"]
        compiles = sorted(r["compile_s"] for r in res
                          if r["status"] == "ok")
        quart = {"p25": compiles[len(compiles) // 4],
                 "p50": compiles[len(compiles) // 2],
                 "p75": compiles[3 * len(compiles) // 4]}
        oracle = min(r["ms"] for r in rows if r["ms"] is not None)
        out[label] = {}
        print(f"\n{label}: refusal compile-time imputation "
              f"{ {k: round(v,1) for k, v in quart.items()} }")
        for qname, qval in quart.items():
            for row, r in zip(rows, res):
                row["cost"] = ((r.get("compile_s") or qval)
                               + REPS * r["median_ms"] / 1000.0
                               if r["status"] == "ok" else qval)
            line = f"  {qname}:"
            out[label][qname] = {}
            for strat in ("random", "tpe", "surrogate"):
                fr = {}
                for t in T_BUDGETS:
                    hits = tot = 0
                    for seed in range(SEEDS_T):
                        random.seed(seed)
                        pool = list(rows)
                        random.shuffle(pool)
                        obs, el, best = [], 0.0, None
                        while pool and el < t:
                            if strat == "random" or len(obs) < INIT_RANDOM:
                                pick = pool.pop()
                            elif strat == "tpe":
                                pick = tpe_pick(obs, pool)
                                pool.remove(pick)
                            else:
                                pick = surrogate_pick(obs, pool)
                                pool.remove(pick)
                            obs.append(pick)
                            el += pick["cost"]
                            if el > t:
                                break
                            if pick["ms"] is not None and \
                                    (best is None or pick["ms"] < best):
                                best = pick["ms"]
                        if best is not None:
                            tot += 1
                            if best / oracle <= 1.05:
                                hits += 1
                    fr[t] = hits / tot if tot else 0.0
                out[label][qname][strat] = fr
                line += f"  {strat}: " + "/".join(
                    f"{fr[t]:.0%}" for t in T_BUDGETS)
            print(line)
    return out


if __name__ == "__main__":
    result = {"transfer": transfer_metrics(), "confusion": confusion(),
              "imputation_sensitivity": imputation_sensitivity()}
    with open("results/round2_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print("\nwrote results/round2_metrics.json")
