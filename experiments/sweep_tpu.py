"""Device-aware Pallas matmul sweep on a TPU.

The TPU counterpart of sweep_gpu.py: enumerate every block configuration
the documented tiling rules allow, measure all of them, and record
failures (VMEM OOM among them) as labelled data points. The resulting
dense landscape serves as the exhaustive oracle for budgeted-search
evaluation, and its failure labels train the learned legality boundary.

Bundle before sending (colab run uploads a single file):

    python tools/bundle.py experiments/sweep_tpu.py -o /tmp/sweep_tpu.py
    colab run --tpu v5e1 --timeout 2400 /tmp/sweep_tpu.py
"""

import functools
import json
import os
import subprocess
import sys
import time

# Colab images ship a libtpu older than the preinstalled jax; align before
# jax is first imported or every pallas_call fails at Mosaic deserialization.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]"],
    check=True,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from andamento import legality  # noqa: E402

M = N = K = 4096
DTYPE = jnp.bfloat16
FLOPS = 2 * M * N * K
TIMING_ITERS = 20

# Powers of two from the smallest MXU-aligned tile worth timing up to half
# the matrix. All satisfy the documented rules (bm % 8, bk/bn % 128); the
# interesting boundary — where Mosaic's scoped-VMEM budget gives out — is
# exactly what we want labelled, so no capacity pre-filtering.
CANDS = [128, 256, 512, 1024, 2048]


def matmul_kernel(x_ref, y_ref, o_ref, acc_ref, *, nk):
    @pl.when(pl.program_id(2) == 0)
    def _init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(
        x_ref[...], y_ref[...], preferred_element_type=jnp.float32
    )

    @pl.when(pl.program_id(2) == nk - 1)
    def _store():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def make_matmul(bm, bk, bn):
    grid = (M // bm, N // bn, K // bk)

    @jax.jit
    def matmul(x, y):
        return pl.pallas_call(
            functools.partial(matmul_kernel, nk=grid[2]),
            grid=grid,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            out_shape=jax.ShapeDtypeStruct((M, N), DTYPE),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        )(x, y)

    return matmul


def bench(fn, *args):
    t0 = time.perf_counter()
    fn(*args).block_until_ready()
    compile_s = time.perf_counter() - t0
    times = []
    for _ in range(TIMING_ITERS):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        times.append(time.perf_counter() - t0)
    return compile_s, float(np.median(times)), float(
        np.percentile(times, 75) - np.percentile(times, 25))


def classify(err):
    text = err.lower()
    if "vmemoom" in text or "vmem" in text:
        return "vmem_oom"
    if "resource_exhausted" in text or "out of memory" in text:
        return "out_of_memory"
    if "unsupported" in text or "failed_precondition" in text:
        return "unsupported"
    return "other"


def main():
    dev = jax.devices()[0]
    kind = dev.device_kind
    print(f"jax {jax.__version__} | {kind} x{jax.device_count()}")

    configs, rejected = [], []
    for bm in CANDS:
        for bk in CANDS:
            for bn in CANDS:
                reasons = legality.tpu_config_reasons(bm, bk, bn, "bfloat16", kind)
                (rejected if reasons else configs).append(
                    ((bm, bk, bn), reasons) if reasons else (bm, bk, bn))
    print(f"{len(configs)} candidates, {len(rejected)} pruned by tiling rules")

    key = jax.random.key(0)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (M, K), dtype=DTYPE)
    y = jax.random.normal(ky, (K, N), dtype=DTYPE)

    xla_dot = jax.jit(
        lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32).astype(DTYPE))
    _, xla_s, _ = bench(xla_dot, x, y)
    ref = np.asarray(xla_dot(x, y), dtype=np.float32)
    print(f"XLA baseline: {xla_s * 1e3:.3f} ms ({FLOPS / xla_s / 1e12:.1f} TFLOP/s)")

    results = []
    for bm, bk, bn in configs:
        feats = legality.tpu_config_features(bm, bk, bn, "bfloat16", kind)
        row = {"config": [bm, bk, bn], "features": feats}
        tag = f"({bm},{bk},{bn})"
        try:
            fn = make_matmul(bm, bk, bn)
            compile_s, med_s, iqr_s = bench(fn, x, y)
            out = np.asarray(fn(x, y), dtype=np.float32)
            row.update(status="ok", median_ms=med_s * 1e3, iqr_ms=iqr_s * 1e3,
                       compile_s=compile_s, tflops=FLOPS / med_s / 1e12,
                       vs_xla=xla_s / med_s,
                       max_abs_err_vs_xla=float(np.max(np.abs(out - ref))))
            print(f"{tag:>18}: {med_s * 1e3:8.3f} ms | {row['tflops']:6.1f} TF/s | "
                  f"{row['vs_xla']:.2f}x XLA")
        except Exception as e:  # noqa: BLE001 - failures are labels
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            row.update(status="fail", failure_class=classify(msg), error=msg)
            print(f"{tag:>18}: FAILED [{row['failure_class']}]")
        results.append(row)

    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        best = min(ok, key=lambda r: r["median_ms"])
        worst = max(ok, key=lambda r: r["median_ms"])
        print(f"\n{len(ok)}/{len(results)} ran | "
              f"best {best['config']} = {best['median_ms']:.3f} ms "
              f"({best['vs_xla']:.2f}x XLA) | spread "
              f"{worst['median_ms'] / best['median_ms']:.1f}x")

    print("\n===RESULTS_JSON===")
    print(json.dumps({
        "device": kind, "jax": jax.__version__, "shape": [M, K, N],
        "dtype": "bf16", "xla_ms": xla_s * 1e3,
        "sweeps": [{"dtype": "bf16", "xla_ms": xla_s * 1e3, "status": "ok",
                    "results": results,
                    "pruned": [{"config": list(c), "reasons": r}
                               for c, r in rejected]}],
    }, default=str))


if __name__ == "__main__":
    main()
