"""Pilot: block-size sensitivity of a Pallas bf16 matmul on a Colab TPU.

Reproduces the effect documented in the official Pallas matmul guide:
utilization swings by an order of magnitude on block choice alone.

Run via:  colab run --tpu v5e1 experiments/pilot_blocksweep.py
The VM is released automatically when the script exits.
"""

import functools
import json
import subprocess
import sys
import time

# Colab images can ship a libtpu older than the preinstalled jax, which makes
# every pallas_call fail with "Unsupported version" at Mosaic deserialization.
# Align libtpu with jax before jax is first imported in this process.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]"],
    check=True,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

M = N = K = 4096
DTYPE = jnp.bfloat16
FLOPS = 2 * M * N * K
TIMING_ITERS = 20

CONFIGS = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (512, 1024, 1024),
    (1024, 1024, 512),
    (2048, 2048, 512),  # deliberately large: probes the VMEM ceiling
]


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
    fn(*args).block_until_ready()  # compile + first run
    compile_s = time.perf_counter() - t0
    times = []
    for _ in range(TIMING_ITERS):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        times.append(time.perf_counter() - t0)
    return compile_s, float(np.median(times)), float(np.percentile(times, 75) - np.percentile(times, 25))


def main():
    dev = jax.devices()[0]
    print(f"jax {jax.__version__} | device: {dev.device_kind} x{jax.device_count()}")

    key = jax.random.key(0)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (M, K), dtype=DTYPE)
    y = jax.random.normal(ky, (K, N), dtype=DTYPE)

    xla_dot = jax.jit(lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32).astype(DTYPE))
    _, xla_ms, _ = bench(xla_dot, x, y)
    ref = np.asarray(xla_dot(x, y), dtype=np.float32)
    print(f"XLA baseline: {xla_ms * 1e3:.3f} ms  ({FLOPS / xla_ms / 1e12:.1f} TFLOP/s)")

    results = []
    for bm, bk, bn in CONFIGS:
        tag = f"({bm},{bk},{bn})"
        try:
            fn = make_matmul(bm, bk, bn)
            compile_s, med_s, iqr_s = bench(fn, x, y)
            out = np.asarray(fn(x, y), dtype=np.float32)
            max_abs = float(np.max(np.abs(out - ref)))
            row = {
                "config": [bm, bk, bn],
                "status": "ok",
                "median_ms": med_s * 1e3,
                "iqr_ms": iqr_s * 1e3,
                "compile_s": compile_s,
                "tflops": FLOPS / med_s / 1e12,
                "vs_xla": xla_ms / med_s,
                "max_abs_err_vs_xla": max_abs,
            }
            print(f"{tag:>18}: {med_s * 1e3:8.3f} ms | {row['tflops']:6.1f} TFLOP/s | "
                  f"{row['vs_xla']:.2f}x XLA | max|err| {max_abs:.3g}")
        except Exception as e:  # noqa: BLE001 - record compile/runtime failures as data
            row = {"config": [bm, bk, bn], "status": "fail",
                   "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"{tag:>18}: FAILED — {row['error']}")
        results.append(row)

    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        best, worst = min(ok, key=lambda r: r["median_ms"]), max(ok, key=lambda r: r["median_ms"])
        print(f"\nspread: worst {worst['median_ms']:.3f} ms -> best {best['median_ms']:.3f} ms "
              f"= {worst['median_ms'] / best['median_ms']:.2f}x from block choice alone")

    print("\n===RESULTS_JSON===")
    print(json.dumps({
        "device": dev.device_kind, "jax": jax.__version__,
        "shape": [M, K, N], "dtype": "bf16",
        "xla_ms": xla_ms * 1e3, "results": results,
    }))


if __name__ == "__main__":
    main()
