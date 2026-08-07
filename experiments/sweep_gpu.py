"""Device-aware Pallas matmul sweep on a GPU.

Supersedes pilot_blocksweep_gpu.py, which used one hand-picked config list
for every device. That list was chosen for Ampere and told us little about
whether a device failed because it cannot run Pallas or because we handed it
the wrong shape. Here the space comes from andamento.legality, every
candidate is measured, and each measurement is stored with the device
properties and static features needed to learn the legality boundary rather
than assume it.

`colab run` uploads a single file into a notebook kernel, so bundle first:

    python tools/bundle.py experiments/sweep_gpu.py -o /tmp/sweep.py
    colab run --gpu A100 --timeout 1800 /tmp/sweep.py bf16

The default --timeout is 30 s, which the CLI spends waiting for the reply
after the script finishes; anything longer is reported as a failure even
though the run succeeded.
"""

import functools
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from andamento import legality  # noqa: E402

M = N = K = 4096
FLOPS = 2 * M * N * K
TIMING_ITERS = 20
DTYPES = {"bf16": jnp.bfloat16, "fp16": jnp.float16}


def _compiler_params(num_warps, num_stages):
    from jax.experimental.pallas import triton as plgpu

    for name in ("CompilerParams", "TritonCompilerParams"):
        cls = getattr(plgpu, name, None)
        if cls is not None:
            return cls(num_warps=num_warps, num_stages=num_stages)
    raise RuntimeError("no Triton compiler-params class in jax.experimental.pallas.triton")


def matmul_kernel(x_ref, y_ref, o_ref, *, bk):
    acc = jnp.zeros(o_ref.shape, jnp.float32)
    for k in range(x_ref.shape[1] // bk):
        acc += jnp.dot(
            x_ref[:, k * bk:(k + 1) * bk],
            y_ref[k * bk:(k + 1) * bk, :],
            preferred_element_type=jnp.float32,
        )
    o_ref[...] = acc.astype(o_ref.dtype)


def make_matmul(bm, bn, bk, num_warps, num_stages, dtype):
    @jax.jit
    def matmul(x, y):
        return pl.pallas_call(
            functools.partial(matmul_kernel, bk=bk),
            grid=(M // bm, N // bn),
            in_specs=[
                pl.BlockSpec((bm, K), lambda i, j: (i, 0)),
                pl.BlockSpec((K, bn), lambda i, j: (0, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
            out_shape=jax.ShapeDtypeStruct((M, N), dtype),
            compiler_params=_compiler_params(num_warps, num_stages),
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
    """Group failures so the dataset carries why, not just that, it failed."""
    text = err.lower()
    if "compute capability" in text:
        return "unsupported_device"
    if "failed to launch" in text or "resource_exhausted" in text:
        return "launch_resources"
    if "out of memory" in text or "oom" in text:
        return "out_of_memory"
    if "shared memory" in text or "smem" in text:
        return "shared_memory"
    return "other"


def sweep(dtype_name, dtype, props):
    print(f"\n=== dtype: {dtype_name} ===")
    legal, rejected = legality.gpu_configs(dtype_name, props)
    print(f"{len(legal)} candidates to measure, {len(rejected)} pruned statically")
    if rejected:
        print(f"  pruned because: {rejected[0][1][0]}")

    key = jax.random.key(0)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (M, K), dtype=dtype)
    y = jax.random.normal(ky, (K, N), dtype=dtype)

    xla_dot = jax.jit(
        lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32).astype(dtype))
    _, xla_s, _ = bench(xla_dot, x, y)
    ref = np.asarray(xla_dot(x, y), dtype=np.float32)
    print(f"XLA baseline: {xla_s * 1e3:.3f} ms ({FLOPS / xla_s / 1e12:.1f} TFLOP/s)")

    results = []
    for bm, bn, bk, nw, ns in legal:
        feats = legality.gpu_config_features(bm, bn, bk, nw, ns, dtype_name, props)
        row = {"config": [bm, bn, bk], "num_warps": nw, "num_stages": ns,
               "features": feats}
        tag = f"({bm},{bn},{bk}) w{nw} s{ns}"
        try:
            fn = make_matmul(bm, bn, bk, nw, ns, dtype)
            compile_s, med_s, iqr_s = bench(fn, x, y)
            out = np.asarray(fn(x, y), dtype=np.float32)
            row.update(status="ok", median_ms=med_s * 1e3, iqr_ms=iqr_s * 1e3,
                       compile_s=compile_s, tflops=FLOPS / med_s / 1e12,
                       vs_xla=xla_s / med_s,
                       max_abs_err_vs_xla=float(np.max(np.abs(out - ref))))
            print(f"{tag:>24}: {med_s * 1e3:8.3f} ms | {row['tflops']:6.1f} TF/s | "
                  f"{row['vs_xla']:.2f}x XLA")
        except Exception as e:  # noqa: BLE001 - failures are labels
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            row.update(status="fail", failure_class=classify(msg), error=msg)
            print(f"{tag:>24}: FAILED [{row['failure_class']}]")
        results.append(row)

    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        best = min(ok, key=lambda r: r["median_ms"])
        worst = max(ok, key=lambda r: r["median_ms"])
        print(f"{len(ok)}/{len(results)} ran | best {best['median_ms']:.3f} ms "
              f"({best['vs_xla']:.2f}x XLA) | spread "
              f"{worst['median_ms'] / best['median_ms']:.1f}x")

    return {"dtype": dtype_name, "xla_ms": xla_s * 1e3, "status": "ok",
            "results": results,
            "pruned": [{"config": list(c), "reasons": r} for c, r in rejected]}


def main():
    props = legality.gpu_properties()
    dev = jax.devices()[0]
    print(f"jax {jax.__version__} | {dev.device_kind}")
    print(f"device properties: {props}")

    wanted = sys.argv[1:] or list(DTYPES)
    sweeps = [sweep(n, DTYPES[n], props) for n in wanted if n in DTYPES]

    print("\n===RESULTS_JSON===")
    print(json.dumps({"device": dev.device_kind, "jax": jax.__version__,
                      "properties": props, "shape": [M, K, N],
                      "sweeps": sweeps}, default=str))


if __name__ == "__main__":
    main()
