"""Pilot: block-size sensitivity of a Pallas matmul on a Colab GPU.

The GPU counterpart to pilot_blocksweep.py. The kernel is structurally
different, not a port: on TPU the grid runs sequentially, so the K-axis
accumulation can live in the grid with a VMEM scratch. On GPU the grid maps
to concurrently scheduled CUDA blocks, so accumulating across a grid axis
would race. The reduction moves inside the kernel, blocks are addressed
lazily instead of being copied into SRAM, and two knobs with no TPU
equivalent appear: num_warps and num_stages.

Run via:  colab run --gpu T4 experiments/pilot_blocksweep_gpu.py
The VM is released automatically when the script exits.
"""

import functools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

M = N = K = 4096
FLOPS = 2 * M * N * K
TIMING_ITERS = 20
DTYPES = {"bf16": jnp.bfloat16, "fp16": jnp.float16}

# (bm, bn, bk, num_warps, num_stages)
CONFIGS = [
    (32, 32, 32, 4, 2),      # far too small: launch/loop overhead dominates
    (64, 64, 32, 4, 2),
    (128, 128, 32, 4, 3),
    (128, 128, 64, 8, 3),
    (128, 256, 64, 8, 3),
    (256, 128, 64, 8, 3),
    (128, 128, 64, 4, 2),    # same tile as above, fewer warps/stages
    (256, 256, 64, 8, 4),    # deliberately large: probes the shared-memory ceiling
]


def _compiler_params(num_warps, num_stages):
    """Triton backend params; the class was renamed across JAX versions."""
    from jax.experimental.pallas import triton as plgpu

    for name in ("CompilerParams", "TritonCompilerParams"):
        cls = getattr(plgpu, name, None)
        if cls is not None:
            return cls(num_warps=num_warps, num_stages=num_stages)
    raise RuntimeError("no Triton compiler-params class found in jax.experimental.pallas.triton")


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
    fn(*args).block_until_ready()  # compile + first run
    compile_s = time.perf_counter() - t0
    times = []
    for _ in range(TIMING_ITERS):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        times.append(time.perf_counter() - t0)
    return compile_s, float(np.median(times)), float(
        np.percentile(times, 75) - np.percentile(times, 25)
    )


def sweep(dtype_name, dtype):
    print(f"\n=== dtype: {dtype_name} ===")
    key = jax.random.key(0)
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (M, K), dtype=dtype)
    y = jax.random.normal(ky, (K, N), dtype=dtype)

    xla_dot = jax.jit(
        lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32).astype(dtype)
    )
    try:
        _, xla_s, _ = bench(xla_dot, x, y)
        ref = np.asarray(xla_dot(x, y), dtype=np.float32)
        print(f"XLA baseline: {xla_s * 1e3:.3f} ms  ({FLOPS / xla_s / 1e12:.1f} TFLOP/s)")
    except Exception as e:  # noqa: BLE001
        print(f"XLA baseline FAILED — {type(e).__name__}: {str(e)[:200]}")
        return {"dtype": dtype_name, "status": "baseline_failed",
                "error": f"{type(e).__name__}: {str(e)[:300]}"}

    results = []
    for bm, bn, bk, nw, ns in CONFIGS:
        tag = f"({bm},{bn},{bk}) w{nw} s{ns}"
        try:
            fn = make_matmul(bm, bn, bk, nw, ns, dtype)
            compile_s, med_s, iqr_s = bench(fn, x, y)
            out = np.asarray(fn(x, y), dtype=np.float32)
            max_abs = float(np.max(np.abs(out - ref)))
            row = {
                "config": [bm, bn, bk], "num_warps": nw, "num_stages": ns,
                "status": "ok",
                "median_ms": med_s * 1e3, "iqr_ms": iqr_s * 1e3,
                "compile_s": compile_s,
                "tflops": FLOPS / med_s / 1e12,
                "vs_xla": xla_s / med_s,
                "max_abs_err_vs_xla": max_abs,
            }
            print(f"{tag:>24}: {med_s * 1e3:8.3f} ms | {row['tflops']:6.1f} TFLOP/s | "
                  f"{row['vs_xla']:.2f}x XLA | max|err| {max_abs:.3g}")
        except Exception as e:  # noqa: BLE001 - failures are data
            row = {"config": [bm, bn, bk], "num_warps": nw, "num_stages": ns,
                   "status": "fail", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"{tag:>24}: FAILED — {type(e).__name__}: {str(e)[:120]}")
        results.append(row)

    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        best = min(ok, key=lambda r: r["median_ms"])
        worst = max(ok, key=lambda r: r["median_ms"])
        print(f"spread: worst {worst['median_ms']:.3f} ms -> best {best['median_ms']:.3f} ms "
              f"= {worst['median_ms'] / best['median_ms']:.2f}x from config choice alone")

    return {"dtype": dtype_name, "status": "ok",
            "xla_ms": xla_s * 1e3, "results": results}


def main():
    dev = jax.devices()[0]
    print(f"jax {jax.__version__} | device: {dev.device_kind} x{jax.device_count()}")
    try:
        import triton
        print(f"triton {triton.__version__}")
    except Exception:  # noqa: BLE001
        print("triton: not importable")

    sweeps = [sweep(name, dt) for name, dt in DTYPES.items()]

    print("\n===RESULTS_JSON===")
    print(json.dumps({
        "device": dev.device_kind, "jax": jax.__version__,
        "shape": [M, K, N], "sweeps": sweeps,
    }))


if __name__ == "__main__":
    main()
