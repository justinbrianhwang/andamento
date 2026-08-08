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

# --- inlined from andamento/legality.py by tools/bundle.py ---
import types as _types
legality = _types.ModuleType('legality')
exec(compile('"""Prune configurations that cannot run, before spending a measurement on them.\n\nEvery real measurement on a rented accelerator costs wall-clock time, so a\nbudgeted autotuner should not spend one discovering that a tile does not fit\nin shared memory. This module answers "could this configuration possibly\nrun here?" from device properties alone.\n\nIt is deliberately conservative. A configuration rejected here is one we are\nconfident cannot run; anything uncertain is left in the space and settled by\nmeasurement. Over-pruning silently removes the optimum, which is worse than\nwasting a trial.\n\nThat conservatism is not theoretical. The shared-memory estimate below does\nnot predict the failures we measured: L4 (100 KiB limit) ran a tile the\nestimate puts at 144 KiB, while A100 (163 KiB limit) refused to launch it.\nTriton evidently reshapes the pipeline rather than allocating naively. So\n`smem_bytes` is recorded as a feature and never used to reject a candidate\nuntil it has been fitted against measured launch failures.\n"""\n\nimport ctypes\nimport functools\n\n# CUDA driver attribute ids (cuda.h, CUdevice_attribute).\n_CU_SMEM_PER_BLOCK_OPTIN = 97\n_CU_SMEM_PER_SM = 41\n_CU_MAX_THREADS_PER_SM = 39\n_CU_SM_COUNT = 16\n_CU_COMPUTE_CAPABILITY_MAJOR = 75\n_CU_COMPUTE_CAPABILITY_MINOR = 76\n\n# The Triton backend in XLA refuses anything below Ampere.\nMIN_TRITON_COMPUTE_CAPABILITY = (8, 0)\n\nDTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "int8": 1,\n               "float8_e4m3fn": 1, "float8_e5m2": 1}\n\n\n@functools.lru_cache(maxsize=1)\ndef gpu_properties():\n    """Device limits read from the CUDA driver, or {} if unavailable."""\n    try:\n        libcuda = ctypes.CDLL("libcuda.so.1")\n        if libcuda.cuInit(0) != 0:\n            return {}\n        handle = ctypes.c_int()\n        if libcuda.cuDeviceGet(ctypes.byref(handle), 0) != 0:\n            return {}\n\n        def attr(attr_id):\n            val = ctypes.c_int()\n            if libcuda.cuDeviceGetAttribute(ctypes.byref(val), attr_id, handle) != 0:\n                return None\n            return val.value\n\n        return {\n            "smem_per_block": attr(_CU_SMEM_PER_BLOCK_OPTIN),\n            "smem_per_sm": attr(_CU_SMEM_PER_SM),\n            "max_threads_per_sm": attr(_CU_MAX_THREADS_PER_SM),\n            "sm_count": attr(_CU_SM_COUNT),\n            "compute_capability": (attr(_CU_COMPUTE_CAPABILITY_MAJOR),\n                                   attr(_CU_COMPUTE_CAPABILITY_MINOR)),\n        }\n    except Exception:  # noqa: BLE001 - no CUDA driver is a normal outcome\n        return {}\n\n\ndef supports_pallas_gpu(props=None):\n    """Whether the Triton backend will accept this device at all."""\n    props = props if props is not None else gpu_properties()\n    cc = props.get("compute_capability")\n    if not cc or cc[0] is None:\n        return None  # unknown; let measurement decide\n    return cc >= MIN_TRITON_COMPUTE_CAPABILITY\n\n\ndef smem_bytes(bm, bn, bk, num_stages, dtype_name):\n    """Naive estimate of the shared memory needed to stage the operand tiles.\n\n    Assumes Triton keeps `num_stages` copies of both operand tiles resident so\n    loads for later iterations overlap the current dot.\n\n    Measurement says this is not what Triton does — see the module docstring.\n    Use it as a recorded feature, not as a decision.\n    """\n    elem = DTYPE_BYTES.get(dtype_name, 2)\n    per_stage = (bm * bk + bk * bn) * elem\n    return per_stage * num_stages\n\n\ndef gpu_config_features(bm, bn, bk, num_warps, num_stages, dtype_name, props=None):\n    """Static features to store next to a measurement, for fitting later."""\n    props = props if props is not None else gpu_properties()\n    elem = DTYPE_BYTES.get(dtype_name, 2)\n    smem = smem_bytes(bm, bn, bk, num_stages, dtype_name)\n    limit = props.get("smem_per_block")\n    return {\n        "smem_estimate_bytes": smem,\n        "smem_limit_bytes": limit,\n        "smem_ratio": (smem / limit) if limit else None,\n        "threads": num_warps * 32,\n        "tile_elems": bm * bn,\n        "operand_elems": bm * bk + bk * bn,\n        "accum_bytes": bm * bn * 4,\n        "arithmetic_intensity": (bm * bn * bk) / max(1, (bm * bk + bk * bn) * elem),\n        "elems_per_thread": (bm * bn) / max(1, num_warps * 32),\n    }\n\n\ndef gpu_config_reasons(bm, bn, bk, num_warps, num_stages, dtype_name, props=None):\n    """Reasons this configuration certainly cannot run.\n\n    Only checks we have evidence for. The compute-capability gate is measured:\n    on a T4 every Pallas kernel fails, down to an elementwise copy with no dot\n    and no grid. Resource limits are deliberately absent — our estimate of\n    them disagrees with what the hardware actually did.\n    """\n    props = props if props is not None else gpu_properties()\n    reasons = []\n\n    if supports_pallas_gpu(props) is False:\n        cc = props["compute_capability"]\n        reasons.append(f"compute capability {cc[0]}.{cc[1]} below Triton minimum "\n                       f"{MIN_TRITON_COMPUTE_CAPABILITY[0]}."\n                       f"{MIN_TRITON_COMPUTE_CAPABILITY[1]}")\n\n    return reasons\n\n\nTILES = [\n    (32, 32, 32), (64, 64, 32), (64, 64, 64),\n    (128, 64, 32), (64, 128, 32),\n    (128, 128, 32), (128, 128, 64),\n    (128, 256, 32), (128, 256, 64),\n    (256, 128, 32), (256, 128, 64),\n    (256, 256, 32), (256, 256, 64),\n]\n\n\ndef gpu_configs(dtype_name, props=None, tiles=None, warps=(4, 8), stages=(2, 3, 4)):\n    """Candidate (bm, bn, bk, num_warps, num_stages) tuples for this device.\n\n    Returns (legal, rejected) so the rejected set can be recorded too — an\n    autotuner that silently drops candidates is indistinguishable from one\n    that never considered them.\n\n    Large tiles are paired with every stage count on purpose. A100 refused to\n    launch (256,128,64) at 3 stages while L4 and G4 ran it, so dropping big\n    tiles wholesale on A100 would compare its small tiles against other\n    devices\' large ones. Offering the same tile at 2 stages gives each device\n    a fair shot at the shape it can actually run.\n    """\n    props = props if props is not None else gpu_properties()\n    tiles = tiles or TILES\n\n    legal, rejected = [], []\n    for bm, bn, bk in tiles:\n        for nw in warps:\n            for ns in stages:\n                cfg = (bm, bn, bk, nw, ns)\n                reasons = gpu_config_reasons(bm, bn, bk, nw, ns, dtype_name, props)\n                (rejected if reasons else legal).append(\n                    (cfg, reasons) if reasons else cfg)\n    return legal, rejected\n\n\n# --- TPU ---------------------------------------------------------------------\n\n# Mosaic requires the last two dimensions of a block to be multiples of the\n# native tile, and VMEM bounds how much of the operands can be resident.\nTPU_TILE = (8, 128)\nTPU_VMEM_BYTES = {  # per TensorCore, from the public hardware tables\n    "TPU v2": 16 * 1024 ** 2,\n    "TPU v3": 16 * 1024 ** 2,\n    "TPU v4": 128 * 1024 ** 2,\n    "TPU v5 lite": 128 * 1024 ** 2,\n    "TPU v5p": 95 * 1024 ** 2,\n    "TPU v6 lite": 128 * 1024 ** 2,\n}\n\n\ndef tpu_vmem_bytes(bm, bk, bn, dtype_name, buffers=2):\n    """Naive estimate of VMEM demand: double-buffered operands plus accumulator.\n\n    Like the GPU estimate, this does not predict the failures we measured.\n    On v5e, (2048, 2048, 512) bf16 works out to roughly 25 MiB against a\n    nominal 128 MiB of VMEM, yet it fails with CompileTimeScopedVmemOom —\n    Mosaic\'s scoped VMEM budget is far smaller than the chip\'s VMEM, and the\n    published capacity is not the number that binds. Record it, do not act\n    on it.\n    """\n    elem = DTYPE_BYTES.get(dtype_name, 2)\n    return (bm * bk + bk * bn) * elem * buffers + bm * bn * 4\n\n\ndef tpu_config_features(bm, bk, bn, dtype_name, device_kind, buffers=2):\n    """Static features to store next to a TPU measurement."""\n    vmem = next((v for k, v in TPU_VMEM_BYTES.items() if device_kind.startswith(k)), None)\n    need = tpu_vmem_bytes(bm, bk, bn, dtype_name, buffers)\n    return {\n        "vmem_estimate_bytes": need,\n        "vmem_nominal_bytes": vmem,\n        "vmem_ratio": (need / vmem) if vmem else None,\n        "operand_elems": bm * bk + bk * bn,\n        "accum_bytes": bm * bn * 4,\n        "tile_elems": bm * bn,\n    }\n\n\ndef tpu_config_reasons(bm, bk, bn, dtype_name, device_kind, buffers=2):\n    """Reasons a Pallas TPU block configuration cannot run.\n\n    Only the tiling rules Mosaic documents, which are cheap and certain.\n    Capacity is deliberately not checked here — see `tpu_vmem_bytes`.\n    """\n    reasons = []\n    if bm % TPU_TILE[0]:\n        reasons.append(f"block_m {bm} not a multiple of {TPU_TILE[0]}")\n    for name, val in (("block_k", bk), ("block_n", bn)):\n        if val % TPU_TILE[1]:\n            reasons.append(f"{name} {val} not a multiple of {TPU_TILE[1]}")\n    return reasons\n', 'andamento/legality.py', 'exec'), legality.__dict__)
# --- end andamento/legality.py ---

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

    payload = {"device": dev.device_kind, "jax": jax.__version__,
               "properties": props, "shape": [M, K, N], "sweeps": sweeps}
    print("\n===RESULTS_JSON===")
    print(json.dumps(payload, default=str))

    # Also write a file so a collaborator running this locally has one
    # obvious artifact to send back, immune to copy-paste truncation.
    safe = "".join(c if c.isalnum() else "_" for c in dev.device_kind)
    out = f"sweep_result_{safe}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, default=str, indent=1)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
