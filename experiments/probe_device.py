"""What can this accelerator actually run, and what are its limits?

Two jobs. First, record the hardware properties an autotuner needs to prune
a configuration space before spending measurements on it. Second, establish
by measurement — not by reading one error message — whether Pallas works on
this device at all, starting from the smallest kernel that could possibly
work and escalating.

Escalating from a trivial elementwise kernel separates "this device cannot
run Pallas" from "the matmul configuration was wrong", which look identical
if you only ever try one big kernel.

Run via:  colab run --gpu T4 experiments/probe_device.py
"""

import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl


def device_properties():
    """Hardware limits that bound the legal configuration space."""
    dev = jax.devices()[0]
    props = {
        "device_kind": dev.device_kind,
        "platform": dev.platform,
        "jax_version": jax.__version__,
        "device_count": jax.device_count(),
    }
    # JAX exposes a few of these directly; the rest come from CUDA if present.
    for attr in ("compute_capability", "core_count", "memory_stats"):
        try:
            val = getattr(dev, attr)
            props[attr] = val() if callable(val) else val
        except Exception:  # noqa: BLE001
            pass

    if dev.platform == "gpu":
        query = ("name,compute_cap,memory.total,clocks.max.sm,"
                 "clocks.max.memory")
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout.strip()
            props["nvidia_smi"] = dict(zip(query.split(","),
                                           [v.strip() for v in out.split(",")]))
        except Exception as e:  # noqa: BLE001
            props["nvidia_smi_error"] = str(e)[:200]

        # Shared memory per block is the hard cap on tile x stages, and it is
        # not exposed through JAX. Read it from the CUDA driver.
        try:
            import ctypes
            libcuda = ctypes.CDLL("libcuda.so.1")
            libcuda.cuInit(0)
            handle = ctypes.c_int()
            libcuda.cuDeviceGet(ctypes.byref(handle), 0)
            # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
            # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 39
            # CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
            for name, attr_id in (("smem_per_block_optin", 97),
                                  ("smem_per_sm", 39),
                                  ("regs_per_block", 12),
                                  ("multiprocessor_count", 16),
                                  ("warp_size", 10)):
                val = ctypes.c_int()
                if libcuda.cuDeviceGetAttribute(
                        ctypes.byref(val), attr_id, handle) == 0:
                    props[name] = val.value
        except Exception as e:  # noqa: BLE001
            props["cuda_driver_error"] = str(e)[:200]

    return props


def try_kernel(name, fn, *args):
    try:
        out = fn(*args)
        out.block_until_ready() if hasattr(out, "block_until_ready") else None
        print(f"  {name:34s} OK")
        return {"stage": name, "status": "ok"}
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {str(e)[:250]}"
        print(f"  {name:34s} FAILED — {type(e).__name__}: {str(e)[:100]}")
        return {"stage": name, "status": "fail", "error": msg}


def escalating_probe():
    """Smallest possible Pallas kernel first, then add one demand at a time."""
    print("\nPallas capability ladder (each step adds one requirement):")
    stages = []

    # 1. Elementwise copy: no dot, no tiling, no reduction. If this fails the
    #    device cannot run Pallas at all.
    def copy_kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...]

    def run_copy(dtype):
        x = jnp.ones((256, 256), dtype)
        return pl.pallas_call(
            copy_kernel, out_shape=jax.ShapeDtypeStruct(x.shape, dtype)
        )(x)

    stages.append(try_kernel("elementwise copy fp32", run_copy, jnp.float32))

    # 2. Same kernel, gridded: exercises BlockSpec and index maps.
    def run_gridded(dtype):
        x = jnp.ones((256, 256), dtype)
        return pl.pallas_call(
            copy_kernel,
            grid=(2, 2),
            in_specs=[pl.BlockSpec((128, 128), lambda i, j: (i, j))],
            out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
            out_shape=jax.ShapeDtypeStruct(x.shape, dtype),
        )(x)

    stages.append(try_kernel("gridded copy fp32", run_gridded, jnp.float32))

    # 3. A dot, at the smallest size the hardware could plausibly accept.
    def dot_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = jnp.dot(x_ref[...], y_ref[...],
                             preferred_element_type=jnp.float32).astype(o_ref.dtype)

    def run_dot(dtype):
        x = jnp.ones((64, 64), dtype)
        y = jnp.ones((64, 64), dtype)
        return pl.pallas_call(
            dot_kernel, out_shape=jax.ShapeDtypeStruct((64, 64), dtype)
        )(x, y)

    for name, dt in (("fp32", jnp.float32), ("fp16", jnp.float16),
                     ("bf16", jnp.bfloat16)):
        stages.append(try_kernel(f"64x64 dot {name}", run_dot, dt))

    return stages


def main():
    props = device_properties()
    print(f"jax {props['jax_version']} | {props['device_kind']} "
          f"({props['platform']}) x{props['device_count']}")
    for k, v in props.items():
        if k not in ("device_kind", "platform", "jax_version", "device_count"):
            print(f"  {k}: {v}")

    stages = escalating_probe()

    ok = [s["stage"] for s in stages if s["status"] == "ok"]
    print(f"\nverdict: {len(ok)}/{len(stages)} stages passed")
    if not ok:
        print("  Pallas is unavailable on this device — no configuration can fix it.")
    elif len(ok) < len(stages):
        print("  Pallas works; some dtypes or shapes are unsupported.")
    else:
        print("  Pallas fully available.")

    print("\n===RESULTS_JSON===")
    print(json.dumps({"properties": props, "stages": stages}, default=str))


if __name__ == "__main__":
    main()
