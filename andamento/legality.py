"""Prune configurations that cannot run, before spending a measurement on them.

Every real measurement on a rented accelerator costs wall-clock time, so a
budgeted autotuner should not spend one discovering that a tile does not fit
in shared memory. This module answers "could this configuration possibly
run here?" from device properties alone.

It is deliberately conservative. A configuration rejected here is one we are
confident cannot run; anything uncertain is left in the space and settled by
measurement. Over-pruning silently removes the optimum, which is worse than
wasting a trial.

That conservatism is not theoretical. The shared-memory estimate below does
not predict the failures we measured: L4 (100 KiB limit) ran a tile the
estimate puts at 144 KiB, while A100 (163 KiB limit) refused to launch it.
Triton evidently reshapes the pipeline rather than allocating naively. So
`smem_bytes` is recorded as a feature and never used to reject a candidate
until it has been fitted against measured launch failures.
"""

import ctypes
import functools

# CUDA driver attribute ids (cuda.h, CUdevice_attribute).
_CU_SMEM_PER_BLOCK_OPTIN = 97
_CU_SMEM_PER_SM = 41
_CU_MAX_THREADS_PER_SM = 39
_CU_SM_COUNT = 16
_CU_COMPUTE_CAPABILITY_MAJOR = 75
_CU_COMPUTE_CAPABILITY_MINOR = 76

# The Triton backend in XLA refuses anything below Ampere.
MIN_TRITON_COMPUTE_CAPABILITY = (8, 0)

DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "int8": 1,
               "float8_e4m3fn": 1, "float8_e5m2": 1}


@functools.lru_cache(maxsize=1)
def gpu_properties():
    """Device limits read from the CUDA driver, or {} if unavailable."""
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        if libcuda.cuInit(0) != 0:
            return {}
        handle = ctypes.c_int()
        if libcuda.cuDeviceGet(ctypes.byref(handle), 0) != 0:
            return {}

        def attr(attr_id):
            val = ctypes.c_int()
            if libcuda.cuDeviceGetAttribute(ctypes.byref(val), attr_id, handle) != 0:
                return None
            return val.value

        return {
            "smem_per_block": attr(_CU_SMEM_PER_BLOCK_OPTIN),
            "smem_per_sm": attr(_CU_SMEM_PER_SM),
            "max_threads_per_sm": attr(_CU_MAX_THREADS_PER_SM),
            "sm_count": attr(_CU_SM_COUNT),
            "compute_capability": (attr(_CU_COMPUTE_CAPABILITY_MAJOR),
                                   attr(_CU_COMPUTE_CAPABILITY_MINOR)),
        }
    except Exception:  # noqa: BLE001 - no CUDA driver is a normal outcome
        return {}


def supports_pallas_gpu(props=None):
    """Whether the Triton backend will accept this device at all."""
    props = props if props is not None else gpu_properties()
    cc = props.get("compute_capability")
    if not cc or cc[0] is None:
        return None  # unknown; let measurement decide
    return cc >= MIN_TRITON_COMPUTE_CAPABILITY


def smem_bytes(bm, bn, bk, num_stages, dtype_name):
    """Naive estimate of the shared memory needed to stage the operand tiles.

    Assumes Triton keeps `num_stages` copies of both operand tiles resident so
    loads for later iterations overlap the current dot.

    Measurement says this is not what Triton does — see the module docstring.
    Use it as a recorded feature, not as a decision.
    """
    elem = DTYPE_BYTES.get(dtype_name, 2)
    per_stage = (bm * bk + bk * bn) * elem
    return per_stage * num_stages


def gpu_config_features(bm, bn, bk, num_warps, num_stages, dtype_name, props=None):
    """Static features to store next to a measurement, for fitting later."""
    props = props if props is not None else gpu_properties()
    elem = DTYPE_BYTES.get(dtype_name, 2)
    smem = smem_bytes(bm, bn, bk, num_stages, dtype_name)
    limit = props.get("smem_per_block")
    return {
        "smem_estimate_bytes": smem,
        "smem_limit_bytes": limit,
        "smem_ratio": (smem / limit) if limit else None,
        "threads": num_warps * 32,
        "tile_elems": bm * bn,
        "operand_elems": bm * bk + bk * bn,
        "accum_bytes": bm * bn * 4,
        "arithmetic_intensity": (bm * bn * bk) / max(1, (bm * bk + bk * bn) * elem),
        "elems_per_thread": (bm * bn) / max(1, num_warps * 32),
    }


def gpu_config_reasons(bm, bn, bk, num_warps, num_stages, dtype_name, props=None):
    """Reasons this configuration certainly cannot run.

    Only checks we have evidence for. The compute-capability gate is measured:
    on a T4 every Pallas kernel fails, down to an elementwise copy with no dot
    and no grid. Resource limits are deliberately absent — our estimate of
    them disagrees with what the hardware actually did.
    """
    props = props if props is not None else gpu_properties()
    reasons = []

    if supports_pallas_gpu(props) is False:
        cc = props["compute_capability"]
        reasons.append(f"compute capability {cc[0]}.{cc[1]} below Triton minimum "
                       f"{MIN_TRITON_COMPUTE_CAPABILITY[0]}."
                       f"{MIN_TRITON_COMPUTE_CAPABILITY[1]}")

    return reasons


TILES = [
    (32, 32, 32), (64, 64, 32), (64, 64, 64),
    (128, 64, 32), (64, 128, 32),
    (128, 128, 32), (128, 128, 64),
    (128, 256, 32), (128, 256, 64),
    (256, 128, 32), (256, 128, 64),
    (256, 256, 32), (256, 256, 64),
]


def gpu_configs(dtype_name, props=None, tiles=None, warps=(4, 8), stages=(2, 3, 4)):
    """Candidate (bm, bn, bk, num_warps, num_stages) tuples for this device.

    Returns (legal, rejected) so the rejected set can be recorded too — an
    autotuner that silently drops candidates is indistinguishable from one
    that never considered them.

    Large tiles are paired with every stage count on purpose. A100 refused to
    launch (256,128,64) at 3 stages while L4 and G4 ran it, so dropping big
    tiles wholesale on A100 would compare its small tiles against other
    devices' large ones. Offering the same tile at 2 stages gives each device
    a fair shot at the shape it can actually run.
    """
    props = props if props is not None else gpu_properties()
    tiles = tiles or TILES

    legal, rejected = [], []
    for bm, bn, bk in tiles:
        for nw in warps:
            for ns in stages:
                cfg = (bm, bn, bk, nw, ns)
                reasons = gpu_config_reasons(bm, bn, bk, nw, ns, dtype_name, props)
                (rejected if reasons else legal).append(
                    (cfg, reasons) if reasons else cfg)
    return legal, rejected


# --- TPU ---------------------------------------------------------------------

# Mosaic requires the last two dimensions of a block to be multiples of the
# native tile, and VMEM bounds how much of the operands can be resident.
TPU_TILE = (8, 128)
TPU_VMEM_BYTES = {  # per TensorCore, from the public hardware tables
    "TPU v2": 16 * 1024 ** 2,
    "TPU v3": 16 * 1024 ** 2,
    "TPU v4": 128 * 1024 ** 2,
    "TPU v5 lite": 128 * 1024 ** 2,
    "TPU v5p": 95 * 1024 ** 2,
    "TPU v6 lite": 128 * 1024 ** 2,
}


def tpu_vmem_bytes(bm, bk, bn, dtype_name, buffers=2):
    """Naive estimate of VMEM demand: double-buffered operands plus accumulator.

    Like the GPU estimate, this does not predict the failures we measured.
    On v5e, (2048, 2048, 512) bf16 works out to roughly 25 MiB against a
    nominal 128 MiB of VMEM, yet it fails with CompileTimeScopedVmemOom —
    Mosaic's scoped VMEM budget is far smaller than the chip's VMEM, and the
    published capacity is not the number that binds. Record it, do not act
    on it.
    """
    elem = DTYPE_BYTES.get(dtype_name, 2)
    return (bm * bk + bk * bn) * elem * buffers + bm * bn * 4


def tpu_config_features(bm, bk, bn, dtype_name, device_kind, buffers=2):
    """Static features to store next to a TPU measurement."""
    vmem = next((v for k, v in TPU_VMEM_BYTES.items() if device_kind.startswith(k)), None)
    need = tpu_vmem_bytes(bm, bk, bn, dtype_name, buffers)
    return {
        "vmem_estimate_bytes": need,
        "vmem_nominal_bytes": vmem,
        "vmem_ratio": (need / vmem) if vmem else None,
        "operand_elems": bm * bk + bk * bn,
        "accum_bytes": bm * bn * 4,
        "tile_elems": bm * bn,
    }


def tpu_config_reasons(bm, bk, bn, dtype_name, device_kind, buffers=2):
    """Reasons a Pallas TPU block configuration cannot run.

    Only the tiling rules Mosaic documents, which are cheap and certain.
    Capacity is deliberately not checked here — see `tpu_vmem_bytes`.
    """
    reasons = []
    if bm % TPU_TILE[0]:
        reasons.append(f"block_m {bm} not a multiple of {TPU_TILE[0]}")
    for name, val in (("block_k", bk), ("block_n", bn)):
        if val % TPU_TILE[1]:
            reasons.append(f"{name} {val} not a multiple of {TPU_TILE[1]}")
    return reasons
