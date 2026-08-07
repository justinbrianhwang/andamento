# Andamento

> *Andamento* — in mosaic art, the visual flow created by the arrangement of tiles.

**Budgeted autotuning and performance-landscape characterization for [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) TPU kernels.**

On TPU, a Pallas kernel's performance is determined not just by block sizes but by the *flow* of the computation: grid traversal order, memory-space placement, pipeline buffering. The gap between a well-chosen configuration and a bad one is routinely 3–10×. Yet JAX ships no equivalent of `@triton.autotune` ([jax#24340](https://github.com/jax-ml/jax/issues/24340)), and existing tools search exhaustively or randomly.

Andamento asks a different question: **given only 20–100 real TPU measurements, how close to the exhaustive optimum can you get?**

## Goals

1. **A general autotuner for arbitrary `pallas_call` kernels** — legality-aware pruning (alignment, divisibility, VMEM capacity) followed by budgeted search (TPE / random-forest surrogates, shape-transfer warm-starts), with persistent shape/device-keyed caching.
2. **Joint search beyond block sizes** — grid order, `dimension_semantics`, memory spaces, scratch buffers, and pipeline buffer counts, which prior work leaves untuned.
3. **An open Pallas configuration–performance dataset** — each measurement labeled with runtime distribution, compile outcome, *and numerical error* (abs/rel/ULP vs. an FP64 reference), on real TPU hardware.

## Related work

- [Tokamax](https://github.com/openxla/tokamax) — exhaustive autotuning inside the `tokamax.Op` abstraction
- [pallas-forge](https://github.com/linhkid/pallas-forge) — grid/random block-size search with profiling discipline
- [TpuGraphs](https://arxiv.org/abs/2308.13490) — XLA/HLO-level configuration dataset (TPU v3)
- JAXBench / Autocomp — LLM-driven Pallas kernel generation on TPU

Andamento sits in the gap between these: source-level Pallas parameters, sample-efficient search, and a dataset that records what the others don't.

## License

MIT
