# MoeEngine architecture

This document separates the implemented reference/runtime contracts from research
proposals. Setup and current entry points are in the [README](../README.md).

## Research boundary

MoeEngine does not change a model's router, Top-K selection, attention,
expert weights, or expert aggregation. A policy decision may change latency
and physical residency only. The numerical baseline must use the same packed
weights and compute kernels so quantization differences are not mistaken for
cache errors.

The first real target is the official `DeepSeek-V4-Flash-0731` checkpoint at
commit `7872f01b1d1fe23eabc4c98b48bffcef5a386062`. The attached DSpark module is
outside the initial paging experiment; it should be evaluated later as a
separate access-pattern and batching variable.

## Logical and physical models

```text
Router-selected (layer_id, expert_id)
                 |
                 v
          ExpertAddress
                 |
          ExpertCache.acquire
          /               \
       cache hit        model fault
          |                 |
          |          ExpertStore.load
          |                 |
          +------> ExpertLease
                         |
                    compute backend
                         |
                completion / release
                         |
                      release
```

An `ExpertLease` pins a slot. Replacement is forbidden until every CPU/GPU
consumer has released the matching slot generation. The Python prototype is
synchronous, but this lease contract is intentionally compatible with a future
asynchronous `Future[ExpertLease]` and Metal completion handler. Those future
interfaces are not implemented by the synchronous Python cache.

The Python cache loads a missing object before evicting residents. Its capacity
therefore limits accounted retained bytes after admission, not the temporary
allocation peak or total process RSS. A backend needs its own allocation and
completion rules to enforce a physical memory limit.

## Storage format

The proposed physical format uses one packed file per layer rather than one
file per expert:

```text
model-store/
  manifest.json
  resident/
  experts/
    layer_000.bin
    layer_001.bin
    ...
```

Each manifest record maps an `ExpertAddress` to a file, byte offset, byte
length, and optional checksum. An expert bundle must contain all compute-native
packed matrices and their quantization metadata. Alignment is a storage/backend
property, not part of the logical address.

The official checkpoint cannot be split by arbitrary byte ranges. The
converter must understand safetensors records, FP4 packing, block scales,
padding, and the Metal kernel layout. Repacking to the final compute layout at
conversion time avoids a costly transpose or copy on every model fault.

## Memory accounting on Apple Silicon

The explicit slot cache and the macOS file cache share the same physical
unified memory. They are logical mechanisms, not independent capacity tiers.
An explicit `pread` path may temporarily leave both file-cache pages and an
anonymous Metal buffer resident, so system-level physical footprint and memory
pressure must be measured in addition to process RSS.

The original native design considered these I/O paths:

1. `mmap` with ordinary OS paging;
2. `mmap` plus macOS advice/read-ahead hints;
3. `pread` directly into a shared `MTLBuffer` slot;
4. mapped file pages exposed to Metal without an intermediate copy, if the
   alignment and lifetime constraints prove safe.

## Prototype modules

- `ExpertAddress` and `ExpertDescriptor`: stable logical/physical metadata.
- `SyntheticExpertStore`: deterministic latency and byte accounting without
  allocating fake multi-gigabyte buffers.
- `PackedExpertStore`: real offset-based `pread` plus optional SHA-256 checks.
- `ExpertCache`: byte-bounded cache with generation-checked leases.
- `LRU`, `LFU`, `Belady`: comparable replacement policies; Belady is offline
  and is never a deployable policy.
- `Generic`, `Router-GDSF`: online route-aware policies; Router-GDSF adds a
  bounded non-resident ghost table and deadline-aware native admission.
- `GenericResidencyOptimizer`: a platform-free tier/candidate/route contract
  that minimizes estimated reload penalty per reclaimed byte.
- `RouteLocalityEstimator`: online cross-step reuse probabilities learned from
  exact router groups rather than cache hits.
- Trace generators: random, Zipf, Markov, and bursty workloads.
- Trace replay: hit/miss, model faults, bytes loaded/evicted, occupancy, and
  fault-latency percentiles.

## Validation stages

These are validation categories, not a current release checklist. Python cache
and store tests, checkpoint analysis, and native decode experiments exist;
end-to-end numerical equivalence and a hard physical-memory limit remain
unverified. The [experiment log](benchmarks.md) and
[GGUF runtime report](qwen4exp-real-runtime-2026-08-28.md) distinguish the tested
paths and their limits.

1. **Runtime correctness:** unit tests, deterministic trace replay, packed-file
   random access, and policy invariants.
2. **Physical feasibility:** measure 12 MiB random reads into the intended
   Metal-visible destination, including sustained and thermal behavior.
3. **Checkpoint analysis:** identify resident tensors, routed expert tensors,
   FP4 scales, and the DSpark-only tensors from the pinned checkpoint index.
4. **One-layer equivalence:** route, fault, load, execute, and compare against a
   baseline using identical packed weights.
5. **Full decode:** separate prefill streaming policy from decode caching and
   compare OS-only, explicit policies, and predictive experiments.
