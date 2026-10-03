# Generic optimizer boundary

MoeEngine is a control plane, not a compute kernel. A portable implementation
must keep these responsibilities separate:

| Input owned by a backend | Decision owned by MoeEngine |
|---|---|
| Expert size and current residency | Which evictable experts to retain |
| Measured transfer latency and bandwidth | Which experts to evict for a demand |
| Exact and advisory router selections | How much value to assign to likely reuse |
| Pin/loading/completion state | Whether a plan is currently feasible |
| CPU, GPU, storage, or network operations | No platform operation |

The reference contract is implemented in `src/moeengine/optimizer.py`. It has no
branches for Apple, Metal, CUDA, CPU instruction sets, operating systems, or
model families. A memory tier is described by capacity, bandwidth, and fixed
latency. A resident expert is described by size, measured reload cost, access
history, predicted reuse, and safety state. The optimizer returns an eviction
plan; it never executes the transfer itself.

## Why hardware acceleration remains

Portability does not mean forcing every backend onto scalar CPU code. NEON,
AVX, Metal, CUDA, direct I/O, pinned memory, and asynchronous copies are backend
capabilities. They may alter measured transfer or execution costs but must not
alter the optimizer contract. A portable binary may runtime-dispatch to an
available instruction set without making the policy specific to that device.

## Current online policy

`GenericCostPolicy` consumes exact route groups when the inference backend can
expose them. It learns per-layer cross-step reuse with an EWMA and ranks
eviction candidates by estimated reload penalty per reclaimed byte. Required,
pinned, and loading entries are ineligible. The configurable score combines:

- learned route reuse probability;
- recency;
- access frequency;
- measured reload latency;
- expert size.

This is a deployable online heuristic, not a claim of optimality. Belady replay
continues to measure the distance to the offline optimum.

## Router-GDSF policy

`RouterGDSFPolicy` is the second deployable policy. Its admission order is the
expected reload penalty per reclaimed byte:

```text
P(reuse before eviction) * measured reload cost / resident bytes
```

Equal-value entries are aged with a global GreedyDual inflation clock. A
bounded ghost table retains only non-resident metadata; short refaults update a
reuse EWMA, with evidence confidence preventing one random refault from
overriding exact router history. Current top-k demand entries are ineligible
for eviction. In mollm, predicted entries for the current forward pass are
ordered by deadline and router confidence before long-term residency value.

The admission order is deliberately scan-resistant. A one-off expert receives
no unconditional frequency credit merely because it was loaded, so a full
sequential pass over unrelated layers cannot automatically flush a learned
routed working set.

## Native integration status

The native standard-C++ contract lives in
[`vendor/Palm-Infra/kernels/moeengine_optimizer.h`](../vendor/Palm-Infra/kernels/moeengine_optimizer.h) and is
selected in mollm with `--ssd-policy moeengine-generic`. mollm supplies the same
tier, route, candidate, and outcome facts without exposing Tensor, Metal, or
DSV4 types to the optimizer. The Python implementation mirrors the contract
for analysis and trace replay; Python is not called on the native hot path.

The C++ API can be consumed by another C++ backend now. A stable C ABI remains
useful for Swift, Rust, and other runtimes, but it is no longer required to
prove the mollm separation.

The same native contract also exposes `moeengine::RouterGdsfOptimizer`, selected
with `--ssd-policy moeengine-gdsf`. Its ghost table defaults to 32,768 entries;
it stores metadata only, never expert tensors.

The portable validation and its limitations are recorded in
[the portable validation report](../benchmark-results/generic-portable-2026-08-25/REPORT.md).
Router-GDSF validation is recorded in
[the Router-GDSF report](../benchmark-results/router-gdsf-2026-08-25/REPORT.md).

The separate llama.cpp GGUF experiment uses frequency-biased LRU and file-page
advice. It does not use this Generic/Router-GDSF optimizer; results from those
paths must not be attributed to the same policy.
