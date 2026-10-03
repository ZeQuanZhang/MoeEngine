# MoeEngine — Initial Architecture Proposal (Historical)

This is the early design snapshot, preserved to explain the research direction.
Its proposed interfaces, task list, and completion markers describe that stage;
they are not a current feature checklist or instructions to run. Generic and
Router-GDSF policies and native decode experiments were added later. See the
[README](../README.md), [current architecture](architecture.md), and
[dated experiments](benchmarks.md) for implemented scope. Proposed GPU,
prefetch, and physical-memory contracts below must be read as design goals
unless a later experiment explicitly validates them.

> **Research proposition**: MoE routed experts should be treated as a
> model-aware Virtual Memory resource — not as ordinary tensors, not as files,
> and not as opaque blobs managed by an inference runtime.  The model's own
> router provides the *memory access signal*; the operating system (or a
> user-level OS-like layer) manages residency, replacement, prefetch, and
> topology-aware placement.

---

## 1. Motivation

A routed MoE model such as `DeepSeek-V4-Flash-0731` has 284 B total parameters
but activates only ~13 B per token. The 11,008 routed expert instances (~129 GiB
in FP4, excluding scales) are the dominant storage cost, yet a single token
touches only 258 of them (43 layers × Top-6). On a 24 GB Apple Silicon MacBook,
the shared core fits comfortably; the problem is the **routed experts**.

The key insight: **expert access is predictable**. The model's own per-layer
router outputs a Top-6 score vector *before* the experts are needed. That vector
is a free, high-quality memory-access hint — far richer than the heuristics a
general-purpose OS page cache can infer from block-level access patterns.

---

## 2. Core Abstraction: Model Virtual Memory

| Traditional OS | MoeEngine |
|---|---|
| Virtual Page | **Virtual Expert** (`ExpertAddress`: layer_id, expert_id) |
| Page Table | **Expert Table** (manifest → physical location) |
| Physical Page Frame | **Expert Slot** (wired Metal/CPU buffer) |
| Page Fault | **Model Fault** (NVMe → RAM, then compute) |
| LRU / LFU | **Replacement Policy** (pluggable: LRU, LFU, Belady, Router-aware) |
| Prefetch Hint | **Router Top-K scores** (free, per-token, per-layer) |
| NUMA Locality | **Expert-Compute Memory Locality** (future: tile-aware scheduling) |

Key design rule:

> The model router *decides* which experts are selected.
> MoeEngine *decides* where selected expert weights *reside*.
> The compute backend executes *unchanged* model mathematics.

No router, attention, embedding, or expert weight is modified. A policy change
affects latency and physical residency only. The numerical baseline uses the
same packed weights and compute kernels — quantization differences must not be
mistaken for cache errors.

---

## 3. Target Model

**`DeepSeek-V4-Flash-0731`** (commit `7872f01b1d1fe23eabc4c98b48bffcef5a386062`).

| Quantity | Value |
|---|---|
| Hidden layers | 43 |
| Routed experts per layer | 256 |
| Selected per token per layer | 6 (Top-6) |
| Total routed expert instances | 11,008 |
| Parameters per routed expert | 25,165,824 (4096×2048×3) |
| FP4 bytes per expert (excl. scales) | ~12 MiB |
| Total routed-expert bytes (excl. scales) | ~129 GiB |
| Cold expert bytes per token | ~3.0 GiB |

The `-0731` release has DSpark fields on layers 40–42. DSpark is **excluded**
from the first cache-policy comparison and should be evaluated later as a
separate access-pattern and batching variable.

---

## 4. Physical Layout (Post-Conversion)

The official Hugging Face checkpoint is distributed as safetensors shards. A
converter must understand: safetensors records, FP4 packing, block scales,
padding, and the Metal kernel layout. Repacking to the final compute layout at
conversion time avoids a costly transpose or copy on every model fault.

```
model-store/
  manifest.json          ← maps ExpertAddress → file, offset, size, checksum
  resident/              ← shared core (attention, embedding, router, norms, lm_head, shared experts)
    core.bin
  experts/
    layer_000.bin        ← 256 experts, packed per-layer
    layer_001.bin
    ...
    layer_042.bin
```

One file per layer (not per expert) keeps the file-descriptor count bounded and
allows the OS to perform contiguous read-ahead within a layer's hot set.

Each manifest record:

```json
{
  "layer_id": 17,
  "expert_id": 93,
  "file": "experts/layer_017.bin",
  "offset": 12345678,
  "size_bytes": 12582912,
  "sha256": "abcdef..."
}
```

---

## 5. Three-Level Memory Hierarchy

```
                        Router
                           │
                     Expert IDs
                           │
              ┌────────────▼────────────┐
 L0           │   Wired Expert Slots    │
              │   (Explicit Cache)      │
              │   Metal-visible buffers │
              │   ~8–12 GiB budget      │
              └────────────┬────────────┘
                           │ miss (model fault)
              ┌────────────▼────────────┐
 L1           │   macOS Unified Memory  │
              │   (File Cache / mmap)   │
              │   MADV_WILLNEED hints   │
              └────────────┬────────────┘
                           │ fault
              ┌────────────▼────────────┐
 L2           │   Internal NVMe         │
              │   ~90–130 GiB experts   │
              │   pread / mmap          │
              └─────────────────────────┘
```

**Critical note for Apple Silicon**: L0 (explicit slot cache) and L1 (macOS
file cache) share the same physical unified memory. They are *logical*
mechanisms, not independent capacity tiers. An explicit `pread` path may
temporarily leave both file-cache pages and an anonymous Metal buffer resident,
so system-level physical footprint and memory pressure must be measured in
addition to process RSS.

---

## 6. 24 GB Unified Memory Budget (Candidate)

```
24 GB Unified Memory
├── macOS / background       ~2–3 GB
├── Shared Core              ~4–6 GB  (attention, embedding, routers, norms, shared experts, lm_head)
├── Expert Hot Cache         ~8–10 GB (explicit slot cache)
├── KV Cache                 ~2–4 GB  (context-dependent)
├── Metal scratch/activation ~1–2 GB
└── safety margin            ~1–2 GB
```

Do not fill 24 GB to 23.9 GB. Once macOS memory pressure triggers aggressive
compression/swap, the entire page-cache and I/O behavior changes and benchmarks
become meaningless.

---

## 7. Single-Token Execution Flow

```
Layer 17
   │
   ▼
Attention (resident)
   │
   ▼
Router (resident)
   │
   ├─ E13  score=0.94
   ├─ E21  score=0.88
   ├─ E49  score=0.71
   ├─ E77  score=0.55
   ├─ E103 score=0.41
   └─ E221 score=0.32
       │
       ▼
Expert Cache Lookup (per ExpertAddress)
       │
   ┌───┴────────┐
   │            │
 HIT          MISS
   │            │
 Metal        async pread()
   │            │
   │        NVMe → slot
   │            │
   │        (overlap: compute shared
   │         expert + resident experts
   │         while waiting)
   │            │
   └──────┬─────┘
          ▼
     grouped GEMM
          │
          ▼
       Layer 18
```

Key optimization: when router returns Top-6, immediately dispatch async reads
for the missing experts while computing the shared expert and any resident
experts. The I/O latency is partially hidden.

---

## 8. Replacement Policies

The prototype supported these pluggable policies at the time of this proposal:

| Policy | Type | Description |
|---|---|---|
| **LRU** | Online | Least-recently-used; standard baseline |
| **LFU** | Online | Least-frequently-used with LRU tie-breaking |
| **Belady** | Offline | Farthest-next-use; theoretical upper bound only |

**Policies proposed at that stage (not a current status list):**

| Policy | Description |
|---|---|
| **Router-Aware LRU** | LRU weighted by router score: evict low-score entries first |
| **Windowed LFU** | Frequency over a sliding window; avoids cold-start retention |
| **Predictive Prefetch** | Use router's Top-K to prefetch before the next layer |
| **Cost-Aware** | eviction cost = f(frequency, router_score, reload_latency) |

### Why Router-Aware?

The model router outputs **per-expert capability scores** every token, every
layer. These are free signals. A router-aware policy can:

1. **Admission**: only cache experts above a score threshold.
2. **Retention**: weight recency by router confidence.
3. **Prefetch**: after layer *n* selects its experts, peek at layer *n+1*'s
   router output (available from the previous token's forward pass or from
   speculative execution) and issue async reads.

This is analogous to a CPU branch predictor, but operating at **model
granularity** instead of instruction granularity.

---

## 9. Benchmark Matrix

### 9.1 Candidate I/O Paths

1. `mmap` with ordinary OS paging (no hints)
2. `mmap` + `MADV_WILLNEED` / `F_RDAHEAD` hints
3. `pread` directly into a shared `MTLBuffer` slot
4. Mapped file pages exposed to Metal without an intermediate copy (if alignment
   and lifetime constraints prove safe)

### 9.2 Policy Comparisons

| Label | Description |
|---|---|
| `os-lru` | mmap only; pure macOS page cache |
| `os-madvise` | mmap + MADV_WILLNEED per expert |
| `explicit-lru` | User-level LRU slot cache + pread |
| `explicit-lfu` | User-level LFU slot cache + pread |
| `router-aware` | LRU weighted by router scores |
| `predictive` | Router-aware + async prefetch |
| `belady` | Offline optimal (upper bound) |

### 9.3 Metrics Per Run

| Metric | Unit |
|---|---|
| tokens/s (decode) | tok/s |
| TTFT (prefill) | ms |
| Expert hit rate | % |
| Model faults / token | count |
| Bytes loaded / token | GiB |
| SSD read bandwidth | GB/s |
| Page faults / token | count |
| Memory pressure (process RSS, system physical) | GiB |
| Expert reuse distance | tokens |
| Prefetch precision / recall | % |
| P50 / P95 / P99 fault latency | ms |

### 9.4 Workloads

| Workload | Description |
|---|---|
| **Synthetic traces** | Random, Zipf, Markov, bursty (already implemented) |
| **Router trace** | Record real DSV4 router Top-6 for a prompt → export JSON |
| **Trace replay** | Replay a recorded router trace against any cache/policy |
| **Live decode** | End-to-end with Metal compute backend |

---

## 10. Original Phase Plan (Historical Snapshot)

### Phase 0 — Trace-level prototype ✅ DONE

- [x] `ExpertAddress`, `ExpertDescriptor`, `LoadedExpert` data types
- [x] `SyntheticExpertStore` (metadata-only, no multi-GB allocation)
- [x] `PackedExpertStore` (real `pread`, manifest, optional SHA-256)
- [x] `ExpertCache` with generation-checked `ExpertLease` (pin-safe)
- [x] `LRU`, `LFU`, `Belady` replacement policies
- [x] Trace generators: random, Zipf, Markov, bursty
- [x] `replay_trace` + `BenchmarkResult` (hit rate, faults, bytes, latencies)
- [x] CLI: `moeengine benchmark`, `inspect-config`, `inspect-index`, `plan-checkpoint`, `inspect-shard`
- [x] Checkpoint planning: safetensors index parser, expert bundle validation, header parser
- [x] Unit tests for all modules
- [x] Initial benchmark results with bursty trace (43 layers, 256 experts, Top-6, 12 MiB)

### Phase 1 — Checkpoint conversion ⬜ Planned at that stage

- [ ] Download `DeepSeek-V4-Flash-0731` safetensors from HF
- [ ] Identify all **resident** tensors (attention, embedding, router, norms, shared experts, lm_head)
- [ ] Identify all **routed expert** tensors (per-layer FFN w1/w2/w3 weights + scales)
- [ ] Identify DSpark-only tensors (layers 40–42, exclude from first experiment)
- [ ] Measure exact byte sizes from safetensors headers (not estimates)
- [ ] Design compute-native packed layout (Metal kernel requirements for FP4 block scales, padding, alignment)
- [ ] Build converter: safetensors → `model-store/` layout (one file per layer, manifest.json)
- [ ] Validate: random `pread` + SHA-256 check on converted store
- [ ] Validate: numerical equivalence on one layer against reference implementation

### Phase 2 — I/O micro-benchmarks ⬜

- [ ] Measure raw NVMe `pread` latency/throughput (4 KiB → 12 MiB block sizes)
- [ ] Measure `mmap` fault latency on Apple Silicon
- [ ] Measure `MADV_WILLNEED` effect on fault latency
- [ ] Measure sustained throughput under 10+ minutes of continuous random reads
- [ ] Measure thermal throttling behavior
- [ ] Determine optimal I/O path for the explicit slot cache

### Phase 3 — Single-layer equivalence ⬜

- [ ] Load resident core tensors into Metal buffers
- [ ] Implement one layer's forward pass: attention, router, Top-6 selection, expert FFN, aggregation
- [ ] Route one test token; trigger model fault for missing experts; load via pread
- [ ] Compare output numerically against reference implementation using identical packed weights
- [ ] Measure: compute time vs. I/O time for one layer

### Phase 4 — Full decode with live cache ⬜

- [ ] Wire the complete 43-layer model
- [ ] Implement per-layer cache lookup + async pread for misses
- [ ] Overlap I/O with compute (shared expert + resident experts while waiting)
- [ ] Implement KV cache management
- [ ] Measure end-to-end tokens/s on 24 GB MacBook
- [ ] Compare: `os-lru`, `os-madvise`, `explicit-lru`, `explicit-lfu`, `router-aware`, `predictive`, `belady`

### Phase 5 — Router trace recording & analysis ⬜

- [ ] Instrument the running model to record every (layer, Top-6 expert IDs, scores) per token
- [ ] Export router traces as JSON for offline analysis
- [ ] Analyze: expert reuse distance distribution
- [ ] Analyze: per-layer working-set size over N tokens
- [ ] Analyze: cross-layer expert co-activation patterns
- [ ] Compute Belady optimal hit rate as theoretical upper bound
- [ ] Measure gap: best online policy vs. Belady

### Phase 6 — Predictive prefetch ⬜

- [ ] Use previous token's layer *n+1* router output to prefetch for next token's layer *n+1*
- [ ] Implement prefetch budget (limit concurrent NVMe reads)
- [ ] Measure prefetch precision (fraction of prefetched experts actually used)
- [ ] Measure prefetch recall (fraction of misses avoided by prefetch)
- [ ] Compare end-to-end tokens/s with and without prefetch

### Phase 7 — OS-aware integration ⬜

- [ ] Experiment with `MADV_WILLNEED` per expert from user-level cache
- [ ] Experiment with `MADV_DONTNEED` on evicted experts
- [ ] Compare explicit cache + OS hints vs. pure OS page cache
- [ ] Measure system-level memory pressure (not just process RSS)
- [ ] Document the interaction between L0 (explicit cache) and L1 (file cache)

---

## 11. Existing Prototype Reference

The Python prototype at `src/moeengine/` implements the **Phase 0** trace-level
simulator. It is deliberately model-independent and validates addressing,
stores, caches, replacement policies, trace generation, trace replay, and
telemetry without requiring real model weights.

### Module map

| Module | Purpose |
|---|---|
| `model.py` | `ExpertAddress`, `ExpertDescriptor`, `LoadedExpert` |
| `store.py` | `ExpertStore` (ABC), `SyntheticExpertStore`, `PackedExpertStore` |
| `cache.py` | `ExpertCache`, `ExpertLease`, `CacheStats`, `CacheAccess` |
| `policies.py` | `ReplacementPolicy` (ABC), `LRUPolicy`, `LFUPolicy`, `BeladyPolicy`, `make_policy` |
| `trace.py` | `random_trace`, `zipf_trace`, `markov_trace`, `bursty_trace`, `generate_trace` |
| `replay.py` | `replay_trace`, `BenchmarkResult` |
| `inspect.py` | `inspect_config`, `inspect_safetensors_index` (classifies resident/base-expert/MTP tensors) |
| `checkpoint.py` | `plan_checkpoint`, `write_checkpoint_plan`, `read_safetensors_header`, `measure_bundles_in_shard` |
| `cli.py` | `moeengine benchmark`, `inspect-config`, `inspect-index`, `plan-checkpoint`, `inspect-shard` |

### Key design decisions preserved from the prototype

1. **ExpertLease is pin-safe**: a generation counter prevents eviction of an
   expert while a GPU/CPU operation is in flight. `release()` must be called after
   the compute fence.
2. **Synthetic stores omit payloads**: cache simulations over 11,008 experts do
   not allocate 129 GiB of fake buffers.
3. **Packed stores use one fd per file**: not per expert.
4. **Belady is offline-only**: it consumes the complete future trace at
   construction time and is never a deployable policy.
5. **Manifest is JSON, versioned**: `PackedExpertStore.MANIFEST_VERSION = 1`.

---

## 12. Original Implementation Tasks (Historical)

### Task A: Router-Aware Replacement Policy

Implement `RouterAwarePolicy` in `src/moeengine/policies.py`:

- Constructor accepts `router_scores: dict[ExpertAddress, float]` or a callable
- on each `on_access` / `on_insert`, store the router score alongside the entry
- `choose_victim` evicts lowest `frequency × router_score` (or configurable formula)
- Unit tests in `tests/test_policies.py`
- Benchmark against LRU/LFU/Belady with bursty trace; compare hit rates

### Task B: Predictive Prefetch in Cache

Extend `ExpertCache` with an optional prefetch budget:

- `prefetch(addresses: list[ExpertAddress])` — issues async `pread` calls, limited by a max-inflight count
- Prefetched experts land in the cache as normal entries (with `on_insert` policy notification)
- A prefetch that arrives after the expert was already requested is still a hit (the `acquire` call waits for the in-flight read)
- Measure prefetch precision/recall in `BenchmarkResult`

### Task C: Cost-Aware Eviction Policy

Implement `CostAwarePolicy`:

- eviction score = `frequency × router_score × reload_cost_factor`
- `reload_cost_factor` could be 1.0 for NVMe, higher for slower storage tiers
- Compare against LRU/LFU on bursty traces

### Task D: Checkpoint Converter Scaffold

Build the `tools/convert.py` skeleton:

- Read the safetensors index JSON
- Classify all tensors as resident / routed-expert / DSpark / other
- Compute packed byte offsets and write `manifest.json`
- Extract resident tensors into `resident/core.bin`
- Extract each layer's 256 experts into `experts/layer_NNN.bin`
- Validate with SHA-256
- **Important**: understand FP4 block scales — the converter must produce
  compute-ready blobs, not raw safetensors slices

### Task E: I/O Benchmarks

Write `tools/io_bench.py`:

- Measure `pread` latency/throughput for block sizes 4 KiB → 12 MiB
- Measure `mmap` page fault latency (touch first byte of each 16 KiB page)
- Measure `MADV_WILLNEED` effectiveness
- Run sustained read test (10 min) and detect thermal throttling
- Output CSV for analysis

### Task F: Router Trace Recorder

Extend the prototype to record real router traces:

- Accept a JSON trace format: `[[(layer, expert_id, score), ...], ...]` per token
- Replay recorded traces through the existing cache simulator
- Compare synthetic vs. real trace hit rates
- Analyze expert reuse distance

---

## 13. Research Design Principles

1. **Gate/Router is a predictor, not an executor.** The router's job is to
   predict which expert can handle the input; the expert does the work. Do not
   make the router another LLM.

2. **Model fault, not page fault.** An expert miss is semantically richer than
   a page miss — the system knows *which* expert is needed and can make
   informed replacement decisions.

3. **Compute follows memory.** For memory-bound autoregressive decode, schedule
   compute on the tile/CPU where the expert weights already reside, not the
   reverse.

4. **Granularity matters.** At 0.8B (or 25M for a single FFN expert), a model
   module becomes a schedulable resource. At 70B+, it's an immovable monolith.
   SLMs make MoeEngine viable.

5. **Don't change the math.** The model's router, attention, expert weights,
   and aggregation are untouched. Only residency and latency change.

6. **Measure the gap, not the absolute.** The key research result is the
   difference between OS-default paging and router-aware caching — not the raw
   tokens/s number.

7. **24 GB is a feature, not a limitation.** A small memory budget forces cache
   misses, evictions, and prefetch to actually happen — exposing the value of
   intelligent policies. A 256 GB Mac would hide all of this.

---

## 14. Evolution Path (Beyond the Current Spec)

```
Gen 1 (current):   0.8B modules + Gate → MoeEngine routing
                    ↓
Gen 2:             Shared Backbone + Domain Expert Banks
                    ↓
Gen 3:             Hardware-aware Expert OS
                   Gate → Model VM → Expert Page Cache → Topology Scheduler
                   DDR0/NPU0  DDR1/NPU1  DDR2/NPU2  NVMe Pool
```

The current spec targets Gen 1 applied to DeepSeek V4's internal MoE structure,
which is a hybrid: the model already *has* a shared backbone and routed experts.
MoeEngine adds the residency/scheduling layer without retraining.