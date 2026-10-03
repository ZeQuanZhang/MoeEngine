# Benchmarks

This is a chronological experiment record. Build paths, completed test counts,
and next-step notes describe each dated run, not the current checkout. Use the
[README](../README.md) and [native build guide](mollm-integration.md) for setup.
Weights are not included. Numerical compute comes from the named upstream
backend; comparing different backends does not isolate a MoeEngine policy gain.

This document records measured results only. Synthetic traces validate runtime
and policy behavior; they do **not** establish that DeepSeek V4 has a small
expert working set.

## Reproduction

```bash
PYTHONPATH=src python3 -m moeengine benchmark \
  --pattern bursty \
  --tokens 200 \
  --layers 43 \
  --experts-per-layer 256 \
  --top-k 6 \
  --expert-size-mib 12 \
  --cache-gib 2,4,8 \
  --policies lru,lfu,belady \
  --seed 20260818
```

## Initial synthetic result

Run on 2026-08-18 with Python 3.14.6. The simulated store used a 5 GB/s
transfer rate plus 80 microseconds fixed latency per 12 MiB expert load.

| Policy | Cache | Hit rate | Model faults | Bytes loaded | P95 modeled fault |
|---|---:|---:|---:|---:|---:|
| LRU | 2 GiB | 0.00% | 51,600 | 604.69 GiB | 2.597 ms |
| LFU | 2 GiB | 0.00% | 51,600 | 604.69 GiB | 2.597 ms |
| Belady | 2 GiB | 38.26% | 31,856 | 373.31 GiB | 2.597 ms |
| LRU | 4 GiB | 19.78% | 41,391 | 485.05 GiB | 2.597 ms |
| LFU | 4 GiB | 3.25% | 49,922 | 585.02 GiB | 2.597 ms |
| Belady | 4 GiB | 54.41% | 23,526 | 275.70 GiB | 2.597 ms |
| LRU | 8 GiB | 45.29% | 28,230 | 330.82 GiB | 2.597 ms |
| LFU | 8 GiB | 11.64% | 45,595 | 534.32 GiB | 2.597 ms |
| Belady | 8 GiB | 71.34% | 14,791 | 173.33 GiB | 2.597 ms |

This intentionally bursty trace changes each layer's hot set every 32 tokens.
It exposes two useful implementation facts:

1. A global LRU cache smaller than one token's 258 selected experts suffers a
   cyclic scan: by the time the next token reaches a layer, later-layer loads
   have displaced its experts. This produced the 2 GiB zero-hit result.
2. Plain LFU retains experts from expired bursts and adapts slowly. An aging or
   windowed frequency policy is a more meaningful future baseline.

Belady's gap shows room for better admission/retention but does not prove an
online predictor can realize it. Real DSV4 router traces remain the decisive
experiment.

## Official 0731 configuration inspection

The pinned official config reports:

| Quantity | Value |
|---|---:|
| Hidden layers | 43 |
| Routed experts per layer | 256 |
| Selected experts per token per layer | 6 |
| Routed expert instances | 11,008 |
| Parameters per routed expert | 25,165,824 |
| Estimated FP4 bytes per expert, excluding scales | 12 MiB |
| Estimated cold routed bytes per token, excluding scales | 3.023 GiB |
| Estimated total routed-expert bytes, excluding scales | 129 GiB |

The `-0731` release retains the preview MoE dimensions and adds DSpark fields,
including target layers 40–42. DSpark is excluded from the first cache-policy
comparison.

## Real DSV4 expert I/O result

Run on 2026-08-23 against the pinned official `DeepSeek-V4-Flash-0731`
checkpoint on the external T7. The source was
`model-00002-of-00048.safetensors`, and macOS `F_NOCACHE` was enabled so each
Model Fault performed storage I/O instead of relying on a warm OS file cache.

The trace visited the first 16 routed experts in layer 0 in a fixed sequence,
repeated for three rounds. Each expert was read from its six real FP4/scale
tensor ranges and materialized as a 13,369,344-byte (12.75 MiB) bundle.

```bash
PYTHONPATH=src python3 -m moeengine benchmark-real \
  /path/to/models/DeepSeek-V4-Flash-0731 \
  --layer 0 \
  --working-set 16 \
  --rounds 3 \
  --cache-mib 128,256 \
  --policy lru \
  --no-os-cache
```

| LRU cache | Experts fitting | Hit rate | Model faults | Bytes loaded | Evictions | P95 fault | Wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 MiB | 10 | 0.00% | 48 | 612 MiB | 38 | 23.796 ms | 1.093 s |
| 256 MiB | 20 | 66.67% | 16 | 204 MiB | 0 | 23.234 ms | 0.362 s |

The measured expert-read throughput was approximately 562–566 MiB/s. The
128 MiB cache cannot hold the 16-expert working set, so the repeated sequential
scan evicts every expert before its next use. Once the 256 MiB cache can retain
the entire 204 MiB working set, the second and third rounds are all explicit
cache hits. This reduces both Model Faults and physical bytes loaded by 3x.

The sample bundle `L0:E0` produced SHA-256
`6c5b6505c6de63a1556a1e1f0460c9a37936c559ccf0ff0f9e5a456de465f706`.
This proves deterministic real-byte addressing and loading. It does not yet
prove end-to-end DSV4 inference performance: the access sequence is constructed,
and router execution, FP4 expert compute, attention, and text generation are
not part of this milestone.

### Complete layer-address sweep

A second run used a 16 MiB cache and visited all 256 routed experts in layer 0
once. It successfully loaded 3,422,552,064 bytes (3.1875 GiB) through 256 Model
Faults with no missing tensor, short read, or range error. Peak explicit cache
occupancy stayed at one 12.75 MiB expert because every preceding expert was
evicted. Measured wall throughput was 564.89 MiB/s, P50 fault latency was
22.482 ms, and P95 was 23.252 ms.

## External T7 versus internal NVMe

The verified 166,898,661,074-byte checkpoint was copied unchanged to the
internal APFS volume and the complete layer-address sweep was repeated with the
same command, `F_NOCACHE`, trace, cache size, and payload digest.

| Storage | Bytes read | Mean fault | P95 fault | Wall throughput | Wall time |
|---|---:|---:|---:|---:|---:|
| External T7 | 3.1875 GiB | 22.496 ms | 23.252 ms | 564.89 MiB/s | 5.778 s |
| Internal NVMe | 3.1875 GiB | 3.536 ms | 4.296 ms | 3,550.87 MiB/s | 0.919 s |

The internal volume was 6.29x faster on this real expert sweep. A shorter
16-expert run measured 4,300–4,468 MiB/s, but the full-layer result is the more
conservative throughput value for projections.

The 16-expert, three-round cache boundary test retained identical policy
behavior on the internal volume:

| LRU cache | Hit rate | Model faults | Bytes loaded | P95 fault | Wall time |
|---:|---:|---:|---:|---:|---:|
| 128 MiB | 0.00% | 48 | 612 MiB | 4.139 ms | 0.142 s |
| 256 MiB | 66.67% | 16 | 204 MiB | 3.304 ms | 0.0457 s |

At 12.75 MiB per expert, a completely cold DSV4 token selects 258 expert
instances and requires 3,289.5 MiB of routed-expert reads. At the conservative
3,550.87 MiB/s internal result, the I/O-only cold-token bound improves from
approximately 0.17 token/s on the T7 to 1.08 token/s internally. Cache hits and
I/O/compute overlap can raise this limit; attention, shared-expert work, router
execution, and synchronization lower the final end-to-end result.

## End-to-end native-checkpoint CPU inference

Run on 2026-08-23 with the official 166,898,661,074-byte checkpoint on the
internal NVMe. The external inference baseline was
`shyringo/deepseek-v4-flash-0731-in-c` at commit
`23dd88d6f439883c942b09395a2fad6105cd5470`, built with Apple Clang 21 and
Homebrew libomp 22.1.8. Its metadata validator accepted all 48 shard headers
and tensor bindings, and its weightless test suite passed before inference.

The runtime is CPU-only on this machine. Its optimized vector kernels are
x86 AVX2-specific, so the Apple Silicon result uses its scalar C kernels with
OpenMP; it is a correctness and storage-policy baseline, not a Metal estimate.
Both measured arms used the same M5 24 GB host, checkpoint, prompt, 4,096-token
context, 12 GiB memory plan, 10 OpenMP threads, greedy decoding, disabled prompt
lookup, and an initially empty application expert cache.

| Expert access | Generated tokens | TTFT | TPOT | Decode | Peak RSS | Cache hit | Expert traffic |
|---|---:|---:|---:|---:|---:|---:|---:|
| Explicit reads + hash-min-SLRU | 64 | 32.129 s | 1.835 s/token | 0.545 token/s | 16.47 GiB | 54.42% | 104.88 GiB read |
| File-backed `mmap` + OS page cache | 39 | 49.936 s | 2.581 s/token | 0.387 token/s | 18.04 GiB | 50.29% | 74.46 GiB advised |

The explicit-read arm completed the full 64-token run and was 40.7% faster in
decode than the `mmap` arm. The `mmap` arm was stopped after 39 generated tokens
once its sustained TPOT was established. Because the run lengths differ, total
fault counts must not be compared directly; normalized major faults were about
2,547 per forward step for explicit reads and 87,799 per step for `mmap`.

A 15 GiB memory plan was also attempted. It regressed to 7.695 seconds/token
over the first 10 generated tokens and accumulated 3,618,797 major faults, so it
was stopped. On this 24 GB host, the larger anonymous expert cache displaced
useful file-backed pages and was materially worse than the 12 GiB plan.

The measured 0.545 token/s is real text generation through all 43 layers, not
the earlier I/O-only projection. It does not yet measure MoeEngine integrated
with a Metal compute backend; the current MoeEngine prototype supplies storage,
cache-policy, and trace-replay components but not the model's numerical forward
implementation.

## Native mollm CPU integration and policy A/B

Run on 2026-08-24 on the same 24 GB M5 MacBook Air. The official checkpoint
remained unchanged on the external T7 and was converted, without requantizing,
to a 156,712,820,637-byte mollm package on the internal NVMe. The package
contains 1,285 streamed weights and preserves checkpoint-native FP8 E4M3 dense
weights and MXFP4 E2M1/E8M0 routed experts.

Palm-Infra/mollm was built with Apple Clang 21, `-mcpu=native`, and embedded
ARM i8mm kernels. Its complete 48-test suite passed. MoeEngine LRU and LFU were
then added as selectable C++ replacement policies inside the same
`MoeSsdCache` I/O and numerical runtime. All arms below used exactly the same
package, prompt, greedy decoding, 4,096-token context, four CPU threads, eight
I/O workers, and 64 generated tokens. Generated text was byte-identical.

```bash
vendor/Palm-Infra/build_moeengine/mollm_bench \
  --package artifacts/deepseek-v4-flash-0731/mollm/deepseek_v4_flash_native.mollm \
  --prompt '给我讲一个故事' \
  --max-new-tokens 64 --n-ctx 4096 --threads 4 --temperature 0 \
  --ssd-cache-mb 1024 --ssd-io-workers 8 \
  --ssd-policy native --warmup 0 --output kv
```

| Policy | Expert cache | TTFT | Decode | TPOT | Peak RSS | App hit rate | Logical expert load/token | Expert wait/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mollm native model-aware | 1 GiB | 3.488 s | **1.85 token/s** | 540.7 ms | 17.09 GB | 21.92% | 3.104 GB | 226.5 ms |
| MoeEngine adaptive overlap | 1 GiB | 3.566 s | **1.84 token/s** | 542.6 ms | 17.19 GB | **23.80%** | **3.091 GB** | 249.7 ms |
| MoeEngine LRU | 1 GiB | 3.588 s | 1.75 token/s | 572.6 ms | 17.28 GB | 0.00% | 3.591 GB | 229.8 ms |
| MoeEngine LFU | 1 GiB | 3.954 s | 1.01 token/s | 994.4 ms | 17.11 GB | 0.00% | 4.174 GB | 311.4 ms |
| mollm native model-aware | 2 GiB | 3.735 s | 1.57 token/s | 636.4 ms | 17.12 GB | 42.26% | 2.526 GB | 174.4 ms |

The 1 GiB native result is 3.39x the earlier scalar-C explicit-cache baseline
of 0.545 token/s. This is a cross-backend observation with different numerical
kernels; it is not a 3.39x speedup caused by the MoeEngine cache policy. Pure
global LRU exhibits the same cyclic-scan failure as the
synthetic trace: all application-level cross-token residency hits disappear.
LFU is materially worse because stale frequency competes with current
cross-layer predictions; 3,413 prefetched entries were evicted unused during
decode.

The first MoeEngine adaptive policy observes consecutive routes directly and
assigns each layer between zero and three cross-token retention slots from an
EWMA of route overlap. It raised the application hit rate by 1.88 percentage
points, or 8.6% relative, without increasing cache capacity. However, current
cross-layer prediction usefulness fell: 879 prefetched entries were evicted
unused versus 519 for native, and foreground expert wait rose by 23.2
ms/token. Final throughput was effectively tied. This is an important metric
boundary: a next-token retention hit is not valuable when obtaining it delays
an expert already predicted for the current token. Future policy scores must
give imminent high-confidence prefetches priority over speculative
cross-token retention.

The 2 GiB native run reduced I/O and doubled the hit rate, but total throughput
fell by 15%. Its expert wait improved while matrix time regressed, so the run
does not support an I/O explanation. Back-to-back sustained workloads on the
fanless Air make thermal throttling and/or larger locked-memory bandwidth
pressure plausible confounders. Cache-size sweeps need alternating run order,
cool-down intervals, and independent-process medians before choosing a final
capacity.

### Time-value-aware adaptive retention

The adaptive policy was then combined with an explicit current-demand epoch.
Previously, a real expert named by the active layer and a speculative expert
for the next layer both appeared to be current-forward entries. A sufficiently
strong prediction could therefore displace a guaranteed, lower-router-rank hit
from the active route. The replacement comparator now sacrifices the future
prediction first. This is a soft ordering rather than an absolute pin, so a
tiny cache can still advance its streaming window.

The change was isolated with two binaries built from identical sources except
for that comparator and run in both orders. Each process used 32 generated
tokens and the same 1 GiB cache configuration above. Fanless thermal variation
still makes the end-to-end rate unsuitable as a precise microbenchmark, but
the policy counters were repeatable across both orderings:

| Adaptive policy counter | Before, two runs | Time-value aware, two runs | Change |
|---|---:|---:|---:|
| Logical expert load/token | 3,149 MB | **3,081 MB** | **-2.2%** |
| Unused prefetch evictions | 493 | **371** | **-24.7%** |
| Short-term reloads | 2,773 | **2,606** | **-6.0%** |
| Useful cross-layer predictions | 2,774 | **3,040** | **+9.6%** |
| Rejected cross-layer predictions | 275 | **141** | **-48.7%** |

The best cool-start decode remained **1.90 token/s**. Across comparable cool
runs, foreground expert wait moved from roughly 246--249 ms/token to about
237 ms/token. Later back-to-back runs again became compute-bound from thermal
throttling, so sustained throughput claims require longer cool-down and more
independent samples. The important result here is narrower: the new ordering
reduces prediction pollution and physical work without increasing cache RAM.

### MXFP4 activation reuse and the 24 GB cache-capacity boundary

The next CPU pass profiled the real routed shapes. Decode is dominated by
MXFP4 `M=1,N=4096,K=4096` gate/up and `M=1,N=4096,K=2048` down projections;
roughly 60% of routed experts still execute as individual GEMVs, while ready
waves are usually batches of two to four. Forcing the runtime to wait for a
larger batch would serialize useful expert I/O, so the optimization instead
shares the gate/up activation quantization across independently arriving
experts.

With the same 1 GiB configuration and 16 generated tokens, dual-Q8 activation
quantization calls fell from 25,259 to 20,833 (**-17.5%**). An isolated hot
matrix microbenchmark also found a small 0.7--2% improvement from replacing 64
dynamic output jobs in a single GEMV with four equal static shards. A same-
capacity real-model A/B measured 1.98 versus 1.99 token/s, so these are valid
low-risk compute savings but not the primary end-to-end lever.

An exploratory cache sweep then used the time-value-aware adaptive policy,
four compute threads, eight I/O workers, and the same deterministic 16-token
request. It was intentionally stopped after the capacity boundary became
clear; these back-to-back fanless runs are not independent thermal samples.

| Expert cache | Pages locked | Decode | TPOT | Logical expert load/token | Expert wait/token | Peak RSS |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 GiB | yes | 1.80 token/s | 556.1 ms | 3.254 GB | 268.1 ms | 15.74 GB |
| **2 GiB** | **yes** | **1.99 token/s** | **501.4 ms** | 2.724 GB | 216.0 ms | 17.50 GB |
| 3 GiB | no | 1.84 token/s | 544.0 ms | 2.317 GB | 170.3 ms | 17.66 GB |
| 4 GiB | no | 1.68 token/s | 595.0 ms | 2.133 GB | 134.9 ms | 17.59 GB |

The monotonic I/O counters prove that extra capacity remains useful, but 3 and
4 GiB crossed the host's lockable-memory/pressure boundary: expert `mlock`
failed and compute time rose enough to erase the I/O win. On this specific
24 GB M5 Air, **2 GiB is therefore the current operational sweet spot** and is
the configuration to reproduce next:

```bash
vendor/Palm-Infra/build_moeengine/mollm_bench \
  --package artifacts/deepseek-v4-flash-0731/mollm/deepseek_v4_flash_native.mollm \
  --prompt '给我讲一个故事' \
  --max-new-tokens 16 --n-ctx 4096 --threads 4 --temperature 0 \
  --ssd-cache-mb 2048 --ssd-io-workers 8 \
  --ssd-policy moeengine-adaptive --warmup 0 --profile --output kv
```

The complete CPU suite passed **48/48** after the activation-cache and static-
sharding changes. A longer, cooled, alternating-order campaign is still needed
before claiming a sustained rate above 2 token/s; the defensible current
statement is that a cool short run reached 1.99 token/s and that 2 GiB has the
best observed capacity trade-off.

### Lossless dense-sidecar compression and the new 4 GiB operating point

The package starts each serialized weight payload at an address whose low four
bits are 9. The CPU loader therefore used to retain about 3.68 GB of aligned
FP16/FP32 construction copies even after a complete interleaved sidecar had
been built. Those transient copies are now released, and FP16 embedding lookup
loads its unaligned package elements with `memcpy`. Native FP8 Q8 sidecars also
store their exact coefficient scale once per original 128-element block rather
than duplicating it for all four 32-element SDOT tiles.

Together these lossless changes reduce locked CPU sidecars from 13,298.6 MB to
9,358.9 MB, a saving of 3,939.7 MB (29.6%). This is enough for the 4 GiB expert
cache to lock successfully on the 24 GB M5 Air. The same deterministic
16-token request then measured 2.34 token/s and 427.7 ms TPOT, compared with
the previous 2 GiB best of 1.99 token/s and 501.4 ms TPOT. Expert wait fell
from 216.0 to 171.2 ms/token. The run produced the same greedy text and passed
the complete 48/48 CPU suite.

An additional exact compact mode keeps the DSV4 `wo_a` sidecar as interleaved
FP8 bytes plus its original E8M0 scales, reconstructing the BF16-defined FP16
value inside the NEON GEMV. It reduces the sidecar total further to 7,916.1 MB:
5,382.5 MB (40.5%) below the original, and 1,442.8 MB below the default layout.
It is intentionally opt-in because exact decoding raises grouped-linear time
enough to reduce the same 4 GiB run from 2.34 to 2.02 token/s even though its
SSD wait is unchanged.

| CPU layout | Locked sidecars | Expert cache | Decode | TPOT | Expert wait/token | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| Original expanded | 13,298.6 MB | 2 GiB | 1.99 token/s | 501.4 ms | 216.0 ms | 17.50 GB |
| **Compressed default** | **9,358.9 MB** | **4 GiB** | **2.34 token/s** | **427.7 ms** | **171.2 ms** | **17.14 GB** |
| Compact `wo_a` | 7,916.1 MB | 4 GiB | 2.02 token/s | 494.5 ms | 171.4 ms | 17.37 GB |

The default is therefore the 9.36 GB lossless layout. Enable the maximum-
compression experiment with:

```bash
MOLLM_DSV4_WOA_COMPACT_SIDECAR=1 \
vendor/Palm-Infra/build_moeengine/mollm_bench \
  --package artifacts/deepseek-v4-flash-0731/mollm/deepseek_v4_flash_native.mollm \
  --prompt '给我讲一个故事' \
  --max-new-tokens 16 --n-ctx 4096 --threads 4 --temperature 0 \
  --ssd-cache-mb 4096 --ssd-io-workers 8 \
  --ssd-policy moeengine-adaptive --warmup 0 --profile --output kv
```

An ensuing three-process, 128-token sustained audit showed that the fanless
M5 Air produced variable decode results: 2.13, 1.28
and 1.57 token/s (median 1.57). The second run's expert wait was lower while
its dense operators took roughly twice as long. This identifies a compute-side
regression consistent with thermal or power effects; neither cause was directly
measured. The earlier 2.34 token/s result is
therefore retained only as a 15-token cold-burst peak. See the full protocol,
raw counters, public comparison boundaries and reproducibility limitations in
[the repeated CPU experiment review](sota-report-2026-08-24.md).

## Xcode-free Metal 4 hybrid inference

Run on 2026-08-24 with the same package, prompt, context, greedy decoding,
thread count, I/O-worker count, and 64-token output. Palm-Infra was changed to
compile `mollm.metal` through `MTLDevice::newLibraryWithSource` at process
startup. This removes the build-time dependency on Xcode's standalone `metal`
and `metallib` executables. On the M5, the runtime compiler accepted the full
Metal 4 tensor source, and the complete Metal-enabled test suite passed 49/49.

The DSV4 run used the supported hybrid path: dense/attention graph work remains
on the CPU, while routed MXFP4 experts use direct Metal I/O and GPU compute.
`--metal-ssd-full` was not used because full-graph DSV4 Metal is not currently
implemented.

```bash
vendor/Palm-Infra/build_metal/mollm_bench \
  --device metal \
  --package artifacts/deepseek-v4-flash-0731/mollm/deepseek_v4_flash_native.mollm \
  --prompt '给我讲一个故事' \
  --max-new-tokens 64 --n-ctx 4096 --threads 4 --temperature 0 \
  --ssd-cache-mb 1024 --ssd-io-workers 8 \
  --warmup 0 --profile --output kv
```

| Backend | Expert cache | TTFT | Decode | TPOT | Peak RSS | Direct-cache hit | Metal expert reads/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU native model-aware | 1 GiB | 3.488 s | **1.85 token/s** | 540.7 ms | 17.09 GB | 21.92% | 3.104 GB logical load |
| Metal 4 hybrid | 1 GiB | 3.577 s | 1.32 token/s | 758.5 ms | 15.54 GB | 45.33% | 3.698 GB physical read |
| Metal 4 hybrid | 2 GiB | 3.752 s | 1.30 token/s | 770.7 ms | 16.86 GB | 53.70% | 3.132 GB physical read |

Doubling the direct Metal cache worked as intended: misses fell from 17,427 to
14,758 and physical expert traffic fell by 15.3%. MoE decode time also improved
from 38.87 to 36.83 seconds. End-to-end throughput did not improve because the
non-MoE CPU portions regressed during the second sustained run; for example,
decode MATMUL time rose from 3.15 to 4.35 seconds. Thermal state and shared
memory-bandwidth pressure remain confounders.

The important negative result is that the current hybrid backend is slower
than the optimized CPU path despite using the M5 GPU and despite a higher
expert-cache hit rate. The Metal implementation is therefore a valid
correctness and instrumentation baseline, not yet a speed win. The next
optimization target is reducing bytes loaded per routed expert and overlapping
Metal I/O, expert GEMV, and the CPU-resident graph rather than merely enlarging
the cache.
