# Router-GDSF validation — 2026-08-25

> Historical measurement archive: project identifiers and local paths are normalized for the MoeEngine release. Measurements were not rerun for this publication. See [archive notes](../README.md).

## Change under test

This iteration replaces the fixed weighted eviction score with router-aware
admission plus operating-system cache mechanisms:

- expected reuse × measured reload cost / resident byte as the admission key;
- a global GreedyDual inflation clock for equal-value aging;
- a bounded 32,768-entry ghost table and confidence-weighted refault EWMA;
- exact top-k demand exclusion from eviction;
- current-forward prediction deadline and confidence before long-term value.

The policy is selected with `--ssd-policy moeengine-gdsf`. It contains no model,
Apple, Metal, or instruction-set branches.

## Synthetic replay

Trace: bursty, 50 tokens, 43 layers, 256 experts/layer, top-6, 12.75 MiB per
expert, seed 25. Belady is an offline upper bound.

| Cache | LRU | Generic | Router-GDSF | Belady | GDSF vs generic |
|---:|---:|---:|---:|---:|---:|
| 1 GiB | 0.00% | 5.12% | **5.26%** | 23.94% | +0.14 pp |
| 2 GiB | 0.00% | 10.12% | **11.12%** | 36.47% | +1.00 pp |
| 4 GiB | 19.79% | 20.49% | **22.25%** | 52.43% | +1.76 pp |

Router-GDSF improves this online baseline at every tested capacity, but a large
gap to the offline bound remains.

## Full DSV4 run

Host state before the first run: AC power, battery 100%, 81% free memory. Both
Router-GDSF iterations used the 156,712.8 MB local package, prompt
`给我讲一个故事`, greedy decoding, 64 generated tokens (63 decode tokens), four
CPU threads, eight I/O workers, a 1 GiB expert cache, no application or OS
swap-backed expert store, and the portable CPU build.

| Policy/iteration | Decode | TTFT | Decode logical load | Demand load | Prefetch load | Expert wait | Matmul |
|---|---:|---:|---:|---:|---:|---:|---:|
| generic historical cool run | 1.94 tok/s | 3.54 s | 195.81 GB | 104.27 GB | 91.54 GB | 12.49 s | 16.61 s |
| GDSF, residency only | 1.69 tok/s | 3.79 s | 194.22 GB | 103.93 GB | 90.28 GB | 16.49 s | 17.00 s |
| GDSF + prediction deadline | **1.80 tok/s** | 3.67 s | 195.37 GB | **99.66 GB** | 95.71 GB | **11.84 s** | 19.14 s |
| generic, subsequent hot control | 1.59 tok/s | 3.56 s | 195.10 GB | 105.35 GB | 89.75 GB | 9.82 s | 24.45 s |

Deadline protection corrected the first GDSF run's critical-path regression:
demand-origin reads fell 4.4% and expert wait fell 5.2% relative to the generic
record. End-to-end decode did not exceed the historical 1.94 tok/s run because
the measured matrix-compute time was 15.2% higher on the fanless host. The I/O
counters support a cache/scheduling improvement; the wall-clock result does
not establish a throughput win.

The same-build generic control was run immediately afterward and fell to 1.59
tok/s while matrix time rose another 27.8%. GDSF therefore sits between a cool
historical generic run and a later hot generic control. This brackets the
thermal effect but is not a randomized A/B. Relative to that control, GDSF
still performed 5.4% fewer demand-origin reads; expert-wait time varied in the
opposite direction, so both counters are reported rather than collapsed into a
single causal claim.

## Verification

- Python reference suite: 26/26 passed.
- Portable native suite before the deadline patch: 49/49 passed.
- Focused native optimizer and MoE SSD cache tests passed after the patch.
- Generated text remained identical to the generic record.

The defensible conclusion is that Router-GDSF improves trace hit rate and the
real model's demand-I/O critical path. A randomized cool-start A/B or active
cooling is still required to measure its end-to-end throughput delta on this
fanless MacBook Air.
