# Qwen3.6-35B-A3B MoeEngine / Mference benchmark

> Historical measurement archive: project identifiers and local paths are normalized for the MoeEngine release. Measurements were not rerun for this publication. See [archive notes](../README.md).

Date: 2026-08-25  
Host: MacBook Air `Mac17,3`, Apple M5, 24 GB unified memory  
OS: macOS 26.5.2 (25F84)  
Power: AC connected, battery 100%  

## Verdict

The Qwen MoE model runs successfully through the same explicit expert-weight streaming design used for the DeepSeek test. The best configuration tested on this 24 GB fanless Air is **32 expert cache slots per layer**:

- warm interactive decode: **about 32 tok/s**;
- six measured 32-slot runs, including cold/cache-fill and sustained-load effects: **28.28 tok/s median**;
- three-run hot-confirm median: **26.53 tok/s**;
- observed sustained range: **16.43-31.96 tok/s**;
- process peak memory: **2,468 MiB**;
- process swaps: **0 in every run**;
- whole-system new swapouts: **0 pages in both 32-slot campaigns**.

The 96-slot automatic policy uses much more memory without improving this workload. It reached a measured median of 27.57 tok/s, used 5,980 MiB peak process memory, and showed a small whole-system swapout increment. For this machine, 32 slots is the better operating point.

This is not a resident-model benchmark. The dense/shared weights and currently selected experts are held in unified memory, while routed expert weights are explicitly read from the model files into an application-managed cache. macOS swap is not being used as the model's weight store.

## Model identity

- Artifact: `/path/to/MoeEngine/artifacts/qwen3.6-35b-a3b-4bit.gturbo`
- Installed size: about 18 GiB
- Architecture: Qwen3.6 MoE, 40 layers, 256 routed experts per layer, top-8 routing
- Expert stride: 1,769,472 bytes
- Source revision: `38740b847e4cb78f352aba30aa41c76e08e6eb46`
- Source snapshot hash: `sha256:0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea`
- Manifest SHA-256: `bfaa17cbca7755969c9a1e02216b5bb28f55701529c0f07dafbc07080517badd`
- Inference executable SHA-256: `d3bb8bd6e5954004c31f21da2e4fd11e7316e8b78c8b1df3a0451557fb990b6c`

## Protocol

Official frozen prompt `real-generation-v1/short-explanation.json`:

- prompt SHA-256: `c57da2677143657be55e03797c1aabe8e012d61c1a71bf9345acdd575117d1e1`
- prompt tokens: 62
- generated tokens: 516
- context limit: 4,096
- generation limit: 1,024
- temperature: 0.2
- top-k: 64
- top-p: 0.95
- seed: 20260721
- verification: trusted receipt
- `F_RDADVISE`: off
- each row is a fresh inference process measured by `/usr/bin/time -l`
- all 12 runs ended with `stop=endOfTurn`
- all 12 outputs had the identical SHA-256 `aed1d8fba029e982c238ef6e83cdc71e8fdd5863397bdf808804b077a9471c32`

## Results

### Automatic policy: 96 slots per layer

| Phase | Decode tok/s |
|---|---:|
| Discarded warm-up | 29.867 |
| Measured 1 | 28.954 |
| Measured 2 | 27.575 |
| Measured 3 | 20.477 |
| Measured median | **27.575** |

- median prefill time: 2.33 s
- peak process memory: 5,980 MiB
- process swaps: 0 for all four processes
- system swap use: 2,143.94 MiB -> 2,130.31 MiB
- system Swapouts: 678,706 -> 678,858, delta **+152 pages** (2.375 MiB at 16 KiB/page)

### Explicit 32 slots: cold/cache-fill campaign

| Phase | Decode tok/s |
|---|---:|
| Discarded warm-up | 6.792 |
| Measured 1 | 13.838 |
| Measured 2 | 30.024 |
| Measured 3 | 31.834 |
| Measured median | **30.024** |

- median prefill time: 1.50 s
- peak process memory: 2,468 MiB
- process swaps: 0 for all four processes
- system swap use: 2,106.31 MiB -> 2,106.31 MiB
- system Swapouts: 678,858 -> 678,858, delta **0 pages**

The first two processes show a material cold/cache-fill penalty. Reporting only the final 31.834 tok/s run would therefore overstate normal performance.

### Explicit 32 slots: immediate hot confirmation

| Phase | Decode tok/s |
|---|---:|
| Discarded warm-up | 32.249 |
| Measured 1 | 31.960 |
| Measured 2 | 26.528 |
| Measured 3 | 16.427 |
| Measured median | **26.528** |

- median prefill time: 1.46 s
- peak process memory: 2,468 MiB
- process swaps: 0 for all four processes
- system swap use: 2,106.31 MiB -> 2,106.31 MiB
- system Swapouts: 678,858 -> 678,858, delta **0 pages**

The decline during the hot campaign is consistent with sustained-load variability on a fanless Air, but temperature telemetry was unavailable, so it should not be presented as a proven thermal diagnosis.

## Cache-size interpretation

The model has 40 MoE layers and a 1,769,472-byte expert stride. The routed-expert cache capacity is therefore approximately:

- 32 slots: 2,264,924,160 bytes, or 2.11 GiB;
- 96 slots: 6,794,772,480 bytes, or 6.33 GiB;
- difference: 4.22 GiB.

The observed process peak fell by 3,512 MiB when moving from 96 to 32 slots. Despite the smaller cache, the 32-slot peak decode rate was higher and its combined six-run median was 28.28 tok/s.

## Honest interpretation of “0 swap”

Before these campaigns, macOS already had about 2.1 GiB of historical swap allocated from earlier work. A reboot was not performed, so the machine did not begin from absolute zero swap use.

What the measurements do prove for the two 32-slot campaigns is:

1. every model process reported zero swaps;
2. the system Swapouts counter did not increase;
3. total system swap allocation did not increase;
4. expert files remained an explicit read-only storage tier managed by the inference runtime, not virtual memory.

Accordingly, the defensible claim is **zero new swapouts during the measured 32-slot inference campaigns**, not “the operating system had never allocated swap.”

## Raw evidence

The paths below identify evidence in the historical experiment workspace. Those directories are not included in this release; only this report is archived here.

- `vendor/Mference/benchmark-results/qwen36-m5air-0swap-2026-08-25-auto96/`
- `vendor/Mference/benchmark-results/qwen36-m5air-0swap-2026-08-25-slots32/`
- `vendor/Mference/benchmark-results/qwen36-m5air-0swap-2026-08-25-slots32-hotconfirm/`

Each directory contains captured stdout/stderr, per-run `/usr/bin/time -l` counters, prompt/model hashes, hardware metadata, summary output, and whole-system `vm.swapusage`/`vm_stat` snapshots.
