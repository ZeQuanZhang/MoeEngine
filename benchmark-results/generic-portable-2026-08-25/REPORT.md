# Generic optimizer / portable runtime validation — 2026-08-25

> Historical measurement archive: project identifiers and local paths are normalized for the MoeEngine release. Measurements were not rerun for this publication. See [archive notes](../README.md).

## Question

Does the MoeEngine control-plane idea still function after removing an M5-native
compile target and Metal from the measured runtime?

This is a portability validation, not a cross-hardware benchmark. Every real
model run below still happened on the same 24 GB M5 MacBook Air because no
second machine was part of the experiment.

## Portable configuration

```text
CMAKE_BUILD_TYPE=Release
MOLLM_ARM_NATIVE=OFF
MOLLM_ARM_I8MM=AUTO
MOLLM_METAL=OFF
MOLLM_CUDA=OFF
```

`MOLLM_ARM_I8MM=AUTO` embeds an optional ARM kernel selected by runtime feature
detection. It is not an M5 compile target. The ordinary ARM64/NEON path remains
available.

The Python reference suite passed 25/25 tests. The portable native mollm suite
passed 49/49 tests after CMake was pointed at the project's Python environment.

## Platform-neutral synthetic replay

Bursty trace: 50 tokens, 43 layers, 256 experts/layer, top-6, 12.75 MiB/expert,
3.5 GB/s measured-tier input, seed 25.

| Cache | LRU hit | LFU hit | Generic hit | Offline Belady hit |
|---:|---:|---:|---:|---:|
| 1 GiB | 0.00% | 0.00% | **5.12%** | 23.94% |
| 2 GiB | 0.00% | 0.00% | **10.12%** | 36.47% |
| 4 GiB | 19.79% | 7.87% | **20.49%** | 52.43% |

The generic heuristic beats the online LRU/LFU baselines in this trace but
remains far below the offline upper bound. Synthetic replay alone does not
establish end-to-end speed.

## Real DSV4 portable run

All arms used the same 156,712,820,637-byte package, prompt, greedy output,
1 GiB cache, four CPU threads, eight I/O workers, and 64 generated tokens.
The output text was identical in all three runs.

```bash
vendor/Palm-Infra/build_portable/mollm_bench \
  --package artifacts/deepseek-v4-flash-0731/mollm/deepseek_v4_flash_native.mollm \
  --prompt '给我讲一个故事' --max-new-tokens 64 --n-ctx 4096 \
  --threads 4 --temperature 0 --ssd-cache-mb 1024 \
  --ssd-io-workers 8 --ssd-policy moeengine-adaptive --warmup 0 --output kv
```

Pre-run state was AC power, 97% battery and charging, with `memory_pressure -Q`
reporting 83% free. Runs were deliberately recorded in execution order:

| Order | Build | Decode | TTFT | Peak RSS | Matmul time | Expert wait |
|---:|---|---:|---:|---:|---:|---:|
| 1 | portable, profile enabled | **1.88 tok/s** | 3.78 s | 13.96 GB | 16.16 s | 14.07 s |
| 2 | `-mcpu=native` control | 1.49 tok/s | 3.89 s | 14.53 GB | 26.77 s | 10.13 s |
| 3 | portable | 1.03 tok/s | 5.98 s | 14.65 GB | 50.22 s | 3.67 s |

The monotonic collapse in matrix throughput while expert wait decreased shows
that the fanless host entered a strongly compute-throttled state. This A/B/A
sequence cannot estimate a portable-versus-native speed delta. It does show:

1. The portable runtime passes the complete native test suite.
2. It loads and generates from the full native DSV4 package.
3. Its first cool-order result is in the same range as the historical
   `moeengine-adaptive` result (1.84 tok/s), so there is no observed catastrophic
   dependency on `-mcpu=native` or Metal.

It does **not** show that the portable build is faster. A defensible speed
comparison requires randomized cool-start runs or active cooling, and a second
non-Apple host is required before claiming cross-hardware portability.

## Native generic optimizer run

After the portable feasibility A/B/A, a standard-C++ optimizer contract was
added and wired to the real runtime as `moeengine-generic`. The policy receives
only exact route groups, expert bytes, measured load-to-ready latency, access
history, and safety flags.

```bash
vendor/Palm-Infra/build_portable/mollm_bench \
  --package artifacts/deepseek-v4-flash-0731/mollm/deepseek_v4_flash_native.mollm \
  --prompt '给我讲一个故事' --max-new-tokens 64 --n-ctx 4096 \
  --threads 4 --temperature 0 --ssd-cache-mb 1024 \
  --ssd-io-workers 8 --ssd-policy moeengine-generic --warmup 0 \
  --profile --output kv
```

| Policy/build | Decode | TTFT | Peak RSS | Matmul time | Expert wait |
|---|---:|---:|---:|---:|---:|
| generic / portable | **1.94 tok/s** | 3.54 s | 13.96 GB | 16.61 s | 12.49 s |
| adaptive / portable, earlier cool-order run | 1.88 tok/s | 3.78 s | 13.96 GB | 16.16 s | 14.07 s |

The generated text was identical. Generic loaded more logical expert bytes
(195.81 GB versus 190.41 GB) and recorded fewer application cache hits, so the
single 3.2% throughput difference is not evidence of a better policy. It is
within the host's observed thermal/run-order variance. The defensible result is
that the newly separated optimizer runs the complete 156.7 GB model at the
same performance class without an M5-native compile target or Metal.

The Python and C++ implementations intentionally share the data model and
heuristic but are separate code for hot-path reasons. A stable C ABI is still
needed before Swift/Rust runtimes can consume the same native implementation.
