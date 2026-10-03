# Qwen3.8 Flash Next × MoeEngine real-runtime experiment

Date: 2026-08-28 (Asia/Shanghai)

This is a dated experiment record. Use the [README](../README.md) for current
installation and [third-party notes](../THIRD_PARTY_NOTICES.md) for the pinned
llama.cpp base and patch. The GGUF model files are not included.

## Scope

This experiment answers a narrow question: can MoeEngine manage the routed
experts of a real split GGUF MoE checkpoint during end-to-end inference on a
24 GB MacBook Air, without using swap as model memory?

Hardware and model:

- MacBook Air `Mac17,3`, Apple M5, 10 CPU cores, 24 GB unified memory.
- Qwen3.8 Flash Next UD IQ4_XS, GGUF V3, 87.24 GiB, 176.94B total / A3B.
- 48 layers, 512 routed experts per layer, top-10 routing.
- Consolidated gate/up/down expert banks total 55.432 GiB.
- llama.cpp commit `4e97ac86ebe2c4cb8212d98d2641ad6768810896`.
- CPU decode with Accelerate selected (`-dev BLAS`), 10 threads, zero GPU
  layers, mmap load, architecture-marked tensors read lazily.

The model files stayed in the normal GGUF format. MoeEngine did not rewrite,
convert, or duplicate the checkpoint.

## What was integrated

The GGUF reader proves that one expert is three exact contiguous file ranges.
For this checkpoint one layer/expert payload is 2,329,600 bytes. It can cross a
split-file boundary; layer 14 was verified to read from shards 2 and 3 without
materializing its complete bank.

The llama.cpp prototype intercepts the real CPU `MUL_MAT_ID` operation after
the router has produced expert ids. For each file-backed expert bank it:

1. protects the experts selected by the current router call;
2. issues `POSIX_MADV_WILLNEED` only for logical misses;
3. tracks frequency-biased recency;
4. issues `POSIX_MADV_DONTNEED` only for interior pages of evicted, clean GGUF
   slices;
5. refuses to operate on anonymous or CPU-repacked tensors.

The policy is model-agnostic at the operator boundary. The loader recognizes
llama.cpp's standard merged `FFN_*_EXPS` tensor roles and marks them lazy only
when MoeEngine is enabled; there is no Qwen-specific cache-policy branch.

## Exact expert I/O checks

Using `GgufExpertStore` with macOS `F_NOCACHE`:

| Test | Payload read | Cache result | Wall storage throughput |
| --- | ---: | ---: | ---: |
| Layer 0, 32 experts × 3 rounds | 74,547,200 B loaded | 66.67% logical hit | 2.89 GiB/s |
| Layer 14, 16 experts × 2 rounds | 37,273,600 B loaded | 50.00% logical hit | 1.86 GiB/s |

These are storage/cache tests, not token-generation throughput.

## End-to-end decode results

All rows used the same model, device selection, mmap mode, 10 threads, 32
decode tokens and no warmup. `reset=on` gives each process one clean expert-page
discard at bank registration, including the baseline. All three repetitions
share one process and accumulate cache history; they are not three independent
cold starts. Hit counters are logical policy counters, and throughput is the
mean and sample standard deviation across the three repetitions.

| Residency policy | Nominal expert-page budget | Logical hit after 32 tokens | Logical hit after 96 tokens | `tg32`, 3 reps |
| --- | ---: | ---: | ---: | ---: |
| macOS-managed mmap, no MoeEngine eviction | unbounded | n/a | n/a | **2.26 ± 0.24 tok/s** |
| MoeEngine, 24 slots/bank | 2.60 GiB | 62.57% | 66.05% | **0.83 ± 0.50 tok/s** |
| MoeEngine, 96 slots/bank | 10.39 GiB | 73.07% | 81.43% | **2.24 ± 0.15 tok/s** |

The nominal cap is `55.432 GiB × slots / 512`. It excludes shared boundary
pages, kernel buffers, dense weights, KV/recurrent state, and OS read-ahead, so
it is a policy budget rather than an exact process-RSS limit. Current required
experts remain protected even if a call exceeds the configured slot count.
Page advice is advisory, so the logical budget does not enforce a physical
resident-page limit either.

An earlier single 32-token run with 24 slots reached 2.73 tok/s versus a
single baseline of 2.33 tok/s, but the three-repetition result exposed severe
re-read thrashing. The repeat result is the one that should guide decisions.

Conclusion from this test: **the 96-slot logical budget produced a three-repeat
mean close to the OS baseline (2.24 versus 2.26 tok/s). This small sample does
not establish throughput equivalence or a speedup.** The 24-slot/roughly-2.6-GiB
configuration suffered repeated reads on this route working set.

## Correctness smoke test

The integrated runtime generated from `The capital of France is`:

> Paris. The capital of Germany is Berlin. The capital of

For 12 generated tokens, the simple runner reported:

- prompt evaluation: 1.77 tok/s for 5 prompt tokens;
- decode-only evaluation: 3.63 tok/s for 11 measured decode steps;
- end-to-end generation section: 2.05 tok/s including the first prompt pass.

The output and unchanged graph math establish a useful smoke test, not a
quality evaluation of the quantization.

## Swap and storage safety

Across the controlled CPU A/B and correctness runs, macOS `vm_stat` stayed at
5,410,984 swapouts: **measured swapout delta = 0 pages / 0 bytes**. MoeEngine
still reads GGUF pages from storage. This counter establishes no new system
swapout during the observed interval; it does not establish a per-process
physical-memory bound or predict swap behavior under another workload.

A separate high-layer Metal offload attempt is not part of the valid result. It
grew swapouts by 1,088,870 16-KiB pages (about 17.0 GiB of writes) and was
stopped immediately. That configuration is excluded from the zero-swap
CPU result and needs separate memory-budget work before further evaluation.

## Objective assessment

What this proves:

- the real Qwen4Exp GGUF can be split logically at exact expert boundaries;
- router ids can drive live expert-page residency in the inference kernel;
- an 87.24-GiB checkpoint can produce coherent text on this 24-GB Air with
  measured zero swapout during the controlled CPU runs;
- the 96-slot configuration produced a mean near the OS baseline in this test.

What this does not prove:

- a Metal speedup or superiority over llama.cpp/macOS caching;
- a hard physical-RAM cap enforced below the VM subsystem;
- benefits for dense models, remote API models, or every MoE route pattern;
- production reliability or long-context/concurrent-session performance.

A proposed next experiment is an active-expert **repacked compute cache**:
keep GGUF as the backing store and repack resident experts into a bounded
compute arena. This is not implemented by the page-residency experiment above.
It would need its own numerical checks, peak-memory accounting, and repeated
benchmark before any benefit could be claimed.

## Reproduction

Install `.[gguf]` as described in the README, prepare the pinned backend, and
build it on the target macOS host. The following uses a CPU build with
Accelerate/BLAS enabled; it does not download the model:

```bash
python -m unittest discover -s tests -v
python3 tools/prepare_backends.py llama.cpp
cmake -S vendor/llama.cpp -B vendor/llama.cpp/build-moeengine \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_BLAS=ON
cmake --build vendor/llama.cpp/build-moeengine --parallel 2
```

With the compatible split GGUF files available locally:

```bash
MOEENGINE_MIN_CTX=1024 \
MOEENGINE_EXPERT_CACHE_SLOTS=96 \
MOEENGINE_EXPERT_CACHE_REPORT=4608 \
MOEENGINE_EXPERT_CACHE_RESET=1 \
vendor/llama.cpp/build-moeengine/bin/llama-bench \
  -m artifacts/Qwen3.8-Flash-Next-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  -p 0 -n 32 -r 3 --no-warmup \
  -t 10 -ngl 0 -dev BLAS -lm mmap --progress
```

Use `MOEENGINE_EXPERT_CACHE_SLOTS=0` with the variable still present for the
same file-backed expert layout without MoeEngine prefetch/eviction. Do not turn
on `MOEENGINE_EXPERT_CACHE_MINCORE` for throughput comparisons.
