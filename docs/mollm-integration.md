# mollm integration

This integration uses the source snapshot in `vendor/Palm-Infra`. Upstream
provenance and licenses are recorded in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Model weights and converted
packages are not included. Run commands from the repository root.

## Responsibility boundary

mollm remains the compute plane: it owns the DSV4 graph, router, attention,
native FP8/MXFP4 kernels, and text generation. MoeEngine is the expert control
plane: it selects admission, replacement, retention, and eventually prefetch
policies. The package and numerical kernels are held constant in every policy
A/B.

The current bridge is intentionally in-process C++. Python remains the
experiment CLI and analysis layer; it is not called for individual expert
loads.

```text
router top-k
    |
    v
MoeSsdCache request/acquire
    |
    +-- native
    +-- moeengine-lru
    +-- moeengine-lfu
    +-- moeengine-adaptive
    +-- moeengine-generic
    +-- moeengine-gdsf
    |
    v
pread native expert bytes -> MXFP4 compute
```

On the CPU decode path, the selected experts in one layer all feed their
gate/up projections from the same graph-owned hidden tensor. Its `storage_id`
is now preserved through both the ready-batch and streaming paths, allowing
MXFP4 to cache the dual Q8 activation sidecar across separate expert
dispatches. The cache is invalidated at the start of every graph execution;
temporary gather/intermediate buffers have no storage identity and remain
uncached. This removes repeated activation quantization without retaining
expert weights or changing numerical output.

Single-expert MXFP4 GEMVs also use one static, equal output-row shard per
worker. Output rows have identical K work, so this avoids dozens of dynamic
queue claims per expert without changing the accumulation order. Multi-expert
batches retain dynamic two-dimensional scheduling because their expert/output
tiles can finish at different times.

Select a policy with:

```bash
--ssd-policy native|moeengine-lru|moeengine-lfu|moeengine-adaptive|moeengine-generic|moeengine-gdsf
```

The policy must be selected before the first expert I/O. The cache rejects a
runtime policy change once an entry exists. This keeps residency accounting,
statistics, and asynchronous request ordering internally consistent.

`moeengine-adaptive` measures actual overlap between consecutive token routes.
After an eight-sample observation window, it assigns each layer zero to three
retention slots from an exponentially weighted overlap estimate. Unlike a hit
counter, this signal remains observable when the cache was too small to retain
the repeated expert.

`moeengine-generic` delegates replacement to the public, standard-C++
`moeengine::ResidencyOptimizer`. The runtime supplies exact route groups, expert
size, measured load-to-ready latency, access history, and current safety state.
The optimizer contains no Apple, Metal, CPU-instruction, or model-family
branches. Its retention budget is learned from expected route reuse rather
than a fixed M5 memory profile.

`moeengine-gdsf` ranks admission by expected reload penalty per byte and uses a
global GreedyDual clock plus bounded ghost/refault metadata for aging. A single
refault is confidence-limited, avoiding random-route cache pollution. Exact
current demand is non-evictable; unconsumed current-forward predictions are
then protected in confidence order so long-term hit-rate decisions do not
delay an imminent expert.

The cache also tracks a separate `demand_epoch` for entries named by the real
router in the active layer. This is deliberately stronger than the ordinary
forward epoch shared by speculative future entries. Native and adaptive victim
ordering sacrifice a future prediction before a guaranteed current-layer hit,
while retaining a soft fallback so extremely small streaming caches cannot
deadlock. This time-value ordering closes the case where a predicted next-layer
expert displaced a lower-ranked current hit that was then reloaded almost
immediately.

## Rebuild

Install CMake and a C++ compiler. For the Python test dependencies, follow the
[README](../README.md). A CPU build of the included backend is:

```bash
cmake -S vendor/Palm-Infra -B build/palm \
  -DCMAKE_BUILD_TYPE=Release \
  -DMOLLM_BUILD_TESTS=ON -DMOLLM_BUILD_EXAMPLE=ON
cmake --build build/palm --parallel 2
ctest --test-dir build/palm --output-on-failure
```

The build directory can be changed. Historical reports used
`vendor/Palm-Infra/build_moeengine`; replace that prefix with `build/palm` when
following their runtime commands. The standalone optimizer test in the README
is smaller and does not validate the full backend or model output.

For portable CPU validation, add `-DMOLLM_ARM_NATIVE=OFF` and
`-DMOLLM_ARM_I8MM=AUTO` to the configure command. Optional instructions are then
selected by runtime feature detection instead of compiling the whole binary
for the build machine's CPU.

The recorded M5 experiments used Apple Command Line Tools without the
standalone `metal` compiler. The experimental runtime-compilation path avoids
that build-time dependency: `MOLLM_METAL_RUNTIME_COMPILE=ON` copies the shader
source into the build tree and compiles it through `Metal.framework` when the
backend starts. This path deliberately targets the Metal 4 tensor kernels on
this M5. Because several quantized prefill kernels have no complete simdgroup
implementation, it fails closed on an older GPU instead of advertising an
unsafe partial fallback.

This optional Metal build requires macOS and compatible Apple GPU features:

```bash
cmake -B build/palm-metal \
  -S vendor/Palm-Infra \
  -DCMAKE_BUILD_TYPE=Release \
  -DMOLLM_ARM_NATIVE=ON \
  -DMOLLM_METAL=ON \
  -DMOLLM_METAL_RUNTIME_COMPILE=ON \
  -DMOLLM_BUILD_TESTS=ON -DMOLLM_BUILD_EXAMPLE=ON
cmake --build build/palm-metal --parallel 2
ctest --test-dir build/palm-metal --output-on-failure
```

The runtime reads the CMake-copied `mollm.metal` and `metal_common.h`. For a
relocated build, override them with `MOLLM_METAL_SOURCE` and
`MOLLM_METAL_COMMON`. Runtime compilation happens once per process; subsequent
pipeline requests are cached by the backend.

## Conversion safety gate

For the checkpoint used in the recorded experiment, the converted package was
145.93 GiB. Before attempting conversion, activate the Python environment and
run the estimator against your local checkpoint:

```bash
python tools/estimate_mollm_dsv4.py \
  /path/to/models/DeepSeek-V4-Flash-0731 \
  artifacts/deepseek-v4-flash-0731/mollm/deepseek_v4_flash_native.mollm
```

The estimator validates the 48 indexed shards, builds the same prefill/decode
graphs, computes the streamed weight byte count, and refuses to proceed when
the target filesystem cannot retain the configured free-space reserve.
