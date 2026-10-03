# Third-party source and backend modifications

MoeEngine includes source snapshots and optional backend patches. The root MIT
license covers this project's original code; it does not replace the licenses
or copyright notices of the following projects.

| Component | Distribution | Upstream baseline | License |
| --- | --- | --- | --- |
| Palm-Infra / mollm | Modified source in `vendor/Palm-Infra` | `051305b467b785b19fb16a90ed6e3604fb86f773` | Apache-2.0; third-party MIT notice for nlohmann/json |
| Mference | Complete local source snapshot in `vendor/Mference` | Exact commit not recorded | MIT; Apache-2.0 derived portions; MLX MIT notices |
| llama.cpp | Fixed revision plus `patches/llama.cpp.patch` | `4e97ac86ebe2c4cb8212d98d2641ad6768810896` (`b10666`) | MIT |
| deepseek-v4-flash-0731-in-c | Fixed revision plus matching patch | `23dd88d6f439883c942b09395a2fad6105cd5470` (`v0.1.1`) | Apache-2.0 |

Source provenance is recorded in `vendor/backends.lock.json`,
`vendor/Palm-Infra.UPSTREAM.md`, and `vendor/Mference.UPSTREAM.md`. The llama.cpp
research checkout used a GitCode mirror; prepared releases fetch the same pinned
commit from the canonical ggml-org repository.

## Preserved notices

- Palm-Infra: `vendor/Palm-Infra/LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`.
- Mference: `vendor/Mference/LICENSE`, `LICENSE-APACHE`, `LICENSE-MLX`, and
  `THIRD_PARTY_NOTICES.md`. These describe derivation from TurboFieldfare and MLX.
- llama.cpp: `patches/licenses/llama.cpp/LICENSE`; the prepared upstream tree
  also retains its bundled third-party license directory and notices.
- DeepSeek C runtime: `patches/licenses/deepseek-v4-flash-0731-in-c/LICENSE` and
  `NOTICE`, including its kimi-k3-in-c provenance.

The C/C++ patches retain the corresponding upstream license. The llama.cpp
patch adds opt-in file-backed expert cache behavior, lazy expert-bank loading,
and benchmark/frontend integration. The DeepSeek C patch adapts automatic
memory detection and peak-RSS reporting to macOS. The Palm-Infra snapshot
contains the native MoeEngine optimizer and its integration; the standalone
`native/` build compiles those same source files rather than duplicating them.

Mference's eleven Swift dependencies are resolved using its checked-in lockfile.
Their license summary is in its `THIRD_PARTY_NOTICES.md`; binary distributors
must carry the applicable license and NOTICE files from the resolved packages.

## Model weights and generated files

No downloaded checkpoints, repacked weights, build outputs, dependency caches,
or nested Git metadata are distributed here. Model weights retain their source
licenses and must be obtained separately. Upstream tokenizer fixtures fetched
by `tools/prepare_backends.py` remain part of the optional ignored source tree;
that command does not download an inference checkpoint.
