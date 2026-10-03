# Palm-Infra source snapshot

- Upstream: https://github.com/TencentYoutuResearch/Palm-Infra
- Baseline commit: `051305b467b785b19fb16a90ed6e3604fb86f773`
- Baseline commit date: `2026-08-23T10:59:09Z`
- Downloaded archive SHA-256: `6b6435e8b2e7b8ce5ff07c76b311649a1883dc3cf9c21b61f58eeb1b5025b3a0`
- License: Apache-2.0; retain `Palm-Infra/LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`.

The research checkout was obtained as an archive and has no Git metadata. This
repository therefore preserves its complete local source snapshot rather than
claiming that a reconstructed patch captures every local change.

The local implementation includes expert eviction policies, a C++ residency
optimizer and Router-GDSF, SSD loading and eviction integration, CLI wiring,
Metal runtime support, and focused behavior tests. The source is modified from
the recorded upstream baseline. Project-specific identifiers, filenames,
policies and environment variables use the MoeEngine name in this distribution.

Only generated CMake build directories, Python bytecode/cache directories,
local metadata, and benchmark output were omitted. Model conversion scripts
under `Palm-Infra/models/` are source code and remain included. No model weights
are included. The original copyright and attribution files are preserved.
