# Historical benchmark archive

These reports and captured measurements come from experiments dated **2026-08-25**. Preparing the MoeEngine public release did not rerun the benchmarks, rebuild the historical binaries, or repeat the reported test suites.

Project names, identifiers, and command examples have been normalized to `MoeEngine`, `moeengine`, and `MOEENGINE`. Machine-specific paths use `/path/to/` placeholders. Measurement values, model and output hashes, system counters, and test-result counts are preserved. Historical binary hashes identify the measured binaries; they do not identify a fresh build of the renamed release.

These are observations for specific devices and configurations. The reports do not establish a cross-platform performance ranking. Read each report's hardware, precision, run-order, thermal, and swap conditions with its results.

## Reports and included evidence

| Experiment | Report | Included evidence |
| --- | --- | --- |
| DeepSeek low-memory and swap checks | [Report](dsv4-0731-m5air-0swap-2026-08-25/REPORT.md) | Captured stdout, stderr, and system-counter files in the [same directory](dsv4-0731-m5air-0swap-2026-08-25/) |
| Portable optimizer validation | [Report](generic-portable-2026-08-25/REPORT.md) | Report only |
| Model-independent policy changes | [Report](model-agnostic-2026-08-25/REPORT.md) | Report only |
| Qwen cache and swap checks | [Report](qwen36-m5air-0swap-2026-08-25/REPORT.md) | Report only |
| Router-GDSF validation | [Report](router-gdsf-2026-08-25/REPORT.md) | Report only |
| Sequential optimization experiments | [Report](sequential-optimization-2026-08-25/REPORT.md) | Report only |

## Paths in historical commands

The release includes curated [Palm-Infra sources](../vendor/Palm-Infra.UPSTREAM.md) and [Mference sources](../vendor/Mference.UPSTREAM.md). Their historical `build_*` / `.build*` outputs and Mference `benchmark-results/` directories are not included. Model files under `artifacts/` are also not included. See [third-party notices](../THIRD_PARTY_NOTICES.md) for the integration and provenance structure.

Commands retain the historical workspace layout and require separately built runtimes, model files, and path adjustments. Historical evidence paths identify original locations rather than files bundled with this release. Placeholder paths are examples, not download locations.

See the [project README](../README.md) for the public release's scope. The relative links in this index and each report's archive note refer to files or directories included in the release tree.
