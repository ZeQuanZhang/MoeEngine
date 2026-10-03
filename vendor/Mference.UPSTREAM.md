# Mference source snapshot

- Upstream named by the local README: https://github.com/NeelM0906/Mference
- Exact upstream commit: not recorded in the available source snapshot.
- Licenses: MIT, with Apache-2.0 derived portions and the MIT MLX notices.
- Dependency versions and revisions: `Mference/Package.resolved`.

The local research directory has no Git metadata. This distribution preserves
the complete runtime, Metal kernels, installer, CLI, server, app, resources,
tests, scripts and source documentation as a snapshot. It does not assert that
all of this code was authored by MoeEngine, or that the snapshot is identical to
a particular upstream commit. Keep `LICENSE`, `LICENSE-APACHE`, `LICENSE-MLX`,
and `THIRD_PARTY_NOTICES.md` with the source.

Swift build caches, scratch/install state, generated benchmark output, and
machine-local Swift package mirror configuration are omitted. The eleven local
Swift dependency copies had clean Git status and matched the revisions in
`Package.resolved`; they are restored by Swift Package Manager from the original
upstreams instead of being duplicated in this repository. No weights are included.
