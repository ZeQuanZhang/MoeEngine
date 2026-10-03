"""Read routed experts directly from an indexed safetensors checkpoint."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .checkpoint import ExpertBundlePlan, SafetensorEntry, plan_checkpoint
from .checkpoint import read_safetensors_header
from .model import ExpertAddress, ExpertDescriptor, LoadedExpert
from .store import ExpertStore


# Darwin's F_NOCACHE disables data caching for a file descriptor.  It is useful
# for measuring storage reads without requiring destructive system-wide cache
# purges.  The experiment is intentionally macOS-specific.
_DARWIN_F_NOCACHE = 48


@dataclass(frozen=True, slots=True)
class ExpertTensorRange:
    """One tensor component belonging to a routed expert."""

    component: str
    tensor_name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SafetensorsExpertLayout:
    """The source shard and six physical ranges for one expert bundle."""

    address: ExpertAddress
    shard: str
    tensors: tuple[ExpertTensorRange, ...]

    @property
    def size_bytes(self) -> int:
        return sum(tensor.size_bytes for tensor in self.tensors)


class SafetensorsExpertStore(ExpertStore):
    """A read-only ExpertStore over an official safetensors checkpoint.

    The loaded payload is the six expert tensors concatenated in the extraction
    plan's stable component order.  This is sufficient for real I/O and cache
    experiments.  A compute converter will later add alignment and a manifest
    describing the internal component offsets.
    """

    def __init__(
        self,
        root: str | Path,
        layouts: Iterable[SafetensorsExpertLayout],
        *,
        no_os_cache: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.no_os_cache = no_os_cache
        if no_os_cache and sys.platform != "darwin":
            raise ValueError("no_os_cache is supported only on macOS")
        self._layouts: dict[ExpertAddress, SafetensorsExpertLayout] = {}
        self._file_descriptors: dict[Path, int] = {}
        for layout in layouts:
            if layout.address in self._layouts:
                raise ValueError(f"duplicate expert layout for {layout.address}")
            if len(layout.tensors) != 6:
                raise ValueError(
                    f"expert {layout.address} requires six tensors, "
                    f"got {len(layout.tensors)}"
                )
            self._layouts[layout.address] = layout
        if not self._layouts:
            raise ValueError("at least one expert layout is required")

    @classmethod
    def from_checkpoint(
        cls,
        root: str | Path,
        *,
        layer_ids: Iterable[int] | None = None,
        no_os_cache: bool = False,
    ) -> "SafetensorsExpertStore":
        root = Path(root).resolve()
        selected_layers = set(layer_ids) if layer_ids is not None else None
        if selected_layers is not None and any(layer_id < 0 for layer_id in selected_layers):
            raise ValueError("layer_ids must be non-negative")
        bundles = [
            bundle
            for bundle in plan_checkpoint(root / "model.safetensors.index.json")
            if selected_layers is None or bundle.group_id in selected_layers
        ]
        if not bundles:
            raise ValueError("checkpoint contains no experts for the selected layers")

        headers: dict[str, dict[str, SafetensorEntry]] = {}
        layouts = []
        for bundle in bundles:
            shard_path = cls._resolve_shard(root, bundle.source_shard)
            header = headers.get(bundle.source_shard)
            if header is None:
                header = read_safetensors_header(shard_path)
                headers[bundle.source_shard] = header
            layouts.append(cls._layout_from_bundle(bundle, header))
        return cls(root, layouts, no_os_cache=no_os_cache)

    @staticmethod
    def _resolve_shard(root: Path, relative_shard: str) -> Path:
        path = (root / relative_shard).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"checkpoint shard escapes root: {relative_shard}") from error
        return path

    @staticmethod
    def _layout_from_bundle(
        bundle: ExpertBundlePlan,
        header: dict[str, SafetensorEntry],
    ) -> SafetensorsExpertLayout:
        tensors = []
        for tensor_source in bundle.tensors:
            try:
                entry = header[tensor_source.tensor_name]
            except KeyError as error:
                raise ValueError(
                    f"index tensor {tensor_source.tensor_name!r} is absent from "
                    f"{bundle.source_shard}"
                ) from error
            tensors.append(
                ExpertTensorRange(
                    component=tensor_source.component,
                    tensor_name=tensor_source.tensor_name,
                    dtype=entry.dtype,
                    shape=entry.shape,
                    offset=entry.offset,
                    size_bytes=entry.size_bytes,
                )
            )
        return SafetensorsExpertLayout(
            address=ExpertAddress(bundle.group_id, bundle.expert_id),
            shard=bundle.source_shard,
            tensors=tuple(tensors),
        )

    @property
    def addresses(self) -> frozenset[ExpertAddress]:
        return frozenset(self._layouts)

    def layout(self, address: ExpertAddress) -> SafetensorsExpertLayout:
        try:
            return self._layouts[address]
        except KeyError as error:
            raise KeyError(f"unknown expert address {address}") from error

    def describe(self, address: ExpertAddress) -> ExpertDescriptor:
        layout = self.layout(address)
        return ExpertDescriptor(
            address=address,
            size_bytes=layout.size_bytes,
            relative_file=layout.shard,
        )

    def _open(self, relative_shard: str) -> int:
        path = self._resolve_shard(self.root, relative_shard)
        file_descriptor = self._file_descriptors.get(path)
        if file_descriptor is None:
            file_descriptor = os.open(path, os.O_RDONLY)
            if self.no_os_cache:
                import fcntl

                fcntl.fcntl(file_descriptor, _DARWIN_F_NOCACHE, 1)
            self._file_descriptors[path] = file_descriptor
        return file_descriptor

    def load(self, address: ExpertAddress) -> LoadedExpert:
        layout = self.layout(address)
        file_descriptor = self._open(layout.shard)
        started_ns = time.perf_counter_ns()
        parts = []
        for tensor in layout.tensors:
            part = os.pread(file_descriptor, tensor.size_bytes, tensor.offset)
            if len(part) != tensor.size_bytes:
                raise EOFError(
                    f"short read for {tensor.tensor_name}: "
                    f"expected {tensor.size_bytes}, got {len(part)}"
                )
            parts.append(part)
        payload = b"".join(parts)
        load_latency_ns = time.perf_counter_ns() - started_ns
        descriptor = self.describe(address)
        return LoadedExpert(
            descriptor=descriptor,
            load_latency_ns=load_latency_ns,
            payload=payload,
        )

    def close(self) -> None:
        for file_descriptor in self._file_descriptors.values():
            os.close(file_descriptor)
        self._file_descriptors.clear()
