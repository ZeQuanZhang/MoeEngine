"""Expert storage backends.

The synthetic backend is metadata-only and is used for large cache sweeps.
The packed backend performs real offset-based reads and verifies optional
checksums, which is the storage contract needed by a future Metal backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from .model import ExpertAddress, ExpertDescriptor, LoadedExpert


class ExpertStore(ABC):
    """Abstract cold store for immutable expert bundles."""

    @abstractmethod
    def describe(self, address: ExpertAddress) -> ExpertDescriptor:
        """Return physical metadata without loading the expert."""

    @abstractmethod
    def load(self, address: ExpertAddress) -> LoadedExpert:
        """Load one complete expert bundle."""

    def close(self) -> None:
        """Release backend resources."""

    def __enter__(self) -> "ExpertStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class SyntheticExpertStore(ExpertStore):
    """A deterministic store that models I/O without allocating payloads."""

    def __init__(
        self,
        *,
        num_layers: int,
        experts_per_layer: int,
        expert_size_bytes: int,
        bandwidth_bytes_per_second: float = 5_000_000_000.0,
        fixed_latency_ns: int = 80_000,
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if experts_per_layer <= 0:
            raise ValueError("experts_per_layer must be positive")
        if expert_size_bytes <= 0:
            raise ValueError("expert_size_bytes must be positive")
        if bandwidth_bytes_per_second <= 0:
            raise ValueError("bandwidth_bytes_per_second must be positive")
        if fixed_latency_ns < 0:
            raise ValueError("fixed_latency_ns must be non-negative")
        self.num_layers = num_layers
        self.experts_per_layer = experts_per_layer
        self.expert_size_bytes = expert_size_bytes
        self.bandwidth_bytes_per_second = bandwidth_bytes_per_second
        self.fixed_latency_ns = fixed_latency_ns

    def _validate(self, address: ExpertAddress) -> None:
        if address.layer_id >= self.num_layers:
            raise KeyError(f"unknown layer in address {address}")
        if address.expert_id >= self.experts_per_layer:
            raise KeyError(f"unknown expert in address {address}")

    def describe(self, address: ExpertAddress) -> ExpertDescriptor:
        self._validate(address)
        return ExpertDescriptor(address=address, size_bytes=self.expert_size_bytes)

    def load(self, address: ExpertAddress) -> LoadedExpert:
        descriptor = self.describe(address)
        transfer_ns = round(
            descriptor.size_bytes / self.bandwidth_bytes_per_second * 1_000_000_000
        )
        return LoadedExpert(
            descriptor=descriptor,
            load_latency_ns=self.fixed_latency_ns + transfer_ns,
        )


class PackedExpertStore(ExpertStore):
    """Offset-addressed expert store backed by one or more regular files."""

    MANIFEST_VERSION = 1

    def __init__(
        self,
        root: str | Path,
        descriptors: Iterable[ExpertDescriptor],
        *,
        verify_checksums: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.verify_checksums = verify_checksums
        self._descriptors: dict[ExpertAddress, ExpertDescriptor] = {}
        self._file_descriptors: dict[Path, int] = {}
        for descriptor in descriptors:
            if descriptor.relative_file is None:
                raise ValueError("packed descriptors require relative_file")
            if descriptor.address in self._descriptors:
                raise ValueError(f"duplicate descriptor for {descriptor.address}")
            self._descriptors[descriptor.address] = descriptor

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        verify_checksums: bool = False,
    ) -> "PackedExpertStore":
        manifest_path = Path(manifest_path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if data.get("version") != cls.MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version: {data.get('version')!r}")
        descriptors = [
            ExpertDescriptor(
                address=ExpertAddress(item["layer_id"], item["expert_id"]),
                relative_file=item["file"],
                offset=item["offset"],
                size_bytes=item["size_bytes"],
                sha256=item.get("sha256"),
            )
            for item in data["experts"]
        ]
        return cls(
            manifest_path.parent,
            descriptors,
            verify_checksums=verify_checksums,
        )

    @staticmethod
    def write_manifest(
        manifest_path: str | Path,
        descriptors: Iterable[ExpertDescriptor],
    ) -> None:
        items = []
        for descriptor in sorted(descriptors, key=lambda item: item.address):
            if descriptor.relative_file is None:
                raise ValueError("packed descriptors require relative_file")
            item: dict[str, object] = {
                "layer_id": descriptor.address.layer_id,
                "expert_id": descriptor.address.expert_id,
                "file": descriptor.relative_file,
                "offset": descriptor.offset,
                "size_bytes": descriptor.size_bytes,
            }
            if descriptor.sha256 is not None:
                item["sha256"] = descriptor.sha256
            items.append(item)
        payload = {"version": PackedExpertStore.MANIFEST_VERSION, "experts": items}
        Path(manifest_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def describe(self, address: ExpertAddress) -> ExpertDescriptor:
        try:
            return self._descriptors[address]
        except KeyError as error:
            raise KeyError(f"unknown expert address {address}") from error

    def _open(self, relative_file: str) -> int:
        path = (self.root / relative_file).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"expert file escapes store root: {relative_file}") from error
        file_descriptor = self._file_descriptors.get(path)
        if file_descriptor is None:
            file_descriptor = os.open(path, os.O_RDONLY)
            self._file_descriptors[path] = file_descriptor
        return file_descriptor

    def load(self, address: ExpertAddress) -> LoadedExpert:
        descriptor = self.describe(address)
        assert descriptor.relative_file is not None
        file_descriptor = self._open(descriptor.relative_file)
        started_ns = time.perf_counter_ns()
        payload = os.pread(file_descriptor, descriptor.size_bytes, descriptor.offset)
        latency_ns = time.perf_counter_ns() - started_ns
        if len(payload) != descriptor.size_bytes:
            raise EOFError(
                f"short read for {address}: expected {descriptor.size_bytes}, got {len(payload)}"
            )
        if self.verify_checksums and descriptor.sha256 is not None:
            actual = hashlib.sha256(payload).hexdigest()
            if actual != descriptor.sha256:
                raise ValueError(
                    f"checksum mismatch for {address}: expected {descriptor.sha256}, got {actual}"
                )
        return LoadedExpert(
            descriptor=descriptor,
            load_latency_ns=latency_ns,
            payload=payload,
        )

    def close(self) -> None:
        for file_descriptor in self._file_descriptors.values():
            os.close(file_descriptor)
        self._file_descriptors.clear()

