"""Read routed experts directly from split GGUF model files.

GGUF stores MoE expert banks as three-dimensional tensors whose outermost
physical dimension is the expert id.  When that dimension is contiguous, one
expert can be loaded with one range read per projection instead of mapping or
copying the complete layer-wide expert bank.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .model import ExpertAddress, ExpertDescriptor, LoadedExpert
from .store import ExpertStore


_DARWIN_F_NOCACHE = 48
_EXPERT_TENSOR = re.compile(
    r"^blk\.(?P<layer>\d+)\.ffn_(?P<component>down|gate|up)_exps\.weight$"
)
_COMPONENT_ORDER = {"gate": 0, "up": 1, "down": 2}


@dataclass(frozen=True, slots=True)
class GgufExpertTensorRange:
    """One contiguous projection slice for one routed expert."""

    component: str
    tensor_name: str
    tensor_type: int
    shape: tuple[int, ...]
    relative_file: str
    offset: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class GgufExpertLayout:
    """Physical GGUF ranges that form one routed expert."""

    address: ExpertAddress
    tensors: tuple[GgufExpertTensorRange, ...]

    @property
    def size_bytes(self) -> int:
        return sum(tensor.size_bytes for tensor in self.tensors)

    @property
    def source_files(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(tensor.relative_file for tensor in self.tensors))


class GgufExpertStore(ExpertStore):
    """A read-only, range-addressed store over routed GGUF expert banks."""

    def __init__(
        self,
        root: str | Path,
        layouts: Iterable[GgufExpertLayout],
        *,
        architecture: str,
        expert_count: int,
        expert_used_count: int,
        no_os_cache: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.architecture = architecture
        self.expert_count = expert_count
        self.expert_used_count = expert_used_count
        self.no_os_cache = no_os_cache
        if no_os_cache and sys.platform != "darwin":
            raise ValueError("no_os_cache is supported only on macOS")
        self._layouts: dict[ExpertAddress, GgufExpertLayout] = {}
        self._file_descriptors: dict[Path, int] = {}
        self._fallback_read_locks: dict[int, threading.Lock] = {}
        for layout in layouts:
            if layout.address in self._layouts:
                raise ValueError(f"duplicate expert layout for {layout.address}")
            components = {tensor.component for tensor in layout.tensors}
            if components != set(_COMPONENT_ORDER):
                raise ValueError(
                    f"expert {layout.address} requires gate/up/down tensors, "
                    f"got {sorted(components)}"
                )
            self._layouts[layout.address] = layout
        if not self._layouts:
            raise ValueError("at least one GGUF expert layout is required")

    @classmethod
    def from_model(
        cls,
        model: str | Path,
        *,
        layer_ids: Iterable[int] | None = None,
        no_os_cache: bool = False,
    ) -> "GgufExpertStore":
        try:
            from gguf import GGUFReader
        except ImportError as error:
            raise RuntimeError(
                "GGUF support requires the optional dependency; "
                "install MoeEngine with the 'gguf' extra"
            ) from error

        files, root = cls._discover_files(model)
        selected_layers = set(layer_ids) if layer_ids is not None else None
        if selected_layers is not None and any(layer_id < 0 for layer_id in selected_layers):
            raise ValueError("layer_ids must be non-negative")

        architecture: str | None = None
        expert_count: int | None = None
        expert_used_count: int | None = None
        banks: dict[tuple[int, str], GgufExpertTensorRange] = {}

        for path in files:
            reader = GGUFReader(str(path), "r")
            architecture_field = reader.get_field("general.architecture")
            if architecture_field is not None:
                shard_architecture = str(architecture_field.contents())
                shard_expert_count = int(
                    cls._required_field(reader, f"{shard_architecture}.expert_count")
                )
                shard_expert_used_count = int(
                    cls._required_field(
                        reader, f"{shard_architecture}.expert_used_count"
                    )
                )
                if architecture is None:
                    architecture = shard_architecture
                    expert_count = shard_expert_count
                    expert_used_count = shard_expert_used_count
                elif (
                    architecture != shard_architecture
                    or expert_count != shard_expert_count
                    or expert_used_count != shard_expert_used_count
                ):
                    raise ValueError(
                        "GGUF shards disagree on architecture or expert counts"
                    )

            relative_file = str(path.relative_to(root))
            for tensor in reader.tensors:
                match = _EXPERT_TENSOR.fullmatch(tensor.name)
                if match is None:
                    continue
                layer_id = int(match.group("layer"))
                if selected_layers is not None and layer_id not in selected_layers:
                    continue
                component = match.group("component")
                shape = tuple(int(value) for value in tensor.shape)
                if expert_count is None:
                    raise ValueError(
                        "GGUF expert tensors were encountered before any shard "
                        "supplied architecture metadata"
                    )
                if len(shape) != 3 or shape[-1] != expert_count:
                    raise ValueError(
                        f"{tensor.name} has unsupported expert layout {shape}; "
                        f"expected a three-dimensional tensor ending in {expert_count}"
                    )
                if tensor.n_bytes % expert_count:
                    raise ValueError(
                        f"{tensor.name} byte length is not divisible by expert count"
                    )
                slice_size = tensor.n_bytes // expert_count
                if (
                    tensor.data.shape[0] != expert_count
                    or tensor.data.strides[0] != slice_size
                    or not tensor.data.flags.c_contiguous
                ):
                    raise ValueError(
                        f"{tensor.name} does not contain contiguous expert slices"
                    )
                key = (layer_id, component)
                if key in banks:
                    raise ValueError(f"duplicate GGUF expert bank {tensor.name}")
                banks[key] = GgufExpertTensorRange(
                    component=component,
                    tensor_name=tensor.name,
                    tensor_type=int(tensor.tensor_type),
                    shape=shape,
                    relative_file=relative_file,
                    offset=int(tensor.data_offset),
                    size_bytes=slice_size,
                )

        if architecture is None or expert_count is None or expert_used_count is None:
            raise ValueError("GGUF model has no readable architecture metadata")

        layer_set = {layer_id for layer_id, _component in banks}
        if selected_layers is not None:
            missing = selected_layers - layer_set
            if missing:
                raise ValueError(f"GGUF model contains no experts for layers {sorted(missing)}")
        layouts = []
        for layer_id in sorted(layer_set):
            layer_banks = [
                bank for (bank_layer, _component), bank in banks.items()
                if bank_layer == layer_id
            ]
            components = {bank.component for bank in layer_banks}
            if components != set(_COMPONENT_ORDER):
                raise ValueError(
                    f"GGUF layer {layer_id} requires gate/up/down expert banks, "
                    f"got {sorted(components)}"
                )
            ordered_banks = sorted(
                layer_banks, key=lambda bank: _COMPONENT_ORDER[bank.component]
            )
            for expert_id in range(expert_count):
                tensors = tuple(
                    GgufExpertTensorRange(
                        component=bank.component,
                        tensor_name=bank.tensor_name,
                        tensor_type=bank.tensor_type,
                        shape=bank.shape,
                        relative_file=bank.relative_file,
                        offset=bank.offset + expert_id * bank.size_bytes,
                        size_bytes=bank.size_bytes,
                    )
                    for bank in ordered_banks
                )
                layouts.append(
                    GgufExpertLayout(
                        address=ExpertAddress(layer_id, expert_id),
                        tensors=tensors,
                    )
                )
        return cls(
            root,
            layouts,
            architecture=architecture,
            expert_count=expert_count,
            expert_used_count=expert_used_count,
            no_os_cache=no_os_cache,
        )

    @staticmethod
    def _required_field(reader: object, name: str) -> object:
        field = reader.get_field(name)  # type: ignore[attr-defined]
        if field is None:
            raise ValueError(f"GGUF metadata is missing {name!r}")
        return field.contents()

    @staticmethod
    def _discover_files(model: str | Path) -> tuple[list[Path], Path]:
        path = Path(model).resolve()
        if path.is_dir():
            root = path
            files = sorted(path.glob("*.gguf"))
        elif path.is_file() and path.suffix.lower() == ".gguf":
            root = path.parent
            split_match = re.match(r"^(.*)-\d{5}-of-\d{5}\.gguf$", path.name)
            files = (
                sorted(root.glob(f"{split_match.group(1)}-*-of-*.gguf"))
                if split_match is not None
                else [path]
            )
        else:
            raise ValueError(f"GGUF model path does not exist: {path}")
        if not files:
            raise ValueError(f"no GGUF files found in {root}")
        return files, root

    @property
    def addresses(self) -> frozenset[ExpertAddress]:
        return frozenset(self._layouts)

    def layout(self, address: ExpertAddress) -> GgufExpertLayout:
        try:
            return self._layouts[address]
        except KeyError as error:
            raise KeyError(f"unknown expert address {address}") from error

    def describe(self, address: ExpertAddress) -> ExpertDescriptor:
        layout = self.layout(address)
        relative_file = layout.source_files[0] if len(layout.source_files) == 1 else None
        return ExpertDescriptor(
            address=address,
            size_bytes=layout.size_bytes,
            relative_file=relative_file,
        )

    def _resolve_file(self, relative_file: str) -> Path:
        path = (self.root / relative_file).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"GGUF shard escapes model root: {relative_file}") from error
        return path

    def _open(self, relative_file: str) -> int:
        path = self._resolve_file(relative_file)
        file_descriptor = self._file_descriptors.get(path)
        if file_descriptor is None:
            file_descriptor = os.open(path, os.O_RDONLY)
            if self.no_os_cache:
                import fcntl

                fcntl.fcntl(file_descriptor, _DARWIN_F_NOCACHE, 1)
            self._file_descriptors[path] = file_descriptor
            self._fallback_read_locks[file_descriptor] = threading.Lock()
        return file_descriptor

    def _read_range(self, file_descriptor: int, size_bytes: int, offset: int) -> bytes:
        if hasattr(os, "pread"):
            return os.pread(file_descriptor, size_bytes, offset)
        lock = self._fallback_read_locks[file_descriptor]
        with lock:
            os.lseek(file_descriptor, offset, os.SEEK_SET)
            return os.read(file_descriptor, size_bytes)

    def load(self, address: ExpertAddress) -> LoadedExpert:
        layout = self.layout(address)
        started_ns = time.perf_counter_ns()
        parts = []
        for tensor in layout.tensors:
            file_descriptor = self._open(tensor.relative_file)
            part = self._read_range(file_descriptor, tensor.size_bytes, tensor.offset)
            if len(part) != tensor.size_bytes:
                raise EOFError(
                    f"short read for {tensor.tensor_name} expert {address.expert_id}: "
                    f"expected {tensor.size_bytes}, got {len(part)}"
                )
            parts.append(part)
        payload = b"".join(parts)
        return LoadedExpert(
            descriptor=self.describe(address),
            load_latency_ns=time.perf_counter_ns() - started_ns,
            payload=payload,
        )

    def close(self) -> None:
        for file_descriptor in self._file_descriptors.values():
            os.close(file_descriptor)
        self._file_descriptors.clear()
        self._fallback_read_locks.clear()
