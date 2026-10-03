"""Stable value types shared by stores, caches, and policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True, slots=True)
class ExpertAddress:
    """Logical address of one routed expert inside a model."""

    layer_id: int
    expert_id: int

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if self.expert_id < 0:
            raise ValueError("expert_id must be non-negative")

    def __str__(self) -> str:
        return f"L{self.layer_id}:E{self.expert_id}"


@dataclass(frozen=True, slots=True)
class ExpertDescriptor:
    """Physical location and integrity metadata for one expert bundle."""

    address: ExpertAddress
    size_bytes: int
    relative_file: str | None = None
    offset: int = 0
    sha256: str | None = None

    def __post_init__(self) -> None:
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.relative_file is not None and not self.relative_file:
            raise ValueError("relative_file must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class LoadedExpert:
    """One store load, optionally carrying materialized bytes.

    Synthetic stores omit ``payload`` so multi-gigabyte cache simulations do
    not actually allocate multi-gigabyte buffers.
    """

    descriptor: ExpertDescriptor
    load_latency_ns: int
    payload: bytes | None = None

    def __post_init__(self) -> None:
        if self.load_latency_ns < 0:
            raise ValueError("load_latency_ns must be non-negative")
        if self.payload is not None and len(self.payload) != self.descriptor.size_bytes:
            raise ValueError("payload length does not match descriptor size")

