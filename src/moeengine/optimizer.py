"""Hardware-independent residency planning for routed model weights.

The optimizer consumes measured costs and router observations.  It deliberately
does not detect CPUs, GPUs, operating systems, or model families.  Runtime
adapters are responsible for turning device capabilities into ``MemoryTier``
values and for executing the returned eviction plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from typing import Iterable

from .model import ExpertAddress


@dataclass(frozen=True, slots=True)
class MemoryTier:
    """One residency tier described only by observable transfer properties."""

    name: str
    capacity_bytes: int
    bandwidth_bytes_per_second: float
    fixed_latency_ns: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tier name must be non-empty")
        if self.capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if self.bandwidth_bytes_per_second <= 0:
            raise ValueError("bandwidth_bytes_per_second must be positive")
        if self.fixed_latency_ns < 0:
            raise ValueError("fixed_latency_ns must be non-negative")

    def transfer_cost_ns(self, size_bytes: int) -> int:
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        return self.fixed_latency_ns + round(
            size_bytes / self.bandwidth_bytes_per_second * 1_000_000_000
        )


@dataclass(frozen=True, slots=True)
class OptimizerWeights:
    """Tunable, platform-neutral weights used by the online cost model."""

    route_reuse: float = 0.60
    recency: float = 0.25
    frequency: float = 0.15

    def __post_init__(self) -> None:
        values = (self.route_reuse, self.recency, self.frequency)
        if any(value < 0 for value in values):
            raise ValueError("optimizer weights must be non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one optimizer weight must be positive")


@dataclass(frozen=True, slots=True)
class ResidentExpert:
    """Backend-neutral state required for one eviction decision."""

    address: ExpertAddress
    size_bytes: int
    last_access_index: int
    access_count: int
    reload_cost_ns: int
    reuse_probability: float = 0.0
    pinned: bool = False
    loading: bool = False
    required: bool = False

    def __post_init__(self) -> None:
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.last_access_index < 0:
            raise ValueError("last_access_index must be non-negative")
        if self.access_count <= 0:
            raise ValueError("access_count must be positive")
        if self.reload_cost_ns < 0:
            raise ValueError("reload_cost_ns must be non-negative")
        if not 0.0 <= self.reuse_probability <= 1.0:
            raise ValueError("reuse_probability must be between zero and one")


@dataclass(frozen=True, slots=True)
class ResidencyPlan:
    """Minimal set of evictions selected to make a requested byte budget."""

    evict: tuple[ExpertAddress, ...]
    freed_bytes: int
    shortfall_bytes: int
    estimated_reload_penalty_ns: float

    @property
    def feasible(self) -> bool:
        return self.shortfall_bytes == 0


class RouteLocalityEstimator:
    """Learn per-layer cross-step expert reuse from exact router selections."""

    def __init__(self, *, alpha: float = 0.125) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be between zero and one")
        self.alpha = alpha
        self._previous: dict[int, frozenset[ExpertAddress]] = {}
        self._probability: dict[ExpertAddress, float] = {}
        self._observed: dict[int, set[ExpertAddress]] = {}

    def observe(self, layer_id: int, experts: Iterable[ExpertAddress]) -> None:
        if layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        current = frozenset(experts)
        if not current:
            raise ValueError("a route must contain at least one expert")
        if any(address.layer_id != layer_id for address in current):
            raise ValueError("every routed expert must belong to layer_id")

        previous = self._previous.get(layer_id)
        if previous is not None:
            observed = self._observed.setdefault(layer_id, set())
            observed.update(previous)
            observed.update(current)
            for address in observed:
                # A first appearance is demand, not yet evidence of reuse.
                # Consecutive membership supplies an admission-resistant
                # probability signal for the following route.
                sample = 1.0 if address in previous and address in current else 0.0
                old = self._probability.get(address, sample)
                self._probability[address] = old + self.alpha * (sample - old)
        self._previous[layer_id] = current

    def probability(self, address: ExpertAddress) -> float:
        return self._probability.get(address, 0.0)


class GenericResidencyOptimizer:
    """Choose evictions by expected reload penalty per reclaimed byte.

    All hardware characteristics arrive as measurements.  The same class can
    therefore plan a CPU-RAM/NVMe cache, a GPU/host-RAM cache, or a distributed
    expert pool without containing platform branches.
    """

    def __init__(self, *, weights: OptimizerWeights | None = None) -> None:
        self.weights = weights or OptimizerWeights()

    def retention_value(
        self,
        candidate: ResidentExpert,
        *,
        access_index: int,
    ) -> float:
        if access_index < candidate.last_access_index:
            raise ValueError("access_index precedes a candidate's last access")
        age = access_index - candidate.last_access_index
        recency = 1.0 / (1.0 + age)
        frequency = min(1.0, log2(1.0 + candidate.access_count) / 8.0)
        weights = self.weights
        normalizer = weights.route_reuse + weights.recency + weights.frequency
        reuse_score = (
            weights.route_reuse * candidate.reuse_probability
            + weights.recency * recency
            + weights.frequency * frequency
        ) / normalizer
        return reuse_score * candidate.reload_cost_ns

    def plan_evictions(
        self,
        candidates: Iterable[ResidentExpert],
        *,
        bytes_to_free: int,
        access_index: int,
    ) -> ResidencyPlan:
        if bytes_to_free < 0:
            raise ValueError("bytes_to_free must be non-negative")
        if access_index < 0:
            raise ValueError("access_index must be non-negative")
        if bytes_to_free == 0:
            return ResidencyPlan((), 0, 0, 0.0)

        ranked: list[tuple[float, int, ExpertAddress, ResidentExpert, float]] = []
        for candidate in candidates:
            if candidate.pinned or candidate.loading or candidate.required:
                continue
            penalty = self.retention_value(candidate, access_index=access_index)
            penalty_per_byte = penalty / candidate.size_bytes
            ranked.append(
                (
                    penalty_per_byte,
                    candidate.last_access_index,
                    candidate.address,
                    candidate,
                    penalty,
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))

        victims: list[ExpertAddress] = []
        freed_bytes = 0
        total_penalty = 0.0
        for _score, _last_access, _address, candidate, penalty in ranked:
            victims.append(candidate.address)
            freed_bytes += candidate.size_bytes
            total_penalty += penalty
            if freed_bytes >= bytes_to_free:
                break
        return ResidencyPlan(
            evict=tuple(victims),
            freed_bytes=freed_bytes,
            shortfall_bytes=max(0, bytes_to_free - freed_bytes),
            estimated_reload_penalty_ns=total_penalty,
        )
