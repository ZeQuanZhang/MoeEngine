"""Pluggable expert-cache replacement policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from collections.abc import Collection, Sequence
from math import inf, sqrt

from .model import ExpertAddress
from .optimizer import (
    GenericResidencyOptimizer,
    ResidentExpert,
    RouteLocalityEstimator,
)


class ReplacementPolicy(ABC):
    """Policy contract used by :class:`moeengine.cache.ExpertCache`."""

    name = "abstract"

    def before_access(self, address: ExpertAddress, access_index: int) -> None:
        """Observe the request before a possible eviction decision."""

    def observe_route(
        self,
        layer_id: int,
        experts: Collection[ExpertAddress],
    ) -> None:
        """Observe one exact router selection when the backend exposes it."""

    def on_load(
        self,
        address: ExpertAddress,
        size_bytes: int,
        load_latency_ns: int,
    ) -> None:
        """Observe a materialized load and its measured backend cost."""

    @abstractmethod
    def on_access(self, address: ExpertAddress, access_index: int) -> None:
        """Record a hit."""

    @abstractmethod
    def on_insert(self, address: ExpertAddress, access_index: int) -> None:
        """Record a newly loaded entry."""

    @abstractmethod
    def on_remove(self, address: ExpertAddress) -> None:
        """Forget an evicted entry."""

    @abstractmethod
    def choose_victim(
        self,
        candidates: Collection[ExpertAddress],
        access_index: int,
    ) -> ExpertAddress:
        """Choose one unpinned address to evict."""


class LRUPolicy(ReplacementPolicy):
    name = "lru"

    def __init__(self) -> None:
        self._last_access: dict[ExpertAddress, int] = {}

    def on_access(self, address: ExpertAddress, access_index: int) -> None:
        self._last_access[address] = access_index

    def on_insert(self, address: ExpertAddress, access_index: int) -> None:
        self._last_access[address] = access_index

    def on_remove(self, address: ExpertAddress) -> None:
        self._last_access.pop(address, None)

    def choose_victim(
        self,
        candidates: Collection[ExpertAddress],
        access_index: int,
    ) -> ExpertAddress:
        if not candidates:
            raise ValueError("no eviction candidates")
        return min(candidates, key=lambda address: (self._last_access[address], address))


class LFUPolicy(ReplacementPolicy):
    """LFU with LRU tie-breaking."""

    name = "lfu"

    def __init__(self) -> None:
        self._frequency: dict[ExpertAddress, int] = {}
        self._last_access: dict[ExpertAddress, int] = {}

    def on_access(self, address: ExpertAddress, access_index: int) -> None:
        self._frequency[address] += 1
        self._last_access[address] = access_index

    def on_insert(self, address: ExpertAddress, access_index: int) -> None:
        self._frequency[address] = 1
        self._last_access[address] = access_index

    def on_remove(self, address: ExpertAddress) -> None:
        self._frequency.pop(address, None)
        self._last_access.pop(address, None)

    def choose_victim(
        self,
        candidates: Collection[ExpertAddress],
        access_index: int,
    ) -> ExpertAddress:
        if not candidates:
            raise ValueError("no eviction candidates")
        return min(
            candidates,
            key=lambda address: (
                self._frequency[address],
                self._last_access[address],
                address,
            ),
        )


class BeladyPolicy(ReplacementPolicy):
    """Offline farthest-next-use policy, used only as a theoretical bound."""

    name = "belady"

    def __init__(self, trace: Sequence[ExpertAddress]) -> None:
        future: dict[ExpertAddress, deque[int]] = defaultdict(deque)
        for index, address in enumerate(trace):
            future[address].append(index)
        self._future = dict(future)

    def before_access(self, address: ExpertAddress, access_index: int) -> None:
        positions = self._future.get(address)
        if positions is None:
            return
        while positions and positions[0] < access_index:
            positions.popleft()
        if positions and positions[0] == access_index:
            positions.popleft()

    def on_access(self, address: ExpertAddress, access_index: int) -> None:
        return None

    def on_insert(self, address: ExpertAddress, access_index: int) -> None:
        return None

    def on_remove(self, address: ExpertAddress) -> None:
        return None

    def choose_victim(
        self,
        candidates: Collection[ExpertAddress],
        access_index: int,
    ) -> ExpertAddress:
        if not candidates:
            raise ValueError("no eviction candidates")

        def next_use(address: ExpertAddress) -> float:
            positions = self._future.get(address)
            return positions[0] if positions else inf

        return max(candidates, key=lambda address: (next_use(address), address))


class GenericCostPolicy(ReplacementPolicy):
    """Online MoeEngine policy based on route reuse and measured reload cost."""

    name = "generic"

    def __init__(self) -> None:
        self.optimizer = GenericResidencyOptimizer()
        self.locality = RouteLocalityEstimator()
        self._last_access: dict[ExpertAddress, int] = {}
        self._frequency: dict[ExpertAddress, int] = {}
        self._size_bytes: dict[ExpertAddress, int] = {}
        self._load_latency_ns: dict[ExpertAddress, int] = {}

    def observe_route(
        self,
        layer_id: int,
        experts: Collection[ExpertAddress],
    ) -> None:
        """Accept exact router output from any inference backend."""

        self.locality.observe(layer_id, experts)

    def on_load(
        self,
        address: ExpertAddress,
        size_bytes: int,
        load_latency_ns: int,
    ) -> None:
        self._size_bytes[address] = size_bytes
        previous = self._load_latency_ns.get(address)
        self._load_latency_ns[address] = (
            load_latency_ns
            if previous is None
            else round(previous * 0.875 + load_latency_ns * 0.125)
        )

    def on_access(self, address: ExpertAddress, access_index: int) -> None:
        self._last_access[address] = access_index
        self._frequency[address] += 1

    def on_insert(self, address: ExpertAddress, access_index: int) -> None:
        self._last_access[address] = access_index
        self._frequency[address] = max(1, self._frequency.get(address, 0))

    def on_remove(self, address: ExpertAddress) -> None:
        self._last_access.pop(address, None)
        self._frequency.pop(address, None)

    def choose_victim(
        self,
        candidates: Collection[ExpertAddress],
        access_index: int,
    ) -> ExpertAddress:
        if not candidates:
            raise ValueError("no eviction candidates")
        resident = [
            ResidentExpert(
                address=address,
                size_bytes=self._size_bytes[address],
                last_access_index=self._last_access[address],
                access_count=self._frequency[address],
                reload_cost_ns=self._load_latency_ns[address],
                reuse_probability=self.locality.probability(address),
            )
            for address in candidates
        ]
        plan = self.optimizer.plan_evictions(
            resident,
            bytes_to_free=1,
            access_index=access_index,
        )
        if not plan.evict:
            raise ValueError("no evictable candidate")
        return plan.evict[0]


class RouterGDSFPolicy(ReplacementPolicy):
    """Router-aware GreedyDual-Size with bounded ghost/refault history."""

    name = "gdsf"

    def __init__(self, *, ghost_limit: int = 32_768, refault_alpha: float = 0.125) -> None:
        if ghost_limit <= 0:
            raise ValueError("ghost_limit must be positive")
        if not 0.0 < refault_alpha <= 1.0:
            raise ValueError("refault_alpha must be between zero and one")
        self.ghost_limit = ghost_limit
        self.refault_alpha = refault_alpha
        self.locality = RouteLocalityEstimator()
        self.inflation = 0.0
        self._route_epoch: dict[int, int] = {}
        self._current_access_index = 0
        self._required: frozenset[ExpertAddress] = frozenset()
        self._last_access: dict[ExpertAddress, int] = {}
        self._frequency: dict[ExpertAddress, int] = {}
        self._size_bytes: dict[ExpertAddress, int] = {}
        self._load_latency_ns: dict[ExpertAddress, int] = {}
        self._priority: dict[ExpertAddress, float] = {}
        self._refault_probability: dict[ExpertAddress, float] = {}
        self._refault_samples: dict[ExpertAddress, int] = {}
        self._pending_refault: dict[ExpertAddress, tuple[float, int]] = {}
        self._ghosts: OrderedDict[
            ExpertAddress, tuple[float, float, int, int]
        ] = OrderedDict()

    @property
    def ghost_count(self) -> int:
        return len(self._ghosts)

    def observe_route(
        self,
        layer_id: int,
        experts: Collection[ExpertAddress],
    ) -> None:
        self.locality.observe(layer_id, experts)
        self._route_epoch[layer_id] = self._route_epoch.get(layer_id, 0) + 1
        self._required = frozenset(experts)

    def _combined_probability(self, address: ExpertAddress) -> float:
        route = self.locality.probability(address)
        refault = self._refault_probability.get(address, 0.0)
        samples = self._refault_samples.get(address, 0)
        # A single ghost hit is useful evidence, but it must not overpower an
        # exact router history.  Treat four observations as the prior strength
        # before admitting the refault signal at full weight.
        refault_confidence = samples / (samples + 4.0)
        evidenced_refault = refault * refault_confidence
        return 1.0 - (1.0 - route) * (1.0 - evidenced_refault)

    def before_access(self, address: ExpertAddress, access_index: int) -> None:
        self._current_access_index = access_index

    def _refresh_priority(self, address: ExpertAddress) -> None:
        size_bytes = self._size_bytes[address]
        load_latency_ns = self._load_latency_ns[address]
        probability = self._combined_probability(address)
        # Demand entries are protected explicitly while in use, so admission
        # does not need an unconditional "+1" credit.  Omitting it is what
        # makes the policy scan-resistant: one-off experts do not advance the
        # GreedyDual clock and flush a learned routed working set.
        self._priority[address] = (
            self.inflation + probability * load_latency_ns / size_bytes
        )

    def on_load(
        self,
        address: ExpertAddress,
        size_bytes: int,
        load_latency_ns: int,
    ) -> None:
        self._size_bytes[address] = size_bytes
        previous = self._load_latency_ns.get(address)
        self._load_latency_ns[address] = (
            load_latency_ns
            if previous is None
            else round(previous * 0.875 + load_latency_ns * 0.125)
        )
        ghost = self._ghosts.get(address)
        if ghost is None:
            return
        _priority, probability, samples, evicted_access = ghost
        distance = max(1, self._current_access_index - evicted_access)
        sample = 1.0 / sqrt(distance)
        if samples == 0:
            probability = sample
        else:
            probability += self.refault_alpha * (sample - probability)
        self._pending_refault[address] = (probability, samples + 1)

    def on_access(self, address: ExpertAddress, access_index: int) -> None:
        self._last_access[address] = access_index
        self._frequency[address] += 1
        self._refresh_priority(address)

    def on_insert(self, address: ExpertAddress, access_index: int) -> None:
        pending = self._pending_refault.pop(address, None)
        if pending is not None:
            self._refault_probability[address], self._refault_samples[address] = pending
            self._ghosts.pop(address, None)
        self._last_access[address] = access_index
        self._frequency[address] = max(1, self._frequency.get(address, 0))
        self._refresh_priority(address)

    def on_remove(self, address: ExpertAddress) -> None:
        priority = self._priority.pop(address, self.inflation)
        self.inflation = max(self.inflation, priority)
        probability = self._refault_probability.pop(address, 0.0)
        samples = self._refault_samples.pop(address, 0)
        evicted_access = self._current_access_index
        self._ghosts.pop(address, None)
        self._ghosts[address] = (priority, probability, samples, evicted_access)
        while len(self._ghosts) > self.ghost_limit:
            self._ghosts.popitem(last=False)
        self._last_access.pop(address, None)
        self._frequency.pop(address, None)

    def choose_victim(
        self,
        candidates: Collection[ExpertAddress],
        access_index: int,
    ) -> ExpertAddress:
        evictable = [address for address in candidates if address not in self._required]
        if not evictable:
            raise ValueError("no eviction candidates outside the current route")
        return min(
            evictable,
            key=lambda address: (
                self._combined_probability(address)
                * self._load_latency_ns[address]
                / self._size_bytes[address],
                self._priority.get(address, self.inflation),
                self._last_access[address],
                address,
            ),
        )


def make_policy(
    name: str,
    *,
    trace: Sequence[ExpertAddress] | None = None,
) -> ReplacementPolicy:
    normalized = name.strip().lower()
    if normalized == "lru":
        return LRUPolicy()
    if normalized == "lfu":
        return LFUPolicy()
    if normalized == "belady":
        if trace is None:
            raise ValueError("Belady policy requires the complete trace")
        return BeladyPolicy(trace)
    if normalized in {"generic", "moeengine"}:
        return GenericCostPolicy()
    if normalized in {"gdsf", "router-gdsf"}:
        return RouterGDSFPolicy()
    raise ValueError(f"unknown policy: {name}")
