"""Bounded expert cache with pin-safe leases and detailed telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from .model import ExpertAddress, LoadedExpert
from .policies import ReplacementPolicy
from .store import ExpertStore


class CacheCapacityError(RuntimeError):
    """Raised when a requested expert cannot fit or no entry can be evicted."""


@dataclass(slots=True)
class CacheEntry:
    expert: LoadedExpert
    pin_count: int = 0
    generation: int = 0


@dataclass(slots=True)
class CacheStats:
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    model_faults: int = 0
    bytes_loaded: int = 0
    bytes_evicted: int = 0
    evictions: int = 0
    occupancy_bytes: int = 0
    peak_occupancy_bytes: int = 0
    fault_latencies_ns: list[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    @property
    def miss_rate(self) -> float:
        return self.misses / self.accesses if self.accesses else 0.0


@dataclass(frozen=True, slots=True)
class CacheAccess:
    hit: bool
    evicted: tuple[ExpertAddress, ...]


class ExpertLease:
    """Pins a cache slot until CPU/GPU work using it has completed."""

    def __init__(
        self,
        cache: "ExpertCache",
        address: ExpertAddress,
        generation: int,
        expert: LoadedExpert,
        access: CacheAccess,
    ) -> None:
        self._cache = cache
        self.address = address
        self.generation = generation
        self.expert = expert
        self.access = access
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._cache._release(self.address, self.generation)
            self._released = True

    def __enter__(self) -> "ExpertLease":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()


class ExpertCache:
    """A byte-bounded cache whose entries cannot be evicted while leased."""

    _INITIAL_GENERATION: Final[int] = 1

    def __init__(
        self,
        *,
        capacity_bytes: int,
        store: ExpertStore,
        policy: ReplacementPolicy,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_bytes = capacity_bytes
        self.store = store
        self.policy = policy
        self.stats = CacheStats()
        self._entries: dict[ExpertAddress, CacheEntry] = {}
        self._generation = self._INITIAL_GENERATION

    def __contains__(self, address: ExpertAddress) -> bool:
        return address in self._entries

    @property
    def resident_addresses(self) -> frozenset[ExpertAddress]:
        return frozenset(self._entries)

    def acquire(self, address: ExpertAddress, access_index: int) -> ExpertLease:
        if access_index < 0:
            raise ValueError("access_index must be non-negative")
        self.stats.accesses += 1
        self.policy.before_access(address, access_index)
        entry = self._entries.get(address)
        if entry is not None:
            self.stats.hits += 1
            entry.pin_count += 1
            self.policy.on_access(address, access_index)
            return ExpertLease(
                self,
                address,
                entry.generation,
                entry.expert,
                CacheAccess(hit=True, evicted=()),
            )

        self.stats.misses += 1
        self.stats.model_faults += 1
        loaded = self.store.load(address)
        size_bytes = loaded.descriptor.size_bytes
        self.policy.on_load(address, size_bytes, loaded.load_latency_ns)
        if size_bytes > self.capacity_bytes:
            raise CacheCapacityError(
                f"expert {address} requires {size_bytes} bytes, cache has {self.capacity_bytes}"
            )

        evicted: list[ExpertAddress] = []
        while self.stats.occupancy_bytes + size_bytes > self.capacity_bytes:
            candidates = {
                candidate
                for candidate, candidate_entry in self._entries.items()
                if candidate_entry.pin_count == 0
            }
            if not candidates:
                raise CacheCapacityError(
                    f"cache needs {size_bytes} bytes but every resident expert is pinned"
                )
            victim = self.policy.choose_victim(candidates, access_index)
            victim_entry = self._entries.pop(victim)
            victim_size = victim_entry.expert.descriptor.size_bytes
            self.stats.occupancy_bytes -= victim_size
            self.stats.bytes_evicted += victim_size
            self.stats.evictions += 1
            self.policy.on_remove(victim)
            evicted.append(victim)

        generation = self._generation
        self._generation += 1
        entry = CacheEntry(expert=loaded, pin_count=1, generation=generation)
        self._entries[address] = entry
        self.policy.on_insert(address, access_index)
        self.stats.bytes_loaded += size_bytes
        self.stats.fault_latencies_ns.append(loaded.load_latency_ns)
        self.stats.occupancy_bytes += size_bytes
        self.stats.peak_occupancy_bytes = max(
            self.stats.peak_occupancy_bytes,
            self.stats.occupancy_bytes,
        )
        return ExpertLease(
            self,
            address,
            generation,
            loaded,
            CacheAccess(hit=False, evicted=tuple(evicted)),
        )

    def _release(self, address: ExpertAddress, generation: int) -> None:
        entry = self._entries.get(address)
        if entry is None or entry.generation != generation:
            raise RuntimeError(f"stale expert lease for {address}")
        if entry.pin_count <= 0:
            raise RuntimeError(f"expert {address} is not pinned")
        entry.pin_count -= 1
