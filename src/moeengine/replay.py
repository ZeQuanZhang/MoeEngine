"""Trace replay and benchmark result calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from statistics import mean

from .cache import ExpertCache
from .model import ExpertAddress
from .policies import make_policy
from .store import ExpertStore


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return ordered[index]


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    policy: str
    capacity_bytes: int
    accesses: int
    hits: int
    misses: int
    hit_rate: float
    miss_rate: float
    model_faults: int
    faults_per_access: float
    bytes_loaded: int
    bytes_evicted: int
    evictions: int
    peak_occupancy_bytes: int
    mean_fault_latency_ns: float
    p50_fault_latency_ns: int
    p95_fault_latency_ns: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def replay_trace(
    trace: list[ExpertAddress],
    *,
    store: ExpertStore,
    capacity_bytes: int,
    policy_name: str,
    route_width: int | None = None,
) -> BenchmarkResult:
    if route_width is not None and route_width <= 0:
        raise ValueError("route_width must be positive")
    policy = make_policy(policy_name, trace=trace)
    cache = ExpertCache(capacity_bytes=capacity_bytes, store=store, policy=policy)
    if route_width is None:
        routes = (trace,)
    else:
        routes = (
            trace[start : start + route_width]
            for start in range(0, len(trace), route_width)
        )
    access_index = 0
    for route in routes:
        if route_width is not None:
            if len(route) != route_width:
                raise ValueError("trace length must be divisible by route_width")
            layer_id = route[0].layer_id
            if any(address.layer_id != layer_id for address in route):
                raise ValueError("one route cannot span multiple layers")
            policy.observe_route(layer_id, route)
        for address in route:
            with cache.acquire(address, access_index):
                pass
            access_index += 1
    stats = cache.stats
    latencies = stats.fault_latencies_ns
    return BenchmarkResult(
        policy=policy.name,
        capacity_bytes=capacity_bytes,
        accesses=stats.accesses,
        hits=stats.hits,
        misses=stats.misses,
        hit_rate=stats.hit_rate,
        miss_rate=stats.miss_rate,
        model_faults=stats.model_faults,
        faults_per_access=stats.model_faults / stats.accesses if stats.accesses else 0.0,
        bytes_loaded=stats.bytes_loaded,
        bytes_evicted=stats.bytes_evicted,
        evictions=stats.evictions,
        peak_occupancy_bytes=stats.peak_occupancy_bytes,
        mean_fault_latency_ns=mean(latencies) if latencies else 0.0,
        p50_fault_latency_ns=_percentile(latencies, 0.50),
        p95_fault_latency_ns=_percentile(latencies, 0.95),
    )
