"""Model-aware virtual memory primitives for MoE expert experiments."""

from .cache import ExpertCache, ExpertLease
from .gguf_store import GgufExpertLayout, GgufExpertStore, GgufExpertTensorRange
from .model import ExpertAddress, ExpertDescriptor, LoadedExpert
from .optimizer import (
    GenericResidencyOptimizer,
    MemoryTier,
    OptimizerWeights,
    ResidencyPlan,
    ResidentExpert,
    RouteLocalityEstimator,
)
from .policies import (
    BeladyPolicy,
    GenericCostPolicy,
    LFUPolicy,
    LRUPolicy,
    ReplacementPolicy,
    RouterGDSFPolicy,
)
from .replay import BenchmarkResult, replay_trace
from .store import ExpertStore, PackedExpertStore, SyntheticExpertStore

__all__ = [
    "BeladyPolicy",
    "BenchmarkResult",
    "ExpertAddress",
    "ExpertCache",
    "ExpertDescriptor",
    "ExpertLease",
    "ExpertStore",
    "GenericCostPolicy",
    "GenericResidencyOptimizer",
    "GgufExpertLayout",
    "GgufExpertStore",
    "GgufExpertTensorRange",
    "LFUPolicy",
    "LRUPolicy",
    "LoadedExpert",
    "MemoryTier",
    "OptimizerWeights",
    "PackedExpertStore",
    "ReplacementPolicy",
    "RouterGDSFPolicy",
    "ResidencyPlan",
    "ResidentExpert",
    "RouteLocalityEstimator",
    "SyntheticExpertStore",
    "replay_trace",
]
