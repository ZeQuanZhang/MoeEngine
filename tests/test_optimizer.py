from __future__ import annotations

import unittest

from moeengine.model import ExpertAddress
from moeengine.optimizer import (
    GenericResidencyOptimizer,
    MemoryTier,
    ResidentExpert,
    RouteLocalityEstimator,
)
from moeengine.replay import replay_trace
from moeengine.policies import RouterGDSFPolicy
from moeengine.store import SyntheticExpertStore


class GenericResidencyOptimizerTests(unittest.TestCase):
    def test_tier_cost_contains_no_platform_assumptions(self) -> None:
        tier = MemoryTier(
            name="remote-expert-pool",
            capacity_bytes=10_000,
            bandwidth_bytes_per_second=1_000_000_000,
            fixed_latency_ns=50_000,
        )
        self.assertEqual(tier.transfer_cost_ns(1_000), 51_000)

    def test_eviction_minimizes_expected_reload_penalty(self) -> None:
        optimizer = GenericResidencyOptimizer()
        cheap = ResidentExpert(
            ExpertAddress(0, 0), 100, 8, 1, 1_000, reuse_probability=0.1
        )
        expensive = ResidentExpert(
            ExpertAddress(0, 1), 100, 8, 1, 100_000, reuse_probability=0.9
        )
        plan = optimizer.plan_evictions(
            [expensive, cheap], bytes_to_free=100, access_index=10
        )
        self.assertEqual(plan.evict, (cheap.address,))
        self.assertTrue(plan.feasible)

    def test_pinned_loading_and_required_entries_are_never_victims(self) -> None:
        optimizer = GenericResidencyOptimizer()
        protected = [
            ResidentExpert(ExpertAddress(0, 0), 100, 0, 1, 1, pinned=True),
            ResidentExpert(ExpertAddress(0, 1), 100, 0, 1, 1, loading=True),
            ResidentExpert(ExpertAddress(0, 2), 100, 0, 1, 1, required=True),
        ]
        victim = ResidentExpert(ExpertAddress(0, 3), 75, 0, 1, 1)
        plan = optimizer.plan_evictions(
            [*protected, victim], bytes_to_free=100, access_index=1
        )
        self.assertEqual(plan.evict, (victim.address,))
        self.assertEqual(plan.shortfall_bytes, 25)
        self.assertFalse(plan.feasible)

    def test_stable_routes_learn_more_reuse_than_changing_routes(self) -> None:
        estimator = RouteLocalityEstimator(alpha=0.5)
        stable = [ExpertAddress(0, 0), ExpertAddress(0, 1)]
        for _ in range(5):
            estimator.observe(0, stable)

        for expert_id in range(6):
            estimator.observe(1, [ExpertAddress(1, expert_id)])

        self.assertGreater(
            estimator.probability(ExpertAddress(0, 0)),
            estimator.probability(ExpertAddress(1, 0)),
        )

    def test_generic_policy_accepts_backend_router_groups(self) -> None:
        a = ExpertAddress(0, 0)
        b = ExpertAddress(0, 1)
        c = ExpertAddress(0, 2)
        trace = [a, b, a, b, a, c, a, b]
        store = SyntheticExpertStore(
            num_layers=1,
            experts_per_layer=3,
            expert_size_bytes=1,
        )
        result = replay_trace(
            trace,
            store=store,
            capacity_bytes=2,
            policy_name="generic",
            route_width=2,
        )
        self.assertGreater(result.hits, 0)

    def test_router_gdsf_advances_clock_and_uses_ghost_refault(self) -> None:
        policy = RouterGDSFPolicy(ghost_limit=2, refault_alpha=0.5)
        address = ExpertAddress(0, 0)
        companion = ExpertAddress(0, 1)
        policy.observe_route(0, [address, companion])
        policy.observe_route(0, [address, companion])
        policy.on_load(address, 100, 10_000)
        policy.on_insert(address, 0)
        policy.on_remove(address)
        self.assertGreater(policy.inflation, 0.0)
        self.assertEqual(policy.ghost_count, 1)

        policy.observe_route(0, [address, companion])
        policy.on_load(address, 100, 10_000)
        policy.on_insert(address, 1)
        self.assertEqual(policy.ghost_count, 0)


if __name__ == "__main__":
    unittest.main()
