from __future__ import annotations

import unittest

from moeengine.model import ExpertAddress
from moeengine.replay import replay_trace
from moeengine.store import SyntheticExpertStore


class PolicyTests(unittest.TestCase):
    def test_belady_is_an_offline_upper_bound_for_known_trace(self) -> None:
        a = ExpertAddress(0, 0)
        b = ExpertAddress(0, 1)
        c = ExpertAddress(0, 2)
        trace = [a, b, c, a, b, c]
        store = SyntheticExpertStore(
            num_layers=1,
            experts_per_layer=3,
            expert_size_bytes=1,
        )
        lru = replay_trace(trace, store=store, capacity_bytes=2, policy_name="lru")
        belady = replay_trace(trace, store=store, capacity_bytes=2, policy_name="belady")
        self.assertEqual(lru.hits, 0)
        self.assertEqual(belady.hits, 2)
        self.assertLessEqual(belady.misses, lru.misses)

    def test_lfu_prefers_frequently_used_entry(self) -> None:
        a = ExpertAddress(0, 0)
        b = ExpertAddress(0, 1)
        c = ExpertAddress(0, 2)
        trace = [a, b, a, c, a]
        store = SyntheticExpertStore(
            num_layers=1,
            experts_per_layer=3,
            expert_size_bytes=1,
        )
        result = replay_trace(trace, store=store, capacity_bytes=2, policy_name="lfu")
        self.assertEqual(result.hits, 2)
        self.assertEqual(result.misses, 3)


if __name__ == "__main__":
    unittest.main()

