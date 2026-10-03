from __future__ import annotations

import unittest

from moeengine.cache import CacheCapacityError, ExpertCache
from moeengine.model import ExpertAddress
from moeengine.policies import LRUPolicy
from moeengine.store import SyntheticExpertStore


class ExpertCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SyntheticExpertStore(
            num_layers=1,
            experts_per_layer=4,
            expert_size_bytes=100,
            bandwidth_bytes_per_second=1_000_000_000,
            fixed_latency_ns=10,
        )

    def test_hit_miss_and_lru_eviction(self) -> None:
        cache = ExpertCache(capacity_bytes=200, store=self.store, policy=LRUPolicy())
        first = ExpertAddress(0, 0)
        second = ExpertAddress(0, 1)
        third = ExpertAddress(0, 2)
        with cache.acquire(first, 0) as lease:
            self.assertFalse(lease.access.hit)
        with cache.acquire(second, 1):
            pass
        with cache.acquire(first, 2) as lease:
            self.assertTrue(lease.access.hit)
        with cache.acquire(third, 3) as lease:
            self.assertEqual(lease.access.evicted, (second,))
        self.assertEqual(cache.stats.hits, 1)
        self.assertEqual(cache.stats.misses, 3)
        self.assertEqual(cache.stats.evictions, 1)
        self.assertEqual(cache.stats.bytes_loaded, 300)
        self.assertEqual(cache.stats.peak_occupancy_bytes, 200)

    def test_pinned_entry_cannot_be_evicted(self) -> None:
        cache = ExpertCache(capacity_bytes=100, store=self.store, policy=LRUPolicy())
        first = cache.acquire(ExpertAddress(0, 0), 0)
        with self.assertRaises(CacheCapacityError):
            cache.acquire(ExpertAddress(0, 1), 1)
        first.release()
        with cache.acquire(ExpertAddress(0, 1), 2):
            pass

    def test_expert_larger_than_cache_is_rejected(self) -> None:
        cache = ExpertCache(capacity_bytes=99, store=self.store, policy=LRUPolicy())
        with self.assertRaises(CacheCapacityError):
            cache.acquire(ExpertAddress(0, 0), 0)


if __name__ == "__main__":
    unittest.main()

