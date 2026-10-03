from __future__ import annotations

import unittest

from moeengine.trace import bursty_trace, markov_trace, random_trace, zipf_trace


class TraceTests(unittest.TestCase):
    def test_all_generators_have_expected_shape_and_unique_top_k(self) -> None:
        generators = (random_trace, zipf_trace, markov_trace, bursty_trace)
        for generator in generators:
            with self.subTest(generator=generator.__name__):
                trace = generator(
                    num_tokens=5,
                    num_layers=3,
                    experts_per_layer=8,
                    top_k=2,
                    seed=42,
                )
                self.assertEqual(len(trace), 5 * 3 * 2)
                for offset in range(0, len(trace), 2):
                    group = trace[offset : offset + 2]
                    self.assertEqual(group[0].layer_id, group[1].layer_id)
                    self.assertEqual(len(set(group)), 2)

    def test_seed_is_reproducible(self) -> None:
        arguments = dict(
            num_tokens=10,
            num_layers=2,
            experts_per_layer=16,
            top_k=3,
            seed=7,
        )
        self.assertEqual(random_trace(**arguments), random_trace(**arguments))


if __name__ == "__main__":
    unittest.main()

