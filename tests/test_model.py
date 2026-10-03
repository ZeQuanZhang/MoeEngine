from __future__ import annotations

import unittest

from moeengine.model import ExpertAddress, ExpertDescriptor, LoadedExpert


class ModelValueTests(unittest.TestCase):
    def test_address_validation_and_string(self) -> None:
        self.assertEqual(str(ExpertAddress(17, 93)), "L17:E93")
        with self.assertRaises(ValueError):
            ExpertAddress(-1, 0)
        with self.assertRaises(ValueError):
            ExpertAddress(0, -1)

    def test_descriptor_and_payload_validation(self) -> None:
        address = ExpertAddress(0, 0)
        descriptor = ExpertDescriptor(address=address, size_bytes=4)
        LoadedExpert(descriptor=descriptor, load_latency_ns=0, payload=b"test")
        with self.assertRaises(ValueError):
            LoadedExpert(descriptor=descriptor, load_latency_ns=0, payload=b"bad")


if __name__ == "__main__":
    unittest.main()

