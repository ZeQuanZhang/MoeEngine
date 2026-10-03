from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from moeengine.model import ExpertAddress, ExpertDescriptor
from moeengine.store import PackedExpertStore


class PackedExpertStoreTests(unittest.TestCase):
    def test_manifest_random_access_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload_a = b"abcd"
            payload_b = b"EFGHI"
            (root / "layer_000.bin").write_bytes(payload_a + payload_b)
            descriptors = [
                ExpertDescriptor(
                    address=ExpertAddress(0, 0),
                    relative_file="layer_000.bin",
                    offset=0,
                    size_bytes=len(payload_a),
                    sha256=hashlib.sha256(payload_a).hexdigest(),
                ),
                ExpertDescriptor(
                    address=ExpertAddress(0, 1),
                    relative_file="layer_000.bin",
                    offset=len(payload_a),
                    size_bytes=len(payload_b),
                    sha256=hashlib.sha256(payload_b).hexdigest(),
                ),
            ]
            manifest_path = root / "manifest.json"
            PackedExpertStore.write_manifest(manifest_path, descriptors)
            with PackedExpertStore.from_manifest(
                manifest_path, verify_checksums=True
            ) as store:
                self.assertEqual(store.load(ExpertAddress(0, 0)).payload, payload_a)
                self.assertEqual(store.load(ExpertAddress(0, 1)).payload, payload_b)

    def test_store_rejects_path_escape(self) -> None:
        descriptor = ExpertDescriptor(
            address=ExpertAddress(0, 0),
            relative_file="../outside.bin",
            offset=0,
            size_bytes=1,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = PackedExpertStore(temporary_directory, [descriptor])
            with self.assertRaises(ValueError):
                store.load(ExpertAddress(0, 0))


if __name__ == "__main__":
    unittest.main()

