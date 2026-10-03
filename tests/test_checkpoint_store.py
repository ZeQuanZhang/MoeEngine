from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from moeengine.checkpoint_store import SafetensorsExpertStore
from moeengine.model import ExpertAddress


def _write_checkpoint(root: Path) -> dict[ExpertAddress, bytes]:
    shard_name = "model-00001-of-00001.safetensors"
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    weight_map: dict[str, str] = {}
    expected_payloads: dict[ExpertAddress, bytes] = {}
    data = bytearray()
    for expert_id in range(2):
        expert_payload = bytearray()
        prefix = f"layers.0.ffn.experts.{expert_id}"
        for projection in ("w1", "w2", "w3"):
            for kind in ("weight", "scale"):
                name = f"{prefix}.{projection}.{kind}"
                value = bytes([expert_id * 10 + len(expert_payload) // 4 + 1]) * 4
                start = len(data)
                data.extend(value)
                header[name] = {
                    "dtype": "U8",
                    "shape": [4],
                    "data_offsets": [start, len(data)],
                }
                weight_map[name] = shard_name
                expert_payload.extend(value)
        expected_payloads[ExpertAddress(0, expert_id)] = bytes(expert_payload)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((-len(header_bytes)) % 8)
    (root / shard_name).write_bytes(
        struct.pack("<Q", len(header_bytes)) + header_bytes + data
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    return expected_payloads


class SafetensorsExpertStoreTests(unittest.TestCase):
    def test_reads_six_real_tensor_ranges_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            expected_payloads = _write_checkpoint(root)
            with SafetensorsExpertStore.from_checkpoint(root, layer_ids=[0]) as store:
                address = ExpertAddress(0, 1)
                loaded = store.load(address)
                layout = store.layout(address)
        self.assertEqual(loaded.payload, expected_payloads[address])
        self.assertEqual(loaded.descriptor.size_bytes, 24)
        self.assertEqual(len(layout.tensors), 6)
        self.assertEqual(
            [tensor.component for tensor in layout.tensors],
            [
                "w1.weight",
                "w1.scale",
                "w2.weight",
                "w2.scale",
                "w3.weight",
                "w3.scale",
            ],
        )

    def test_payload_digest_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            expected_payloads = _write_checkpoint(root)
            with SafetensorsExpertStore.from_checkpoint(root) as store:
                first = store.load(ExpertAddress(0, 0))
                second = store.load(ExpertAddress(0, 0))
        assert first.payload is not None
        assert second.payload is not None
        self.assertEqual(
            hashlib.sha256(first.payload).digest(),
            hashlib.sha256(second.payload).digest(),
        )
        self.assertEqual(first.payload, expected_payloads[ExpertAddress(0, 0)])

    def test_unknown_layer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_checkpoint(root)
            with self.assertRaisesRegex(ValueError, "no experts"):
                SafetensorsExpertStore.from_checkpoint(root, layer_ids=[3])


if __name__ == "__main__":
    unittest.main()
