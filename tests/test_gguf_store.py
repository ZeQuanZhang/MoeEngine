from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFWriter

from moeengine.gguf_store import GgufExpertStore
from moeengine.model import ExpertAddress


def _write_tiny_qwen4exp(path: Path) -> dict[ExpertAddress, bytes]:
    writer = GGUFWriter(path, "qwen4exp")
    writer.add_block_count(1)
    writer.add_expert_count(4)
    writer.add_expert_used_count(2)
    for component_index, component in enumerate(("down", "gate", "up")):
        values = (
            np.arange(24, dtype=np.float32) + component_index * 100
        ).reshape(4, 3, 2)
        writer.add_tensor(f"blk.0.ffn_{component}_exps.weight", values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    reader = GGUFReader(str(path), "r")
    expected: dict[ExpertAddress, bytes] = {}
    for expert_id in range(4):
        parts = []
        for component in ("gate", "up", "down"):
            tensor = next(
                tensor
                for tensor in reader.tensors
                if tensor.name == f"blk.0.ffn_{component}_exps.weight"
            )
            parts.append(bytes(tensor.data[expert_id]))
        expected[ExpertAddress(0, expert_id)] = b"".join(parts)
    return expected


class GgufExpertStoreTests(unittest.TestCase):
    def test_reads_one_expert_without_materializing_the_bank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tiny-qwen4exp.gguf"
            expected = _write_tiny_qwen4exp(path)
            with GgufExpertStore.from_model(path, layer_ids=[0]) as store:
                address = ExpertAddress(0, 2)
                layout = store.layout(address)
                loaded = store.load(address)
                descriptor = store.describe(address)

        self.assertEqual(store.architecture, "qwen4exp")
        self.assertEqual(store.expert_count, 4)
        self.assertEqual(store.expert_used_count, 2)
        self.assertEqual([tensor.component for tensor in layout.tensors], ["gate", "up", "down"])
        self.assertEqual(loaded.payload, expected[address])
        self.assertEqual(descriptor.size_bytes, 72)
        self.assertEqual(len(loaded.payload or b""), 72)

    def test_directory_discovery_and_unknown_layer_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_tiny_qwen4exp(root / "tiny.gguf")
            with GgufExpertStore.from_model(root) as store:
                self.assertEqual(len(store.addresses), 4)
            with self.assertRaisesRegex(ValueError, r"layers \[3\]"):
                GgufExpertStore.from_model(root, layer_ids=[3])


if __name__ == "__main__":
    unittest.main()
