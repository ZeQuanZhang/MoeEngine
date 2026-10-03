from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from moeengine.checkpoint import (
    measure_bundles_in_shard,
    plan_checkpoint,
    read_safetensors_header,
    write_checkpoint_plan,
)


def _expert_tensors(prefix: str, shard: str) -> dict[str, str]:
    return {
        f"{prefix}.{projection}.{kind}": shard
        for projection in ("w1", "w2", "w3")
        for kind in ("weight", "scale")
    }


class CheckpointTests(unittest.TestCase):
    def test_plan_checkpoint_groups_complete_experts(self) -> None:
        weight_map = {"embed.weight": "model-00001-of-00002.safetensors"}
        weight_map.update(
            _expert_tensors(
                "layers.2.ffn.experts.7", "model-00002-of-00002.safetensors"
            )
        )
        weight_map.update(
            _expert_tensors(
                "mtp.0.ffn.experts.7", "model-00002-of-00002.safetensors"
            )
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            index_path = root / "model.safetensors.index.json"
            output_path = root / "plan.json"
            index_path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
            bundles = plan_checkpoint(index_path)
            write_checkpoint_plan(output_path, bundles, source_index=index_path.name)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].namespace, "base")
        self.assertEqual(bundles[0].group_id, 2)
        self.assertEqual(bundles[0].expert_id, 7)
        self.assertEqual(len(bundles[0].tensors), 6)
        self.assertEqual(payload["bundle_count"], 1)

    def test_plan_rejects_cross_shard_expert(self) -> None:
        weight_map = _expert_tensors("layers.0.ffn.experts.0", "a.safetensors")
        weight_map["layers.0.ffn.experts.0.w3.scale"] = "b.safetensors"
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "index.json"
            path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "spans shards"):
                plan_checkpoint(path)

    def test_reads_header_and_measures_expert(self) -> None:
        shard_name = "model-00001-of-00001.safetensors"
        weight_map = _expert_tensors("layers.0.ffn.experts.0", shard_name)
        header: dict[str, object] = {"__metadata__": {"format": "pt"}}
        offset = 0
        for name in weight_map:
            header[name] = {
                "dtype": "U8",
                "shape": [4],
                "data_offsets": [offset, offset + 4],
            }
            offset += 4
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        padding = (-len(header_bytes)) % 8
        header_bytes += b" " * padding
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shard_path = root / shard_name
            shard_path.write_bytes(
                struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(offset)
            )
            index_path = root / "model.safetensors.index.json"
            index_path.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
            entries = read_safetensors_header(shard_path)
            result = measure_bundles_in_shard(plan_checkpoint(index_path), shard_path)
        self.assertEqual(len(entries), 6)
        self.assertEqual(result["expert_bundle_count"], 1)
        self.assertEqual(result["expert_size_bytes_unique"], [24])

    def test_rejects_truncated_payload(self) -> None:
        header = {
            "tensor": {"dtype": "U8", "shape": [10], "data_offsets": [0, 10]}
        }
        header_bytes = json.dumps(header).encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "bad.safetensors"
            path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes)
            with self.assertRaisesRegex(ValueError, "payload exceeds"):
                read_safetensors_header(path)


if __name__ == "__main__":
    unittest.main()
