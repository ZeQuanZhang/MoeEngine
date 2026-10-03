from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from moeengine.inspect import inspect_config, inspect_safetensors_index


class InspectTests(unittest.TestCase):
    def test_deepseek_style_config_estimate(self) -> None:
        config = {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 43,
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "hidden_size": 4096,
            "moe_intermediate_size": 2048,
            "expert_dtype": "fp4",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            result = inspect_config(path)
        self.assertEqual(result["parameters_per_expert"], 25_165_824)
        self.assertEqual(
            result["estimated_expert_bytes_excluding_scales"], 12 * 1024 * 1024
        )
        self.assertEqual(result["routed_expert_instances"], 43 * 256)

    def test_safetensors_index_classifies_base_and_mtp_experts(self) -> None:
        weight_map = {"embed.weight": "model-00001.safetensors"}
        for prefix in ("layers.0.ffn.experts.0", "mtp.0.ffn.experts.0"):
            for projection in ("w1", "w2", "w3"):
                for kind in ("weight", "scale"):
                    weight_map[f"{prefix}.{projection}.{kind}"] = "model-00002.safetensors"
        index = {"metadata": {"total_size": 123}, "weight_map": weight_map}
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.safetensors.index.json"
            path.write_text(json.dumps(index), encoding="utf-8")
            result = inspect_safetensors_index(path)
        self.assertEqual(result["tensor_count"], 13)
        self.assertEqual(result["base_experts"]["bundle_count"], 1)
        self.assertEqual(result["mtp_experts"]["bundle_count"], 1)
        self.assertEqual(result["base_experts"]["malformed_bundle_count"], 0)
        self.assertEqual(result["other_tensor_count"], 1)


if __name__ == "__main__":
    unittest.main()
