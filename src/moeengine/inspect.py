"""Model metadata inspection helpers."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def inspect_config(path: str | Path) -> dict[str, object]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = int(config["num_hidden_layers"])
    experts_per_layer = int(config["n_routed_experts"])
    top_k = int(config["num_experts_per_tok"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["moe_intermediate_size"])
    parameters_per_expert = 3 * hidden_size * intermediate_size
    expert_dtype = str(config.get("expert_dtype", "unknown")).lower()
    bytes_per_parameter = {"fp4": 0.5, "int4": 0.5, "fp8": 1.0}.get(expert_dtype)
    expert_bytes = (
        round(parameters_per_expert * bytes_per_parameter)
        if bytes_per_parameter is not None
        else None
    )
    result: dict[str, object] = {
        "model_type": config.get("model_type"),
        "num_hidden_layers": layers,
        "n_routed_experts": experts_per_layer,
        "num_experts_per_tok": top_k,
        "hidden_size": hidden_size,
        "moe_intermediate_size": intermediate_size,
        "expert_dtype": expert_dtype,
        "parameters_per_expert": parameters_per_expert,
        "routed_expert_instances": layers * experts_per_layer,
        "activated_expert_instances_per_token": layers * top_k,
        "routed_expert_parameters": layers * experts_per_layer * parameters_per_expert,
        "activated_routed_parameters_per_token": layers * top_k * parameters_per_expert,
    }
    if expert_bytes is not None:
        result.update(
            {
                "estimated_expert_bytes_excluding_scales": expert_bytes,
                "estimated_routed_expert_bytes_excluding_scales": (
                    expert_bytes * layers * experts_per_layer
                ),
                "estimated_cold_expert_bytes_per_token_excluding_scales": (
                    expert_bytes * layers * top_k
                ),
            }
        )
    return result


_BASE_EXPERT_PATTERN = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
_MTP_EXPERT_PATTERN = re.compile(
    r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
_EXPECTED_EXPERT_COMPONENTS = {
    (projection, kind)
    for projection in ("w1", "w2", "w3")
    for kind in ("weight", "scale")
}


def _bundle_summary(
    bundles: dict[tuple[int, int], list[tuple[str, str, str]]],
) -> dict[str, object]:
    component_counts = Counter(len(components) for components in bundles.values())
    shards_per_bundle = Counter(
        len({shard for _projection, _kind, shard in components})
        for components in bundles.values()
    )
    malformed = []
    for address, components in bundles.items():
        actual = {(projection, kind) for projection, kind, _shard in components}
        if actual != _EXPECTED_EXPERT_COMPONENTS:
            malformed.append(address)
    groups = Counter(group_id for group_id, _expert_id in bundles)
    return {
        "bundle_count": len(bundles),
        "tensor_count": sum(len(components) for components in bundles.values()),
        "group_count": len(groups),
        "bundles_per_group": dict(sorted(Counter(groups.values()).items())),
        "component_count_distribution": dict(sorted(component_counts.items())),
        "shards_per_bundle_distribution": dict(sorted(shards_per_bundle.items())),
        "malformed_bundle_count": len(malformed),
        "malformed_bundle_examples": malformed[:10],
    }


def inspect_safetensors_index(path: str | Path) -> dict[str, object]:
    """Classify resident, base-expert, and MTP-expert checkpoint tensors."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    weight_map: dict[str, str] = data["weight_map"]
    base_bundles: dict[tuple[int, int], list[tuple[str, str, str]]] = defaultdict(list)
    mtp_bundles: dict[tuple[int, int], list[tuple[str, str, str]]] = defaultdict(list)
    classified_tensor_count = 0
    for tensor_name, shard in weight_map.items():
        base_match = _BASE_EXPERT_PATTERN.match(tensor_name)
        if base_match:
            base_bundles[(int(base_match.group(1)), int(base_match.group(2)))].append(
                (base_match.group(3), base_match.group(4), shard)
            )
            classified_tensor_count += 1
            continue
        mtp_match = _MTP_EXPERT_PATTERN.match(tensor_name)
        if mtp_match:
            mtp_bundles[(int(mtp_match.group(1)), int(mtp_match.group(2)))].append(
                (mtp_match.group(3), mtp_match.group(4), shard)
            )
            classified_tensor_count += 1
    return {
        "metadata_total_size": data.get("metadata", {}).get("total_size"),
        "tensor_count": len(weight_map),
        "shard_count": len(set(weight_map.values())),
        "base_experts": _bundle_summary(base_bundles),
        "mtp_experts": _bundle_summary(mtp_bundles),
        "other_tensor_count": len(weight_map) - classified_tensor_count,
    }
