"""Checkpoint planning and safetensors metadata helpers.

This module deliberately reads only JSON indexes and safetensors headers.  It
does not interpret or transform DeepSeek's FP4 payload; that belongs to the
native converter and compute backend.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


_BASE_EXPERT_PATTERN = re.compile(
    r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
_MTP_EXPERT_PATTERN = re.compile(
    r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$"
)
_EXPECTED_COMPONENTS = tuple(
    f"{projection}.{kind}"
    for projection in ("w1", "w2", "w3")
    for kind in ("weight", "scale")
)


@dataclass(frozen=True, slots=True)
class TensorSource:
    """One logical tensor and the Hugging Face shard that contains it."""

    component: str
    tensor_name: str
    shard: str


@dataclass(frozen=True, slots=True)
class ExpertBundlePlan:
    """All source tensors required to reconstruct one routed expert."""

    namespace: str
    group_id: int
    expert_id: int
    source_shard: str
    tensors: tuple[TensorSource, ...]


@dataclass(frozen=True, slots=True)
class SafetensorEntry:
    """A tensor's type, shape, and absolute payload range in one shard."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    size_bytes: int


def _collect_bundles(
    weight_map: dict[str, str],
    pattern: re.Pattern[str],
    namespace: str,
) -> list[ExpertBundlePlan]:
    grouped: dict[tuple[int, int], list[TensorSource]] = {}
    for tensor_name, shard in weight_map.items():
        match = pattern.match(tensor_name)
        if match is None:
            continue
        address = (int(match.group(1)), int(match.group(2)))
        component = f"{match.group(3)}.{match.group(4)}"
        grouped.setdefault(address, []).append(
            TensorSource(component=component, tensor_name=tensor_name, shard=shard)
        )

    result = []
    expected = set(_EXPECTED_COMPONENTS)
    for (group_id, expert_id), tensors in sorted(grouped.items()):
        actual = {tensor.component for tensor in tensors}
        if actual != expected or len(tensors) != len(expected):
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"incomplete {namespace} expert {group_id}:{expert_id}; "
                f"missing={missing}, extra={extra}, tensor_count={len(tensors)}"
            )
        shards = {tensor.shard for tensor in tensors}
        if len(shards) != 1:
            raise ValueError(
                f"{namespace} expert {group_id}:{expert_id} spans shards: "
                f"{sorted(shards)}"
            )
        by_component = {tensor.component: tensor for tensor in tensors}
        result.append(
            ExpertBundlePlan(
                namespace=namespace,
                group_id=group_id,
                expert_id=expert_id,
                source_shard=next(iter(shards)),
                tensors=tuple(by_component[name] for name in _EXPECTED_COMPONENTS),
            )
        )
    return result


def plan_checkpoint(
    index_path: str | Path,
    *,
    include_mtp: bool = False,
) -> list[ExpertBundlePlan]:
    """Build and validate an extraction plan from a safetensors index."""

    data = json.loads(Path(index_path).read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("checkpoint index has no weight_map object")
    result = _collect_bundles(weight_map, _BASE_EXPERT_PATTERN, "base")
    if include_mtp:
        result.extend(_collect_bundles(weight_map, _MTP_EXPERT_PATTERN, "mtp"))
    return result


def write_checkpoint_plan(
    output_path: str | Path,
    bundles: Iterable[ExpertBundlePlan],
    *,
    source_index: str | Path,
) -> None:
    """Write a deterministic, converter-friendly extraction plan."""

    bundle_list = list(bundles)
    payload = {
        "version": 1,
        "source_index": str(source_index),
        "bundle_count": len(bundle_list),
        "bundles": [asdict(bundle) for bundle in bundle_list],
    }
    Path(output_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_safetensors_header(path: str | Path) -> dict[str, SafetensorEntry]:
    """Read tensor byte ranges without importing torch, numpy, or safetensors."""

    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors length prefix")
        (header_length,) = struct.unpack("<Q", prefix)
        if header_length == 0:
            raise ValueError("empty safetensors header")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError("truncated safetensors header")
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid safetensors JSON header") from error

    payload_start = 8 + header_length
    file_size = path.stat().st_size
    entries: dict[str, SafetensorEntry] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        try:
            start, end = metadata["data_offsets"]
            dtype = str(metadata["dtype"])
            shape = tuple(int(dimension) for dimension in metadata["shape"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid tensor metadata for {name!r}") from error
        if start < 0 or end < start:
            raise ValueError(f"invalid data_offsets for {name!r}: {[start, end]}")
        absolute_offset = payload_start + start
        if payload_start + end > file_size:
            raise ValueError(f"tensor payload exceeds shard for {name!r}")
        entries[name] = SafetensorEntry(
            name=name,
            dtype=dtype,
            shape=shape,
            offset=absolute_offset,
            size_bytes=end - start,
        )
    return entries


def measure_bundles_in_shard(
    bundles: Iterable[ExpertBundlePlan],
    shard_path: str | Path,
) -> dict[str, object]:
    """Measure exact expert payload sizes once a source shard is available."""

    shard_path = Path(shard_path)
    entries = read_safetensors_header(shard_path)
    selected = [bundle for bundle in bundles if bundle.source_shard == shard_path.name]
    measurements = []
    for bundle in selected:
        try:
            components = [entries[tensor.tensor_name] for tensor in bundle.tensors]
        except KeyError as error:
            raise ValueError(
                f"index tensor {error.args[0]!r} is missing from {shard_path.name}"
            ) from error
        measurements.append(
            {
                "namespace": bundle.namespace,
                "group_id": bundle.group_id,
                "expert_id": bundle.expert_id,
                "size_bytes": sum(component.size_bytes for component in components),
                "components": [asdict(component) for component in components],
            }
        )
    sizes = [int(item["size_bytes"]) for item in measurements]
    return {
        "shard": shard_path.name,
        "tensor_count": len(entries),
        "expert_bundle_count": len(measurements),
        "expert_size_bytes_min": min(sizes) if sizes else None,
        "expert_size_bytes_max": max(sizes) if sizes else None,
        "expert_size_bytes_unique": sorted(set(sizes)),
        "experts": measurements,
    }
