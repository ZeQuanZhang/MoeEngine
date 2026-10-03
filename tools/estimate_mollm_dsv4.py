#!/usr/bin/env python3
"""Validate a DSV4 checkpoint and estimate a native mollm package size.

The estimator builds the same prefill/decode graphs as mollm's converter but
does not copy any model bytes.  It is intended as a cheap safety gate before a
roughly 150 GB package conversion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


GIB = 1024**3
DEFAULT_RESERVE_BYTES = 16 * GIB


def _format_gib(value: int) -> str:
    return f"{value / GIB:.2f} GiB"


def _load_mollm(repo: Path):
    models_dir = repo / "models"
    if not (models_dir / "deepseek_v4.py").is_file():
        raise FileNotFoundError(f"mollm converter not found under {models_dir}")
    sys.path.insert(0, str(models_dir))
    from deepseek_v4 import (  # type: ignore[import-not-found]
        _add_engine_boundary_weights,
        build_full_graph,
    )

    return build_full_graph, _add_engine_boundary_weights


def estimate(
    *,
    model_dir: Path,
    mollm_repo: Path,
    output: Path,
    prefill_seq_len: int,
    n_ctx: int,
    reserve_bytes: int,
) -> dict[str, object]:
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    tokenizer_path = model_dir / "tokenizer.json"
    for required in (config_path, index_path, tokenizer_path):
        if not required.is_file():
            raise FileNotFoundError(f"required checkpoint file is missing: {required}")

    cfg = json.loads(config_path.read_text())
    if cfg.get("model_type") != "deepseek_v4":
        raise ValueError(f"expected model_type=deepseek_v4, got {cfg.get('model_type')!r}")

    index_json = json.loads(index_path.read_text())
    shard_names = sorted(set(index_json["weight_map"].values()))
    missing_shards = [name for name in shard_names if not (model_dir / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f"checkpoint is missing {len(missing_shards)} indexed shard(s): "
            + ", ".join(missing_shards[:3])
        )

    build_full_graph, add_engine_boundary_weights = _load_mollm(mollm_repo)
    prefill, prefill_weights = build_full_graph(
        model_dir, cfg, prefill_seq_len, n_ctx
    )
    decode, decode_weights = build_full_graph(model_dir, cfg, 1, n_ctx)
    streamed = dict(prefill_weights)
    streamed.update(decode_weights)

    # The converter adds these after constructing both model graphs.
    from safetensors_stream import SafeTensorIndex  # type: ignore[import-not-found]

    add_engine_boundary_weights(
        (prefill, decode), streamed, SafeTensorIndex(model_dir), cfg
    )

    weights_bytes = sum(weight.size for weight in streamed.values())
    tokenizer_bytes = tokenizer_path.stat().st_size
    tokenizer_config = model_dir / "tokenizer_config.json"
    tokenizer_config_bytes = tokenizer_config.stat().st_size if tokenizer_config.exists() else 0

    # Serialize the two graphs to include their exact sizes. Package metadata
    # is small relative to the weights; one MiB is a deliberately generous
    # allowance for metadata, headers, and the extracted chat template.
    with tempfile.TemporaryDirectory(prefix="moeengine_mollm_estimate_") as tmp:
        tmp_path = Path(tmp)
        prefill.save(str(tmp_path / "prefill"))
        decode.save(str(tmp_path / "decode"))
        graph_bytes = (tmp_path / "prefill.graph").stat().st_size
        graph_bytes += (tmp_path / "decode.graph").stat().st_size

    package_upper_bound = (
        weights_bytes
        + tokenizer_bytes
        + tokenizer_config_bytes
        + graph_bytes
        + 1024**2
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output.parent).free
    safe = free_bytes >= package_upper_bound + reserve_bytes

    return {
        "model_type": cfg["model_type"],
        "layers": int(cfg["num_hidden_layers"]),
        "experts_per_layer": int(cfg["n_routed_experts"]),
        "indexed_shards": len(shard_names),
        "streamed_weights": len(streamed),
        "weights_bytes": weights_bytes,
        "graph_bytes": graph_bytes,
        "package_upper_bound_bytes": package_upper_bound,
        "free_bytes": free_bytes,
        "reserve_bytes": reserve_bytes,
        "safe_to_convert": safe,
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mollm-repo",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "vendor" / "Palm-Infra",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--reserve-gib", type=int, default=16)
    args = parser.parse_args()

    result = estimate(
        model_dir=args.model_dir.resolve(),
        mollm_repo=args.mollm_repo.resolve(),
        output=args.output.resolve(),
        prefill_seq_len=args.prefill_seq_len,
        n_ctx=args.n_ctx,
        reserve_bytes=args.reserve_gib * GIB,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        "package upper bound:",
        _format_gib(int(result["package_upper_bound_bytes"])),
    )
    print("filesystem free:", _format_gib(int(result["free_bytes"])))
    print("required reserve:", _format_gib(int(result["reserve_bytes"])))
    if not result["safe_to_convert"]:
        print("refusing conversion: insufficient free space", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
