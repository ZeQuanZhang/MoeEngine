"""Command-line interface for the first MoeEngine milestone."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

from .checkpoint import measure_bundles_in_shard, plan_checkpoint, write_checkpoint_plan
from .checkpoint_store import SafetensorsExpertStore
from .gguf_store import GgufExpertStore
from .inspect import inspect_config, inspect_safetensors_index
from .model import ExpertAddress
from .replay import BenchmarkResult, replay_trace
from .store import SyntheticExpertStore
from .trace import generate_trace


MIB = 1024**2
GIB = 1024**3


def _csv_floats(value: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return result


def _csv_strings(value: str) -> list[str]:
    result = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated names")
    unknown = set(result) - {"lru", "lfu", "belady", "generic", "gdsf"}
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown policies: {sorted(unknown)}")
    return result


def _human_bytes(value: int) -> str:
    if value >= GIB:
        return f"{value / GIB:.2f} GiB"
    if value >= MIB:
        return f"{value / MIB:.2f} MiB"
    return f"{value} B"


def _print_results(results: Sequence[BenchmarkResult]) -> None:
    header = (
        f"{'policy':<8} {'cache':>10} {'hit rate':>10} "
        f"{'faults':>10} {'loaded':>12} {'p95 fault':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.policy:<8} "
            f"{_human_bytes(result.capacity_bytes):>10} "
            f"{result.hit_rate:>9.2%} "
            f"{result.model_faults:>10,} "
            f"{_human_bytes(result.bytes_loaded):>12} "
            f"{result.p95_fault_latency_ns / 1_000_000:>9.3f} ms"
        )


def _benchmark(args: argparse.Namespace) -> int:
    trace_kwargs: dict[str, object] = {
        "num_tokens": args.tokens,
        "num_layers": args.layers,
        "experts_per_layer": args.experts_per_layer,
        "top_k": args.top_k,
        "seed": args.seed,
    }
    if args.pattern == "zipf":
        trace_kwargs["alpha"] = args.zipf_alpha
    elif args.pattern == "markov":
        trace_kwargs["stay_probability"] = args.stay_probability
    elif args.pattern == "bursty":
        trace_kwargs["burst_tokens"] = args.burst_tokens
        trace_kwargs["hot_set_size"] = args.hot_set_size
        trace_kwargs["hot_probability"] = args.hot_probability
    trace = generate_trace(args.pattern, **trace_kwargs)
    store = SyntheticExpertStore(
        num_layers=args.layers,
        experts_per_layer=args.experts_per_layer,
        expert_size_bytes=round(args.expert_size_mib * MIB),
        bandwidth_bytes_per_second=args.bandwidth_gbps * 1_000_000_000,
        fixed_latency_ns=round(args.fixed_latency_us * 1_000),
    )
    results = [
        replay_trace(
            trace,
            store=store,
            capacity_bytes=round(cache_gib * GIB),
            policy_name=policy_name,
            route_width=args.top_k,
        )
        for cache_gib in args.cache_gib
        for policy_name in args.policies
    ]
    if args.json:
        print(json.dumps([result.as_dict() for result in results], indent=2))
    else:
        _print_results(results)
    return 0


def _inspect_config(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_config(args.path), indent=2, sort_keys=True))
    return 0


def _inspect_index(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_safetensors_index(args.path), indent=2, sort_keys=True))
    return 0


def _plan_checkpoint(args: argparse.Namespace) -> int:
    bundles = plan_checkpoint(args.index, include_mtp=args.include_mtp)
    write_checkpoint_plan(args.output, bundles, source_index=args.index)
    namespaces: dict[str, int] = {}
    shards: set[str] = set()
    for bundle in bundles:
        namespaces[bundle.namespace] = namespaces.get(bundle.namespace, 0) + 1
        shards.add(bundle.source_shard)
    print(
        json.dumps(
            {
                "output": str(Path(args.output)),
                "bundle_count": len(bundles),
                "namespaces": namespaces,
                "source_shard_count": len(shards),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _inspect_shard(args: argparse.Namespace) -> int:
    bundles = plan_checkpoint(args.index, include_mtp=args.include_mtp)
    result = measure_bundles_in_shard(bundles, args.shard)
    if not args.verbose:
        result.pop("experts")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _benchmark_real(args: argparse.Namespace) -> int:
    if args.layer < 0:
        raise ValueError("layer must be non-negative")
    if args.working_set <= 0:
        raise ValueError("working-set must be positive")
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")

    model_path = Path(args.model)
    is_gguf = model_path.suffix.lower() == ".gguf" or (
        model_path.is_dir() and any(model_path.glob("*.gguf"))
    )
    store_factory = (
        GgufExpertStore.from_model
        if is_gguf
        else SafetensorsExpertStore.from_checkpoint
    )
    with store_factory(
        args.model, layer_ids=[args.layer], no_os_cache=args.no_os_cache
    ) as store:
        layer_addresses = sorted(
            address for address in store.addresses if address.layer_id == args.layer
        )
        if args.working_set > len(layer_addresses):
            raise ValueError(
                f"working-set {args.working_set} exceeds the layer's "
                f"{len(layer_addresses)} experts"
            )
        selected = layer_addresses[: args.working_set]
        trace = selected * args.rounds

        sample = store.load(selected[0])
        assert sample.payload is not None
        sample_sha256 = hashlib.sha256(sample.payload).hexdigest()
        layout = store.layout(selected[0])
        results = []
        for cache_mib in args.cache_mib:
            started_ns = time.perf_counter_ns()
            benchmark = replay_trace(
                trace,
                store=store,
                capacity_bytes=round(cache_mib * MIB),
                policy_name=args.policy,
                route_width=args.working_set,
            )
            wall_ns = time.perf_counter_ns() - started_ns
            fault_io_ns = benchmark.mean_fault_latency_ns * benchmark.model_faults
            results.append(
                {
                    **benchmark.as_dict(),
                    "wall_time_ns": wall_ns,
                    "wall_throughput_mib_per_second": (
                        benchmark.bytes_loaded / MIB / (wall_ns / 1_000_000_000)
                        if wall_ns
                        else 0.0
                    ),
                    "fault_io_throughput_mib_per_second": (
                        benchmark.bytes_loaded / MIB / (fault_io_ns / 1_000_000_000)
                        if fault_io_ns
                        else 0.0
                    ),
                }
            )
        source_files = (
            list(layout.source_files)
            if hasattr(layout, "source_files")
            else [layout.shard]
        )
        payload = {
            "model": str(Path(args.model)),
            "format": "gguf" if is_gguf else "safetensors",
            "layer": args.layer,
            "experts_in_layer": len(layer_addresses),
            "working_set_experts": args.working_set,
            "rounds": args.rounds,
            "trace_accesses": len(trace),
            "no_os_cache": args.no_os_cache,
            "sample_expert": {
                "address": str(selected[0]),
                "source_files": source_files,
                "tensor_count": len(layout.tensors),
                "size_bytes": sample.descriptor.size_bytes,
                "sha256": sample_sha256,
            },
            "results": results,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moeengine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="replay a synthetic trace")
    benchmark.add_argument(
        "--pattern",
        choices=("random", "zipf", "markov", "bursty"),
        default="bursty",
    )
    benchmark.add_argument("--tokens", type=int, default=200)
    benchmark.add_argument("--layers", type=int, default=43)
    benchmark.add_argument("--experts-per-layer", type=int, default=256)
    benchmark.add_argument("--top-k", type=int, default=6)
    benchmark.add_argument("--expert-size-mib", type=float, default=12.0)
    benchmark.add_argument("--cache-gib", type=_csv_floats, default=[2.0, 4.0, 8.0])
    benchmark.add_argument(
        "--policies", type=_csv_strings, default=["lru", "lfu", "belady"]
    )
    benchmark.add_argument("--bandwidth-gbps", type=float, default=5.0)
    benchmark.add_argument("--fixed-latency-us", type=float, default=80.0)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--zipf-alpha", type=float, default=1.2)
    benchmark.add_argument("--stay-probability", type=float, default=0.75)
    benchmark.add_argument("--burst-tokens", type=int, default=32)
    benchmark.add_argument("--hot-set-size", type=int, default=None)
    benchmark.add_argument("--hot-probability", type=float, default=0.9)
    benchmark.add_argument("--json", action="store_true")
    benchmark.set_defaults(handler=_benchmark)

    inspect_parser = subparsers.add_parser(
        "inspect-config", help="estimate expert layout from a model config"
    )
    inspect_parser.add_argument("path")
    inspect_parser.set_defaults(handler=_inspect_config)

    index_parser = subparsers.add_parser(
        "inspect-index", help="classify tensors in a safetensors index"
    )
    index_parser.add_argument("path")
    index_parser.set_defaults(handler=_inspect_index)

    plan_parser = subparsers.add_parser(
        "plan-checkpoint", help="validate and emit a per-expert extraction plan"
    )
    plan_parser.add_argument("index")
    plan_parser.add_argument("output")
    plan_parser.add_argument("--include-mtp", action="store_true")
    plan_parser.set_defaults(handler=_plan_checkpoint)

    shard_parser = subparsers.add_parser(
        "inspect-shard", help="measure expert payloads from a safetensors header"
    )
    shard_parser.add_argument("index")
    shard_parser.add_argument("shard")
    shard_parser.add_argument("--include-mtp", action="store_true")
    shard_parser.add_argument("--verbose", action="store_true")
    shard_parser.set_defaults(handler=_inspect_shard)

    real_parser = subparsers.add_parser(
        "benchmark-real",
        help="replay a bounded trace against safetensors or GGUF expert weights",
    )
    real_parser.add_argument("model")
    real_parser.add_argument("--layer", type=int, default=0)
    real_parser.add_argument("--working-set", type=int, default=16)
    real_parser.add_argument("--rounds", type=int, default=3)
    real_parser.add_argument("--cache-mib", type=_csv_floats, default=[128.0, 256.0])
    real_parser.add_argument(
        "--policy",
        choices=("lru", "lfu", "belady", "generic", "gdsf"),
        default="lru",
    )
    real_parser.add_argument("--no-os-cache", action="store_true")
    real_parser.set_defaults(handler=_benchmark_real)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
