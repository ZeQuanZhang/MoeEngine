"""Synthetic routed-expert access traces."""

from __future__ import annotations

import random
from collections.abc import Sequence

from .model import ExpertAddress


def _validate_shape(
    num_tokens: int,
    num_layers: int,
    experts_per_layer: int,
    top_k: int,
) -> None:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if experts_per_layer <= 0:
        raise ValueError("experts_per_layer must be positive")
    if not 1 <= top_k <= experts_per_layer:
        raise ValueError("top_k must be between 1 and experts_per_layer")


def _weighted_unique_sample(
    rng: random.Random,
    population: Sequence[int],
    weights: Sequence[float],
    count: int,
) -> list[int]:
    selected: list[int] = []
    selected_set: set[int] = set()
    while len(selected) < count:
        item = rng.choices(population, weights=weights, k=1)[0]
        if item not in selected_set:
            selected.append(item)
            selected_set.add(item)
    return selected


def random_trace(
    *,
    num_tokens: int,
    num_layers: int,
    experts_per_layer: int,
    top_k: int,
    seed: int = 0,
) -> list[ExpertAddress]:
    _validate_shape(num_tokens, num_layers, experts_per_layer, top_k)
    rng = random.Random(seed)
    trace: list[ExpertAddress] = []
    for _token in range(num_tokens):
        for layer_id in range(num_layers):
            for expert_id in rng.sample(range(experts_per_layer), top_k):
                trace.append(ExpertAddress(layer_id, expert_id))
    return trace


def zipf_trace(
    *,
    num_tokens: int,
    num_layers: int,
    experts_per_layer: int,
    top_k: int,
    alpha: float = 1.2,
    seed: int = 0,
) -> list[ExpertAddress]:
    _validate_shape(num_tokens, num_layers, experts_per_layer, top_k)
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    rng = random.Random(seed)
    population = list(range(experts_per_layer))
    rank_weights = [1.0 / ((rank + 1) ** alpha) for rank in population]
    layer_orders = []
    for _layer in range(num_layers):
        order = population.copy()
        rng.shuffle(order)
        layer_orders.append(order)
    trace: list[ExpertAddress] = []
    for _token in range(num_tokens):
        for layer_id in range(num_layers):
            selected_ranks = _weighted_unique_sample(
                rng,
                population,
                rank_weights,
                top_k,
            )
            for rank in selected_ranks:
                trace.append(ExpertAddress(layer_id, layer_orders[layer_id][rank]))
    return trace


def markov_trace(
    *,
    num_tokens: int,
    num_layers: int,
    experts_per_layer: int,
    top_k: int,
    stay_probability: float = 0.75,
    seed: int = 0,
) -> list[ExpertAddress]:
    _validate_shape(num_tokens, num_layers, experts_per_layer, top_k)
    if not 0.0 <= stay_probability <= 1.0:
        raise ValueError("stay_probability must be between 0 and 1")
    rng = random.Random(seed)
    previous = [rng.sample(range(experts_per_layer), top_k) for _ in range(num_layers)]
    trace: list[ExpertAddress] = []
    for token_index in range(num_tokens):
        for layer_id in range(num_layers):
            if token_index == 0:
                selected = previous[layer_id]
            else:
                retained = [
                    expert_id
                    for expert_id in previous[layer_id]
                    if rng.random() < stay_probability
                ]
                rng.shuffle(retained)
                retained = retained[:top_k]
                available = [
                    expert_id
                    for expert_id in range(experts_per_layer)
                    if expert_id not in retained
                ]
                selected = retained + rng.sample(available, top_k - len(retained))
                previous[layer_id] = selected
            for expert_id in selected:
                trace.append(ExpertAddress(layer_id, expert_id))
    return trace


def bursty_trace(
    *,
    num_tokens: int,
    num_layers: int,
    experts_per_layer: int,
    top_k: int,
    burst_tokens: int = 32,
    hot_set_size: int | None = None,
    hot_probability: float = 0.9,
    seed: int = 0,
) -> list[ExpertAddress]:
    _validate_shape(num_tokens, num_layers, experts_per_layer, top_k)
    if burst_tokens <= 0:
        raise ValueError("burst_tokens must be positive")
    if not 0.0 <= hot_probability <= 1.0:
        raise ValueError("hot_probability must be between 0 and 1")
    if hot_set_size is None:
        hot_set_size = min(experts_per_layer, max(top_k, top_k * 4))
    if not top_k <= hot_set_size <= experts_per_layer:
        raise ValueError("hot_set_size must be between top_k and experts_per_layer")
    rng = random.Random(seed)
    hot_sets: list[list[int]] = []
    trace: list[ExpertAddress] = []
    for token_index in range(num_tokens):
        if token_index % burst_tokens == 0:
            hot_sets = [
                rng.sample(range(experts_per_layer), hot_set_size)
                for _ in range(num_layers)
            ]
        for layer_id in range(num_layers):
            selected: list[int] = []
            selected_set: set[int] = set()
            while len(selected) < top_k:
                if rng.random() < hot_probability:
                    candidate = rng.choice(hot_sets[layer_id])
                else:
                    candidate = rng.randrange(experts_per_layer)
                if candidate not in selected_set:
                    selected.append(candidate)
                    selected_set.add(candidate)
            for expert_id in selected:
                trace.append(ExpertAddress(layer_id, expert_id))
    return trace


def generate_trace(pattern: str, **kwargs: object) -> list[ExpertAddress]:
    normalized = pattern.strip().lower()
    if normalized == "random":
        return random_trace(**kwargs)  # type: ignore[arg-type]
    if normalized == "zipf":
        return zipf_trace(**kwargs)  # type: ignore[arg-type]
    if normalized == "markov":
        return markov_trace(**kwargs)  # type: ignore[arg-type]
    if normalized == "bursty":
        return bursty_trace(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown trace pattern: {pattern}")

