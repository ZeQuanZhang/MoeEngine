#include "kernels/moeengine_optimizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace moeengine {

uint64_t recommended_expert_cache_bytes(uint64_t physical_memory_bytes) {
    constexpr uint64_t kConservativeCeiling =
        uint64_t{1024} * 1024 * 1024;
    return std::min(physical_memory_bytes / 8, kConservativeCeiling);
}

double bernoulli_upper_confidence_bound(uint64_t successes,
                                        uint64_t attempts,
                                        double z_score) {
    if (successes > attempts || !(z_score > 0.0) ||
        !std::isfinite(z_score)) {
        throw std::invalid_argument("invalid Bernoulli confidence inputs");
    }
    if (attempts == 0) return 1.0;
    const double n = static_cast<double>(attempts);
    const double p = static_cast<double>(successes) / n;
    const double z2 = z_score * z_score;
    const double denominator = 1.0 + z2 / n;
    const double center = p + z2 / (2.0 * n);
    const double radius = z_score * std::sqrt(
        (p * (1.0 - p) + z2 / (4.0 * n)) / n);
    return std::clamp((center + radius) / denominator, 0.0, 1.0);
}

uint64_t MemoryTier::transfer_cost_ns(size_t size_bytes) const {
    if (size_bytes == 0 || bandwidth_bytes_per_second <= 0.0)
        throw std::invalid_argument("invalid memory tier transfer");
    const long double transfer =
        static_cast<long double>(size_bytes) /
        static_cast<long double>(bandwidth_bytes_per_second) * 1.0e9L;
    const long double total =
        static_cast<long double>(fixed_latency_ns) + transfer;
    if (total >= static_cast<long double>(
                     std::numeric_limits<uint64_t>::max())) {
        return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(std::llround(total));
}

RouteLocalityEstimator::RouteLocalityEstimator(double alpha) : alpha_(alpha) {
    if (!(alpha > 0.0 && alpha <= 1.0))
        throw std::invalid_argument("route-locality alpha must be in (0, 1]");
}

void RouteLocalityEstimator::observe(int layer,
                                     const std::vector<int>& experts) {
    if (layer < 0 || experts.empty())
        throw std::invalid_argument("invalid routed expert group");
    std::unordered_set<int> current;
    for (int expert : experts) {
        if (expert < 0)
            throw std::invalid_argument("expert id must be non-negative");
        current.insert(expert);
    }
    LayerState& state = layers_[layer];
    if (state.has_previous) {
        state.observed.insert(state.previous.begin(), state.previous.end());
        state.observed.insert(current.begin(), current.end());
        for (int expert : state.observed) {
            const double sample =
                state.previous.count(expert) && current.count(expert)
                    ? 1.0 : 0.0;
            const auto found = state.probability.find(expert);
            const double old = found == state.probability.end()
                                   ? sample : found->second;
            state.probability[expert] =
                old + alpha_ * (sample - old);
        }
    }
    state.previous = std::move(current);
    state.has_previous = true;
}

double RouteLocalityEstimator::probability(int layer, int expert) const {
    const auto layer_it = layers_.find(layer);
    if (layer_it == layers_.end()) return 0.0;
    const auto probability_it = layer_it->second.probability.find(expert);
    return probability_it == layer_it->second.probability.end()
               ? 0.0 : probability_it->second;
}

double RouteLocalityEstimator::expected_reuse(
    int layer, const std::vector<int>& experts) const {
    std::unordered_set<int> unique;
    double expected = 0.0;
    for (int expert : experts) {
        if (unique.insert(expert).second)
            expected += probability(layer, expert);
    }
    return expected;
}

void RouteLocalityEstimator::clear() {
    layers_.clear();
}

ResidencyOptimizer::ResidencyOptimizer(OptimizerWeights weights)
    : weights_(weights) {
    if (weights.route_reuse < 0.0 || weights.recency < 0.0 ||
        weights.frequency < 0.0 ||
        weights.route_reuse + weights.recency + weights.frequency <= 0.0) {
        throw std::invalid_argument("invalid optimizer weights");
    }
}

double ResidencyOptimizer::retention_value(
    const ResidentExpert& candidate, uint64_t access_index) const {
    if (candidate.size_bytes == 0 || candidate.access_count == 0 ||
        access_index < candidate.last_access) {
        throw std::invalid_argument("invalid residency candidate");
    }
    const uint64_t age = access_index - candidate.last_access;
    const double recency = 1.0 / (1.0 + static_cast<double>(age));
    const double frequency = std::min(
        1.0, std::log2(1.0 + static_cast<double>(candidate.access_count)) / 8.0);
    const double probability =
        std::clamp(candidate.reuse_probability, 0.0, 1.0);
    const double normalizer =
        weights_.route_reuse + weights_.recency + weights_.frequency;
    const double reuse_score =
        (weights_.route_reuse * probability + weights_.recency * recency +
         weights_.frequency * frequency) / normalizer;
    return reuse_score * static_cast<double>(candidate.reload_cost_ns);
}

bool ResidencyOptimizer::victim_before(
    const ResidentExpert& candidate, const ResidentExpert& current,
    uint64_t access_index) const {
    if (candidate.pinned || candidate.loading || candidate.required)
        return false;
    if (current.pinned || current.loading || current.required)
        return true;
    const double candidate_score =
        retention_value(candidate, access_index) / candidate.size_bytes;
    const double current_score =
        retention_value(current, access_index) / current.size_bytes;
    if (candidate_score != current_score)
        return candidate_score < current_score;
    if (candidate.last_access != current.last_access)
        return candidate.last_access < current.last_access;
    return candidate.key < current.key;
}

EvictionPlan ResidencyOptimizer::plan_evictions(
    const std::vector<ResidentExpert>& candidates, size_t bytes_to_free,
    uint64_t access_index) const {
    EvictionPlan plan;
    if (bytes_to_free == 0) return plan;
    std::vector<const ResidentExpert*> ranked;
    for (const ResidentExpert& candidate : candidates) {
        if (!candidate.pinned && !candidate.loading && !candidate.required)
            ranked.push_back(&candidate);
    }
    std::sort(ranked.begin(), ranked.end(),
              [&](const ResidentExpert* left, const ResidentExpert* right) {
                  return victim_before(*left, *right, access_index);
              });
    for (const ResidentExpert* candidate : ranked) {
        plan.evict.push_back(candidate->key);
        plan.freed_bytes += candidate->size_bytes;
        plan.estimated_reload_penalty_ns +=
            retention_value(*candidate, access_index);
        if (plan.freed_bytes >= bytes_to_free) break;
    }
    plan.shortfall_bytes = plan.freed_bytes >= bytes_to_free
                               ? 0 : bytes_to_free - plan.freed_bytes;
    return plan;
}

RouterGdsfOptimizer::RouterGdsfOptimizer(size_t ghost_limit,
                                         double refault_alpha)
    : ghost_limit_(ghost_limit),
      refault_alpha_(refault_alpha) {
    if (ghost_limit == 0)
        throw std::invalid_argument("ghost limit must be positive");
    if (!(refault_alpha > 0.0 && refault_alpha <= 1.0))
        throw std::invalid_argument("refault alpha must be in (0, 1]");
}

void RouterGdsfOptimizer::observe_route(
    int layer, const std::vector<int>& experts) {
    locality_.observe(layer, experts);
}

double RouterGdsfOptimizer::combined_probability(
    const ExpertKey& key, double refault_probability,
    uint64_t refault_samples) const {
    const double route = locality_.probability(key.layer, key.expert);
    const double refault = std::clamp(refault_probability, 0.0, 1.0);
    const double confidence = static_cast<double>(refault_samples) /
        (static_cast<double>(refault_samples) + 4.0);
    const double evidenced_refault = refault * confidence;
    return 1.0 - (1.0 - route) * (1.0 - evidenced_refault);
}

void RouterGdsfOptimizer::refresh_priority(
    const ResidentExpert& candidate, ResidentState& state) {
    const double size = static_cast<double>(std::max<size_t>(candidate.size_bytes, 1));
    const double cost = static_cast<double>(candidate.reload_cost_ns);
    state.priority = inflation_ +
        combined_probability(candidate.key, state.refault_probability,
                             state.refault_samples) * cost / size;
}

void RouterGdsfOptimizer::observe_load(
    const ResidentExpert& candidate, uint64_t route_epoch) {
    ResidentState state;
    const auto ghost = ghosts_.find(candidate.key);
    if (ghost != ghosts_.end()) {
        state.refault_probability = ghost->second.refault_probability;
        state.refault_samples = ghost->second.refault_samples;
        const uint64_t distance =
            route_epoch > ghost->second.evicted_epoch
                ? route_epoch - ghost->second.evicted_epoch : 1;
        const double sample = 1.0 / std::sqrt(static_cast<double>(distance));
        if (state.refault_samples == 0) {
            state.refault_probability = sample;
        } else {
            state.refault_probability += refault_alpha_ *
                (sample - state.refault_probability);
        }
        ++state.refault_samples;
        ghosts_.erase(ghost);
    }
    refresh_priority(candidate, state);
    resident_[candidate.key] = state;
}

void RouterGdsfOptimizer::observe_access(
    const ResidentExpert& candidate) {
    ResidentState& state = resident_[candidate.key];
    refresh_priority(candidate, state);
}

double RouterGdsfOptimizer::priority_for(
    const ResidentExpert& candidate) const {
    const auto resident = resident_.find(candidate.key);
    if (resident != resident_.end()) return resident->second.priority;
    const double size = static_cast<double>(std::max<size_t>(candidate.size_bytes, 1));
    const double cost = static_cast<double>(candidate.reload_cost_ns);
    return inflation_ +
        combined_probability(candidate.key, 0.0, 0) * cost / size;
}

double RouterGdsfOptimizer::expected_penalty_per_byte(
    const ResidentExpert& candidate) const {
    const auto resident = resident_.find(candidate.key);
    const double refault_probability = resident == resident_.end()
        ? 0.0 : resident->second.refault_probability;
    const uint64_t refault_samples = resident == resident_.end()
        ? 0 : resident->second.refault_samples;
    const double size = static_cast<double>(
        std::max<size_t>(candidate.size_bytes, 1));
    return combined_probability(candidate.key, refault_probability,
                                refault_samples) *
        static_cast<double>(candidate.reload_cost_ns) / size;
}

void RouterGdsfOptimizer::observe_eviction(
    const ResidentExpert& candidate, uint64_t route_epoch) {
    GhostState ghost;
    const auto resident = resident_.find(candidate.key);
    if (resident != resident_.end()) {
        ghost.priority = resident->second.priority;
        ghost.refault_probability = resident->second.refault_probability;
        ghost.refault_samples = resident->second.refault_samples;
        resident_.erase(resident);
    } else {
        ghost.priority = priority_for(candidate);
    }
    inflation_ = std::max(inflation_, ghost.priority);
    ghost.evicted_epoch = route_epoch;
    ghost.serial = next_ghost_serial_++;
    ghosts_[candidate.key] = ghost;
    ghost_order_.push_back({candidate.key, ghost.serial});
    trim_ghosts();
}

bool RouterGdsfOptimizer::victim_before(
    const ResidentExpert& candidate, const ResidentExpert& current) const {
    if (candidate.pinned || candidate.loading || candidate.required)
        return false;
    if (current.pinned || current.loading || current.required)
        return true;
    // Admission value is the primary order.  The GreedyDual clock ages only
    // candidates with equivalent expected reload penalty, which prevents a
    // sequential pass over unrelated layers from flushing a learned route.
    const double candidate_penalty = expected_penalty_per_byte(candidate);
    const double current_penalty = expected_penalty_per_byte(current);
    if (candidate_penalty != current_penalty)
        return candidate_penalty < current_penalty;
    const double candidate_priority = priority_for(candidate);
    const double current_priority = priority_for(current);
    if (candidate_priority != current_priority)
        return candidate_priority < current_priority;
    if (candidate.last_access != current.last_access)
        return candidate.last_access < current.last_access;
    return candidate.key < current.key;
}

double RouterGdsfOptimizer::reuse_probability(int layer, int expert) const {
    const ExpertKey key{layer, expert};
    const auto resident = resident_.find(key);
    if (resident != resident_.end()) {
        return combined_probability(key, resident->second.refault_probability,
                                    resident->second.refault_samples);
    }
    const auto ghost = ghosts_.find(key);
    return combined_probability(
        key, ghost == ghosts_.end() ? 0.0 : ghost->second.refault_probability,
        ghost == ghosts_.end() ? 0 : ghost->second.refault_samples);
}

double RouterGdsfOptimizer::expected_reuse(
    int layer, const std::vector<int>& experts) const {
    std::unordered_set<int> unique;
    double expected = 0.0;
    for (int expert : experts) {
        if (unique.insert(expert).second)
            expected += reuse_probability(layer, expert);
    }
    return expected;
}

void RouterGdsfOptimizer::trim_ghosts() {
    while (ghosts_.size() > ghost_limit_ && !ghost_order_.empty()) {
        const GhostOrder oldest = ghost_order_.front();
        ghost_order_.pop_front();
        const auto ghost = ghosts_.find(oldest.key);
        if (ghost != ghosts_.end() && ghost->second.serial == oldest.serial)
            ghosts_.erase(ghost);
    }
}

void RouterGdsfOptimizer::clear() {
    inflation_ = 0.0;
    next_ghost_serial_ = 1;
    locality_.clear();
    resident_.clear();
    ghosts_.clear();
    ghost_order_.clear();
}

}  // namespace moeengine
