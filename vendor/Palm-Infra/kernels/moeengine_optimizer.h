#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace moeengine {

struct MemoryTier {
    size_t capacity_bytes = 0;
    double bandwidth_bytes_per_second = 0.0;
    uint64_t fixed_latency_ns = 0;

    uint64_t transfer_cost_ns(size_t size_bytes) const;
};

// Conservative hardware-only budget for frontends that cannot inspect core
// and sidecar residency before constructing the runtime. The ceiling prevents
// an explicitly locked cache from forcing unrelated system pages into swap;
// package-aware frontends can instead account for their complete working set.
uint64_t recommended_expert_cache_bytes(uint64_t physical_memory_bytes);

struct ExpertKey {
    int layer = -1;
    int expert = -1;

    bool operator==(const ExpertKey& other) const {
        return layer == other.layer && expert == other.expert;
    }
    bool operator<(const ExpertKey& other) const {
        return layer != other.layer ? layer < other.layer
                                    : expert < other.expert;
    }
};

struct ExpertKeyHash {
    size_t operator()(const ExpertKey& key) const {
        const uint64_t layer = static_cast<uint32_t>(key.layer);
        const uint64_t expert = static_cast<uint32_t>(key.expert);
        return static_cast<size_t>((layer << 32) ^ expert);
    }
};

struct ResidentExpert {
    ExpertKey key;
    size_t size_bytes = 0;
    uint64_t last_access = 0;
    uint64_t access_count = 0;
    uint64_t reload_cost_ns = 0;
    double reuse_probability = 0.0;
    bool pinned = false;
    bool loading = false;
    bool required = false;
};

struct OptimizerWeights {
    double route_reuse = 0.60;
    double recency = 0.25;
    double frequency = 0.15;
};

struct EvictionPlan {
    std::vector<ExpertKey> evict;
    size_t freed_bytes = 0;
    size_t shortfall_bytes = 0;
    double estimated_reload_penalty_ns = 0.0;

    bool feasible() const { return shortfall_bytes == 0; }
};

// Upper end of a Wilson score interval for a Bernoulli process.  With no
// observations the bound is 1.0, so callers explore until evidence—not a
// model-specific warm-up count—shows that an action cannot meet its required
// success rate.
double bernoulli_upper_confidence_bound(uint64_t successes,
                                        uint64_t attempts,
                                        double z_score = 3.890592);

class RouteLocalityEstimator {
public:
    explicit RouteLocalityEstimator(double alpha = 0.125);

    void observe(int layer, const std::vector<int>& experts);
    double probability(int layer, int expert) const;
    double expected_reuse(int layer, const std::vector<int>& experts) const;
    void clear();

private:
    struct LayerState {
        std::unordered_set<int> previous;
        std::unordered_set<int> observed;
        std::unordered_map<int, double> probability;
        bool has_previous = false;
    };

    double alpha_;
    std::unordered_map<int, LayerState> layers_;
};

class ResidencyOptimizer {
public:
    explicit ResidencyOptimizer(OptimizerWeights weights = {});

    double retention_value(const ResidentExpert& candidate,
                           uint64_t access_index) const;
    bool victim_before(const ResidentExpert& candidate,
                       const ResidentExpert& current,
                       uint64_t access_index) const;
    EvictionPlan plan_evictions(const std::vector<ResidentExpert>& candidates,
                                size_t bytes_to_free,
                                uint64_t access_index) const;

private:
    OptimizerWeights weights_;
};

// Router-aware GreedyDual-Size with bounded non-resident history.  The
// inflation clock ages priorities without scanning resident entries, while
// ghost refaults reveal reuse that a too-small cache could not observe as hits.
class RouterGdsfOptimizer {
public:
    explicit RouterGdsfOptimizer(size_t ghost_limit = 32768,
                                 double refault_alpha = 0.125);

    void observe_route(int layer, const std::vector<int>& experts);
    void observe_load(const ResidentExpert& candidate, uint64_t route_epoch);
    void observe_access(const ResidentExpert& candidate);
    void observe_eviction(const ResidentExpert& candidate,
                          uint64_t route_epoch);
    bool victim_before(const ResidentExpert& candidate,
                       const ResidentExpert& current) const;
    double expected_reuse(int layer, const std::vector<int>& experts) const;
    double reuse_probability(int layer, int expert) const;
    double inflation() const { return inflation_; }
    size_t ghost_count() const { return ghosts_.size(); }
    void clear();

private:
    struct ResidentState {
        double priority = 0.0;
        double refault_probability = 0.0;
        uint64_t refault_samples = 0;
    };
    struct GhostState {
        double priority = 0.0;
        double refault_probability = 0.0;
        uint64_t refault_samples = 0;
        uint64_t evicted_epoch = 0;
        uint64_t serial = 0;
    };
    struct GhostOrder {
        ExpertKey key;
        uint64_t serial = 0;
    };

    double combined_probability(const ExpertKey& key,
                                double refault_probability,
                                uint64_t refault_samples) const;
    double expected_penalty_per_byte(
        const ResidentExpert& candidate) const;
    double priority_for(const ResidentExpert& candidate) const;
    void refresh_priority(const ResidentExpert& candidate,
                          ResidentState& state);
    void trim_ghosts();

    size_t ghost_limit_;
    double refault_alpha_;
    double inflation_ = 0.0;
    uint64_t next_ghost_serial_ = 1;
    RouteLocalityEstimator locality_;
    std::unordered_map<ExpertKey, ResidentState, ExpertKeyHash> resident_;
    std::unordered_map<ExpertKey, GhostState, ExpertKeyHash> ghosts_;
    std::deque<GhostOrder> ghost_order_;
};

}  // namespace moeengine
