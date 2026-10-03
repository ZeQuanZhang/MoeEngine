#include "kernels/moeengine_optimizer.h"

#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

}  // namespace

int main() {
    check(moeengine::recommended_expert_cache_bytes(
              uint64_t{24} * 1024 * 1024 * 1024) ==
              uint64_t{1} * 1024 * 1024 * 1024,
          "cache budget applies a model-independent locked-memory ceiling");
    check(moeengine::recommended_expert_cache_bytes(
              uint64_t{4} * 1024 * 1024 * 1024) ==
              uint64_t{512} * 1024 * 1024,
          "small hosts still scale the cache from physical memory");
    check(moeengine::bernoulli_upper_confidence_bound(0, 0) == 1.0,
          "unobserved Bernoulli action remains eligible");
    check(moeengine::bernoulli_upper_confidence_bound(0, 3) > 0.80 &&
              moeengine::bernoulli_upper_confidence_bound(0, 4) < 0.80,
          "confidence bound replaces a model-specific sample threshold");

    moeengine::MemoryTier tier{10'000, 1.0e9, 50'000};
    check(tier.transfer_cost_ns(1'000) == 51'000,
          "tier cost uses measured values");

    moeengine::RouteLocalityEstimator locality(0.5);
    for (int sample = 0; sample < 5; ++sample)
        locality.observe(0, {0, 1});
    for (int expert = 0; expert < 6; ++expert)
        locality.observe(1, {expert});
    check(locality.probability(0, 0) > locality.probability(1, 0),
          "stable route learns higher reuse");

    moeengine::ResidentExpert cheap{{0, 0}, 100, 8, 1, 1'000, 0.1};
    moeengine::ResidentExpert expensive{
        {0, 1}, 100, 8, 1, 100'000, 0.9};
    moeengine::ResidencyOptimizer optimizer;
    auto plan = optimizer.plan_evictions({expensive, cheap}, 100, 10);
    check(plan.feasible() && plan.evict.size() == 1 &&
              plan.evict.front() == cheap.key,
          "optimizer evicts lower expected reload penalty");

    moeengine::ResidentExpert pinned{{0, 2}, 100, 0, 1, 1, 0.0, true};
    plan = optimizer.plan_evictions({pinned, cheap}, 150, 10);
    check(plan.evict.size() == 1 && plan.evict.front() == cheap.key &&
              plan.shortfall_bytes == 50,
          "optimizer never evicts protected entries");

    moeengine::RouterGdsfOptimizer gdsf(2, 0.5);
    gdsf.observe_route(0, {0, 1});
    gdsf.observe_route(0, {0, 1});
    moeengine::ResidentExpert low_cost{{0, 0}, 100, 10, 1, 1'000};
    moeengine::ResidentExpert high_cost{{0, 1}, 100, 11, 1, 100'000};
    gdsf.observe_load(low_cost, 2);
    gdsf.observe_load(high_cost, 2);
    check(gdsf.victim_before(low_cost, high_cost),
          "Router-GDSF retains the higher refetch cost");
    gdsf.observe_eviction(low_cost, 2);
    check(gdsf.inflation() > 0.0 && gdsf.ghost_count() == 1,
          "Router-GDSF advances inflation and records a ghost");
    gdsf.observe_load(low_cost, 3);
    check(gdsf.ghost_count() == 0 &&
              gdsf.reuse_probability(0, 0) > 0.0,
          "short refault restores reuse without resident history");

    gdsf.observe_eviction(low_cost, 3);
    gdsf.observe_eviction(high_cost, 3);
    moeengine::ResidentExpert third{{0, 2}, 100, 12, 1, 10'000};
    gdsf.observe_eviction(third, 3);
    check(gdsf.ghost_count() == 2,
          "Router-GDSF bounds non-resident metadata");

    return failures == 0 ? 0 : 1;
}
