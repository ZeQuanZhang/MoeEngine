import Foundation
import Testing
@testable import Mference

/// Auto always uses the same geometry- and memory-driven slot rule. Resident
/// and fixed rungs stay explicit controls.
@Suite struct ResidencyAutoProfileTests {

    static let gib = UInt64(1) << 30
    /// Approximate Qwen 3.6 sizes: 18.1 GB expert pool, 1.45 GB core.
    static let qwenPool = UInt64(18_100_000_000)
    static let qwenCore = UInt64(1_450_000_000)

    static let qwenProfile = ExpertPoolProfile(
        poolByteSize: qwenPool,
        coreWeightsByteSize: qwenCore,
        expertStride: 1_769_472,
        routedLayerCount: 40,
        expertsPerLayer: 256)

    @Test("24 GiB host picks a geometry-bounded slot rung")
    func geometryOn24GiBPicksSlots() {
        let mode = RuntimeConfiguration.defaultExpertStreamingMode(
            physicalMemoryBytes: 24 * Self.gib,
            profile: Self.qwenProfile)
        guard case .pread(let slots) = mode, slots == 32 else {
            Issue.record("expected 32-slot geometry default, got \(mode)")
            return
        }
    }

    @Test("8 GiB host chooses the smallest supported rung")
    func geometryOn8GiBKeepsSmallSlots() {
        let mode = RuntimeConfiguration.defaultExpertStreamingMode(
            physicalMemoryBytes: 8 * Self.gib,
            profile: Self.qwenProfile)
        guard case .pread(let slots) = mode, slots == 8 else {
            Issue.record("expected 8-slot fallback, got \(mode)")
            return
        }
    }

    @Test("Large experts reduce slots without consulting model family")
    func largeExpertsUseSmallerRung() {
        let largeExpertProfile = ExpertPoolProfile(
            poolByteSize: UInt64(91) * Self.gib,
            coreWeightsByteSize: UInt64(10) * Self.gib,
            expertStride: 8_388_608,
            routedLayerCount: 43,
            expertsPerLayer: 256)
        let mode = RuntimeConfiguration.defaultExpertStreamingMode(
            physicalMemoryBytes: 24 * Self.gib,
            profile: largeExpertProfile)
        guard case .pread(let slots) = mode, slots == 8 else {
            Issue.record("expected 8-slot geometry default, got \(mode)")
            return
        }
    }
}
