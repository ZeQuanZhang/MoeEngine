import Testing
@testable import Mference

@Suite struct RuntimeConfigurationTests {
    @Test func productionDefaultsAreStable() {
        let runtime = RuntimeConfiguration.production
        #expect(runtime.fp16RingEnabled)
        #expect(runtime.expertCacheSlots == 16)
        #expect(runtime.expertCachePolicy == .lfu)
        #expect(runtime.rdadvisePolicy == .off)
        #expect(!runtime.rdadviseEnabled)
        #expect(runtime.prefillPolicy == .chunked)
        #expect(runtime.prefillChunkTokens == 128)
        #expect(runtime.prefillAttentionPath == .fullTensorOps2DPreferred)
        #expect(runtime.headPath == .fusedRows)
        #expect(!runtime.useMapleFlashHead)
    }

    @Test func automaticExpertCacheSlotsUseMemoryAndGeometry() {
        let eightGiB = UInt64(8) * 1024 * 1024 * 1024
        let twentyFourGiB = UInt64(24) * 1024 * 1024 * 1024
        let sixteenGiB = UInt64(16) * 1024 * 1024 * 1024
        let profile = ExpertPoolProfile(
            poolByteSize: 18_119_393_280,
            coreWeightsByteSize: 1_389_476_096,
            expertStride: 1_769_472,
            routedLayerCount: 40,
            expertsPerLayer: 256)
        #expect(RuntimeConfiguration.defaultExpertCacheSlots(
            physicalMemoryBytes: twentyFourGiB, profile: profile) == 32)
        #expect(RuntimeConfiguration.defaultExpertCacheSlots(
            physicalMemoryBytes: sixteenGiB, profile: profile) == 24)
        #expect(RuntimeConfiguration.defaultExpertCacheSlots(
            physicalMemoryBytes: eightGiB, profile: profile) == 8)

        let sameGeometryDifferentPoolIdentity = ExpertPoolProfile(
            poolByteSize: profile.poolByteSize,
            coreWeightsByteSize: profile.coreWeightsByteSize,
            expertStride: profile.expertStride,
            routedLayerCount: profile.routedLayerCount,
            expertsPerLayer: profile.expertsPerLayer)
        #expect(RuntimeConfiguration.defaultExpertCacheSlots(
            physicalMemoryBytes: twentyFourGiB,
            profile: sameGeometryDifferentPoolIdentity) == 32)
    }

    @Test func retainedControlsReachTypedRuntime() {
        let runtime = RuntimeConfiguration(
            expertCacheSlots: 32,
            expertCachePolicy: .lru,
            rdadvisePolicy: .adaptive,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            prefillAttentionPath: .causalTiled,
            forceLogitsHead: true,
            useMapleFlashHead: true)
        #expect(runtime.expertCacheSlots == 32)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.rdadviseEnabled)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.prefillAttentionPath == .causalTiled)
        #expect(runtime.headPath == .logits)
        #expect(runtime.useMapleFlashHead)
    }

    @Test(arguments: [32, 64, 128, 256, 512, 1024, 2048, 4096])
    func productionPrefillSupportsPublicChunkSizes(_ chunkTokens: Int) {
        let runtime = RuntimeConfiguration(prefillChunkTokens: chunkTokens)
        #expect(runtime.prefillConfig.mode == .chunked)
        #expect(runtime.prefillConfig.chunkTokens == chunkTokens)
    }
}
