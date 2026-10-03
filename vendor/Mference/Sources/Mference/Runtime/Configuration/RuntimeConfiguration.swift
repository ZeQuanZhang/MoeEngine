import Foundation

public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
}

public enum RuntimePrefillAttentionPath: String, Codable, Sendable {
    case causalTiled = "causal-tiled"
    case fullTensorOps2DPreferred = "full-tensorops-2d-preferred"
    case fullTensorOps2DValidityV2 = "full-tensorops-2d-validity-v2"
}

public enum RuntimeExpertCachePolicy: String, Codable, Sendable {
    case lfu
    case lru
}

public struct RuntimeConfiguration: Sendable, Equatable {
    /// 96 and 128 are the near-resident rungs: large wired LFU sets for hosts
    /// with RAM to spare but not enough to cache the whole expert pool.
    public static let allowedExpertCacheSlots = [8, 16, 24, 32, 64, 96, 128]
    public static let allowedPrefillChunkTokens = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let headPath: RuntimeHeadPath
    /// Opt-in approximate singleton-decode head for Maple checkpoints that
    /// retain FlashHead tensors. Prefill continues to use the exact head.
    public let useMapleFlashHead: Bool

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 128,
                prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
                forceLogitsHead: Bool = false,
                useMapleFlashHead: Bool = false) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.useMapleFlashHead = useMapleFlashHead
    }

    public static var production: RuntimeConfiguration {
        RuntimeConfiguration()
    }

    /// Size the cache from host memory and package geometry, never a model
    /// family. One eighth of physical memory is the maximum automatic cache
    /// budget; the core weights plus OS headroom must still leave twice that
    /// budget free. The result is rounded down to a supported slot rung.
    public static func defaultExpertCacheSlots(
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        profile: ExpertPoolProfile
    ) -> Int {
        guard profile.bytesPerCacheSlot > 0,
              profile.bytesPerCacheSlot < UInt64.max,
              profile.expertsPerLayer > 0 else {
            return allowedExpertCacheSlots.min()!
        }
        let reserved = profile.coreWeightsByteSize.addingReportingOverflow(
            residentHeadroomBytes)
        let spareBytes: UInt64
        if reserved.overflow || reserved.partialValue >= physicalMemoryBytes {
            spareBytes = 0
        } else {
            spareBytes = physicalMemoryBytes - reserved.partialValue
        }
        let budget = min(physicalMemoryBytes / 8, spareBytes / 2)
        let geometryLimit = min(
            UInt64(profile.expertsPerLayer),
            budget / profile.bytesPerCacheSlot)
        return allowedExpertCacheSlots.reversed().first {
            UInt64($0) <= geometryLimit
        } ?? allowedExpertCacheSlots.min()!
    }

    /// Fixed reserve the resident rung leaves for the KV cache, scratch, the
    /// process, and the OS. Clean file-backed expert pages degrade toward
    /// page-cache streaming under pressure, so the rung only needs the nominal
    /// working set to fit.
    static let residentHeadroomBytes = UInt64(4) * 1024 * 1024 * 1024

    /// Auto remains explicit pread-based streaming. Resident mapping and fixed
    /// slot rungs are opt-in controls; the default never changes storage mode
    /// based on a model identity.
    public static func defaultExpertStreamingMode(
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        profile: ExpertPoolProfile
    ) -> ExpertStreamingMode {
        return .pread(slotCount: defaultExpertCacheSlots(
            physicalMemoryBytes: physicalMemoryBytes,
            profile: profile))
    }

    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    public var modelExpertCachePolicy: ExpertCachePolicy {
        expertCachePolicy == .lru ? .lru : .lfu
    }
}
