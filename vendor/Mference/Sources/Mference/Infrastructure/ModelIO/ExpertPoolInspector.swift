import Foundation

/// Model-name-independent inputs for sizing a routed-expert cache. A cache
/// slot exists once per routed layer, so `bytesPerCacheSlot` is the actual RAM
/// cost of increasing the slot count by one across the model.
public struct ExpertPoolProfile: Equatable, Sendable {
    public let poolByteSize: UInt64
    public let coreWeightsByteSize: UInt64
    public let expertStride: UInt64
    public let routedLayerCount: Int
    public let expertsPerLayer: Int
    public let bytesPerCacheSlot: UInt64

    public init(poolByteSize: UInt64,
                coreWeightsByteSize: UInt64,
                expertStride: UInt64,
                routedLayerCount: Int,
                expertsPerLayer: Int) {
        self.poolByteSize = poolByteSize
        self.coreWeightsByteSize = coreWeightsByteSize
        self.expertStride = expertStride
        self.routedLayerCount = routedLayerCount
        self.expertsPerLayer = expertsPerLayer
        let (slotBytes, overflow) = expertStride.multipliedReportingOverflow(
            by: UInt64(max(0, routedLayerCount)))
        self.bytesPerCacheSlot = overflow ? UInt64.max : slotBytes
    }
}

/// Pre-load sizing for the expert-cache auto profile: how many bytes the
/// routed-expert pool and core weights occupy on disk, read from
/// `packed_experts/layout.json` and `model_weights.bin` without loading
/// the model.
public enum ExpertPoolInspector {

    public static func profile(directoryURL: URL) throws -> ExpertPoolProfile {
        let layout = try PackedExpertsLayoutReader.load(directoryURL: directoryURL)
        let poolBytes = layout.layers.reduce(UInt64(0)) { total, layer in
            total + UInt64(layer.experts.count) * layout.expertStride
        }
        let activeLayers = layout.layers.reduce(0) { count, layer in
            count + (layer.experts.isEmpty ? 0 : 1)
        }
        return ExpertPoolProfile(
            poolByteSize: poolBytes,
            coreWeightsByteSize: try coreWeightsByteSize(directoryURL: directoryURL),
            expertStride: layout.expertStride,
            routedLayerCount: activeLayers,
            expertsPerLayer: layout.expertsPerLayer)
    }

    /// Total routed-expert bytes across all layers. Dense-FFN layers with no
    /// routed experts contribute nothing.
    public static func poolByteSize(directoryURL: URL) throws -> UInt64 {
        let layout = try PackedExpertsLayoutReader.load(directoryURL: directoryURL)
        return layout.layers.reduce(UInt64(0)) { total, layer in
            total + UInt64(layer.experts.count) * layout.expertStride
        }
    }

    /// On-disk size of `model_weights.bin` (resident core weights + index).
    public static func coreWeightsByteSize(directoryURL: URL) throws -> UInt64 {
        let url = directoryURL.appendingPathComponent("model_weights.bin")
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let size = attrs[.size] as? UInt64 else {
            throw ModelError.missingFile(name: "model_weights.bin")
        }
        return size
    }
}
