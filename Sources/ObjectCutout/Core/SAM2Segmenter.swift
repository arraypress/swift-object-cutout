//
//  SAM2Segmenter.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//
//  "Click a point, get the object": SAM 2.1 tiny (Meta, Apache-2.0) on Core
//  AI, from the downloaded asset (``ClickModel``). The picture is stretched
//  to the network's 1024² square and encoded once; each click on it is a
//  decode of about 5 ms, which returns three candidate masks and the
//  network's own estimate of each one's IoU; the best is stretched back over
//  the picture. Load it once and reuse it: the first load in a program
//  compiles the model for this Mac (cached by the OS after that).
//
//  Compiled only where the SDK has Core AI: the iOS Simulator does not.
//

#if canImport(CoreAI)
import CoreAI
import CoreGraphics
import CoreImage
import Foundation

/// The object under a click, from the downloaded SAM 2.1 tiny.
///
/// ```swift
/// let segmenter = try await SAM2Segmenter.shared()        // ClickModel.installedURL
/// let found = try await segmenter.maskWithScore(for: image, at: CGPoint(x: 0.4, y: 0.6))
/// ```
@available(macOS 27, iOS 27, *)
public final class SAM2Segmenter: Sendable {

    /// The asset this was loaded from.
    public let url: URL

    private let encoder: InferenceFunction
    private let decoder: InferenceFunction
    private let context = CIContext()

    /// One encoded picture: what every click on it is decoded against.
    public struct Encoding: Sendable {
        /// The picture's extent, which the masks are stretched back over.
        public let extent: CGRect
        fileprivate let embedding: NDArray
        fileprivate let featuresS0: NDArray
        fileprivate let featuresS1: NDArray
    }

    /// What one click decodes to: three candidate masks and their scores.
    public struct Prediction: Sendable {
        /// Three `ClickModel.maskSize`² logit maps, one after another, rows from the top.
        public let logits: [Float]
        /// The network's estimate of each mask's IoU.
        public let scores: [Float]
        /// The best-scoring mask's index.
        public var best: Int { ClickMask.best(scores) ?? 0 }
    }

    /// Loads the asset, on the GPU (Core AI's Neural Engine preference
    /// measured up to 90× slower on another fleet model and still ended on
    /// the GPU).
    public init(contentsOf url: URL) async throws(MatteError) {
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent("main.hash").path) else {
            throw .modelUnavailable("no Core AI asset at \(url.path)")
        }
        self.url = url
        do {
            let model = try await AIModel(contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
            guard let encode = try model.loadFunction(named: "encode"), let decode = try model.loadFunction(named: "decode") else {
                throw MatteError.modelUnavailable("\(url.lastPathComponent) lacks 'encode' and 'decode'; it has \(model.functionNames)")
            }
            encoder = encode
            decoder = decode
        } catch let error as MatteError {
            throw error
        } catch {
            throw .modelUnavailable("\(url.lastPathComponent) could not be loaded: \(error)")
        }
    }

    /// The model at `url`, loaded once per program and kept.
    public static func shared(at url: URL = ClickModel.installedURL) async throws(MatteError) -> SAM2Segmenter {
        try await shelf.model(at: url) { try await SAM2Segmenter(contentsOf: $0) }
    }

    private static let shelf = ModelShelf<SAM2Segmenter>()

    // MARK: Encode, decode

    /// Encodes `image` (stretched to the network's square) for any number of clicks.
    public func encode(_ image: CIImage) async throws(MatteError) -> Encoding {
        let side = ClickModel.inputSize
        let extent = image.extent.integral
        guard let picture = MattePixels.render(image, context: context),
              let input = MattePixels.planarRGB(of: picture, side: side) else {
            throw .imageUnreadable("the image could not be rendered")
        }
        var pixels = NDArray(shape: [1, 3, side, side], scalarType: .float32)
        pixels.mutableView(as: Float.self).withUnsafeMutablePointer { pointer, _, _ in
            input.withUnsafeBufferPointer { pointer.update(from: $0.baseAddress!, count: input.count) }
        }
        var embedding = NDArray(shape: [1, 256, 64, 64], scalarType: .float32)
        var s0 = NDArray(shape: [1, 32, 256, 256], scalarType: .float32)
        var s1 = NDArray(shape: [1, 64, 128, 128], scalarType: .float32)
        var views = InferenceFunction.MutableViews()
        views.insert(embedding.mutableRawView(), for: "image_embedding")
        views.insert(s0.mutableRawView(), for: "feats_s0")
        views.insert(s1.mutableRawView(), for: "feats_s1")
        do {
            _ = try await encoder.run(inputs: ["image": pixels], outputViews: consume views)
        } catch {
            throw .inference("encode: \(error)")
        }
        return Encoding(extent: extent, embedding: embedding, featuresS0: s0, featuresS1: s1)
    }

    /// The three masks for a click at `point` (normalised `0…1`, top-left) on an encoded picture.
    public func predict(_ encoding: Encoding, at point: CGPoint) async throws(MatteError) -> Prediction {
        let side = Float(ClickModel.inputSize), maskSide = ClickModel.maskSize
        var click = NDArray(shape: [1, 1, 2], scalarType: .float32)
        click.mutableView(as: Float.self).withUnsafeMutablePointer { pointer, _, _ in
            pointer[0] = Float(point.x) * side
            pointer[1] = Float(point.y) * side
        }
        var label = NDArray(shape: [1, 1], scalarType: .float32)
        label.mutableView(as: Float.self).withUnsafeMutablePointer { pointer, _, _ in pointer[0] = 1 }
        var masks = NDArray(shape: [1, 3, maskSide, maskSide], scalarType: .float32)
        var scores = NDArray(shape: [1, 3], scalarType: .float32)
        var views = InferenceFunction.MutableViews()
        views.insert(masks.mutableRawView(), for: "masks")
        views.insert(scores.mutableRawView(), for: "scores")
        do {
            _ = try await decoder.run(inputs: [
                "image_embedding": encoding.embedding, "feats_s0": encoding.featuresS0, "feats_s1": encoding.featuresS1,
                "point": click, "label": label,
            ], outputViews: consume views)
        } catch {
            throw .inference("decode: \(error)")
        }
        let count = 3 * maskSide * maskSide
        var logits = [Float](repeating: 0, count: count)
        masks.view(as: Float.self).withUnsafePointer { pointer, _, _ in
            logits.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: pointer, count: count) }
        }
        var values = [Float](repeating: 0, count: 3)
        scores.view(as: Float.self).withUnsafePointer { pointer, _, _ in
            values.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: pointer, count: 3) }
        }
        return Prediction(logits: logits, scores: values)
    }

    // MARK: Masks

    /// The best mask for a click on an encoded picture, as a soft grey
    /// `CIImage` over its extent, with the network's IoU estimate.
    public func maskWithScore(for encoding: Encoding, at point: CGPoint) async throws(MatteError) -> (mask: CIImage, score: Float)? {
        let prediction = try await predict(encoding, at: point)
        let best = prediction.best
        if ProcessInfo.processInfo.environment["SAM2_DEBUG"] != nil { print("SAM2 scores \(prediction.scores) best=\(best)") }
        guard let mask = ClickMask.image(logits: prediction.logits, index: best, side: ClickModel.maskSize,
                                         extent: encoding.extent) else { return nil }
        return (mask, prediction.scores[best])
    }

    /// The object at `point` (normalised `0…1`, top-left) in `image`: a soft
    /// grey mask over the image's extent (white = object) and the network's
    /// IoU estimate — trackers use it to notice a lost object.
    public func maskWithScore(for image: CIImage, at point: CGPoint) async throws(MatteError) -> (mask: CIImage, score: Float)? {
        guard image.extent.width >= 1, image.extent.height >= 1, !image.extent.isInfinite else { return nil }
        return try await maskWithScore(for: try await encode(image), at: point)
    }

    /// The object at `point` in `image`, as a soft grey mask over its extent.
    public func mask(for image: CIImage, at point: CGPoint) async throws(MatteError) -> CIImage? {
        try await maskWithScore(for: image, at: point)?.mask
    }

    /// Cuts out the object at `point` (normalised `0…1`, top-left), its edge
    /// finished as `edge` asks (crisp and decontaminated by default).
    public func cutout(from image: CIImage, at point: CGPoint, edge: EdgeOptions = .crisp) async throws(MatteError) -> ObjectCutout.Result? {
        guard let mask = try await mask(for: image, at: point) else { return nil }
        return Matte.cutout(from: image, mask: mask, edge: edge)
    }
}

#endif
