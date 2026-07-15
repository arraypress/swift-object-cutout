//
//  SAM2Segmenter.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//

import CoreImage
import CoreML
import Foundation

/// Wraps the three SAM 2 (tiny) Core ML models — image encoder, prompt encoder,
/// mask decoder — into a single "click a point, get the object mask" call.
/// Apache-licensed, on-device.
public final class SAM2Segmenter: @unchecked Sendable {

    private let encoder: MLModel
    private let promptEncoder: MLModel
    private let decoder: MLModel
    private let ciContext = CIContext()

    static let inputSize = 1024

    /// Loads the three `.mlpackage`/`.mlmodelc` components.
    public init(encoderURL: URL, promptEncoderURL: URL, decoderURL: URL) throws {
        func load(_ url: URL) throws -> MLModel {
            let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
            let config = MLModelConfiguration()
            config.computeUnits = .all
            return try MLModel(contentsOf: compiled, configuration: config)
        }
        encoder = try load(encoderURL)
        promptEncoder = try load(promptEncoderURL)
        decoder = try load(decoderURL)
    }

    /// Segments the object at `point` (normalized `0…1`, top-left) in `image`.
    /// Returns a soft grayscale mask (`CIImage`, white = object) at the image's
    /// extent, or `nil` on failure.
    public func mask(for image: CIImage, at point: CGPoint) -> CIImage? {
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else { return nil }
        let N = Self.inputSize

        // 1) Encode the image (resized to the model's 1024² input).
        guard let buffer = pixelBuffer(from: image, side: N) else { return nil }
        guard let encOut = try? encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": buffer])),
              let embedding = encOut.featureValue(for: "image_embedding")?.multiArrayValue,
              let s0 = encOut.featureValue(for: "feats_s0")?.multiArrayValue,
              let s1 = encOut.featureValue(for: "feats_s1")?.multiArrayValue
        else { return nil }

        // 2) Encode the point prompt (a single foreground click).
        guard let points = try? MLMultiArray(shape: [1, 1, 2], dataType: .float32),
              let labels = try? MLMultiArray(shape: [1, 1], dataType: .int32) else { return nil }
        points[[0, 0, 0]] = NSNumber(value: Float(point.x) * Float(N))
        points[[0, 0, 1]] = NSNumber(value: Float(point.y) * Float(N))
        labels[[0, 0]] = 1
        guard let pOut = try? promptEncoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["points": points, "labels": labels])),
              let sparse = pOut.featureValue(for: "sparse_embeddings")?.multiArrayValue,
              let dense = pOut.featureValue(for: "dense_embeddings")?.multiArrayValue
        else { return nil }

        // 3) Decode masks and pick the highest-scoring one.
        guard let dOut = try? decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "image_embedding": embedding, "sparse_embedding": sparse, "dense_embedding": dense,
            "feats_s0": s0, "feats_s1": s1,
        ])),
              let masks = dOut.featureValue(for: "low_res_masks")?.multiArrayValue,
              let scores = dOut.featureValue(for: "scores")?.multiArrayValue
        else { return nil }

        var best = 0, bestScore = -Float.greatestFiniteMagnitude
        var all: [Float] = []
        for i in 0..<scores.shape[1].intValue {
            let s = scores[[0, NSNumber(value: i)]].floatValue
            all.append(s)
            if s > bestScore { bestScore = s; best = i }
        }
        if ProcessInfo.processInfo.environment["SAM2_DEBUG"] != nil { print("SAM2 scores \(all) best=\(best)") }
        return maskImage(from: masks, index: best, extent: extent)
    }

    // MARK: - Helpers

    private func pixelBuffer(from image: CIImage, side: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        CVPixelBufferCreate(nil, side, side, kCVPixelFormatType_32BGRA, attrs, &pb)
        guard let buffer = pb else { return nil }
        let sx = CGFloat(side) / image.extent.width, sy = CGFloat(side) / image.extent.height
        let scaled = image
            .transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        ciContext.render(scaled, to: buffer)
        return buffer
    }

    /// One of the 3 low-res mask channels → a soft grayscale `CIImage` at extent.
    private func maskImage(from masks: MLMultiArray, index: Int, extent: CGRect) -> CIImage? {
        let side = masks.shape[2].intValue       // 256
        var bytes = [UInt8](repeating: 0, count: side * side)
        for y in 0..<side {
            for x in 0..<side {
                let logit = masks[[0, NSNumber(value: index), NSNumber(value: y), NSNumber(value: x)]].floatValue
                // Ramp from the decision boundary up — excludes the ambiguous
                // near-zero region (deconv checkerboard) that a soft sigmoid
                // would show as speckle, while keeping a soft object edge.
                let p = max(0, min(1, logit * 0.6))
                bytes[y * side + x] = UInt8(p * 255)
            }
        }
        guard let cg = bytes.withUnsafeMutableBytes({ ptr -> CGImage? in
            guard let ctx = CGContext(data: ptr.baseAddress, width: side, height: side, bitsPerComponent: 8,
                                      bytesPerRow: side, space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
            return ctx.makeImage()
        }) else { return nil }

        let sx = extent.width / CGFloat(side), sy = extent.height / CGFloat(side)
        return CIImage(cgImage: cg)
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
            .cropped(to: extent)
    }
}
