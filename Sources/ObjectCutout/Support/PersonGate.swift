//
//  PersonGate.swift
//  ObjectCutout
//
//  `--person` with the downloaded model: the model draws the edge (hair,
//  fingers), Vision's person segmentation says where the people are. The
//  model's alpha is kept only inside Vision's mask grown by 5 % of the short
//  side and softened, so a salient lamp beside the person goes and a strand
//  of hair Vision's coarse mask missed stays. Measured on AIM-500 portraits:
//  SAD 7.5 against 24.9 for Vision's person mask alone (crisp edge) and 4.9
//  for the model ungated — the gate costs a little where the true matte
//  includes what a person holds.
//
//  The gate is coarse by nature, so it is built at no more than 512 px on the
//  long side, on plain buffers (testable without a model), and stretched back.
//

import CoreImage
import Foundation

enum PersonGate {

    /// The gate's reach: this share of the picture's short side.
    static let reach = 0.05
    /// Vision's mask counts as a person from this level (0…1).
    static let threshold: Float = 0.1
    /// The long side the gate is built at.
    static let workingSide = 512

    /// The gate over a `width`×`height` person mask (0…1, rows from the
    /// top): on where a person is or within `radius` of one, ramping to off
    /// over about `radius`/2 beyond — a binary dilation, a Gaussian of
    /// sigma `radius`/2 (three box passes), doubled and clamped.
    static func gate(person: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        var on = person.map { $0 > threshold ? Float(1) : 0 }
        on = dilate(on, width: width, height: height, radius: max(1, radius))
        let sigma = Double(max(1, radius)) / 2
        // Three box passes of width w approximate a Gaussian of sigma √((w²−1)/4).
        let box = max(1, Int((sqrt(4 * sigma * sigma + 1)).rounded()) / 2)
        var soft = on
        for _ in 0..<3 { soft = boxBlur(soft, width: width, height: height, radius: box) }
        return soft.map { min(1, $0 * 2) }
    }

    /// Binary dilation by a square of half-side `radius` (separable maximum).
    static func dilate(_ values: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        var rows = [Float](repeating: 0, count: values.count)
        for y in 0..<height {
            for x in 0..<width {
                var m: Float = 0
                for dx in max(0, x - radius)...min(width - 1, x + radius) where values[y * width + dx] > m { m = values[y * width + dx] }
                rows[y * width + x] = m
            }
        }
        var out = [Float](repeating: 0, count: values.count)
        for y in 0..<height {
            for x in 0..<width {
                var m: Float = 0
                for dy in max(0, y - radius)...min(height - 1, y + radius) where rows[dy * width + x] > m { m = rows[dy * width + x] }
                out[y * width + x] = m
            }
        }
        return out
    }

    /// A box blur of half-width `radius`, rows then columns, zero beyond the edge.
    static func boxBlur(_ values: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        let norm = 1 / Float(2 * radius + 1)
        var rows = [Float](repeating: 0, count: values.count)
        for y in 0..<height {
            var sum: Float = 0
            for x in -radius...radius where x >= 0 && x < width { sum += values[y * width + x] }
            for x in 0..<width {
                rows[y * width + x] = sum * norm
                let leaving = x - radius, entering = x + radius + 1
                if leaving >= 0 { sum -= values[y * width + leaving] }
                if entering < width { sum += values[y * width + entering] }
            }
        }
        var out = [Float](repeating: 0, count: values.count)
        for x in 0..<width {
            var sum: Float = 0
            for y in -radius...radius where y >= 0 && y < height { sum += rows[y * width + x] }
            for y in 0..<height {
                out[y * width + x] = sum * norm
                let leaving = y - radius, entering = y + radius + 1
                if leaving >= 0 { sum -= rows[leaving * width + x] }
                if entering < height { sum += rows[entering * width + x] }
            }
        }
        return out
    }

    /// `model` (the subject model's mask) kept where `person` (Vision's)
    /// says people are, both over `extent`.
    static func apply(model: CIImage, person: CIImage, extent: CGRect, context: CIContext) -> CIImage {
        let scale = min(1, CGFloat(workingSide) / max(extent.width, extent.height))
        let width = max(1, Int((extent.width * scale).rounded())), height = max(1, Int((extent.height * scale).rounded()))
        let small = person
            .transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
            .transformed(by: CGAffineTransform(scaleX: CGFloat(width) / extent.width, y: CGFloat(height) / extent.height))
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        context.render(small, toBitmap: &bytes, rowBytes: width * 4, bounds: CGRect(x: 0, y: 0, width: width, height: height),
                       format: .RGBA8, colorSpace: nil)
        let values = (0..<(width * height)).map { Float(bytes[$0 * 4]) / 255 }
        let radius = Int((Double(min(width, height)) * reach).rounded())
        let gate = gate(person: values, width: width, height: height, radius: radius)
        let data = gate.withUnsafeBufferPointer { Data(buffer: $0) }
        let gateImage = CIImage(bitmapData: data, bytesPerRow: width * MemoryLayout<Float>.size,
                                size: CGSize(width: width, height: height), format: .Lf, colorSpace: nil)
            .clampedToExtent()
            .transformed(by: CGAffineTransform(scaleX: extent.width / CGFloat(width), y: extent.height / CGFloat(height)))
            .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
            .cropped(to: extent)
        return model.applyingFilter("CIMultiplyCompositing", parameters: [kCIInputBackgroundImageKey: gateImage])
            .cropped(to: extent)
    }
}
