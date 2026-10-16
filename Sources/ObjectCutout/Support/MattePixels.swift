//
//  MattePixels.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//
//  What the subject model is given and what comes back, on plain buffers:
//  the picture resized to the network's square as planar RGB in 0…1, and
//  the square alpha as a mask on the picture's own extent. Separate from the
//  runtime so it is testable without a model.
//

import CoreGraphics
import CoreImage
import Foundation

enum MattePixels {

    /// `image` drawn into a `side`×`side` square (stretched, as the network
    /// was trained), high-quality (area-averaged) resampling, sRGB, as
    /// planar float RGB in 0…1: all red, then all green, then all blue.
    static func planarRGB(of image: CGImage, side: Int) -> [Float]? {
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: side, height: side, bitsPerComponent: 8,
                                          bytesPerRow: side * 4, space: srgb,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
            return true
        }
        guard drawn else { return nil }
        return planar(rgbx: pixels, count: side * side)
    }

    /// RGBX8 (any fourth byte) as planar float RGB in 0…1.
    static func planar(rgbx: [UInt8], count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count * 3)
        let scale: Float = 1 / 255
        rgbx.withUnsafeBufferPointer { source in
            out.withUnsafeMutableBufferPointer { target in
                for i in 0..<count {
                    target[i] = Float(source[i * 4]) * scale
                    target[count + i] = Float(source[i * 4 + 1]) * scale
                    target[2 * count + i] = Float(source[i * 4 + 2]) * scale
                }
            }
        }
        return out
    }

    /// A `side`×`side` alpha (rows from the top, 0…1) as a float grey mask
    /// stretched over `extent` — bilinear, which is how the authors resize
    /// the prediction back.
    static func mask(alpha: [Float], side: Int, extent: CGRect) -> CIImage {
        let clamped = alpha.map { min(max($0, 0), 1) }
        let data = clamped.withUnsafeBufferPointer { Data(buffer: $0) }
        // Rows from the top in memory; Core Image's origin is the bottom left,
        // and a bitmap's first row lands at the top — no flip needed.
        let square = CIImage(bitmapData: data, bytesPerRow: side * MemoryLayout<Float>.size,
                             size: CGSize(width: side, height: side), format: .Lf, colorSpace: nil)
        return square
            .clampedToExtent()
            .transformed(by: CGAffineTransform(scaleX: extent.width / CGFloat(side), y: extent.height / CGFloat(side)))
            .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
            .cropped(to: extent)
    }
}
