//
//  ClickMask.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//
//  What the click segmenter returns, turned into a mask: the best-scoring of
//  its three 256² logit maps, ramped up from the decision boundary and
//  stretched over the picture. Separate from the runtime so it is testable
//  without a model.
//

import CoreGraphics
import CoreImage
import Foundation

enum ClickMask {

    /// The index of the highest score (the first on a tie), or `nil` for none.
    static func best(_ scores: [Float]) -> Int? {
        scores.indices.max { scores[$0] < scores[$1] || (scores[$0] == scores[$1] && $0 > $1) }
    }

    /// Channel `index` of `logits` (channels of `side`² values, rows from the
    /// top) as a soft grey mask over `extent`.
    ///
    /// The ramp starts at the decision boundary (logit 0) and is fully on by
    /// logit 1.67: it leaves out the ambiguous near-zero region (the deconv
    /// checkerboard), which a sigmoid would show as speckle, and keeps a soft
    /// edge. Lanczos stretches it back — a plain affine scale leaves
    /// stair-steps.
    static func image(logits: [Float], index: Int, side: Int, extent: CGRect) -> CIImage? {
        let count = side * side
        guard index >= 0, logits.count >= (index + 1) * count, extent.width > 0, extent.height > 0 else { return nil }
        var bytes = [UInt8](repeating: 0, count: count)
        logits.withUnsafeBufferPointer { source in
            let base = index * count
            for i in 0..<count { bytes[i] = UInt8(max(0, min(1, source[base + i] * 0.6)) * 255) }
        }
        guard let cg = bytes.withUnsafeMutableBytes({ pointer -> CGImage? in
            CGContext(data: pointer.baseAddress, width: side, height: side, bitsPerComponent: 8, bytesPerRow: side,
                      space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)?.makeImage()
        }) else { return nil }
        let up = CIImage(cgImage: cg).applyingFilter("CILanczosScaleTransform", parameters: [
            "inputScale": extent.height / CGFloat(side), "inputAspectRatio": extent.width / extent.height,
        ])
        return up
            .transformed(by: CGAffineTransform(translationX: extent.minX - up.extent.minX, y: extent.minY - up.extent.minY))
            .cropped(to: extent)
    }
}
