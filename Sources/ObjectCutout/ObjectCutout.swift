//
//  ObjectCutout.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//

import CoreImage
import Foundation

/// Tap a point in an image, get a transparent-background cutout of that object.
/// Wraps ``SAM2Segmenter`` and turns its mask into an alpha cutout plus the
/// tight content rect (for making a sticker).
public struct ObjectCutout: Sendable {

    public let segmenter: SAM2Segmenter
    private let ciContext = CIContext()

    public init(segmenter: SAM2Segmenter) {
        self.segmenter = segmenter
    }

    /// A cutout: the object on transparency (full extent) + its tight bounds.
    public struct Result: Sendable {
        /// The object masked onto a transparent background, at the source extent.
        public let image: CIImage
        /// Tight content rectangle in source pixels (crop here for a sticker).
        public let bounds: CGRect
    }

    /// Cuts out the object at `point` (normalized `0…1`, top-left). `feather`
    /// `nil` = auto (scaled to the frame so the 256² mask edge reads smooth).
    public func cutout(from image: CIImage, at point: CGPoint, feather: CGFloat? = nil) -> Result? {
        guard let mask = segmenter.mask(for: image, at: point) else { return nil }
        return cutout(from: image, mask: mask, feather: feather)
    }

    /// Cuts out using an already-computed `mask` (avoids a second segmentation
    /// when the caller also needs the mask, e.g. for a track centroid).
    public func cutout(from image: CIImage, mask maskInput: CIImage, feather: CGFloat? = nil) -> Result? {
        var mask = maskInput
        let extent = image.extent
        let bounds = contentBounds(of: mask, extent: extent) ?? extent

        // Auto feather ~ one model-mask-pixel at frame resolution — enough to
        // blend the upscaled edge without eating the object.
        let radius = feather ?? max(1.5, extent.width / 256 * 1.2)
        if radius > 0 {
            mask = mask.clampedToExtent()
                .applyingFilter("CIGaussianBlur", parameters: ["inputRadius": radius])
                .cropped(to: extent)
        }
        let clear = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0)).cropped(to: extent)
        let cut = image.applyingFilter("CIBlendWithMask", parameters: [
            "inputBackgroundImage": clear, "inputMaskImage": mask,
        ]).cropped(to: extent)
        return Result(image: cut, bounds: bounds)
    }

    /// The tight bounds of the mask's content, found by scanning a small render.
    private func contentBounds(of mask: CIImage, extent: CGRect) -> CGRect? {
        let side = 128
        var buf = [UInt8](repeating: 0, count: side * side * 4)
        let scaled = mask
            .transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
            .transformed(by: CGAffineTransform(scaleX: CGFloat(side) / extent.width, y: CGFloat(side) / extent.height))
        ciContext.render(scaled, toBitmap: &buf, rowBytes: side * 4,
                         bounds: CGRect(x: 0, y: 0, width: side, height: side),
                         format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())

        var minX = side, minY = side, maxX = -1, maxY = -1
        for y in 0..<side {
            for x in 0..<side where buf[(y * side + x) * 4] > 128 {   // R channel = mask value
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
            }
        }
        guard maxX >= minX, maxY >= minY else { return nil }
        // The render bitmap is top-left; map back to the image's bottom-left extent.
        let fx = extent.width / CGFloat(side), fy = extent.height / CGFloat(side)
        let x = extent.minX + CGFloat(minX) * fx
        let w = CGFloat(maxX - minX + 1) * fx
        let h = CGFloat(maxY - minY + 1) * fy
        let yTop = CGFloat(minY) * fy
        let y = extent.minY + extent.height - yTop - h
        return CGRect(x: x, y: y, width: w, height: h).intersection(extent)
    }
}
